"""Bounded scale-conditioned corrections of final fused boxes, scores frozen."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TARGETS = (1, 3, 4)


class PrecisionHead(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).float())
        self.register_buffer('std', torch.as_tensor(std).float().clamp_min(.05))
        self.register_buffer('cap', torch.tensor([.05, .05, .08, .08]))
        self.local = nn.Sequential(nn.LayerNorm(1152), nn.Linear(1152, 64), nn.SiLU())
        self.context = nn.Sequential(nn.LayerNorm(1152), nn.Linear(1152, 64), nn.SiLU())
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(196, 64), nn.SiLU(), nn.Linear(64, 8)) for _ in TARGETS)
        for head in self.heads:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self, visual, meta, category, boxes):
        x = torch.cat([self.local(visual[:, :1152].float()), self.context(visual[:, 1152:].float()),
            ((meta.float()-self.mean)/self.std).clamp(-8, 8)], 1)
        sizes = (boxes[:, 2:]-boxes[:, :2]).clamp_min(1)
        # Deterministic smooth size gate; tiny and larger objects learn separate outputs.
        tiny = torch.sigmoid((torch.log(boxes.new_tensor(32.))-torch.log(sizes.prod(1).sqrt()))*3)[:, None]
        output = x.new_zeros((len(x), 4))
        for c, head in zip(TARGETS, self.heads):
            keep = category == c
            if keep.any():
                pair = head(x[keep]).reshape(-1, 2, 4)
                output[keep] = (tiny[keep]*torch.tanh(pair[:, 0])+(1-tiny[keep])*torch.tanh(pair[:, 1]))*self.cap
        return output


def decode(boxes, delta):
    size = (boxes[:, 2:]-boxes[:, :2]).clamp_min(1e-4)
    center = (boxes[:, 2:]+boxes[:, :2])*.5+delta[:, :2]*size
    adjusted = size*delta[:, 2:].exp()
    return torch.cat([center-adjusted*.5, center+adjusted*.5], 1)


def aligned_iou(a, b):
    inter = (torch.minimum(a[:, 2:], b[:, 2:])-torch.maximum(a[:, :2], b[:, :2])).clamp_min(0).prod(1)
    union = (a[:, 2:]-a[:, :2]).clamp_min(0).prod(1)+(b[:, 2:]-b[:, :2]).clamp_min(0).prod(1)-inter
    iou = inter/union.clamp_min(1e-7)
    enclosing = (torch.maximum(a[:, 2:], b[:, 2:])-torch.minimum(a[:, :2], b[:, :2])).clamp_min(0).prod(1)
    return iou, iou-(enclosing-union)/enclosing.clamp_min(1e-7)


def objective(model, b):
    delta = model(b['visual'], b['meta'], b['category'], b['boxes'])
    boxes = decode(b['boxes'], delta); pos = b['positive']
    loss = delta.square().mean()*.02
    if pos.any():
        iou, giou = aligned_iou(boxes[pos], b['targets'][pos])
        size = (b['targets'][pos, 2:]-b['targets'][pos, :2]).clamp_min(2)
        scale = torch.cat([size, size], 1)
        edges = F.smooth_l1_loss((boxes[pos]-b['targets'][pos])/scale,
            torch.zeros_like(boxes[pos]), beta=.02, reduction='none').mean(1)
        # High-overlap boxes retain explicit pressure on sub-box boundary error.
        loss = loss + (1-giou+2*edges).mean()
    if (~pos).any(): loss = loss + 2*delta[~pos].square().mean()
    return loss


def apply_boxes(rows, indices, delta, strengths, width, height):
    delta = np.asarray(delta, dtype=np.float32)
    if delta.shape != (len(indices), 4) or not np.isfinite(delta).all(): raise ValueError('Invalid box corrections')
    output = copy.deepcopy(rows)
    for j, d in zip(indices, delta):
        row = output[int(j)]; strength = float(strengths.get(str(row['category_id']), 0.))
        if not strength or not np.any(d): continue
        if row['category_id'] not in TARGETS: raise ValueError('Protected category correction')
        x,y,w,h = row['bbox']; d = d*strength
        cx,cy = x+w*.5+d[0]*w, y+h*.5+d[1]*h
        nw,nh = w*np.exp(d[2]), h*np.exp(d[3])
        left,top = np.clip(cx-nw*.5,0,width), np.clip(cy-nh*.5,0,height)
        right,bottom = np.clip(cx+nw*.5,0,width), np.clip(cy+nh*.5,0,height)
        if right-left >= 1 and bottom-top >= 1:
            row['bbox'] = [float(left),float(top),float(right-left),float(bottom-top)]
    return output
