"""Spatial, class-conditioned edge distributions around a frozen native refiner.

This is a custom residual refinement architecture, not an implementation of GFL.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss
import native_core as c


def deployed_edges(raw, blend=.5):
    """xyxy in the normalized 2x-context crop, for the deployed correction."""
    d = raw.float() * blend
    center = .5 + .5 * d[:, :2]
    size = .5 * d[:, 2:].exp()
    return torch.cat((center - size / 2, center + size / 2), 1)


def encode_edges(edges):
    center = (edges[:, :2] + edges[:, 2:]) / 2
    size = (edges[:, 2:] - edges[:, :2]).clamp_min(1e-5)
    return torch.cat(((center - .5) / .5, (size / .5).log()), 1)


def distribution_loss(logits, target):
    """Linear interpolation between adjacent coordinate bins; all four edges."""
    bins = logits.shape[-1]
    position = target.clamp(0, 1) * (bins - 1)
    left = position.floor().long().clamp(max=bins-2)
    weight = position - left
    logp = F.log_softmax(logits.float(), dim=-1)
    low = logp.gather(-1, left[..., None]).squeeze(-1)
    high = logp.gather(-1, (left+1)[..., None]).squeeze(-1)
    return -((1-weight)*low + weight*high).mean()


class BoundaryRefiner(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        if checkpoint is None: checkpoint = c.read(c.ROOT/'reference.json')['checkpoint']
        self.anchor = c.NativeRefiner()
        state = torch.load(checkpoint, map_location='cpu')['state_dict']
        self.anchor.load_state_dict(state, strict=True)
        self.anchor.requires_grad_(False)
        self.laterals = nn.ModuleList(nn.Conv2d(n, 64, 1) for n in (64, 128, 256))
        self.class_embed = nn.Embedding(2, 64)
        self.spatial = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8,64), nn.SiLU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8,64), nn.SiLU())
        # 48x48 spatial features -> four 48-bin edge distributions (x1,y1,x2,y2).
        self.edge_logits = nn.Conv2d(64, 4, 1)
        nn.init.zeros_(self.edge_logits.weight)
        nn.init.zeros_(self.edge_logits.bias)
        self.register_buffer('bins', torch.linspace(0, 1, 48))
        self.eval()

    def forward(self, images, classes, return_aux=False):
        # The successful previous model is a fixed reference, including BatchNorm.
        self.anchor.eval()
        with torch.no_grad():
            p1 = self.anchor.stem(images)
            p2 = self.anchor.layer2(p1)
            p3 = self.anchor.layer3(p2)
            vector = F.adaptive_avg_pool2d(p3, (6,6)).flatten(1)
            out = self.anchor.head(torch.cat((vector, self.anchor.class_embed(classes)), 1))
            old_raw = torch.tanh(out[:, :4]) * out.new_tensor([.25,.25,.4,.4])
            quality = out[:, 4]
        feat = self.laterals[0](p1)
        for lateral, level in zip(self.laterals[1:], (p2,p3)):
            feat = feat + F.interpolate(lateral(level), size=p1.shape[-2:], mode='bilinear', align_corners=False)
        feat = feat + self.class_embed(classes)[:, :, None, None]
        maps = self.edge_logits(self.spatial(feat)).float()
        spatial_logits = torch.stack((maps[:,0].mean(1), maps[:,1].mean(2),
                                      maps[:,2].mean(1), maps[:,3].mean(2)), 1)
        # Float32 geometry also under AMP. Zero final weights make this an exact
        # residual identity, including the previous quality gate and blend.
        base_edges = deployed_edges(old_raw)
        prior = -.5 * ((self.bins[None,None,:] - base_edges[:,:,None]) / .07).square()
        logits = prior + spatial_logits
        prior_mean = (prior.softmax(-1) * self.bins).sum(-1)
        mean = (logits.softmax(-1) * self.bins).sum(-1)
        shift = .08 * torch.tanh((mean - prior_mean) / .08)
        new_edges = base_edges + shift
        raw = old_raw.float() + (encode_edges(new_edges) - encode_edges(base_edges)) / .5
        if not return_aux: return raw, quality
        return raw, quality, dict(logits=logits, old_raw=old_raw.float(), shift=shift)


def objective(raw, quality, aux, boxes, targets, positive, dfl_weight=.15):
    """Train the actual blend=.5, quality-gated geometry used in validation."""
    deployed = raw.float() * .5
    accepted = quality.float().sigmoid() >= .5
    good = positive & accepted
    loss = raw.sum()*0
    reg = loss; dfl = loss; preserve = loss
    if good.any():
        reg = F.smooth_l1_loss(deployed[good], c.encode(boxes[good], targets[good]), beta=.03)
        reg = reg + 2*generalized_box_iou_loss(c.decode(boxes[good], deployed[good]), targets[good], reduction='mean')
        size = (boxes[:,2:]-boxes[:,:2]).clamp_min(2.)
        left = (boxes[:,:2]+boxes[:,2:])/2-size
        target_edges = (targets-torch.cat((left,left),1))/torch.cat((2*size,2*size),1)
        dfl = distribution_loss(aux['logits'][good], target_edges[good])
    if (~positive & accepted).any():
        preserve = aux['shift'][~positive & accepted].square().mean()
    loss = reg + dfl_weight*dfl + 5*preserve
    return loss, dict(regression=float(reg.detach()), boundary_distribution=float(dfl.detach()),
                      negative_preservation=float(preserve.detach()), accepted_positives=int(good.sum()))


@torch.no_grad()
def update_ema(model, ema, decay):
    # EMA arithmetic on frozen weights causes rounding drift; copy them exactly.
    source = model.state_dict()
    trainable = {key for key, value in model.named_parameters() if value.requires_grad}
    for key, value in ema.state_dict().items():
        if key not in trainable or not value.is_floating_point(): value.copy_(source[key])
        else: value.mul_(decay).add_(source[key], alpha=1-decay)
