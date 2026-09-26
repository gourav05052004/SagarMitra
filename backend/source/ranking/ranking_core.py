"""Frozen-box scoring head. Inference features never use labels or image IDs."""
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TARGETS = (1, 3, 4)
THRESHOLDS = np.arange(.50, .951, .05, dtype=np.float64)


def boxes(rows):
    b = np.asarray([r['bbox'] for r in rows], np.float64).reshape(-1, 4)
    b[:, 2:] += b[:, :2]
    return b


def overlap(a, b, crowd=False):
    wh = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    np.maximum(a[:, None, :2], b[None, :, :2]), 0.)
    inter = wh.prod(2)
    aa = np.maximum(a[:, 2:]-a[:, :2], 0).prod(1)
    bb = np.maximum(b[:, 2:]-b[:, :2], 0).prod(1)
    denom = aa[:, None] if crowd else aa[:, None]+bb[None, :]-inter
    return inter / np.maximum(denom, 1e-12)


def eligible(rows, score_min=.001, max_side=1024):
    return [j for j, r in enumerate(rows) if r['category_id'] in TARGETS
            and r['score'] >= score_min and min(r['bbox'][2:]) >= 2
            and max(r['bbox'][2:]) <= max_side]


def metadata(rows, sources, image, indices):
    """Geometry, local competitors and cross-view agreement; no annotations/meta."""
    all_b = boxes(rows)
    all_s = np.asarray([r['score'] for r in rows], np.float64)
    all_c = np.asarray([r['category_id'] for r in rows])
    result = []
    for j in indices:
        b, s, cat = all_b[j], all_s[j], all_c[j]
        wh = np.maximum(b[2:]-b[:2], 1.)
        same = (all_c == cat); same[j] = False
        other = all_c != cat
        o = overlap(b[None], all_b)[0]
        ss = np.clip(s, 1e-5, 1-1e-5)
        v = [np.log(ss/(1-ss)), np.log(wh[0]), np.log(wh[1]),
             np.log(wh[0]/wh[1]), wh[0]/image['width'], wh[1]/image['height'],
             min(b[0], image['width']-b[2])/wh[0],
             min(b[1], image['height']-b[3])/wh[1]]
        for threshold in (.3, .5, .7, .9):
            mask = same & (o >= threshold)
            strengths = all_s[mask]
            v.extend([np.log1p(mask.sum()), np.log1p((strengths > s).sum()),
                      np.log((strengths.max(initial=0)+1e-5)/(s+1e-5)),
                      float(o[mask].max(initial=0))])
        for category in (1, 2, 3, 4, 5):
            mask = other & (all_c == category) & (o >= .3)
            v.extend([float(all_s[mask].max(initial=0)), float(o[mask].max(initial=0))])
        # Separate full-image observations from tiles. View names/counts are not IDs.
        for full in (True, False):
            votes, strengths, ious = [], [], []
            for name, items in sorted(sources.items()):
                if (name == 'full') != full:
                    continue
                same_rows = [r for r in items if r['category_id'] == cat]
                if not same_rows:
                    continue
                view_b = boxes(same_rows)
                io = overlap(b[None], view_b)[0]
                k = int(io.argmax())
                if io[k] < .3:
                    continue
                votes.append((view_b[k]-b)/np.tile(wh, 2))
                strengths.append(float(same_rows[k]['score']))
                ious.append(float(io[k]))
            v.append(np.log1p(len(votes)))
            if votes:
                votes = np.asarray(votes); io = np.asarray(ious)
                st = np.asarray(strengths)
                v.extend([st.max(), st.mean(), st.std(), io.max(), io.mean(),
                          np.mean(io >= .5), np.mean(io >= .75), np.mean(io >= .9)])
                v.extend(votes.mean(0).tolist()); v.extend(votes.std(0).tolist())
            else:
                v.extend([0.]*16)
        result.append(v)
    return np.clip(np.asarray(result, np.float32).reshape(len(indices), 68), -20, 20)


def training_targets(rows, annotations, indices):
    """Ten IoU-threshold targets, assigning the best-localized unique candidates.

    This is a surrogate for ranking, not differentiable COCO AP. Duplicate
    candidates compete for each GT independently at each threshold. All ambiguous
    boxes are retained; only crowd/ignored-region matches are masked from loss.
    """
    labels = np.zeros((len(rows), 10), np.float32)
    all_b = boxes(rows)
    for category in TARGETS:
        loc = [j for j, r in enumerate(rows) if r['category_id'] == category]
        gt = [a for a in annotations if a['category_id'] == category
              and not a.get('iscrowd', 0) and not a.get('ignore', 0)]
        ignored = [a for a in annotations if a['category_id'] == category
                   and (a.get('iscrowd', 0) or a.get('ignore', 0))]
        if not loc:
            continue
        m = overlap(all_b[loc], boxes(gt))
        pairs = [(float(m[j, k]), j, k) for j, k in zip(*np.where(m >= .5-1e-9))]
        pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
        for t, threshold in enumerate(THRESHOLDS):
            used_rows, used_gt = set(), set()
            for value, j, k in pairs:
                if value+1e-9 < threshold:
                    break
                if j not in used_rows and k not in used_gt:
                    labels[loc[j], t] = 1.; used_rows.add(j); used_gt.add(k)
            if ignored:
                hit = overlap(all_b[loc], boxes(ignored), crowd=True).max(1) >= threshold
                for j in np.where(hit)[0]:
                    if j not in used_rows:
                        labels[loc[j], t] = -1.
    return labels[indices]


class ScoringHead(nn.Module):
    """Three independent heads; original box coordinates/classes remain frozen."""
    def __init__(self, mean, std, cap=2.0):
        super().__init__()
        self.register_buffer('meta_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('meta_std', torch.as_tensor(std, dtype=torch.float32).clamp_min(.05))
        self.cap = float(cap)
        self.visual_norm = nn.LayerNorm(2304, elementwise_affine=False)
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(2304+68, 64), nn.SiLU(),
                                                nn.Linear(64, 10)) for _ in TARGETS)
        for head in self.heads:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self, visual, meta, category, scores):
        x = torch.cat([self.visual_norm(visual.float()),
                       ((meta.float()-self.meta_mean)/self.meta_std).clamp(-8, 8)], 1)
        delta = x.new_zeros((len(x), 10))
        for j, category_id in enumerate(TARGETS):
            mask = category == category_id
            if mask.any():
                delta[mask] = self.cap*torch.tanh(self.heads[j](x[mask]))
        logits = torch.logit(scores.float().clamp(1e-5, 1-1e-5))[:, None]+delta
        return logits, delta.mean(1)


def objective(model, batch):
    logits, delta = model(batch['visual'], batch['meta'], batch['category'], batch['scores'])
    target = batch['target']; valid = target >= 0
    bce = F.binary_cross_entropy_with_logits(logits, target.clamp_min(0), reduction='none')
    # Preserve class-balanced calibration without discarding difficult negatives.
    loss = (bce*valid).sum()/valid.sum().clamp_min(1)
    mean_target = (target.clamp_min(0)*valid).sum(1)/valid.sum(1).clamp_min(1)
    rank_logit = logits.mean(1)
    pair_losses = []
    for category in TARGETS:
        ids = torch.where((batch['category'] == category) & valid.any(1))[0]
        if len(ids) < 2:
            continue
        # Deterministic cross-image pairs; no RNG state hidden in the loss function.
        right = ids.roll(1)
        difference = mean_target[ids]-mean_target[right]
        keep = difference.abs() >= .2
        if keep.any():
            margin = (rank_logit[ids]-rank_logit[right])*difference.sign()
            pair_losses.append((F.softplus(-margin[keep])*difference[keep].abs()).mean())
    if pair_losses:
        loss = loss+.1*torch.stack(pair_losses).mean()
    # Probabilities should not rise as the required IoU becomes stricter.
    loss = loss+.02*F.relu(logits[:, 1:]-logits[:, :-1]).mean()+.005*delta.square().mean()
    return loss


def apply_scores(rows, indices, deltas, blends):
    output = copy.deepcopy(rows)
    if len(indices) != len(deltas) or not np.isfinite(deltas).all():
        raise ValueError('Invalid scoring-head outputs.')
    for j, delta in zip(indices, deltas):
        r = output[j]; strength = float(blends.get(str(r['category_id']), 0.))
        if not strength or float(delta) == 0.:
            continue  # Exact identity, including original score ties.
        if r['category_id'] not in TARGETS:
            raise ValueError('Attempted to change a protected category.')
        s = np.clip(r['score'], 1e-5, 1-1e-5)
        r['score'] = float(1/(1+np.exp(-(np.log(s/(1-s))+strength*float(delta)))))
    return output
