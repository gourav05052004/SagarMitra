"""Fine-resolution, class-specific residual localization; original detector frozen."""
import torch
from torch import nn
from mmdet.models.builder import HEADS, build_roi_extractor
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class DetailCascadeRoIHead(CascadeRoIHead):
    def __init__(self, *args, detail_max_side=192., detail_cap=2., **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_stages == 3 and self.bbox_head[-1].num_classes == 5
        assert not self.bbox_head[-1].reg_class_agnostic
        self.detail_max_side = float(detail_max_side)
        self.detail_cap = float(detail_cap)
        self.detail_enabled = True
        self.detail_project = nn.Conv2d(256, 32, 1)
        self.detail_pool = build_roi_extractor(dict(type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=32, featmap_strides=[4]))
        self.detail_predict = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,groups=8), nn.ReLU(), nn.AvgPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7,128), nn.ReLU(), nn.Linear(128,8))
        nn.init.zeros_(self.detail_predict[-1].weight)
        nn.init.zeros_(self.detail_predict[-1].bias)

    def _bbox_forward(self, stage, x, rois):
        output = super()._bbox_forward(stage,x,rois)
        if stage != self.num_stages-1 or not self.detail_enabled:
            return output
        # P2 already exists at stride 4. Preserve it and sample its detail more
        # finely; this does not invent additional pixels or add a new backbone.
        features = self.detail_project(x[0].detach())
        tight = self.detail_pool([features],rois)
        context_rois = self.detail_pool.roi_rescale(rois,1.5)
        context = self.detail_pool([features],context_rois)
        correction = self.detail_cap * torch.tanh(self.detail_predict(torch.cat([tight,context],dim=1)))
        sizes = (rois[:,3:5]-rois[:,1:3]).clamp(min=0)
        eligible = (sizes.max(dim=1).values <= self.detail_max_side).to(correction.dtype)
        correction = correction.reshape(-1,2,4)*eligible[:,None,None]
        boxes = output['bbox_pred'].reshape(-1,5,4).clone()
        # Category IDs 1 and 4 are class indices 0 and 3. Other regression
        # columns and all classification logits remain those of the source.
        boxes[:,0,:] = boxes[:,0,:]+correction[:,0,:]
        boxes[:,3,:] = boxes[:,3,:]+correction[:,1,:]
        output['bbox_pred'] = boxes.reshape(-1,20)
        return output
