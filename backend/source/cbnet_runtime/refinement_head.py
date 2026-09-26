"""Zero-start regression-only residual; original classification is unchanged."""
import torch
from torch import nn
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead

@HEADS.register_module()
class RegressionRefinementHead(ConvFCBBoxHead):
    def __init__(self, *args, deform_indices=None, nwd_weight=0., nwd_constant=16., **kwargs):
        super().__init__(*args, **kwargs)
        self.nwd_weight = float(nwd_weight)
        self.nwd_constant = float(nwd_constant)
        if self.nwd_constant <= 0:
            raise ValueError('nwd_constant must be positive')
        self.reg_refine = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
            nn.Linear(128*9, self.fc_reg.out_features))
        self.reset_refinement()

    def reset_refinement(self):
        nn.init.zeros_(self.reg_refine[-1].weight)
        nn.init.zeros_(self.reg_refine[-1].bias)

    def init_weights(self):
        super().init_weights()
        self.reset_refinement()

    def forward(self, x):
        cls, boxes = super().forward(x)
        return cls, boxes + self.reg_refine(x)

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights,
             bbox_targets, bbox_weights, reduction_override=None):
        losses = super().loss(cls_score, bbox_pred, rois, labels, label_weights,
                              bbox_targets, bbox_weights, reduction_override)
        if not self.nwd_weight:
            return losses
        if not self.reg_decoded_bbox or self.reg_class_agnostic:
            raise ValueError('NWD experiment requires decoded, class-specific regression')
        widths = (bbox_targets[:, 2]-bbox_targets[:, 0]).clamp(min=0)
        heights = (bbox_targets[:, 3]-bbox_targets[:, 1]).clamp(min=0)
        # Class indices 0/3 correspond to swimmer / life saving appliance.
        # Small is defined in resized training coordinates here, not COCO original area.
        selected = ((labels == 0) | (labels == 3)) & (widths > 0) & (heights > 0)
        selected &= (torch.minimum(widths, heights) < 32)
        if selected.any():
            with torch.cuda.amp.autocast(enabled=False):
                decoded = self.bbox_coder.decode(rois[:, 1:].float(), bbox_pred.float())
                boxes = decoded.reshape(-1, self.num_classes, 4)[selected, labels[selected]]
                target = bbox_targets[selected].float()
                centers = (boxes[:, :2]+boxes[:, 2:])/2
                target_centers = (target[:, :2]+target[:, 2:])/2
                sizes = boxes[:, 2:]-boxes[:, :2]
                target_sizes = target[:, 2:]-target[:, :2]
                distance = ((centers-target_centers).square().sum(1) +
                            .25*(sizes-target_sizes).square().sum(1) + 1e-9).sqrt()
                losses['loss_nwd_aux'] = self.nwd_weight * (1-torch.exp(-distance/self.nwd_constant)).mean()
        else:
            losses['loss_nwd_aux'] = bbox_pred.sum()*0
        return losses
