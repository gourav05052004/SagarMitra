"""Optional localization-quality head for the final Cascade stage."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox_overlaps, multiclass_nms
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class QualityConvFCBBoxHead(ConvFCBBoxHead):
    """ConvFC head that learns class-specific IoU and uses it for ranking.

    Zero initialization makes the initial quality constant, preserving the
    within-class classification ranking before the new layer is trained.
    """
    def __init__(self, quality_alpha=.35, quality_loss_weight=4., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_alpha = float(quality_alpha)
        self.quality_loss_weight = float(quality_loss_weight)
        self.fc_quality = nn.Linear(self.reg_last_dim, self.num_classes)
        self._quality_pred = None
        if not self.reg_decoded_bbox or self.reg_class_agnostic:
            raise ValueError('Quality head requires decoded targets and class-specific boxes')

    def init_weights(self):
        super().init_weights()
        nn.init.constant_(self.fc_quality.weight, 0.)
        nn.init.constant_(self.fc_quality.bias, 0.)

    def forward(self, x):
        for conv in self.shared_convs:
            x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        x_cls = x
        x_reg = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        self._quality_pred = self.fc_quality(x_reg)
        return cls_score, bbox_pred

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights,
             bbox_targets, bbox_weights, reduction_override=None):
        losses = super().loss(cls_score, bbox_pred, rois, labels, label_weights,
                              bbox_targets, bbox_weights, reduction_override)
        pos = (labels >= 0) & (labels < self.num_classes)
        if pos.any():
            decoded = self.bbox_coder.decode(rois[:, 1:].float(), bbox_pred.detach().float())
            selected = decoded.view(decoded.size(0), -1, 4)[pos, labels[pos]]
            targets = bbox_targets[pos]
            quality_target = bbox_overlaps(selected.float(), targets.float(),
                                           is_aligned=True).clamp_(0, 1).detach()
            quality_pred = self._quality_pred[pos, labels[pos]].float()
            losses['loss_quality'] = F.binary_cross_entropy_with_logits(
                quality_pred, quality_target, reduction='mean') * self.quality_loss_weight
            losses['quality_target'] = quality_target.mean()
        else:
            losses['loss_quality'] = self._quality_pred.sum() * 0
        neg = labels == self.num_classes
        if neg.any():
            # Teach the most likely foreground class on background RoIs low quality.
            chosen = cls_score[neg, :self.num_classes].detach().argmax(dim=1)
            background = self._quality_pred[neg].float().gather(1, chosen[:, None]).flatten()
            losses['loss_quality_background'] = .25 * self.quality_loss_weight * (
                F.binary_cross_entropy_with_logits(background, torch.zeros_like(background)))
        return losses

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor,
                   rescale=False, cfg=None):
        scores = F.softmax(cls_score, dim=-1)
        if self._quality_pred is not None and self.quality_alpha > 0:
            quality = self._quality_pred.sigmoid().to(scores.dtype)
            foreground = scores[:, :-1].clamp_min(1e-8).pow(1-self.quality_alpha)
            foreground = foreground * quality.clamp_min(1e-8).pow(self.quality_alpha)
            scores = torch.cat([foreground, scores[:, -1:]], dim=1)
        bboxes = self.bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=img_shape)
        if rescale and bboxes.size(0) > 0:
            factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / factor).view(bboxes.size(0), -1)
        if cfg is None:
            return bboxes, scores
        return multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
