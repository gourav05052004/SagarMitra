"""Checkpoint-preserving deformable refinement inside the Cascade ROI heads.

Adaptation of modulated deformable convolution, not a reproduction of DCNv2's
full training recipe. Mask = 2*sigmoid(logit) preserves ordinary convolution at
zero initialization. All existing conv/norm/FC parameter paths are unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead


class IdentityStartDeformConv(ModulatedDeformConv2d):
    def __init__(self, conv):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None)
        k = self.kernel_size[0]*self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels, 3*k, kernel_size=3,padding=1)
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if conv.bias is not None: self.bias.copy_(conv.bias)
        self.reset_offsets()

    def reset_offsets(self):
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self,x):
        dtype=x.dtype
        # Explicit float32: MMCV CUDA DCNv2 half support varies across builds.
        with torch.cuda.amp.autocast(enabled=False):
            x=x.float()
            prediction=F.conv2d(x,self.conv_offset.weight.float(),self.conv_offset.bias.float(),padding=1)
            ox,oy,logit=prediction.chunk(3,dim=1)
            offset=torch.cat((ox,oy),dim=1)
            mask=2*logit.sigmoid()
            out=modulated_deform_conv2d(x,offset,mask,self.weight.float(),
                self.bias.float() if self.bias is not None else None,
                self.stride,self.padding,self.dilation,self.groups,self.deform_groups)
        return out.to(dtype)


@HEADS.register_module()
class DeformableConvFCBBoxHead(ConvFCBBoxHead):
    def __init__(self,*args,deform_indices=(3,),**kwargs):
        super().__init__(*args,**kwargs)
        self.deform_indices=tuple(deform_indices)
        for index in self.deform_indices:
            self.shared_convs[index].conv=IdentityStartDeformConv(self.shared_convs[index].conv)

    def init_weights(self):
        super().init_weights()
        for module in self.modules():
            if isinstance(module,IdentityStartDeformConv): module.reset_offsets()
