"""Three independent RGB/gradient boundary-distribution heads."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

_spec=importlib.util.spec_from_file_location('pixel_box_geometry',Path(__file__).resolve().parent.parent/'Maritime_CBNet62_PrecisionBox_backend/box_core.py')
geometry=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(geometry)
TARGETS=geometry.TARGETS
apply_boxes=geometry.apply_boxes
decode=geometry.decode
aligned_iou=geometry.aligned_iou
objective=geometry.objective


class PrecisionHead(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        # Preserve constructor/state format used by the verified training controller.
        self.register_buffer('mean',torch.as_tensor(mean).float())
        self.register_buffer('std',torch.as_tensor(std).float().clamp_min(.05))
        self.register_buffer('positions',torch.linspace(-1,1,9))
        self.heads=nn.ModuleList(nn.Sequential(nn.Conv2d(7,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv2d(16,16,3,padding=1,groups=4),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv2d(16,1,1)) for _ in TARGETS)
        for head in self.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def forward(self,visual,meta,category,boxes):
        if visual.shape[1:]!=(4,6,24,9):raise ValueError('Expected N x 4 x 6 x 24 x 9 boundary profiles')
        output=visual.new_zeros((len(visual),4),dtype=torch.float32)
        side=torch.arange(4,device=visual.device,dtype=torch.float32).view(1,4,1,1,1)/3
        for c,head in zip(TARGETS,self.heads):
            use=category==c
            if not use.any():continue
            x=visual[use].float(); side_channel=side.expand(len(x),-1,1,24,9)
            x=torch.cat([x,side_channel],2).reshape(-1,7,24,9)
            logits=head(x).mean(2).reshape(-1,4,9)
            probability=logits.softmax(-1)
            # Subtract exact uniform probability: zero logits give exact zero shift.
            shift=((probability-torch.full_like(probability,1/9))*self.positions).sum(-1)*.06
            left,top,right,bottom=shift.unbind(1)
            output[use]=torch.stack([(left+right)*.5,(top+bottom)*.5,
                torch.log1p(right-left),torch.log1p(bottom-top)],1)
        return output
