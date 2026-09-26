"""Independent target-class heads initialized from the measured boundary winner."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
import native_core as c
from boundary_model import BoundaryRefiner, deployed_edges, encode_edges


class AdaptiveBoundary(BoundaryRefiner):
    def __init__(self, checkpoint=None):
        super().__init__()
        if checkpoint is None:checkpoint=c.read(c.ROOT/'boundary_reference.json')['checkpoint']
        self.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True)
        self.appliance_laterals=copy.deepcopy(self.laterals)
        self.appliance_class_embed=copy.deepcopy(self.class_embed)
        self.appliance_spatial=copy.deepcopy(self.spatial)
        self.appliance_edge_logits=copy.deepcopy(self.edge_logits)
        self.eval()

    def forward(self, images, classes, return_aux=False):
        self.anchor.eval()
        with torch.no_grad():
            p1=self.anchor.stem(images);p2=self.anchor.layer2(p1);p3=self.anchor.layer3(p2)
            vector=F.adaptive_avg_pool2d(p3,(6,6)).flatten(1)
            out=self.anchor.head(torch.cat((vector,self.anchor.class_embed(classes)),1))
            old_raw=torch.tanh(out[:,:4])*out.new_tensor([.25,.25,.4,.4]);quality=out[:,4]
        maps=None
        for cls,prefix in ((0,''),(1,'appliance_')):
            if not (classes==cls).any():continue
            laterals=getattr(self,prefix+'laterals')
            feat=laterals[0](p1)
            for lateral,level in zip(laterals[1:],(p2,p3)):
                feat=feat+F.interpolate(lateral(level),size=p1.shape[-2:],mode='bilinear',align_corners=False)
            feat=feat+getattr(self,prefix+'class_embed')(classes)[:,:,None,None]
            values=getattr(self,prefix+'edge_logits')(getattr(self,prefix+'spatial')(feat)).float()
            # Full-batch arithmetic preserves the original head's AMP numerics.
            mask=(classes==cls)[:,None,None,None]
            maps=torch.where(mask,values,torch.zeros_like(values) if maps is None else maps)
        spatial_logits=torch.stack((maps[:,0].mean(1),maps[:,1].mean(2),maps[:,2].mean(1),maps[:,3].mean(2)),1)
        base_edges=deployed_edges(old_raw)
        prior=-.5*((self.bins[None,None,:]-base_edges[:,:,None])/.07).square()
        logits=prior+spatial_logits
        prior_mean=(prior.softmax(-1)*self.bins).sum(-1)
        mean=(logits.softmax(-1)*self.bins).sum(-1)
        shift=.08*torch.tanh((mean-prior_mean)/.08)
        raw=old_raw.float()+(encode_edges(base_edges+shift)-encode_edges(base_edges))/.5
        if not return_aux:return raw,quality
        return raw,quality,dict(logits=logits,old_raw=old_raw.float(),shift=shift)
