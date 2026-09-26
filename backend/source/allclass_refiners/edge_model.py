"""Second-stage RGB border refinement around the best model's predicted edges."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss
import native_core as c
from adaptive_model import AdaptiveBoundary
from boundary_model import deployed_edges,encode_edges


def strip_grid(edges, samples=32, width=5, radius=.04, feature_size=96):
    """Four xyxy edge strips; stride-2 convolution centers correspond to pixel 2*j."""
    t=torch.linspace(0,1,samples,device=edges.device,dtype=edges.dtype)[None,:,None]
    offsets=torch.linspace(-radius,radius,width,device=edges.device,dtype=edges.dtype)[None,None,:]
    left,top,right,bottom=[edges[:,i,None,None] for i in range(4)]
    along_x=(left+(right-left)*t).expand(-1,-1,width)
    along_y=(top+(bottom-top)*t).expand(-1,-1,width)
    grids=[]
    for x,y in (((left+offsets).expand(-1,samples,-1),along_y),
                (along_x,(top+offsets).expand(-1,samples,-1)),
                ((right+offsets).expand(-1,samples,-1),along_y),
                (along_x,(bottom+offsets).expand(-1,samples,-1))):
        grids.append(torch.stack((x,y),-1))
    coords=torch.stack(grids,1)
    return coords*2-1+1/feature_size


def sample_strips(features, edges):
    batch,channels,height,width=features.shape
    assert height==width==96, 'The verified crop geometry uses stride-2 192px RGB crops'
    grid=strip_grid(edges,feature_size=height).reshape(batch,4*32,5,2)
    # Bilinear border samples stay float32 under AMP for small-box geometry.
    with torch.cuda.amp.autocast(enabled=False):
        sampled=F.grid_sample(features.float(),grid.float(),mode='bilinear',padding_mode='border',align_corners=False)
    return sampled.reshape(batch,channels,4,32,5).permute(0,2,1,3,4)


class EdgeAlignedRefiner(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        if checkpoint is None:checkpoint=c.read(c.ROOT/'edge_reference.json')['checkpoint']
        self.anchor=AdaptiveBoundary()
        self.anchor.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True)
        self.anchor.requires_grad_(False);self.anchor.eval()
        stem=self.anchor.anchor.stem
        self.rgb_stem=copy.deepcopy(nn.Sequential(stem[0],stem[1],stem[2]))
        self.rgb_stem[0].requires_grad_(True)
        self.rgb_stem[1].requires_grad_(False)
        self.detail=nn.Sequential(nn.Conv2d(64,32,1),nn.GroupNorm(8,32),nn.SiLU(),
                                  nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU())
        # Mean/max along an edge retain its five across-edge positions.
        self.edge_hidden=nn.Sequential(nn.Linear(32*2*5+32+4+4,128),nn.SiLU())
        self.edge_logits=nn.Linear(128,1)
        nn.init.zeros_(self.edge_logits.weight);nn.init.zeros_(self.edge_logits.bias)
        for name in ('rgb_stem','detail','edge_hidden','edge_logits'):
            setattr(self,'appliance_'+name,copy.deepcopy(getattr(self,name)))
        self.eval()

    def forward(self, images, classes, return_aux=False):
        self.anchor.eval();self.rgb_stem[1].eval();self.appliance_rgb_stem[1].eval()
        with torch.no_grad():old_raw,quality=self.anchor(images,classes)
        edges=deployed_edges(old_raw)
        shift=torch.zeros_like(edges)
        for cls,prefix in ((0,''),(1,'appliance_')):
            selected=classes==cls
            if not selected.any():continue
            features=getattr(self,prefix+'detail')(getattr(self,prefix+'rgb_stem')(images[selected]))
            local=sample_strips(features,edges[selected])
            pooled=torch.cat((local.mean(3),local.amax(3)),2).flatten(2)
            context=features.float().mean((2,3))[:,None,:].expand(-1,4,-1)
            geometry=edges[selected,None,:].expand(-1,4,-1)
            side=torch.eye(4,device=images.device)[None].expand(int(selected.sum()),-1,-1)
            vector=torch.cat((pooled,context,geometry,side),-1)
            offsets=getattr(self,prefix+'edge_logits')(getattr(self,prefix+'edge_hidden')(vector)).squeeze(-1).float()
            shift=shift.index_copy(0,selected.nonzero().flatten(),.04*torch.tanh(offsets))
        raw=old_raw.float()+(encode_edges(edges+shift)-encode_edges(edges))/.5
        if not return_aux:return raw,quality
        return raw,quality,dict(shift=shift,old_raw=old_raw.float(),base_edges=edges)


def objective(raw,quality,aux,boxes,targets,positive,edge_weight=.5):
    accepted=quality.float().sigmoid()>=.5;good=positive & accepted
    zero=raw.sum()*0;reg=edge_loss=preserve=zero
    if good.any():
        deployed=raw.float()*.5
        reg=F.smooth_l1_loss(deployed[good],c.encode(boxes[good],targets[good]),beta=.03)
        reg=reg+2*generalized_box_iou_loss(c.decode(boxes[good],deployed[good]),targets[good],reduction='mean')
        size=(boxes[:,2:]-boxes[:,:2]).clamp_min(2.)
        left=(boxes[:,:2]+boxes[:,2:])/2-size
        expected=(targets-torch.cat((left,left),1))/torch.cat((2*size,2*size),1)
        edge_loss=F.smooth_l1_loss((aux['base_edges']+aux['shift'])[good],expected[good],beta=.02)
    if (~positive & accepted).any():preserve=aux['shift'][~positive & accepted].square().mean()
    loss=reg+edge_weight*edge_loss+5*preserve
    return loss,dict(regression=float(reg.detach()),edge_alignment=float(edge_loss.detach()),
        negative_preservation=float(preserve.detach()),accepted_positives=int(good.sum()))
