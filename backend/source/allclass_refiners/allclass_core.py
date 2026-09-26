"""All-class dual-context refinement initialized as the measured 60.8066 pipeline."""
import copy,hashlib,json,math,os,random
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.ops import generalized_box_iou_loss
import native_core as native
from edge_model import sample_strips

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT/'allclass_reference.json'
CATEGORIES=(1,2,3,4,5)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def atomic_torch(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    torch.save(obj,tmp);os.replace(tmp,path)
def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)

def settings(name='allclass_001'):
    if not name.replace('_','').replace('-','').isalnum():raise ValueError('Simple run name required')
    ref=read(REFERENCE)
    return dict(run_name=name,run_dir=str(ROOT/'runs'/name),max_updates=3200,eval_every=200,patience=5,
        batch_size=40,base_lr=5e-5,stem_lr_multiplier=.1,warmup=30,lr_factor=.5,min_lr=1e-6,
        min_gain=.01,ema_decay=.98,seed=1547,wall_hours=8.,target_map=62.15,
        crop_size=192,local_context=2.,wide_context=4.,score_min=.001,max_box_side=384.,cache_format='metadata_v2',
        train_image_limit=3500,max_real_per_class=24,jitters_per_gt=3,
        sampling_probabilities=[.65,.25,.10],quality_weight=.3,edge_weight=.5,
        source_predictions=ref['predictions'],source_predictions_sha=sha(ref['predictions']),
        source_checkpoint=ref['checkpoint'],source_checkpoint_sha=ref['checkpoint_sha256'],
        source_AP=ref['AP'],architecture='all_class_dual_context_edge_quality_v1')

def data_signature(s):
    keys=('seed','crop_size','local_context','wide_context','score_min','max_box_side','cache_format','train_image_limit',
          'max_real_per_class','jitters_per_gt','source_predictions_sha','source_checkpoint_sha')
    files=[ROOT/'prepared/train.json',ROOT/'prepared/val.json',ROOT/'starting_model.json',ROOT/'allclass_core.py',ROOT/'allclass_data.py',
           ROOT/'fusion_reference.json',ROOT/'allclass_reference.json']
    payload=dict(settings={k:s[k] for k in keys},files={str(p):sha(p) for p in files})
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
def signature(s):
    files=[p for p in ROOT.glob('*.py') if not p.name.startswith(('build_','test_','verify_'))]
    files += [ROOT/'allclass_reference.json',ROOT/'prepared/train.json',ROOT/'prepared/val.json']
    return hashlib.sha256(json.dumps(dict(settings=s,files={str(p):sha(p) for p in sorted(files)}),sort_keys=True).encode()).hexdigest()

def xyxy(row):
    x,y,w,h=row['bbox'];return np.asarray([x,y,x+w,y+h],np.float32)
def overlap(box,targets):return native.overlap(box,targets)
def clip_box(box,width,height):return native.clip_box(box,width,height)
def eligible(row,s):return row['category_id'] in CATEGORIES and row['score']>=s['score_min'] and min(row['bbox'][2:])>=2 and max(row['bbox'][2:])<=s['max_box_side']
def crop_rgb(image,box,s,context):
    custom=dict(s);custom['context']=context
    return native.crop_rgb(image,box,custom)
def tensor_crops(patches,device='cuda'):return native.tensor_crops(patches,device)
def encode(boxes,targets):return native.encode(boxes,targets)
def decode(boxes,deltas):return native.decode(boxes,deltas)

class AllClassContextRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        net=resnet18(weights=None);net.load_state_dict(torch.load(native.WEIGHTS,map_location='cpu'))
        self.stem=nn.Sequential(net.conv1,net.bn1,net.relu)
        self.stem[1].requires_grad_(False)
        self.detail=nn.Sequential(nn.Conv2d(64,32,1),nn.GroupNorm(8,32),nn.SiLU(),
                                  nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU())
        self.class_embed=nn.Embedding(5,32)
        side_dim=32*2*5+32+32+4+4
        self.edge_hidden=nn.Sequential(nn.Linear(side_dim,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.edge_output=nn.ModuleList(nn.Linear(64,1) for _ in CATEGORIES)
        self.quality_hidden=nn.Sequential(nn.Linear(32+32+32,96),nn.SiLU())
        self.quality_output=nn.ModuleList(nn.Linear(96,1) for _ in CATEGORIES)
        for layer in list(self.edge_output)+list(self.quality_output):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.eval()

    def forward(self,local,wide,classes,return_aux=False):
        self.stem[1].eval()
        lf=self.detail(self.stem(local));wf=self.detail(self.stem(wide))
        base=local.new_tensor([.25,.25,.75,.75]).float()[None].expand(len(local),-1)
        strips=sample_strips(lf,base)
        pooled=torch.cat((strips.mean(3),strips.amax(3)),2).flatten(2)
        local_context=lf.float().mean((2,3));wide_context=wf.float().mean((2,3));embedding=self.class_embed(classes).float()
        context=torch.cat((wide_context,embedding),1)[:,None,:].expand(-1,4,-1)
        geometry=base[:,None,:].expand(-1,4,-1);side=torch.eye(4,device=local.device)[None].expand(len(local),-1,-1)
        hidden=self.edge_hidden(torch.cat((pooled,context,geometry,side),-1))
        offsets=torch.zeros(len(local),4,device=local.device)
        quality=torch.zeros(len(local),device=local.device)
        qvector=self.quality_hidden(torch.cat((local_context,wide_context,embedding),1))
        for cls in range(5):
            mask=classes==cls
            if mask.any():
                offsets[mask]=self.edge_output[cls](hidden[mask]).squeeze(-1).float()
                quality[mask]=self.quality_output[cls](qvector[mask]).squeeze(-1).float()
        shift=.06*torch.tanh(offsets)
        edges=base+shift
        center=(edges[:,:2]+edges[:,2:])/2;size=(edges[:,2:]-edges[:,:2]).clamp_min(1e-4)
        delta=torch.cat(((center-.5)/.5,(size/.5).log()),1)
        score_multiplier=torch.exp(.35*torch.tanh(quality))
        if not return_aux:return delta,score_multiplier
        return delta,score_multiplier,dict(shift=shift,edges=edges,quality=quality)

def aligned_iou(boxes,targets):
    wh=(torch.minimum(boxes[:,2:],targets[:,2:])-torch.maximum(boxes[:,:2],targets[:,:2])).clamp_min(0)
    inter=wh.prod(1);area1=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(1);area2=(targets[:,2:]-targets[:,:2]).clamp_min(0).prod(1)
    return inter/(area1+area2-inter).clamp_min(1e-6)

def objective(delta,multiplier,aux,boxes,targets,positive,s):
    zero=delta.sum()*0;reg=edge=preserve=zero
    if positive.any():
        reg=F.smooth_l1_loss(delta[positive],encode(boxes[positive],targets[positive]),beta=.03)
        reg=reg+2*generalized_box_iou_loss(decode(boxes[positive],delta[positive]),targets[positive],reduction='mean')
        size=(boxes[:,2:]-boxes[:,:2]).clamp_min(2.);left=(boxes[:,:2]+boxes[:,2:])/2-size
        expected=(targets-torch.cat((left,left),1))/torch.cat((2*size,2*size),1)
        edge=F.smooth_l1_loss(aux['edges'][positive],expected[positive],beta=.02)
    if (~positive).any():preserve=aux['shift'][~positive].square().mean()
    with torch.no_grad():quality_target=aligned_iou(decode(boxes,delta).detach(),targets).clamp(0,1)*positive.float()
    predicted=((multiplier.log()/.35).clamp(-1,1)+1)/2
    quality=F.binary_cross_entropy(predicted.clamp(1e-5,1-1e-5),quality_target)
    loss=reg+s['edge_weight']*edge+5*preserve+s['quality_weight']*quality
    return loss,dict(regression=float(reg.detach()),edge=float(edge.detach()),quality=float(quality.detach()),
                     negative_preservation=float(preserve.detach()),positives=int(positive.sum()))

def apply(rows,indices,deltas,multipliers,width,height):
    result=copy.deepcopy(rows)
    if not indices:return result
    if not np.isfinite(deltas).all() or not np.isfinite(multipliers).all():raise FloatingPointError('Nonfinite inference')
    source=torch.tensor(np.stack([xyxy(rows[i]) for i in indices]))
    updated=decode(source,torch.as_tensor(deltas,dtype=torch.float32)).numpy()
    for j,index in enumerate(indices):
        box=clip_box(updated[j],width,height)
        if min(box[2:]-box[:2])<1:continue
        if not np.all(np.asarray(deltas[j])==0):
            result[index]['bbox']=[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])]
        if float(multipliers[j])!=1.0:
            result[index]['score']=float(np.clip(rows[index]['score']*multipliers[j],0,1))
    return result

@torch.no_grad()
def update_ema(model,ema,decay):
    trainable={k for k,p in model.named_parameters() if p.requires_grad};source=model.state_dict()
    for key,value in ema.state_dict().items():
        if key not in trainable or not value.is_floating_point():value.copy_(source[key])
        else:value.mul_(decay).add_(source[key],alpha=1-decay)
