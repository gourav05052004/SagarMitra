"""Original-resolution RGB crop refinement; frozen CBNet detections are the input."""
import copy, hashlib, json, os, random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.ops import generalized_box_iou_loss

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'Maritime_CBNet61_DetailHead_backend/runs/detail_001/validation_00200/fused_predictions.json'
WEIGHTS=ROOT/'weights/resnet18-f37072fd.pth'
TARGETS=(1,4)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(temp,path)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def atomic_torch(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial');torch.save(state,temp);os.replace(temp,path)
def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
def settings(name='native_001'):
    if not name.replace('_','').replace('-','').isalnum():raise ValueError('Simple run name required')
    return dict(run_name=name,run_dir=str(ROOT/'runs'/name),max_updates=2400,eval_every=200,patience=1,
        batch_size=32,base_lr=2e-4,backbone_lr=2e-5,warmup=20,seed=46,crop_size=192,context=2.,
        score_min=.05,max_box_side=192.,blend=.5,quality_min=.5,train_image_limit=1536,
        jitters_per_gt=3,max_real_per_class=32,ema_decay=.99,target_map=62.15)
def signature(s):
    paths=[p for p in ROOT.glob('*.py') if not p.name.startswith(('build_','test_','verify_'))]
    paths += [ROOT/'prepared/train.json',ROOT/'prepared/val.json',ROOT/'source_identity.json',
              ROOT/'starting_model.json',ROOT/'fusion_reference.json',WEIGHTS,SOURCE]
    return hashlib.sha256(json.dumps(dict(settings=s,files={str(p):sha(p) for p in sorted(paths)}),sort_keys=True).encode()).hexdigest()
def ensure_weights():
    WEIGHTS.parent.mkdir(exist_ok=True)
    if not WEIGHTS.exists():
        torch.hub.download_url_to_file('https://download.pytorch.org/models/resnet18-f37072fd.pth',
            str(WEIGHTS.with_suffix('.partial')),hash_prefix='f37072fd',progress=True)
        os.replace(WEIGHTS.with_suffix('.partial'),WEIGHTS)
    assert sha(WEIGHTS).startswith('f37072fd'),'ImageNet weight checksum mismatch'

def xyxy(row):
    x,y,w,h=row['bbox'];return np.array([x,y,x+w,y+h],dtype=np.float32)
def clip_box(box,width,height):
    box=np.asarray(box,np.float32).copy();box[0::2]=np.clip(box[0::2],0,width);box[1::2]=np.clip(box[1::2],0,height)
    return box
def overlap(box,targets):
    if not len(targets):return np.zeros(0,np.float32)
    targets=np.asarray(targets,np.float32)
    wh=np.maximum(np.minimum(box[2:],targets[:,2:])-np.maximum(box[:2],targets[:,:2]),0)
    intersection=wh.prod(1)
    area=np.maximum(box[2:]-box[:2],0).prod()+np.maximum(targets[:,2:]-targets[:,:2],0).prod(1)
    return intersection/np.maximum(area-intersection,1e-6)
def crop_rgb(image,box,s):
    box=np.asarray(box,np.float32);center=(box[:2]+box[2:])/2
    extent=np.maximum(box[2:]-box[:2],2)*s['context'];left=center-extent/2
    scale=s['crop_size']/extent
    matrix=np.array([[scale[0],0,-left[0]*scale[0]],[0,scale[1],-left[1]*scale[1]]],np.float32)
    return cv2.warpAffine(image,matrix,(s['crop_size'],s['crop_size']),flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,borderValue=(123,116,103))
def eligible(row,s):
    return row['category_id'] in TARGETS and row['score']>=s['score_min'] and 2<=min(row['bbox'][2:]) and max(row['bbox'][2:])<=s['max_box_side']
def tensor_crops(patches,device='cuda'):
    data=torch.from_numpy(np.asarray(patches)).permute(0,3,1,2).to(device=device,dtype=torch.float32)/255.
    return (data-data.new_tensor([.485,.456,.406])[None,:,None,None])/data.new_tensor([.229,.224,.225])[None,:,None,None]
def encode(boxes,targets):
    size=(boxes[:,2:]-boxes[:,:2]).clamp_min(1.)
    return torch.cat((((targets[:,:2]+targets[:,2:])-(boxes[:,:2]+boxes[:,2:]))/(2*size),
        ((targets[:,2:]-targets[:,:2]).clamp_min(1.)/size).log()),1)
def decode(boxes,deltas):
    size=(boxes[:,2:]-boxes[:,:2]).clamp_min(1.)
    center=(boxes[:,:2]+boxes[:,2:])/2+deltas[:,:2]*size
    size=size*deltas[:,2:].exp()
    return torch.cat((center-size/2,center+size/2),1)

class NativeRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        net=resnet18(weights=None);net.load_state_dict(torch.load(WEIGHTS,map_location='cpu'))
        self.stem=nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool,net.layer1)
        self.layer2=net.layer2;self.layer3=net.layer3
        self.class_embed=nn.Embedding(2,8)
        self.head=nn.Sequential(nn.Linear(256*6*6+8,256),nn.ReLU(),nn.Linear(256,5))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.stem.requires_grad_(False)
        for module in self.modules():
            if isinstance(module,nn.BatchNorm2d):module.requires_grad_(False)
        self.eval()
    def forward(self,images,classes):
        x=self.layer3(self.layer2(self.stem(images)))
        x=F.adaptive_avg_pool2d(x,(6,6)).flatten(1)
        out=self.head(torch.cat((x,self.class_embed(classes)),1))
        delta=torch.tanh(out[:,:4])*out.new_tensor([.25,.25,.4,.4])
        return delta,out[:,4]

def losses(delta,quality,boxes,targets,positive):
    expected=encode(boxes,targets)
    zero=delta.sum()*0
    regression=zero
    if positive.any():
        # Scale-free regression and GIoU preserve sensitivity to tiny-object boundaries.
        regression=F.smooth_l1_loss(delta[positive],expected[positive],beta=.05)
        regression+=2*generalized_box_iou_loss(decode(boxes[positive],delta[positive]),targets[positive],reduction='mean')
    if (~positive).any():regression+=.1*delta[~positive].square().mean()
    quality_loss=.2*F.binary_cross_entropy_with_logits(quality,positive.float())
    return regression+quality_loss,dict(regression=float(regression.detach()),quality=float(quality_loss.detach()))

def apply_deltas(rows,indices,deltas,quality,width,height,s):
    result=copy.deepcopy(rows)
    if len(indices)==0:return result
    assert all(eligible(rows[i],s) for i in indices),'Only eligible target-class boxes can be refined'
    if not np.isfinite(np.asarray(deltas)).all() or not np.isfinite(np.asarray(quality)).all():
        raise FloatingPointError('Nonfinite refiner predictions')
    source=torch.tensor(np.stack([xyxy(rows[i]) for i in indices]))
    updated=decode(source,torch.as_tensor(deltas,dtype=torch.float32)*s['blend']).numpy()
    for j,index in enumerate(indices):
        if quality[j]<s['quality_min']:continue
        box=clip_box(updated[j],width,height)
        if min(box[2:]-box[:2])<1:continue
        # Identity is bitwise preserved when zero initialized; do not rewrite rounded coordinates.
        if np.all(np.asarray(deltas[j])==0):continue
        result[index]['bbox']=[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])]
    return result
