"""Fresh pixel profiles, frozen detection scores, original data read-only."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import pixel_core

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'Maritime_CBNet62_PrecisionBox_backend'
sys.path.insert(0,str(OLD))
_spec=importlib.util.spec_from_file_location('pixel_training_controller',OLD/'precision_run.py')
base=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=base
_spec.loader.exec_module(base)
_prepare=base.prepare
base.core=pixel_core


def defaults(name='pixel_001'):
    s=base.defaults(name)
    s.update(run_dir=str(ROOT/'runs'/name),max_updates=600,batch_size=48,lr=3e-4,min_lr=3.75e-5,
        eval_every=200,patience=3,warmup=25,threads=4,max_minutes=45.,seed=621529,
        correction_score_min=.05,architecture='rgb_gradient_boundary_distribution_v1',
        extension_hashes={str(p):base.f.sha(p) for p in (Path(__file__),ROOT/'pixel_core.py')})
    return s


def setup(s):
    for p,digest in s['extension_hashes'].items():
        if base.f.sha(p)!=digest:raise RuntimeError('Pixel architecture changed: reload the kernel and use a new run name.')
    return base.setup(s)


def image_fields(bgr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)*.25
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)*.25
    return np.concatenate([rgb,gray[:,:,None],gx[:,:,None],gy[:,:,None]],2)


def extract_profiles(fields,boxes):
    """Pixel-center maps on four edges,24along x9across; labels never accepted."""
    boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    if not len(boxes):return np.empty((0,4,6,24,9),np.float16)
    output=[];along=np.linspace(.05,.95,24,dtype=np.float32)[None,:,None]
    across=np.linspace(-.06,.06,9,dtype=np.float32)[None,None,:]
    for start in range(0,len(boxes),128):
        b=boxes[start:start+128];x0,y0,x1,y1=[b[:,i,None,None] for i in range(4)]
        w=np.maximum(x1-x0,1.);h=np.maximum(y1-y0,1.)
        along_x=np.broadcast_to(x0+w*along,(len(b),24,9));along_y=np.broadcast_to(y0+h*along,(len(b),24,9))
        grids=[(np.broadcast_to(x0+w*across,along_y.shape),along_y),
            (along_x,np.broadcast_to(y0+h*across,along_x.shape)),
            (np.broadcast_to(x1+w*across,along_y.shape),along_y),
            (along_x,np.broadcast_to(y1+h*across,along_x.shape))]
        sides=[]
        for x,y in grids:
            # Detector coordinates describe continuous pixel edges; array samples
            # have pixel centers at index+.5. Preserve that half-pixel conversion.
            values=cv2.remap(fields,(x-.5).reshape(-1,9).astype(np.float32),
                (y-.5).reshape(-1,9).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            sides.append(values.reshape(len(b),24,9,6).transpose(0,3,1,2))
        output.append(np.stack(sides,1).astype(np.float16))
    return np.concatenate(output)


def pixel_item(item,s,record,tag):
    iid=item['image']['id'];source=Path(item['image']['file_name']);st=source.stat()
    chosen=np.array([i for i,j in enumerate(item['indices']) if item['rows'][int(j)]['score']>=s['correction_score_min']],dtype=np.int64)
    path=Path(s['run_dir'])/'pixel_profiles'/tag/f'{iid}.pt'
    stamp=[st.st_size,st.st_mtime_ns]
    if path.exists():
        saved=torch.load(path,map_location='cpu')
        assert saved['signature']==record['signature'] and saved['image_stamp']==stamp
        assert np.array_equal(saved['chosen'],chosen)
    else:
        if len(chosen):
            bgr=cv2.imread(str(source))
            if bgr is None or bgr.shape[:2]!=(item['image']['height'],item['image']['width']):raise RuntimeError('Invalid image: '+str(source))
            profiles=extract_profiles(image_fields(bgr),item['data']['boxes'][chosen].numpy())
        else:profiles=np.empty((0,4,6,24,9),np.float16)
        saved=dict(signature=record['signature'],image_stamp=stamp,chosen=chosen,profiles=torch.from_numpy(profiles))
        base.q.store(path,saved)
    item['indices']=[item['indices'][i] for i in chosen]
    item['data']={k:v[chosen] for k,v in item['data'].items() if k!='visual'}
    item['data']['visual']=saved['profiles']
    return item


def prepare(s,record,train,val,deadline):
    # Existing preparation validates all descriptor/row identities and target masks.
    # We discard descriptors and replace them with fresh raw pixel profiles.
    fit,tune,validation=_prepare(s,record,train,val,deadline)
    del fit;gc.collect()
    fitted=[]
    for name,items,ids in [('fit',None,record['split']['fit_ids']),('tune',tune,None),('val',validation,None)]:
        if items is None:
            items=[]
            for iid in ids:
                a=torch.load(Path(s['run_dir'])/'prepared/fit'/f'{iid}.pt',map_location='cpu')
                shard=base.q.shard(base.q.PARENT/'cache/train_features'/f'{iid}.pt',record['parent_signature'])
                a['data']=dict(meta=shard['meta'],category=shard['category'],
                    boxes=torch.tensor(base.q.old.boxes(a['rows'])[a['indices']],dtype=torch.float32),
                    targets=a['targets'],positive=a['positive'],valid=a['valid'])
                items.append(a)
        for n,a in enumerate(items,1):
            base.f.check(s,deadline);pixel_item(a,s,record,name)
            if n%25==0 or n==len(items):base.f.log(s,stage='pixel_profiles_'+name,images=n,total=len(items))
        if name=='fit':fitted=items
    data={k:torch.cat([a['data'][k] for a in fitted]) for k in fitted[0]['data']}
    valid=data.pop('valid');data={k:v[valid] for k,v in data.items()}
    counts={str(c):int(((data['category']==c)&data['positive']).sum()) for c in pixel_core.TARGETS}
    assert min(counts.values())>30,counts
    base.f.save(Path(s['run_dir'])/'pixel_training_counts.json',dict(positives=counts,
        fit_boxes=len(data['category']),val_boxes=sum(len(a['indices']) for a in validation),score_min=s['correction_score_min']))
    return data,tune,validation


base.prepare=prepare


def run(s):
    setup(s)
    return base.run(s)


def verify_best(s):
    setup(s)
    return base.verify_best(s)
