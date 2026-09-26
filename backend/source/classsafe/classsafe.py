"""Preserve the measured buoy-only gain, then learn bounded independent residuals."""
import collections,copy,gc,hashlib,json,os,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'Maritime_CBNet6215_Fast_backend'))
import fast
import allclass_data as proposals

def settings(name='classsafe_001'):
    return dict(run_dir=str(ROOT/'runs'/name),run_name=name,seed=2215,max_updates=150,eval_every=25,
        patience=2,batch_size=64,lr=2e-5,min_lr=2.5e-6,screen_min_gain=.05,images_per_class=16,
        max_per_class_image=24,identity_weight=10.,target=62.15,feature_batch=96,
        source_run=str(fast.ROOT/'runs/fast_001'),source_head=str(fast.ROOT/'runs/fast_001/checkpoints/update_00400.pt'))

def sources(s):
    base_s=fast.read(Path(s['source_run'])/'settings.json');base=fast.read(Path(s['source_run'])/'baseline.json')
    assert base['signature']==fast.identity(base_s),'Source feature cache changed; do not silently mix experiments.'
    return base_s,base

def signature(s):
    base_s,base=sources(s)
    files=[Path(__file__),Path(s['source_head']),fast.COMMON/'allclass_data.py',fast.COMMON/'allclass_core.py']
    return hashlib.sha256(json.dumps(dict(settings=s,source_signature=base['signature'],
        files={str(p):fast.sha(p) for p in files}),sort_keys=True).encode()).hexdigest()

class Model(nn.Module):
    def __init__(self,s):
        super().__init__();self.anchor=fast.ResidualHead()
        self.anchor.load_state_dict(torch.load(s['source_head'],map_location='cpu')['state_dict'])
        self.anchor.requires_grad_(False);self.anchor.eval()
        self.heads=nn.ModuleList(nn.Sequential(nn.LayerNorm(2304),nn.Linear(2304,64),nn.SiLU(),nn.Linear(64,4)) for _ in range(5))
        for head in self.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    def forward(self,features,classes,aux=False):
        with torch.no_grad():anchor=self.anchor(features,classes)*(classes==4)[:,None]
        correction=torch.zeros_like(anchor)
        for category in range(5):
            mask=classes==category
            if mask.any():correction[mask]=torch.tanh(self.heads[category](features[mask].float()))*anchor.new_tensor([.025,.025,.04,.04])
        return (anchor+correction,correction,anchor) if aux else anchor+correction

def metrics_for(model,s,ids,tag,write_rows=False):
    """Use existing 31k feature cache; no CBNet or encoder inference at validation."""
    bs,base=sources(s);doc=fast.read(fast.COMMON/'prepared/val.json');images={i['id']:i for i in doc['images']}
    by_image=collections.defaultdict(list)
    for row in fast.read(bs['source']['predictions']):by_image[row['image_id']].append(row)
    output=[];model.eval();started=time.monotonic()
    with torch.inference_mode():
        for iid in ids:
            fast.stop(s);rows=copy.deepcopy(by_image[iid]);shard=torch.load(Path(s['source_run'])/'val_features'/f'{iid}.pt',map_location='cpu')
            assert shard['signature']==base['signature']
            if len(shard['indices']):
                delta=model(shard['features'].cuda(),shard['classes'].cuda()).cpu()
                if not torch.isfinite(delta).all():raise FloatingPointError('Nonfinite boxes')
                boxes=torch.tensor(np.stack([fast.native.xyxy(rows[j]) for j in shard['indices']]))
                updated=fast.native.decode(boxes,delta).numpy()
                for k,j in enumerate(shard['indices']):
                    if torch.count_nonzero(delta[k])==0:continue
                    box=fast.native.clip_box(updated[k],images[iid]['width'],images[iid]['height'])
                    if min(box[2:]-box[:2])>=1:rows[j]['bbox']=[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])]
            output.extend(rows)
    m=fast.score(output,doc,ids);dest=Path(s['run_dir'])/tag;fast.save(dest/'metrics.json',m)
    if write_rows:fast.save(dest/'predictions.json',output)
    fast.progress(s,stage='evaluation',tag=tag,images=len(ids),AP=m['AP'],seconds=round(time.monotonic()-started,2))
    return m

def verify(s):
    torch.set_num_threads(8);root=Path(s['run_dir']);sig=signature(s);bs,base=sources(s)
    if (root/'settings.json').exists():assert fast.read(root/'settings.json')==s,'Use a new run_name for changed settings.'
    fast.save(root/'settings.json',s)
    if (root/'baseline.json').exists():
        record=fast.read(root/'baseline.json');assert record['signature']==sig;print('Reusing verified class-safe source:',record['full']['AP']);return record
    model=Model(s).cuda();ids=[i['id'] for i in fast.read(fast.COMMON/'prepared/val.json')['images']]
    # Missing source shards can be rebuilt once without retraining a detector.
    missing=[iid for iid in ids if not (Path(s['source_run'])/'val_features'/f'{iid}.pt').exists()]
    if missing:
        enc=fast.Encoder().cuda();fast.prepare_val(bs,enc,base['signature'],missing);del enc
    screen=metrics_for(model,s,base['screen_ids'],'baseline_screen')
    full=metrics_for(model,s,ids,'baseline_full',True)
    assert full['AP']>=base['full']['AP'],'Buoy correction failed to reproduce a full gain; stop.'
    cp=root/'checkpoints/update_00000.pt';fast.store(cp,dict(state_dict=model.state_dict(),settings=s,signature=sig,update=0))
    remaining=sorted(set(ids)-set(base['screen_ids']));doc=fast.read(fast.COMMON/'prepared/val.json')
    previous_remainder=fast.score(fast.read(bs['source']['predictions']),doc,remaining)
    new_remainder=fast.score(fast.read(root/'baseline_full/predictions.json'),doc,remaining)
    record=dict(signature=sig,full=full,screen=screen,screen_ids=base['screen_ids'],
        source_full=base['full'],remaining1291_before=previous_remainder,remaining1291_after=new_remainder,
        checkpoint=str(cp),predictions=str(root/'baseline_full/predictions.json'))
    fast.save(root/'baseline.json',record);del model;gc.collect();torch.cuda.empty_cache();return record

def select_training(doc,s):
    rng=np.random.default_rng(s['seed']);images={i['id']:i for i in doc['images']};bycat=collections.defaultdict(set)
    for a in doc['annotations']:
        if not a.get('iscrowd',0) and 2<=min(a['bbox'][2:]) and max(a['bbox'][2:])<=384:bycat[a['category_id']].add(a['image_id'])
    chosen=set()
    for category in range(1,6):
        pool=sorted(bycat[category]);rng.shuffle(pool);chosen.update(pool[:s['images_per_class']])
    return [images[i] for i in sorted(chosen)]

def prepare(s):
    baseline=verify(s);sig=baseline['signature'];root=Path(s['run_dir']);path=root/'real_features.pt'
    if path.exists():
        t=torch.load(path,map_location='cpu');assert t['signature']==sig;print('Reusing real proposal features:',len(t['classes']));return t
    bs,_=sources(s);doc=fast.read(fast.COMMON/'prepared/train.json');images=select_training(doc,s)
    val=fast.read(fast.COMMON/'prepared/val.json');assert not ({i['file_name'] for i in images}&{i['file_name'] for i in val['images']})
    gt=collections.defaultdict(list)
    for a in doc['annotations']:
        if not a.get('iscrowd',0):gt[a['image_id']].append(a)
    old_s=dict(source_checkpoint=bs['source']['checkpoint']);pending=[im for im in images if not (root/'train_predictions'/f"{im['id']}.json").exists()]
    if pending:
        models=proposals._load_pipeline(old_s);started=time.monotonic()
        try:
            for k,im in enumerate(pending,1):
                fast.stop(s);bgr=cv2.imread(im['file_name'])
                if bgr is None:raise FileNotFoundError(im['file_name'])
                with torch.inference_mode():rows=proposals._infer_pipeline(models,bgr,im,old_s)
                fast.save(root/'train_predictions'/f"{im['id']}.json",dict(signature=sig,rows=rows))
                if k%5==0 or k==len(pending):fast.progress(s,stage='real_fused_train_proposals',images=k,total=len(pending),seconds=round(time.monotonic()-started,1))
        finally:del models;gc.collect();torch.cuda.empty_cache()
    encoder=fast.Encoder().cuda();all_features=[];boxes=[];targets=[];classes=[];positive=[];image_ids=[]
    try:
        for k,im in enumerate(images,1):
            fast.stop(s);dest=root/'train_features'/f"{im['id']}.pt"
            if dest.exists():shard=torch.load(dest,map_location='cpu');assert shard['signature']==sig
            else:
                stored=fast.read(root/'train_predictions'/f"{im['id']}.json");assert stored['signature']==sig
                selected=[]
                for category in range(1,6):
                    gtboxes=np.array([fast.native.xyxy(a) for a in gt[im['id']] if a['category_id']==category],np.float32).reshape(-1,4)
                    rows=sorted((r for r in stored['rows'] if r['category_id']==category and fast.eligible(r,bs)),key=lambda r:-r['score'])[:s['max_per_class_image']]
                    for row in rows:
                        box=fast.native.xyxy(row);ious=fast.native.overlap(box,gtboxes);j=int(ious.argmax()) if len(ious) else -1;overlap=float(ious[j]) if j>=0 else 0.
                        if overlap>=.5:selected.append((box,gtboxes[j],True,category-1))
                        elif overlap<.1:selected.append((box,box,False,category-1))
                rgb=cv2.cvtColor(cv2.imread(im['file_name']),cv2.COLOR_BGR2RGB)
                shard=dict(signature=sig,features=fast.descriptors(encoder,rgb,[r[0] for r in selected],bs),
                    boxes=torch.tensor(np.array([r[0] for r in selected],np.float32).reshape(-1,4)),
                    targets=torch.tensor(np.array([r[1] for r in selected],np.float32).reshape(-1,4)),
                    positive=torch.tensor([r[2] for r in selected],dtype=torch.bool),classes=torch.tensor([r[3] for r in selected],dtype=torch.long))
                fast.store(dest,shard)
            all_features.append(shard['features']);boxes.append(shard['boxes']);targets.append(shard['targets']);positive.append(shard['positive']);classes.append(shard['classes'])
            image_ids.extend([im['id']]*len(shard['classes']))
        result=dict(signature=sig,features=torch.cat(all_features),boxes=torch.cat(boxes),targets=torch.cat(targets),
            positive=torch.cat(positive),classes=torch.cat(classes),image_ids=torch.tensor(image_ids))
        counts={str(i+1):dict(all=int((result['classes']==i).sum()),positive=int(((result['classes']==i)&result['positive']).sum())) for i in range(5)}
        assert all(v['positive']>=5 for v in counts.values()),f'Too few real positives; use more images: {counts}'
        result['counts']=counts;fast.store(path,result);fast.save(root/'training_counts.json',counts)
        fast.progress(s,stage='ready',training_images=len(images),real_examples=len(result['classes']),counts=counts)
        return result
    finally:del encoder;gc.collect();torch.cuda.empty_cache()

def objective(model,batch,s):
    delta,correction,anchor=model(batch['features'],batch['classes'],True);positive=batch['positive'];loss=delta.sum()*0
    if positive.any():
        loss=F.smooth_l1_loss(delta[positive],fast.native.encode(batch['boxes'][positive],batch['targets'][positive]),beta=.03)
        loss+=.5*generalized_box_iou_loss(fast.native.decode(batch['boxes'][positive],delta[positive]),batch['targets'][positive],reduction='mean')
    loss+=s['identity_weight']*correction.square().mean()
    if (~positive).any():loss+=20*correction[~positive].square().mean()
    return loss

def run(s):
    import psutil
    root=Path(s['run_dir']);sig=signature(s);baseline=verify(s)
    if (root/'result.json').exists():
        result=fast.read(root/'result.json');assert result['signature']==sig;print(json.dumps(result,indent=2));return result
    lock=root/'RUNNING.json'
    if lock.exists():
        state=fast.read(lock)
        if psutil.pid_exists(state['pid']) and abs(psutil.Process(state['pid']).create_time()-state['created'])<1:raise RuntimeError('Run already active.')
    fast.save(lock,dict(pid=os.getpid(),created=psutil.Process().create_time()))
    try:return _run(s,sig,baseline)
    finally:lock.unlink(missing_ok=True);gc.collect();torch.cuda.empty_cache()

def _run(s,sig,baseline):
    torch.manual_seed(s['seed']);torch.set_num_threads(8);root=Path(s['run_dir']);data=torch.load(root/'real_features.pt',map_location='cpu')
    assert data['signature']==sig
    tensors={k:data[k].cuda() for k in ('features','boxes','targets','classes','positive')}
    pool={i:torch.where(data['classes']==i)[0].numpy() for i in range(5)}
    model=Model(s).cuda();winner=copy.deepcopy(model);rates=[s['lr']]*5
    def optimizer():return torch.optim.AdamW([dict(params=model.heads[i].parameters(),lr=rates[i]) for i in range(5)],weight_decay=1e-4)
    opt=optimizer();best=dict(AP=baseline['full']['AP'],checkpoint=baseline['checkpoint'],predictions=baseline['predictions'],update=0)
    screen_best=baseline['screen'];history=[];update=stale=0;last=root/'last.pt';started=time.monotonic()
    if last.exists():
        st=torch.load(last,map_location='cpu');assert st['signature']==sig
        model.load_state_dict(st['model']);winner.load_state_dict(st['winner']);rates=st['rates'];opt=optimizer();opt.load_state_dict(st['optimizer'])
        best,screen_best,history,update,stale=(st[k] for k in ('best','screen_best','history','update','stale'))
    def persist():
        fast.store(last,dict(signature=sig,model=model.state_dict(),winner=winner.state_dict(),optimizer=opt.state_dict(),rates=rates,
            best=best,screen_best=screen_best,history=history,update=update,stale=stale))
        fast.save(root/'history.json',history);fast.save(root/'best_pipeline.json',dict(**best,settings=s,signature=sig,target_reached=best['AP']>=s['target'],metric='Full1547 COCO bbox AP50:95'))
    def validate():
        nonlocal best,screen_best,stale,opt
        raw=metrics_for(model,s,baseline['screen_ids'],f'screen_raw_{update:05d}')
        selected=[i for i in range(5) if raw['per_class_AP'][str(i+1)]>screen_best['per_class_AP'][str(i+1)]+s['screen_min_gain']]
        candidate=copy.deepcopy(winner)
        for i in selected:candidate.heads[i].load_state_dict(model.heads[i].state_dict())
        full=None;accepted=False;path=root/f'checkpoints/update_{update:05d}.pt'
        fast.store(path,dict(signature=sig,state_dict=candidate.state_dict(),settings=s,update=update,screen_selected_categories=[i+1 for i in selected]))
        if selected:
            ids=[i['id'] for i in fast.read(fast.COMMON/'prepared/val.json')['images']]
            full=metrics_for(candidate,s,ids,f'full_{update:05d}',True)
            accepted=full['AP']>best['AP']+.001
        if accepted:
            winner.load_state_dict(candidate.state_dict());best=dict(AP=full['AP'],checkpoint=str(path),predictions=str(root/f'full_{update:05d}/predictions.json'),update=update)
            # Mixed class metrics are rescored; no assumption that screen AP combines linearly.
            screen_best=metrics_for(winner,s,baseline['screen_ids'],f'screen_retained_{update:05d}');stale=0
        else:stale+=1
        for i in range(5):
            if not accepted or i not in selected:rates[i]=max(s['min_lr'],rates[i]*.5)
        model.load_state_dict(winner.state_dict());opt=optimizer()
        row=dict(update=update,screen_raw_AP=raw['AP'],screen_selected_categories=[i+1 for i in selected],
            FULL1547_AP=None if full is None else full['AP'],best_verified_full_AP=best['AP'],accepted=accepted,
            patience=stale,learning_rates=rates.copy(),raw_per_class_AP=raw['per_class_AP'])
        history.append(row);persist();fast.progress(s,stage='validated',**row)
    try:
        while True:
            fast.stop(s)
            if update and update%s['eval_every']==0 and (not history or history[-1]['update']!=update):validate()
            if best['AP']>=s['target'] or stale>=s['patience'] or update>=s['max_updates']:break
            rng=np.random.default_rng(s['seed']+update);indices=[int(rng.choice(pool[j%5])) for j in range(s['batch_size'])]
            idx=torch.tensor(indices,device='cuda');batch={k:v[idx] for k,v in tensors.items()}
            opt.zero_grad(set_to_none=True);loss=objective(model,batch,s)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
            opt.step();update+=1
            if update%10==0:persist();fast.progress(s,stage='train',update=update,loss=float(loss.detach()),best_verified_full_AP=best['AP'])
        persist();result=dict(signature=sig,best=best,history=history,updates=update,seconds=time.monotonic()-started,
            target_reached=best['AP']>=s['target'],reason='target' if best['AP']>=s['target'] else 'patience' if stale>=s['patience'] else 'updates')
        fast.save(root/'result.json',result);fast.progress(s,stage='complete',best_verified_full_AP=best['AP'],target_reached=result['target_reached']);return result
    except KeyboardInterrupt:
        persist();fast.progress(s,stage='paused',update=update);return dict(paused=True,update=update)
