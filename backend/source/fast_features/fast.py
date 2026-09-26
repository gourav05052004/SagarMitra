"""Cached frozen features, train-only residual learning, screen then full COCO evaluation."""
import collections,contextlib,copy,gc,hashlib,io,json,os,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.ops import generalized_box_iou_loss
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT=Path(__file__).resolve().parent
COMMON=ROOT.parent/'Maritime_CBNet6215_AllClassContext_backend'
for name in ('native_core','native_data','native_run','edge_model','adaptive_model','boundary_model'):
    module=sys.modules.get(name)
    if module and Path(module.__file__).resolve().parent!=COMMON:
        raise RuntimeError('Restart the Jupyter kernel before importing the fast notebook.')
sys.path.insert(0,str(COMMON))
import native_core as native
from edge_model import EdgeAlignedRefiner

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def store(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    torch.save(obj,tmp);os.replace(tmp,path)
def settings(name='fast_001'):
    assert name.replace('_','').replace('-','').isalnum()
    ref=read(COMMON/'allclass_reference.json')
    return dict(run_name=name,run_dir=str(ROOT/'runs'/name),seed=215,max_updates=600,eval_every=200,
        patience=2,batch_size=256,lr=1e-4,min_lr=1e-5,warmup=20,screen_images=256,
        screen_min_gain=.02,target=62.15,score_min=.05,max_box_side=384,crop_size=96,context=2.,
        train_images_per_class=48,jitters_per_gt=2,gt_per_image=8,real_train_images=96,
        real_per_image=12,source=ref,feature_batch=96,threads=8)
def identity(s):
    files=[Path(__file__),COMMON/'prepared/train.json',COMMON/'prepared/val.json',
        Path(s['source']['checkpoint']),Path(s['source']['predictions']),native.WEIGHTS,
        COMMON/'native_core.py',COMMON/'edge_model.py',COMMON/'boundary_model.py',COMMON/'adaptive_model.py',
        ROOT.parent/'Maritime_CBNet62_NativeOvernight_backend/cache/ecb02a11eae7437b/train/manifest.json']
    return hashlib.sha256(json.dumps(dict(settings=s,files={str(p):sha(p) for p in files}),sort_keys=True).encode()).hexdigest()
def eligible(row,s):return row['score']>=s['score_min'] and min(row['bbox'][2:])>=2 and max(row['bbox'][2:])<=s['max_box_side']
def progress(s,**values):
    values['time']=time.strftime('%Y-%m-%d %H:%M:%S');save(Path(s['run_dir'])/'progress.json',values)
    print(json.dumps(values),flush=True)
def stop(s):
    if (Path(s['run_dir'])/'STOP').exists():raise KeyboardInterrupt('STOP requested; rerun after removing STOP.')
def score(rows,doc,ids):
    ids=list(map(int,ids));wanted=set(ids);rows=[r for r in rows if r['image_id'] in wanted]
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=copy.deepcopy(doc);gt.createIndex()
        if rows:dt=gt.loadRes(copy.deepcopy(rows))
        else:
            dt=COCO();dt.dataset=dict(images=doc['images'],categories=doc['categories'],annotations=[]);dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=ids;ev.params.catIds=[1,2,3,4,5];ev.params.maxDets=[1,10,100]
        ev.evaluate();ev.accumulate();ev.summarize()
    per={}
    for k in range(5):
        p=ev.eval['precision'][:,:,k,0,-1];valid=p[p>=0];per[str(k+1)]=float(valid.mean()*100) if valid.size else None
    return dict(AP=float(ev.stats[0]*100),AP50=float(ev.stats[1]*100),AP75=float(ev.stats[2]*100),
        AP_small=float(ev.stats[3]*100),per_class_AP=per,images=len(ids),metric='COCO bbox AP50:95',maxDets=100)

def screen_ids(doc,s):
    # Fixed random draw plus a label-based class coverage check; never choose by detector errors/AP.
    rng=np.random.default_rng(s['seed']);ids=sorted(im['id'] for im in doc['images'])
    chosen=sorted(map(int,rng.choice(ids,size=s['screen_images'],replace=False)))
    covered={a['category_id'] for a in doc['annotations'] if a['image_id'] in set(chosen)}
    assert covered=={1,2,3,4,5},'Screen needs class coverage; increase screen_images.'
    return chosen

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();net=resnet18(weights=None);net.load_state_dict(torch.load(native.WEIGHTS,map_location='cpu'))
        self.net=nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool,net.layer1,net.layer2)
        self.requires_grad_(False);self.eval()
    def forward(self,x):return F.adaptive_avg_pool2d(self.net(x),(3,3)).flatten(1)

class ResidualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(2304);self.shared=nn.Sequential(nn.Linear(2304,128),nn.SiLU())
        self.heads=nn.ModuleList(nn.Linear(128,4) for _ in range(5))
        for head in self.heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def forward(self,features,classes):
        hidden=self.shared(self.norm(features.float()));all_out=torch.stack([head(hidden) for head in self.heads],1)
        out=all_out[torch.arange(len(classes),device=classes.device),classes]
        return torch.tanh(out)*out.new_tensor([.08,.08,.12,.12])

def loss_fn(model,features,classes,boxes,targets,positive):
    delta=model(features,classes);loss=delta.sum()*0
    if positive.any():
        loss=F.smooth_l1_loss(delta[positive],native.encode(boxes[positive],targets[positive]),beta=.03)
        loss+=2*generalized_box_iou_loss(native.decode(boxes[positive],delta[positive]),targets[positive],reduction='mean')
    if (~positive).any():loss+=5*delta[~positive].square().mean()
    return loss+.01*delta.square().mean()

def descriptors(encoder,rgb,boxes,s):
    if not len(boxes):return torch.empty(0,2304,dtype=torch.float16)
    output=[]
    with torch.inference_mode():
        for offset in range(0,len(boxes),s['feature_batch']):
            selected=boxes[offset:offset+s['feature_batch']]
            local=np.stack([native.crop_rgb(rgb,b,s) for b in selected]);wide_s=dict(s,context=4.)
            wide=np.stack([native.crop_rgb(rgb,b,wide_s) for b in selected])
            with torch.cuda.amp.autocast():
                f=torch.cat((encoder(native.tensor_crops(local)),encoder(native.tensor_crops(wide))),1)
            output.append(f.cpu().half())
    return torch.cat(output)

def check(s):
    assert torch.cuda.is_available(),'Select CBNet Diagnostics (PyTorch 2.0.1 CUDA).'
    assert s['max_updates']>0 and s['max_updates']%s['eval_every']==0 and s['patience']>=1
    assert s['crop_size']==96 and 0<s['min_lr']<=s['lr'] and 0<s['screen_images']<1547
    torch.set_num_threads(s['threads']);root=Path(s['run_dir']);root.mkdir(parents=True,exist_ok=True)
    sig=identity(s)
    if (root/'settings.json').exists():assert read(root/'settings.json')==s,'Changed settings require a new run_name.'
    save(root/'settings.json',s)
    train=read(COMMON/'prepared/train.json');val=read(COMMON/'prepared/val.json')
    assert len(train['images'])==8930 and len(val['images'])==1547
    assert not ({i['file_name'] for i in train['images']} & {i['file_name'] for i in val['images']})
    assert sha(s['source']['checkpoint'])==s['source']['checkpoint_sha256']
    baseline=score(read(s['source']['predictions']),val,[im['id'] for im in val['images']])
    assert abs(baseline['AP']-s['source']['AP'])<1e-8
    ids=screen_ids(val,s);screen=score(read(s['source']['predictions']),val,ids)
    result=dict(signature=sig,full=baseline,screen=screen,screen_ids=ids)
    save(root/'baseline.json',result);progress(s,stage='verified',FULL1547_AP=baseline['AP'],SCREEN256_AP=screen['AP'])
    return result

def prepare_train(s,encoder,sig):
    root=Path(s['run_dir']);path=root/'train_features.pt'
    if path.exists():
        state=torch.load(path,map_location='cpu');assert state['signature']==sig;return state
    rng=np.random.default_rng(s['seed']);doc=read(COMMON/'prepared/train.json');images={i['id']:i for i in doc['images']}
    annotations=collections.defaultdict(list)
    for a in doc['annotations']:
        if not a.get('iscrowd',0):annotations[a['image_id']].append(a)
    examples=collections.defaultdict(list)
    # Existing real detector proposals: no fresh CBNet full-image/tile inference.
    manifest=read(ROOT.parent/'Maritime_CBNet62_NativeOvernight_backend/cache/ecb02a11eae7437b/train/manifest.json')
    assert set(manifest['image_ids'])<=set(images)
    pairs=collections.defaultdict(list)
    for key,group in manifest['groups'].items():
        if int(key.split('_')[1]) in (0,2):
            for iid,index in group:pairs[int(iid)].append(int(index))
    candidates=sorted(pairs);rng.shuffle(candidates);selected=candidates[:s['real_train_images']]
    old=torch.load(s['source']['checkpoint'],map_location='cpu');old_s=old['settings']
    anchor=EdgeAlignedRefiner(old_s['boundary_checkpoint']).cuda().eval();anchor.load_state_dict(old['state_dict']);del old
    anchor.requires_grad_(False)
    for number,iid in enumerate(selected,1):
        stop(s)
        with np.load(Path(manifest['root'])/f'{iid}.npz') as shard:
            candidates=sorted(set(pairs[iid]));rng.shuffle(candidates);idx=candidates[:s['real_per_image']]
            patches=shard['patches'][idx];classes=shard['classes'][idx];boxes=shard['boxes'][idx];targets=shard['targets'][idx];positive=shard['positive'][idx]
        with torch.inference_mode(),torch.cuda.amp.autocast():d,q=anchor(native.tensor_crops(patches),torch.as_tensor(classes,device='cuda'))
        new=native.decode(torch.tensor(boxes),d.float().cpu()*old_s['blend']).numpy();accepted=q.float().sigmoid().cpu().numpy()>=old_s['quality_min']
        for j in range(len(idx)):
            box=native.clip_box(new[j] if accepted[j] else boxes[j],images[iid]['width'],images[iid]['height'])
            if min(box[2:]-box[:2])>=2:examples[iid].append((box,targets[j],bool(positive[j]),[0,3][int(classes[j])]))
        if number%24==0:progress(s,stage='cached_real_proposals',images=number,total=len(selected))
    del anchor;gc.collect();torch.cuda.empty_cache()
    real_count=sum(map(len,examples.values()))
    # Add a bounded, balanced set of jittered train GT for ALL classes; no validation fitting.
    for category in range(1,6):
        ids=[iid for iid,aa in annotations.items() if any(a['category_id']==category and 2<=min(a['bbox'][2:]) and max(a['bbox'][2:])<=s['max_box_side'] for a in aa)]
        rng.shuffle(ids)
        for iid in ids[:s['train_images_per_class']]:
            aa=[a for a in annotations[iid] if a['category_id']==category and 2<=min(a['bbox'][2:]) and max(a['bbox'][2:])<=s['max_box_side']]
            rng.shuffle(aa)
            for a in aa[:s['gt_per_image']]:
                target=native.xyxy(a)
                for j in range(s['jitters_per_gt']):
                    delta=np.r_[rng.normal(0,.025 if j==0 else .055,2),rng.normal(0,.055 if j==0 else .10,2)]
                    box=native.decode(torch.tensor(target[None]),torch.tensor(delta[None],dtype=torch.float32))[0].numpy()
                    box=native.clip_box(box,images[iid]['width'],images[iid]['height'])
                    if min(box[2:]-box[:2])>=2:examples[iid].append((box,target,True,category-1))
    features=[];boxes=[];targets=[];positive=[];classes=[];started=time.monotonic()
    # Per-image feature shards allow interrupted setup to resume.
    for number,(iid,rows) in enumerate(sorted(examples.items()),1):
        stop(s);dest=root/'train_shards'/f'{iid}.pt'
        if dest.exists():
            shard=torch.load(dest,map_location='cpu');assert shard['signature']==sig
        else:
            rgb=cv2.cvtColor(cv2.imread(images[iid]['file_name']),cv2.COLOR_BGR2RGB)
            shard=dict(signature=sig,features=descriptors(encoder,rgb,[r[0] for r in rows],s));store(dest,shard)
        features.append(shard['features']);boxes.extend(r[0] for r in rows);targets.extend(r[1] for r in rows)
        positive.extend(r[2] for r in rows);classes.extend(r[3] for r in rows)
        if number%25==0 or number==len(examples):progress(s,stage='train_feature_cache',images=number,total=len(examples),seconds=round(time.monotonic()-started,1))
    state=dict(signature=sig,features=torch.cat(features),boxes=torch.tensor(np.stack(boxes)),targets=torch.tensor(np.stack(targets)),
        positive=torch.tensor(positive),classes=torch.tensor(classes),train_ids=sorted(examples),real_examples=real_count)
    assert set(state['classes'].tolist())==set(range(5));store(path,state);return state

def prepare_val(s,encoder,sig,ids):
    root=Path(s['run_dir'])/'val_features';root.mkdir(parents=True,exist_ok=True)
    doc=read(COMMON/'prepared/val.json');images={i['id']:i for i in doc['images']};by_image=collections.defaultdict(list)
    for row in read(s['source']['predictions']):by_image[row['image_id']].append(row)
    started=time.monotonic();new=0
    for number,iid in enumerate(ids,1):
        stop(s);path=root/f'{iid}.pt'
        if not path.exists():
            rows=by_image[iid];indices=[j for j,r in enumerate(rows) if eligible(r,s)];boxes=[native.xyxy(rows[j]) for j in indices]
            if boxes:
                rgb=cv2.cvtColor(cv2.imread(images[iid]['file_name']),cv2.COLOR_BGR2RGB);features=descriptors(encoder,rgb,boxes,s)
            else:features=torch.empty(0,2304,dtype=torch.float16)
            store(path,dict(signature=sig,features=features,indices=indices,classes=torch.tensor([rows[j]['category_id']-1 for j in indices],dtype=torch.long)));new+=1
        if number%100==0 or number==len(ids):progress(s,stage='validation_feature_cache',images=number,total=len(ids),new_images=new,seconds=round(time.monotonic()-started,1))

@torch.inference_mode()
def evaluate(model,s,sig,ids,tag):
    root=Path(s['run_dir']);doc=read(COMMON/'prepared/val.json');images={im['id']:im for im in doc['images']}
    by_image=collections.defaultdict(list)
    for r in read(s['source']['predictions']):by_image[r['image_id']].append(r)
    predictions=[];model.eval()
    for iid in ids:
        stop(s);rows=copy.deepcopy(by_image[iid]);shard=torch.load(root/'val_features'/f'{iid}.pt',map_location='cpu');assert shard['signature']==sig
        if len(shard['indices']):
            deltas=model(shard['features'].cuda(),shard['classes'].cuda()).cpu()
            if not torch.isfinite(deltas).all():raise FloatingPointError('Nonfinite inference; best predictions remain intact.')
            updated=native.decode(torch.tensor(np.stack([native.xyxy(rows[j]) for j in shard['indices']])),deltas).numpy()
            for k,j in enumerate(shard['indices']):
                if torch.count_nonzero(deltas[k])==0:continue
                box=native.clip_box(updated[k],images[iid]['width'],images[iid]['height'])
                if min(box[2:]-box[:2])>=1:rows[j]['bbox']=[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])]
        predictions.extend(rows)
    metrics=score(predictions,doc,ids);save(root/tag/'metrics.json',metrics)
    if len(ids)==1547:save(root/tag/'predictions.json',predictions)
    return metrics

def prepare(s):
    baseline=check(s);sig=baseline['signature'];torch.manual_seed(s['seed']);encoder=Encoder().cuda()
    try:
        train=prepare_train(s,encoder,sig);prepare_val(s,encoder,sig,baseline['screen_ids'])
        head=ResidualHead().cuda();initial=evaluate(head,s,sig,baseline['screen_ids'],'screen_initial')
        assert abs(initial['AP']-baseline['screen']['AP'])<1e-8,'Identity screen mismatch'
        # Real cached-feature backward check; discard the updated model.
        batch={k:train[k][:32].cuda() for k in ('features','classes','boxes','targets','positive')}
        opt=torch.optim.AdamW(head.parameters(),lr=s['lr']);loss=loss_fn(head,**batch);loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None and bool((p.grad!=0).any()) for p in head.parameters())
        opt.step();progress(s,stage='ready',source_full_AP=baseline['full']['AP'],source_screen_AP=initial['AP'],
            training_examples=len(train['classes']),real_proposals=train['real_examples'],train_images=len(train['train_ids']))
        return baseline
    finally:
        del encoder;gc.collect();torch.cuda.empty_cache()

def run(s):
    import psutil
    root=Path(s['run_dir']);sig=identity(s);baseline=read(root/'baseline.json');assert baseline['signature']==sig
    lock=root/'RUNNING.json'
    if lock.exists():
        old=read(lock)
        if psutil.pid_exists(old['pid']) and abs(psutil.Process(old['pid']).create_time()-old['created'])<1:raise RuntimeError('This run is already active.')
    save(lock,dict(pid=os.getpid(),created=psutil.Process().create_time()))
    try:return _run(s,sig,baseline)
    finally:lock.unlink(missing_ok=True);gc.collect();torch.cuda.empty_cache()

def _run(s,sig,baseline):
    root=Path(s['run_dir']);result=root/'result.json'
    if result.exists():
        done=read(result);assert done['signature']==sig;print(json.dumps(done,indent=2));return done
    torch.manual_seed(s['seed']);torch.set_num_threads(s['threads'])
    train=torch.load(root/'train_features.pt',map_location='cpu');assert train['signature']==sig
    pool={cls:torch.where(train['classes']==cls)[0] for cls in range(5)}
    tensors={k:train[k].cuda() for k in ('features','classes','boxes','targets','positive')}
    model=ResidualHead().cuda();opt=torch.optim.AdamW(model.parameters(),lr=s['lr'],weight_decay=1e-4)
    best=dict(AP=baseline['full']['AP'],checkpoint=s['source']['checkpoint'],predictions=s['source']['predictions'],update=0,new_head=False)
    best_screen=baseline['screen']['AP'];screen_weights=copy.deepcopy(model.state_dict());history=[];lr=s['lr'];stale=0;update=0;reason=None;last=root/'last.pt'
    if last.exists():
        st=torch.load(last,map_location='cpu');assert st['signature']==sig;model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer'])
        best,best_screen,screen_weights,history,lr,stale,update=(st[k] for k in ('best','best_screen','screen_weights','history','lr','stale','update'))
    started=time.monotonic()
    def persist():
        store(last,dict(signature=sig,model=model.state_dict(),optimizer=opt.state_dict(),best=best,best_screen=best_screen,
            screen_weights=screen_weights,history=history,lr=lr,stale=stale,update=update))
        save(root/'history.json',history);save(root/'best_pipeline.json',dict(**best,source=s['source'],settings=s,
             target_reached=best['AP']>=s['target'],metric='Full1547 COCO bbox AP50:95'))
    try:
        if best['AP']>=s['target']:reason='target'
        elif stale>=s['patience']:reason='patience'
        while update<s['max_updates'] and reason is None:
            stop(s)
            if update and update%s['eval_every']==0 and (not history or history[-1]['update']!=update):
                tag=f'screen_{update:05d}';screen=evaluate(model,s,sig,baseline['screen_ids'],tag);full=None;improves=screen['AP']>best_screen+s['screen_min_gain']
                store(root/f'checkpoints/update_{update:05d}.pt',dict(state_dict=model.state_dict(),settings=s,signature=sig,update=update))
                # Screen scores never replace the verified full-validation winner.
                if improves:
                    encoder=Encoder().cuda();ids=[im['id'] for im in read(COMMON/'prepared/val.json')['images']]
                    prepare_val(s,encoder,sig,ids);del encoder;torch.cuda.empty_cache()
                    full=evaluate(model,s,sig,ids,f'full_{update:05d}')
                    if full['AP']>best['AP']:
                        best=dict(AP=full['AP'],checkpoint=str(root/f'checkpoints/update_{update:05d}.pt'),predictions=str(root/f'full_{update:05d}/predictions.json'),update=update,new_head=True)
                if improves and full is not None and full['AP']>=best['AP']:
                    best_screen=screen['AP'];screen_weights=copy.deepcopy(model.state_dict());stale=0
                else:
                    stale+=1;lr=max(s['min_lr'],lr*.5);model.load_state_dict(screen_weights)
                    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
                row=dict(update=update,SCREEN256_AP=screen['AP'],screen_gain=screen['AP']-baseline['screen']['AP'],
                    FULL1547_AP=None if full is None else full['AP'],best_verified_full_AP=best['AP'],patience=stale,next_lr=lr)
                history.append(row);persist();progress(s,stage='validated',**row)
                if best['AP']>=s['target']:reason='target';break
                if stale>=s['patience']:reason='patience';break
            rng=np.random.default_rng(s['seed']+update);indices=[]
            for i in range(s['batch_size']):
                candidates=pool[i%5];indices.append(int(candidates[rng.integers(len(candidates))]))
            idx=torch.tensor(indices,device='cuda');batch={k:v[idx] for k,v in tensors.items()}
            rate=lr*min(1.,(update+1)/s['warmup'])
            for g in opt.param_groups:g['lr']=rate
            opt.zero_grad(set_to_none=True);loss=loss_fn(model,**batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; retained source remains safe.')
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),2.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm.')
            opt.step();update+=1
            if update%25==0:persist();progress(s,stage='train',update=update,max_updates=s['max_updates'],loss=float(loss.detach()),lr=rate,seconds=round(time.monotonic()-started,1))
        # The final scheduled candidate also needs evaluation.
        if update==s['max_updates'] and (not history or history[-1]['update']!=update):
            screen=evaluate(model,s,sig,baseline['screen_ids'],f'screen_{update:05d}');full=None
            path=root/f'checkpoints/update_{update:05d}.pt';store(path,dict(state_dict=model.state_dict(),settings=s,signature=sig,update=update))
            if screen['AP']>best_screen+s['screen_min_gain']:
                encoder=Encoder().cuda();ids=[im['id'] for im in read(COMMON/'prepared/val.json')['images']];prepare_val(s,encoder,sig,ids);del encoder
                full=evaluate(model,s,sig,ids,f'full_{update:05d}')
                if full['AP']>best['AP']:best=dict(AP=full['AP'],checkpoint=str(path),predictions=str(root/f'full_{update:05d}/predictions.json'),update=update,new_head=True)
            history.append(dict(update=update,SCREEN256_AP=screen['AP'],FULL1547_AP=None if full is None else full['AP'],best_verified_full_AP=best['AP']))
        persist();out=dict(signature=sig,best=best,history=history,reason=reason or 'updates',target_reached=best['AP']>=s['target'],seconds=time.monotonic()-started)
        save(result,out);progress(s,stage='complete',best_verified_full_AP=best['AP'],target_reached=out['target_reached']);return out
    except KeyboardInterrupt:
        persist();progress(s,stage='paused',update=update);return dict(paused=True,update=update)
