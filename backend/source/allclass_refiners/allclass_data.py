"""Resumable train-proposal generation and validation crops for all five classes."""
import collections,copy,gc,json,os,sys,time
from pathlib import Path
import cv2,numpy as np,torch
import allclass_core as c

DETECTOR_BACKEND=c.ROOT.parent/'Maritime_CBNet61_DetailHead_backend'

def stop(s):
    if (Path(s['run_dir'])/'STOP').exists():raise KeyboardInterrupt('STOP requested')
def save_npz(path,**arrays):
    path=Path(path);tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)

def check_inputs(s):
    ref=c.read(c.REFERENCE)
    assert c.sha(s['source_checkpoint'])==s['source_checkpoint_sha']==ref['checkpoint_sha256']
    assert c.sha(s['source_predictions'])==s['source_predictions_sha']
    train=c.read(c.ROOT/'prepared/train.json');val=c.read(c.ROOT/'prepared/val.json')
    assert len(train['images'])==8930 and len(val['images'])==1547
    assert not ({im['file_name'] for im in train['images']} & {im['file_name'] for im in val['images']})
    assert [cat['id'] for cat in train['categories']]==list(c.CATEGORIES)
    for doc in (train,val):
        for im in doc['images']:
            if not Path(im['file_name']).is_file():raise FileNotFoundError(im['file_name'])
    return dict(train_images=8930,val_images=1547,source_AP=s['source_AP'])

def training_selection(doc,s):
    by_image=collections.defaultdict(set)
    for a in doc['annotations']:
        if a['category_id'] in c.CATEGORIES and not a.get('iscrowd',0):by_image[a['image_id']].add(a['category_id'])
    images={im['id']:im for im in doc['images']};rng=np.random.default_rng(s['seed']);chosen=[];seen=set()
    # Preserve every rare appliance and jetski image when the limit permits.
    for category in (4,3,5,1,2):
        pool=[iid for iid,cats in by_image.items() if category in cats and iid not in seen]
        rng.shuffle(pool)
        quota=len(pool) if category in (4,3) else max(0,s['train_image_limit']//5)
        for iid in pool[:quota]:
            if len(chosen)>=s['train_image_limit']:break
            chosen.append(images[iid]);seen.add(iid)
    remaining=[im for im in doc['images'] if im['id'] in by_image and im['id'] not in seen]
    rng.shuffle(remaining);chosen+=remaining[:max(0,s['train_image_limit']-len(chosen))]
    assert len(chosen)==s['train_image_limit']
    return chosen

def _load_pipeline(s):
    # These module names belong to the detector backend. Require a fresh notebook kernel.
    for name in ('worker','model_config','recovery','inference_tools','fixed_fusion','data_tools'):
        module=sys.modules.get(name)
        if module and not Path(module.__file__).resolve().is_relative_to(DETECTOR_BACKEND):
            raise RuntimeError('Restart kernel: a different detector backend is imported')
    sys.path.insert(0,str(DETECTOR_BACKEND))
    import worker,recovery
    from inference_tools import infer_at_scale,_translate_tile
    from fixed_fusion import fuse
    from data_tools import tile_windows
    detector_settings=copy.deepcopy(c.read(c.ROOT/'starting_model.json')['settings'])
    detector_settings.update(root=str(DETECTOR_BACKEND),init_checkpoint=c.read(c.ROOT/'starting_model.json')['checkpoint'])
    detector,_=worker.create_model(detector_settings,checkpoint=detector_settings['init_checkpoint'])
    detector.requires_grad_(False);detector.eval();recovery.SESSION_STOP=None;recovery.DEADLINE=None
    from edge_model import EdgeAlignedRefiner
    saved=torch.load(s['source_checkpoint'],map_location='cpu');old_settings=saved['settings']
    refiner=EdgeAlignedRefiner(old_settings['boundary_checkpoint']).cuda()
    refiner.load_state_dict(saved['state_dict'],strict=True);del saved
    refiner.requires_grad_(False);refiner.eval()
    return dict(detector=detector,refiner=refiner,settings=detector_settings,old_settings=old_settings,
                worker=worker,infer=infer_at_scale,translate=_translate_tile,fuse=fuse,windows=tile_windows)

def _infer_pipeline(models,bgr,im,s):
    ds=models['settings'];worker=models['worker'];infer=models['infer'];detector=models['detector']
    sources={'full':worker.prediction_rows(infer(detector,bgr,ds['eval_scale']),im['id'])}
    for index,(left,top,right,bottom) in enumerate(models['windows'](im['width'],im['height'],ds['tile_size'],ds['tile_overlap'])):
        rows=worker.prediction_rows(infer(detector,bgr[top:bottom,left:right],ds['eval_scale']),im['id'])
        sources[f'tile{index}']=models['translate'](rows,im['id'],left,top,right,bottom,im['width'],im['height'])
    reference=c.read(c.ROOT/'fusion_reference.json');params=c.read(DETECTOR_BACKEND/'fusion_parameters.json')
    variants={'full':sources['full']}
    for name in set(reference['selection'].values())-{'full'}:
        cfg=params[name];variants[name]=models['fuse'](sources,im['id'],cfg['full_weight'],cfg['iou'])
    fused=[]
    for category,name in reference['selection'].items():fused.extend(row for row in variants[name] if row['category_id']==int(category))
    # Reproduce the prior best stage for its two target classes before making new all-class examples.
    import native_core as native
    old_settings=models['old_settings']
    indices=[i for i,row in enumerate(fused) if native.eligible(row,old_settings)]
    if indices:
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);deltas=[];quality=[]
        for offset in range(0,len(indices),64):
            selected=indices[offset:offset+64]
            patches=np.stack([native.crop_rgb(rgb,native.xyxy(fused[i]),old_settings) for i in selected])
            classes=torch.tensor([native.TARGETS.index(fused[i]['category_id']) for i in selected],device='cuda')
            with torch.inference_mode(),torch.cuda.amp.autocast():d,q=models['refiner'](native.tensor_crops(patches),classes)
            deltas.extend(d.float().cpu().numpy());quality.extend(q.float().sigmoid().cpu().numpy())
        fused=native.apply_deltas(fused,indices,deltas,quality,im['width'],im['height'],old_settings)
    return fused

def prepare_training(s):
    check_inputs(s);sig=c.data_signature(s);root=c.ROOT/'cache'/sig[:16]/'train';root.mkdir(parents=True,exist_ok=True)
    marker=root/'manifest.json'
    if marker.exists():
        result=c.read(marker);assert result['data_signature']==sig;print('Reusing complete all-class training cache.');return result
    doc=c.read(c.ROOT/'prepared/train.json');images=training_selection(doc,s);annotations=collections.defaultdict(list)
    for a in doc['annotations']:
        if a['category_id'] in c.CATEGORIES and not a.get('iscrowd',0):annotations[a['image_id']].append(a)
    models=_load_pipeline(s);started=time.monotonic()
    try:
        for k,im in enumerate(images,1):
            stop(s);dest=root/f"{im['id']}.npz"
            if not dest.exists():
                bgr=cv2.imread(im['file_name'])
                if bgr is None:raise FileNotFoundError(im['file_name'])
                with torch.inference_mode():rows=_infer_pipeline(models,bgr,im,s)
                boxes=[];targets=[];positive=[];classes=[];kinds=[]
                def add(box,target,good,category,kind):
                    if min(box[2:]-box[:2])<2 or max(box[2:]-box[:2])>s['max_box_side']:return
                    boxes.append(box);targets.append(target);positive.append(good);classes.append(category-1);kinds.append(kind)
                rng=np.random.default_rng(s['seed']+im['id'])
                for category in c.CATEGORIES:
                    gt=np.asarray([c.xyxy(a) for a in annotations[im['id']] if a['category_id']==category],np.float32).reshape(-1,4)
                    candidates=sorted((r for r in rows if r['category_id']==category and c.eligible(r,s)),key=lambda r:-r['score'])[:s['max_real_per_class']]
                    for row in candidates:
                        box=c.xyxy(row);ious=c.overlap(box,gt);best=int(ious.argmax()) if len(ious) else -1;value=float(ious[best]) if best>=0 else 0.
                        if value>=.4:add(box,gt[best],True,category,0)
                        elif value<.1:add(box,box,False,category,2)
                    for target in gt:
                        wh=target[2:]-target[:2]
                        if min(wh)<2 or max(wh)>s['max_box_side']:continue
                        for j in range(s['jitters_per_gt']):
                            delta=np.r_[rng.normal(0,.08,2),rng.normal(0,.14,2)] if j else np.zeros(4)
                            box=c.decode(torch.tensor(target[None]),torch.tensor(delta[None],dtype=torch.float32))[0].numpy()
                            box=c.clip_box(box,im['width'],im['height'])
                            if c.overlap(box,target[None])[0]>=.35:add(box,target,True,category,1)
                save_npz(dest,boxes=np.asarray(boxes,np.float32).reshape(-1,4),targets=np.asarray(targets,np.float32).reshape(-1,4),
                    positive=np.asarray(positive,bool),classes=np.asarray(classes,np.int64),kinds=np.asarray(kinds,np.int64))
            if k%10==0 or k==len(images):
                status=dict(stage='prepare_training_proposals',images=k,total=len(images),elapsed_seconds=time.monotonic()-started)
                c.save(Path(s['run_dir'])/'progress.json',status);print(json.dumps(status),flush=True)
    finally:
        del models;gc.collect();torch.cuda.empty_cache()
    groups={f'{cls}_{kind}':[] for cls in range(5) for kind in range(3)}
    for im in images:
        with np.load(root/f"{im['id']}.npz") as shard:
            for index,(cls,kind) in enumerate(zip(shard['classes'],shard['kinds'])):groups[f'{cls}_{kind}'].append([im['id'],index])
    for cls in range(5):assert groups[f'{cls}_1'],f'No jitter examples for class {cls+1}'
    result=dict(data_signature=sig,root=str(root),image_ids=[im['id'] for im in images],
                file_names={str(im['id']):im['file_name'] for im in images},groups=groups,
                counts={key:len(value) for key,value in groups.items()},split='train',source_pipeline_AP=s['source_AP'])
    c.save(marker,result);return result

def prepare_validation(s):
    check_inputs(s);sig=c.data_signature(s);root=c.ROOT/'cache'/sig[:16]/'val';root.mkdir(parents=True,exist_ok=True)
    marker=root/'manifest.json'
    if marker.exists():result=c.read(marker);assert result['data_signature']==sig;return result
    doc=c.read(c.ROOT/'prepared/val.json');by_image=collections.defaultdict(list)
    for row in c.read(s['source_predictions']):by_image[row['image_id']].append(row)
    total=0;started=time.monotonic()
    for k,im in enumerate(doc['images'],1):
        stop(s);rows=by_image[im['id']];indices=[i for i,row in enumerate(rows) if c.eligible(row,s)];total+=len(indices)
        dest=root/f"{im['id']}.npz"
        if not dest.exists():
            save_npz(dest,indices=np.asarray(indices,np.int64),
                     classes=np.asarray([rows[i]['category_id']-1 for i in indices],np.int64))
        if k%50==0 or k==len(doc['images']):
            status=dict(stage='prepare_validation_crops',images=k,total=len(doc['images']),crops=total,elapsed_seconds=time.monotonic()-started)
            c.save(Path(s['run_dir'])/'progress.json',status);print(json.dumps(status),flush=True)
    result=dict(data_signature=sig,root=str(root),images=1547,crops=total,split='validation_inputs_only')
    c.save(marker,result);return result

class Batches:
    def __init__(self,manifest,s):
        self.manifest=manifest;self.s=s;self.cache=collections.OrderedDict();self.images=collections.OrderedDict()
        self.entries=collections.defaultdict(list)
        for key,items in manifest['groups'].items():
            cls,kind=map(int,key.split('_'))
            for image_id,index in items:self.entries[(cls,image_id)].append((index,kind))
        self.image_choices={cls:sorted({image_id for (group_cls,image_id) in self.entries if group_cls==cls}) for cls in range(5)}
    def get(self,image_id):
        if image_id not in self.cache:
            with np.load(Path(self.manifest['root'])/f'{image_id}.npz') as shard:self.cache[image_id]={k:shard[k] for k in shard.files}
            while len(self.cache)>8:self.cache.popitem(last=False)
        self.cache.move_to_end(image_id);return self.cache[image_id]
    def image(self,image_id):
        if image_id not in self.images:
            bgr=cv2.imread(self.manifest['file_names'][str(image_id)])
            if bgr is None:raise FileNotFoundError(self.manifest['file_names'][str(image_id)])
            self.images[image_id]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            while len(self.images)>10:self.images.popitem(last=False)
        self.images.move_to_end(image_id);return self.images[image_id]
    def batch(self,update,device='cuda'):
        rng=np.random.default_rng(self.s['seed']+update*104729);rows=[];assert self.s['batch_size']%5==0
        for cls in range(5):
            choices=self.image_choices[cls];image_id=choices[rng.integers(len(choices))]
            shard=self.get(image_id);rgb=self.image(image_id);entries=self.entries[(cls,image_id)]
            for _ in range(self.s['batch_size']//5):
                kind=int(rng.choice(3,p=self.s['sampling_probabilities']));pool=[x for x in entries if x[1]==kind] or entries
                index,_=pool[rng.integers(len(pool))];box=shard['boxes'][index]
                rows.append(dict(local=c.crop_rgb(rgb,box,self.s,self.s['local_context']),
                                 wide=c.crop_rgb(rgb,box,self.s,self.s['wide_context']),
                                 **{key:shard[key][index] for key in ('boxes','targets','positive','classes')}))
        tensor=lambda key:torch.as_tensor(np.stack([row[key] for row in rows]),device=device)
        return c.tensor_crops(np.stack([r['local'] for r in rows]),device),c.tensor_crops(np.stack([r['wide'] for r in rows]),device),tensor('classes'),tensor('boxes'),tensor('targets'),tensor('positive')
