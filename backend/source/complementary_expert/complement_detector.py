"""Separate complementary detector; keep original CBNet/dataset intact."""
import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent/'complement_detector'
ROOT.mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'settings')
os.environ['YOLO_AUTOINSTALL']='false'
sys.path.insert(0,str(ROOT.parent/'detector_packages'))
import torch
import yaml
from ultralytics import YOLO, settings

settings.update({k:False for k in ('sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','tensorboard','wandb','vscode_msg')})
settings.update(weights_dir=str(ROOT/'weights'),runs_dir=str(ROOT/'runs'),datasets_dir=str(ROOT/'dataset'))


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(temp,path)


def log(stage,**kw):
    item=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),**kw)
    print(json.dumps(item),flush=True);save(ROOT/'progress.json',item)
    with (ROOT/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(item)+'\n')


def prepare():
    import hashlib
    source=ROOT.parent.parent/'Maritime_CBNet61_DetailHead_backend/prepared'
    docs={tag:json.loads((source/f'{tag}.json').read_text(encoding='utf-8')) for tag in ('train','val')}
    assert [len(docs[t]['images']) for t in ('train','val')]==[8930,1547]
    assert {a['id'] for a in docs['train']['categories']}=={1,2,3,4,5}
    names={a['id']-1:a['name'] for a in docs['train']['categories']}
    file_sets=[]
    for tag,doc in docs.items():
        file_sets.append({Path(im['file_name']).resolve() for im in doc['images']})
        labels=collections.defaultdict(list)
        for a in doc['annotations']:
            if not a.get('iscrowd',0) and not a.get('ignore',0):labels[a['image_id']].append(a)
        paths=[]
        for n,im in enumerate(doc['images'],1):
            original=Path(im['file_name']).resolve();dest=ROOT/'dataset/images'/tag/f"{im['id']}.jpg"
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Independent files: dataset scanners may repair JPEGs or create caches.
            # Never expose the source images to those writes through hardlinks.
            if dest.exists() and os.path.samefile(original,dest):
                assert dest.resolve().is_relative_to((ROOT/'dataset/images').resolve())
                dest.unlink()  # Only this new experiment's link, never the original.
            if not dest.exists():shutil.copy2(original,dest)
            assert not os.path.samefile(original,dest)
            assert original.stat().st_size==dest.stat().st_size
            rows=[]
            for a in labels[im['id']]:
                x,y,w,h=a['bbox'];left=max(0.,x);top=max(0.,y)
                right=min(float(im['width']),x+w);bottom=min(float(im['height']),y+h)
                if right<=left or bottom<=top:continue
                row=[(left+right)/2/im['width'],(top+bottom)/2/im['height'],
                    (right-left)/im['width'],(bottom-top)/im['height']]
                assert all(0<=v<=1 for v in row)
                rows.append(str(a['category_id']-1)+' '+' '.join(f'{v:.10f}' for v in row))
            path=ROOT/'dataset/labels'/tag/f"{im['id']}.txt";path.parent.mkdir(parents=True,exist_ok=True)
            text='\n'.join(rows)+'\n'
            if not path.exists():path.write_text(text,encoding='utf-8')
            else:assert path.read_text(encoding='utf-8')==text
            paths.append(str(dest))
            if n%1000==0 or n==len(doc['images']):log('prepare_'+tag,images=n,total=len(doc['images']))
        (ROOT/f'{tag}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
        # Fixed small smoke subset, with every class represented when possible.
        chosen=[]
        for cat in range(1,6):
            chosen.extend(im for im in doc['images'] if any(a['category_id']==cat for a in labels[im['id']]))
        unique={im['id']:im for im in chosen}; selected=[]
        for cat in range(1,6):
            selected.extend([im['id'] for im in unique.values() if any(a['category_id']==cat for a in labels[im['id']])][:20 if tag=='train' else 4])
        selected=list(dict.fromkeys(selected))
        (ROOT/f'smoke_{tag}.txt').write_text('\n'.join(str(ROOT/'dataset/images'/tag/f'{iid}.jpg') for iid in selected)+'\n',encoding='utf-8')
    assert not file_sets[0].intersection(file_sets[1])
    for mode in ('full','smoke'):
        prefix='' if mode=='full' else 'smoke_'
        config=dict(path=str(ROOT/'dataset'),train=str(ROOT/f'{prefix}train.txt'),val=str(ROOT/f'{prefix}val.txt'),names=names)
        (ROOT/f'{mode}.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    save(ROOT/'dataset_manifest.json',dict(source_hashes={t:hashlib.sha256((source/f'{t}.json').read_bytes()).hexdigest() for t in docs},
        train_images=8930,val_images=1547,category_mapping=names,images='Independent image copies; all labels/cache writes in new folder.'))


def run(smoke=False,batch=4,hours=2.,model_scale='s',workers=2):
    torch.set_num_threads(4)
    os.chdir(ROOT)
    weights=ROOT/'weights'/f'yolo11{model_scale}.pt';weights.parent.mkdir(exist_ok=True)
    if not weights.exists():
        log('download_pretrained_weights')
        torch.hub.download_url_to_file('https://github.com/ultralytics/assets/releases/download/v8.3.0/'+weights.name,str(weights),progress=True)
    model=YOLO(str(weights));start=time.monotonic();counter=[0]
    assert model.names[0]=='person' and model.names[8]=='boat'
    source_heads=[(branch[-1].weight.detach().cpu().clone(),branch[-1].bias.detach().cpu().clone())
        for branch in model.model.model[-1].cv3]
    def initialize_categories(trainer):
        # Preserve known COCO semantic filters when adapting 80 outputs to five.
        # Swimmer <- person; boat/jetski <- boat. Other categories learn normally.
        mapping={0:0,1:8,2:8}
        with torch.no_grad():
            for detector in (trainer.model,trainer.ema.ema):
                for branch,(weight,bias) in zip(detector.model[-1].cv3,source_heads):
                    target=branch[-1]
                    assert target.weight.shape[1:]==weight.shape[1:]
                    for destination,source in mapping.items():
                        target.weight[destination].copy_(weight[source]);target.bias[destination].copy_(bias[source])
        log('semantic_head_initialized',mapping=mapping)
    model.add_callback('on_pretrain_routine_end',initialize_categories)
    def step(trainer):
        counter[0]+=1
        assert max(group['lr'] for group in trainer.optimizer.param_groups)<=.000501
        if counter[0]==1 or counter[0]%25==0:
            log('train',batches=counter[0],epoch=trainer.epoch+1,elapsed_seconds=time.monotonic()-start,
                losses=[float(x) for x in trainer.tloss],reserved_GiB=torch.cuda.memory_reserved()/1024**3,
                smoke=smoke,batches_per_epoch=len(trainer.train_loader))
        if (ROOT/'STOP').exists():trainer.stop=True
    def epoch(trainer):
        log('epoch',epoch=trainer.epoch+1,metrics=trainer.metrics,elapsed_seconds=time.monotonic()-start,
            smoke=smoke,validation_images=18 if smoke else 1547,
            note='Ultralytics validation metrics; separate unchanged COCO evaluation required for benchmark claim.')
    model.add_callback('on_train_batch_end',step);model.add_callback('on_fit_epoch_end',epoch)
    result=model.train(data=str(ROOT/('smoke.yaml' if smoke else 'full.yaml')),epochs=1 if smoke else 40,
        time=None if smoke else hours,imgsz=1280,batch=batch,device=0,workers=workers,cache=False,
        optimizer='AdamW',lr0=.0005,lrf=.05,cos_lr=True,warmup_epochs=1.,warmup_bias_lr=0.,momentum=.9,patience=10,
        amp=True,seed=92463,deterministic=False,project=str(ROOT/'runs'),name='smoke' if smoke else 'expert_001',
        exist_ok=False,plots=False,save=True,save_period=1,rect=False,mosaic=.2,close_mosaic=3,
        mixup=0.,degrees=0.,translate=.05,scale=.25,fliplr=.5,hsv_h=.01,hsv_s=.3,hsv_v=.2,
        max_det=100,nbs=16,val=True,verbose=False)
    log('completed',smoke=smoke,batches=counter[0],elapsed_seconds=time.monotonic()-start,
        save_dir=str(model.trainer.save_dir),metrics=result.results_dict)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','smoke','train']);parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--hours',type=float,default=2.);parser.add_argument('--model',choices=['s','m'],default='s');parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:run(args.action=='smoke',args.batch,args.hours,args.model,args.workers)
