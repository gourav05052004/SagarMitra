"""Single-GPU, resumable CBNetV2 Cascade training and full-image COCO evaluation."""
import argparse, copy, gc, json, logging, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from mmcv.parallel import collate, scatter
from mmcv.runner import build_optimizer, wrap_fp16_model
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.apis import inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from model_config import make_config
from data_tools import (NAMES, save_json, read_json, annotated, montage, gt_boxes,
                        tile_windows, prepare, preprocessing_visuals)
from checkpoint_io import MODEL_FILE, sha256, compatible_state

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def atomic_torch(obj, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial')
    torch.save(obj,temp); os.replace(temp,path)

def state_cpu(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items()}

def check_environment(s):
    import mmcv, mmdet, torchvision
    from mmcv.ops import nms, roi_align
    assert torch.__version__.startswith('2.0.1'), torch.__version__
    assert mmcv.__version__=='1.7.2' and mmdet.__version__=='2.28.2'
    assert torch.cuda.is_available(), 'The isolated worker cannot access an NVIDIA GPU.'
    dev=torch.device('cuda:0')
    x=torch.randn(1,8,32,32,device=dev,requires_grad=True)
    roi_align(x,torch.tensor([[0.,1.,1.,20.,20.]],device=dev),7).sum().backward()
    nms(torch.tensor([[0.,0.,10.,10.],[1.,1.,10.,10.]],device=dev),torch.tensor([.9,.8],device=dev),.5)
    info=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,
              mmcv=mmcv.__version__,mmdet=mmdet.__version__,gpu=torch.cuda.get_device_name(0),
              vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30,compiled_ops='CUDA NMS and RoIAlign backward passed')
    save_json(Path(s['root'])/'environment.json',info)
    print(json.dumps(info,indent=2),flush=True)

def create_model(s, checkpoint=None, transfer=True, device='cuda:0'):
    cfg=make_config(s)
    logging.getLogger('mmdet').setLevel(logging.ERROR)
    model=build_detector(cfg.model)
    # Full pretrained state is loaded below; only new classification/regression outputs initialize randomly.
    previous_logging_disable=logging.root.manager.disable
    try:
        logging.disable(logging.INFO)
        model.init_weights()
    finally:
        logging.disable(previous_logging_disable)
    if checkpoint:
        state=torch.load(checkpoint,map_location='cpu')['state_dict']
        state, excluded, quality_missing = compatible_state(
            state, model.state_dict().keys(), cfg.model.roi_head.num_stages,
            allow_quality_init=bool(s.get('quality_head', False)),
            allow_deform_init=bool(s.get('deformable_roi', False)),
            allow_detail_init=bool(s.get('detail_head', False)))
        result = model.load_state_dict(state, strict=not quality_missing)
        if quality_missing:
            unexpected = list(result.unexpected_keys)
            missing = sorted(result.missing_keys)
            if unexpected or missing != quality_missing:
                raise RuntimeError(f'Unexpected quality transfer result: missing={missing}, '
                                   f'unexpected={unexpected}')
            print(f'Architecture transfer: initialized only {len(quality_missing)} new tensors; '
                  'all pretrained tensors loaded strictly.', flush=True)
        if excluded:
            print(f'Two-stage ablation: omitted {len(excluded)} third-stage tensors; '
                  'all retained parameters loaded strictly.', flush=True)
    elif transfer:
        path=Path(s['root'])/'weights'/MODEL_FILE
        source=torch.load(path,map_location='cpu')
        state=source.get('state_dict',source)
        state={k.removeprefix('module.'):v for k,v in state.items()}
        target=model.state_dict()
        matched={k:v for k,v in state.items() if k in target and v.shape==target[k].shape}
        missing=[k for k in target if k not in matched]
        unexpected=[k for k in missing if not ('.fc_cls.' in k or '.fc_reg.' in k)]
        if unexpected:
            raise RuntimeError(f'Unexpected pretrained mismatch; refusing partial backbone transfer: {unexpected[:20]}')
        model.load_state_dict(matched,strict=False)
        fraction=sum(v.numel() for v in matched.values())/sum(v.numel() for v in target.values())
        if fraction<.97: raise RuntimeError(f'Too little pretrained transfer: {fraction:.2%}')
        save_json(Path(s['root'])/'transfer_report.json',dict(transferred_fraction=fraction,
            reset_parameters=missing,source=str(path),source_sha256=sha256(path),
            note='COCO detector initialization, not continuation of the 51.85 AP D-FINE model.'))
        print(f'COCO pretrained transfer: {fraction:.3%}; reset ONLY the 5-class outputs.',flush=True)
        del source,state,matched
    if s.get('train_scope'):
        from train_policy import trainable
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(trainable(name, s['train_scope']))
    elif s.get('refinement_only'):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('.reg_refine.' in name)
    model.CLASSES=NAMES; model.cfg=cfg
    model.to(device)
    if str(device).startswith('cuda'):
        wrap_fp16_model(model)
    return model,cfg

def parse_loss(model, losses):
    loss, details=model._parse_losses(losses)
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite training loss: {details}')
    return loss,details

def prediction_rows(result, image_id):
    if isinstance(result,tuple): result=result[0]
    rows=[]
    for cls,boxes in enumerate(result):
        for x1,y1,x2,y2,score in boxes:
            if score<.001 or x2<=x1 or y2<=y1: continue
            rows.append(dict(image_id=int(image_id),category_id=cls+1,
                bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
    return rows

def preview_predictions(coco, predictions, selected, dest, title):
    by_image={i:[] for i in selected}
    for p in predictions:
        if p['image_id'] in by_image: by_image[p['image_id']].append(p)
    panels=[]
    for im in coco['images']:
        if im['id'] not in selected: continue
        image=Image.open(im['file_name']).convert('RGB')
        boxes=gt_boxes(coco,im['id'])
        rows=sorted(by_image[im['id']],key=lambda p:-p['score'])
        pb=[[p['bbox'][0],p['bbox'][1],p['bbox'][0]+p['bbox'][2],p['bbox'][1]+p['bbox'][3],p['category_id']-1] for p in rows]
        scores=[p['score'] for p in rows]
        panels.extend([annotated(image,boxes,f'GT | {im["id"]}'),
                       annotated(image,pb,f'{title} | display >=0.25; AP uses >=0.001',scores)])
        # A fixed GT-centered close-up is visualization only. Inference covers the whole frame.
        if boxes:
            target=min(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
            cx,cy=(target[0]+target[2])/2,(target[1]+target[3])/2
            w,h=min(640,image.width),min(640,image.height)
            left=max(0,min(int(cx-w/2),image.width-w)); top=max(0,min(int(cy-h/2),image.height-h))
            def clipped(rows):
                return [[max(a,left)-left,max(b,top)-top,min(c,left+w)-left,min(d,top+h)-top,k]
                        for a,b,c,d,k in rows if c>left and d>top and a<left+w and b<top+h]
            zoom=image.crop((left,top,left+w,top+h))
            visible=[(b,sc) for b,sc in zip(pb,scores) if b[2]>left and b[3]>top and b[0]<left+w and b[1]<top+h]
            panels.extend([annotated(zoom,clipped(boxes),'GT small-object close-up'),
                annotated(zoom,clipped([b for b,sc in visible]),'Prediction close-up',[sc for b,sc in visible])])
    montage(panels,dest)

def coco_metrics(coco_path, predictions, image_ids):
    gt=COCO(str(coco_path))
    if predictions:
        dt=gt.loadRes(predictions)
    else:
        dt=COCO(); dt.dataset=dict(images=gt.dataset['images'],categories=gt.dataset['categories'],annotations=[]); dt.createIndex()
    ev=COCOeval(gt,dt,'bbox'); ev.params.imgIds=list(image_ids); ev.params.catIds=list(range(1,6))
    ev.params.maxDets=[1,10,100]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    by_class={}
    for k,name in enumerate(NAMES):
        values=ev.eval['precision'][:,:,k,0,-1]; values=values[values>=0]
        by_class[name]=float(values.mean()*100) if len(values) else None
    return dict(map=float(ev.stats[0]*100),ap50=float(ev.stats[1]*100),ap75=float(ev.stats[2]*100),
                ap_small=float(ev.stats[3]*100),per_class_ap=by_class,images=len(image_ids),
                metric='COCO AP50:95, original image coordinates, maxDets=100')

def merge_rows(rows,image_id,iou=.5):
    from torchvision.ops import batched_nms
    if not rows:return []
    boxes=torch.tensor([[p['bbox'][0],p['bbox'][1],p['bbox'][0]+p['bbox'][2],p['bbox'][1]+p['bbox'][3]] for p in rows])
    scores=torch.tensor([p['score'] for p in rows]); labels=torch.tensor([p['category_id'] for p in rows])
    keep=batched_nms(boxes,scores,labels,iou)[:300]
    return [rows[i] for i in keep.tolist()]

def extra_inference(model, im, settings, mode):
    import cv2
    bgr=cv2.imread(im['file_name'])
    if bgr is None: raise OSError(im['file_name'])
    rows=prediction_rows(inference_detector(model,bgr),im['id'])
    if mode=='tiles':
        for x,y,x2,y2 in tile_windows(im['width'],im['height'],settings['tile_size']):
            if x==y==0 and x2==im['width'] and y2==im['height']: continue
            tile_rows=prediction_rows(inference_detector(model,bgr[y:y2,x:x2]),im['id'])
            for p in tile_rows:
                # Clip local coordinates before translating to the original frame.
                a,b,w,h=p['bbox']; c=min(x2-x,max(0,a+w)); d=min(y2-y,max(0,b+h))
                a=max(0,a); b=max(0,b)
                if c>a and d>b:
                    p['bbox']=[a+x,b+y,c-a,d-b]; rows.append(p)
    elif mode=='tta':
        # Original scale plus a larger scale, each with its horizontal flip.
        aug=model.cfg.data.test.pipeline[1]
        old=aug['img_scale']
        for scale in (tuple(settings['eval_scale']),tuple(settings['tta_scale'])):
            aug['img_scale']=scale
            for flip in (False,True):
                if scale==tuple(settings['eval_scale']) and not flip: continue
                image=np.ascontiguousarray(bgr[:,::-1]) if flip else bgr
                for p in prediction_rows(inference_detector(model,image),im['id']):
                    if flip: p['bbox'][0]=im['width']-p['bbox'][0]-p['bbox'][2]
                    rows.append(p)
        aug['img_scale']=old
    return merge_rows(rows,im['id'])

def evaluate(model,s,directory,mode='single',limit=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    coco_path=Path(s['root'])/'prepared/val.json'; coco=read_json(coco_path)
    ims=coco['images'][:limit] if limit else coco['images']
    expected=int(s.get('expected_val_images',1547))
    if not limit and len(ims)!=expected:
        raise ValueError(f'Full validation must contain all {expected} expected images')
    selected=read_json(Path(s['root'])/'prepared/preview_ids.json')
    model.eval(); predictions=[]; start=time.perf_counter()
    with torch.no_grad():
        for index,im in enumerate(ims):
            rows=(prediction_rows(inference_detector(model,im['file_name']),im['id']) if mode=='single'
                  else extra_inference(model,im,s,mode))
            predictions.extend(rows)
            if (index+1)%100==0: print(f'Validation {mode}: {index+1}/{len(ims)}',flush=True)
    metrics=coco_metrics(coco_path,predictions,[im['id'] for im in ims])
    metrics.update(seconds=time.perf_counter()-start,smoke_only=bool(limit) or expected!=1547,inference=mode)
    save_json(directory/'predictions.json',predictions); save_json(directory/'metrics.json',metrics)
    preview_predictions(coco,predictions,selected if not limit else [im['id'] for im in ims],directory/'detections.jpg',mode)
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics

def plots(history, path):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(18,5)); xs=[h['epoch'] for h in history]
    for key in ('map','ap50','ap75','ap_small'): axes[0].plot(xs,[h[key] for h in history],label=key)
    axes[0].axhline(62,ls='--',color='gray',label='target AP50:95'); axes[0].legend(); axes[0].set_title('Full validation: 1547 frames')
    for name in NAMES:axes[1].plot(xs,[h['per_class_ap'][name] for h in history],label=name)
    axes[1].legend(fontsize=8);axes[1].set_title('AP50:95 per class')
    axes[2].plot(xs,[h['train_loss'] for h in history]);axes[2].set_title('Training loss')
    for ax in axes: ax.set_xlabel('Completed epoch');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path);plt.close(fig)

def feature_visuals(model,s,batch):
    import matplotlib.pyplot as plt
    model.eval()
    with torch.no_grad(),torch.cuda.amp.autocast():
        feats=model.extract_feat(batch['img'])
    fig,axes=plt.subplots(1,len(feats),figsize=(20,5))
    for i,(f,ax) in enumerate(zip(feats,axes)):
        a=f[0].float().abs().mean(0).cpu().numpy()
        ax.imshow(a,cmap='magma');ax.set_title(f'P{i+2}, stride {2**(i+2)}\n{f.shape[-1]} x {f.shape[-2]}');ax.axis('off')
    fig.suptitle('Actual pretrained FPN activation magnitude; not detection probabilities')
    fig.savefig(Path(s['root'])/'visuals/07_pretrained_features.png',bbox_inches='tight');plt.close(fig)

def signature(s):
    import hashlib
    keys=['train_scale','eval_scale','anchor_scale','repeat_threshold','repeat_cap','flip_probability',
          'epochs','lr_steps','learning_rate','accumulate','seed']
    content={k:s[k] for k in keys}
    # maritime_class_weights_v1
    if s.get('class_weights') is not None:
        content['class_weights']=s['class_weights']
    content['annotations']={split:sha256(Path(s['root'])/f'prepared/{split}.json') for split in ('train','val')}
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()

def preflight(s,mini=False):
    check_environment(s)
    if not mini and torch.cuda.get_device_properties(0).total_memory<18*2**30:
        raise RuntimeError('This notebook targets >=20 GB GPUs. A 4 GB laptop is for developer smoke tests only.')
    model,cfg=create_model(s)
    if mini:
        for sampler in model.roi_head.bbox_sampler:sampler.num=32
        model.train_cfg.rpn_proposal.max_per_img=128
        model.train_cfg.rpn_proposal.nms_pre=256
        model.test_cfg.rpn.max_per_img=128
    dataset=build_dataset(cfg.data.train)
    sample=dataset[0]
    batch=scatter(collate([sample],samples_per_gpu=1),[0])[0]
    feature_visuals(model,s,batch)
    if not mini:
        # Test the maximum padded rectangle, even if dataset[0] has a narrower aspect ratio.
        long,short=max(s['train_scale']),min(s['train_scale'])
        h,w=math.ceil(short/32)*32,math.ceil(long/32)*32
        batch['img']=torch.zeros((1,3,h,w),device='cuda')
        batch['img_metas'][0].update(img_shape=(h,w,3),pad_shape=(h,w,3),batch_input_shape=(h,w))
        batch['gt_bboxes']=[torch.tensor([[16.,16.,32.,32.],[w/3,h/3,w/3+150,h/3+90]],device='cuda')]
        batch['gt_labels']=[torch.tensor([0,1],device='cuda')]
    optimizer=build_optimizer(model,cfg.optimizer)
    scaler=torch.cuda.amp.GradScaler(init_scale=512.)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    model.train(); values=[]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(): loss,details=parse_loss(model,model(**batch,return_loss=True))
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),35.)
        if not torch.isfinite(norm):raise FloatingPointError('Preflight gradient overflow; lower initial scale and inspect losses.')
        scaler.step(optimizer);scaler.update();values.append(float(loss))
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()/2**30
    # Check empty-frame loss too: empty images must not cause a crash or NaN.
    empty=copy.copy(batch);empty['gt_bboxes']=[torch.empty((0,4),device='cuda')];empty['gt_labels']=[torch.empty((0,),dtype=torch.long,device='cuda')]
    model.train()
    with torch.no_grad(),torch.cuda.amp.autocast():empty_loss,_=parse_loss(model,model(**empty,return_loss=True))
    report=dict(passed=True,mini_only=mini,peak_allocated_gib=peak,seconds=time.perf_counter()-start,
                losses=values,empty_frame_loss=float(empty_loss),parameter_count=sum(p.numel() for p in model.parameters()),
                training_images_after_repetition=len(dataset),signature=signature(s))
    save_json(Path(s['root'])/('developer_smoke.json' if mini else 'preflight.json'),report)
    print(json.dumps(report,indent=2),flush=True)
    # These optimization steps are deliberately discarded. Real training reloads COCO weights.

def train(s):
    root=Path(s['root']);run=root/'runs'/s['run_name'];run.mkdir(parents=True,exist_ok=True)
    pre=read_json(root/'preflight.json')
    sig=signature(s)
    if not pre['passed'] or pre['mini_only'] or pre['signature']!=sig:
        raise RuntimeError('Run the full GPU preflight again with these exact settings before training.')
    resume=run/'last.pth';model,cfg=create_model(s,checkpoint=resume if resume.exists() else None)
    model.cfg=cfg;cfg.dump(str(run/'resolved_model_config.py'))
    save_json(run/'settings.json',s)
    dataset=build_dataset(cfg.data.train)
    loader=build_dataloader(dataset,samples_per_gpu=1,workers_per_gpu=s['workers'],num_gpus=1,
        dist=False,shuffle=True,seed=s['seed'],persistent_workers=s['workers']>0)
    optimizer=build_optimizer(model,cfg.optimizer);scaler=torch.cuda.amp.GradScaler(init_scale=512.)
    start_epoch,steps,best,history=0,0,-1,[]
    if resume.exists():
        state=torch.load(resume,map_location='cpu')
        if state['signature']!=sig:raise RuntimeError('Settings changed: use a new run_name or restore the saved settings.')
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        for state_values in optimizer.state.values():
            for key,value in state_values.items():
                if torch.is_tensor(value) and key!='step':state_values[key]=value.to('cuda')
        start_epoch=state['epoch'];steps=state['steps'];best=state['best'];history=state['history']
        random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy']);torch.set_rng_state(state['rng_torch'])
        torch.cuda.set_rng_state_all(state['rng_cuda']);del state
        print(f'Resuming completed epoch {start_epoch}; optimizer, scaler and history restored.',flush=True)
    if start_epoch>=s['epochs']:
        print('Configured training is already complete.',flush=True);return
    for epoch in range(start_epoch,s['epochs']):
        # Epoch-seeded order; persistent worker augmentation RNG may differ after a process restart.
        seed_all(s['seed']+epoch)
        model.train();optimizer.zero_grad(set_to_none=True)
        started=time.perf_counter();total=0.;overflows=0;updates=0;diag={}
        for i,raw in enumerate(loader):
            batch=scatter(raw,[0])[0]
            lr=s['learning_rate']*(.1**sum(epoch>=e for e in s['lr_steps']))*min(1.,(steps+1)/s['warmup_updates'])
            for group in optimizer.param_groups:group['lr']=lr
            group_start=(i//s['accumulate'])*s['accumulate']
            group_size=min(s['accumulate'],len(loader)-group_start)
            with torch.cuda.amp.autocast():loss,detail=parse_loss(model,model(**batch,return_loss=True))
            scaler.scale(loss/group_size).backward();total+=float(loss.detach())
            for k,v in detail.items():diag[k]=diag.get(k,0.)+v
            if (i+1)%s['accumulate']==0 or i+1==len(loader):
                scaler.unscale_(optimizer)
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),35.)
                before=scaler.get_scale()
                if not torch.isfinite(grad):
                    scaler.update(new_scale=max(1.,before/2));overflows+=1
                else:
                    scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()<before:overflows+=1
                    else:updates+=1
                if overflows>max(20,int((steps+1)*.05)):
                    raise FloatingPointError('Repeated gradient overflows. Last completed checkpoint is safe; inspect diagnostics.')
                optimizer.zero_grad(set_to_none=True);steps+=1
            if i%50==0:
                progress=dict(stage='train',epoch=epoch+1,total_epochs=s['epochs'],iteration=i+1,
                    iterations=len(loader),loss=total/(i+1),lr=lr,seconds=time.perf_counter()-started)
                save_json(run/'progress.json',progress)
                print(f'Epoch {epoch+1}/{s["epochs"]} | {i+1}/{len(loader)} | loss {total/(i+1):.4f} | LR {lr:.2g}',flush=True)
        train_seconds=time.perf_counter()-started
        directory=run/f'epoch_{epoch+1:02d}'
        save_json(run/'progress.json',dict(stage='validation',epoch=epoch+1,images=1547))
        metrics=evaluate(model,s,directory)
        metrics.update(epoch=epoch+1,train_loss=total/len(loader),train_seconds=train_seconds,
                       learning_rate=lr,overflow_updates=overflows,successful_updates=updates)
        save_json(directory/'training_diagnostics.json',{k:v/len(loader) for k,v in diag.items()})
        history.append(metrics);cpu=state_cpu(model)
        if metrics['map']>best:
            best=metrics['map'];atomic_torch(dict(state_dict=cpu,epoch=epoch+1,metrics=metrics,settings=s,signature=sig),run/'best.pth')
        checkpoint=dict(state_dict=cpu,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=epoch+1,
            steps=steps,best=best,history=history,signature=sig,rng_python=random.getstate(),rng_numpy=np.random.get_state(),
            rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        atomic_torch(checkpoint,run/'last.pth')
        save_json(run/'history.json',history);plots(history,run/'learning_curves.png')
        remaining=(s['epochs']-epoch-1)*np.mean([h['train_seconds']+h['seconds'] for h in history[-3:]])/3600
        save_json(run/'progress.json',dict(stage='epoch_complete',epoch=epoch+1,map=metrics['map'],best=best,
                  remaining_hours_estimate=float(remaining),target_crossed=best>62))
        print(f'Epoch {epoch+1} saved | AP {metrics["map"]:.3f} | best {best:.3f} | estimated remaining {remaining:.1f} h',flush=True)
        del checkpoint,cpu;gc.collect()
        if (run/'STOP_AFTER_EPOCH').exists():
            print('Stopped cleanly after saving the completed epoch. Remove STOP_AFTER_EPOCH to resume.',flush=True);return
    save_json(run/'completed.json',dict(best_map=best,epochs=s['epochs'],target_crossed=best>62))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['check','prepare','visuals','preflight','smoke','train','evaluate'])
    ap.add_argument('--settings',required=True);ap.add_argument('--mode',default='single',choices=['single','tta','tiles'])
    ap.add_argument('--limit',type=int);a=ap.parse_args();s=read_json(a.settings)
    seed_all(s['seed']);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.backends.cudnn.benchmark=False
    if a.command=='check':check_environment(s)
    elif a.command=='prepare':prepare(s)
    elif a.command=='visuals':preprocessing_visuals(s)
    elif a.command in ('preflight','smoke'):preflight(s,a.command=='smoke')
    elif a.command=='train':train(s)
    elif a.command=='evaluate':
        run=Path(s['root'])/'runs'/s['run_name'];model,cfg=create_model(s,checkpoint=run/'best.pth')
        evaluate(model,s,run/f'final_{a.mode}',a.mode,a.limit)

if __name__=='__main__':main()
