"""Exact resumable SeaDronesSee ODv2 export for the 61.162515 validation pipeline."""
import collections, contextlib, copy, gc, hashlib, json, math, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
RUN=ROOT/'runs/submission_611625_001'
TEST=WORK/'Maritime_CBNet63_TestSubmission_backend'
DETAIL=WORK/'Maritime_CBNet61_DetailHead_backend'
ALLCLASS=WORK/'Maritime_CBNet6215_AllClassContext_backend'
CLASS_SAFE=WORK/'Maritime_CBNet6215_ClassSafe_backend'
UPDATED=WORK/'Maritime_CBNet615_Consensus_backend'
FASTSEARCH=WORK/'Maritime_CBNet61_FastSearch_backend'
FINAL_RESULT=UPDATED/'runs/calibration_001/result.json'
PREPARED=TEST/'prepared_test.json'
OFFICIAL=TEST/'instances_test_nogt.json'
OUTPUT=WORK/'macvi_cbnet_61_1625_updated_odv2.json'

sys.path.insert(0,str(TEST))
import submission as sub

FUSION={1:(.85,.70),2:(1.5,.85),3:(2.,.60),4:(4.,.90),5:(4.,.85)}
CONSENSUS={1:dict(kind='vote',iou=.55,blend=.25,max_shift=.08),
           2:dict(kind='vote',iou=.85,blend=.50,max_shift=.08),
           3:dict(kind='identity'),
           4:dict(kind='support',iou=.85,strength=.25),
           5:dict(kind='vote',iou=.85,blend=.50,max_shift=.08)}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def log(**values):
    values['time']=time.strftime('%Y-%m-%d %H:%M:%S');save(RUN/'progress.json',values);print(json.dumps(values),flush=True)
def stop():
    if (RUN/'STOP').exists():raise KeyboardInterrupt('STOP requested; remove the file and rerun to resume.')

def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    final=read(FINAL_RESULT);assert abs(final['metrics']['AP']-61.16251511566342)<1e-8
    detail=read(ALLCLASS/'starting_model.json');checkpoint=Path(detail['checkpoint'])
    edge_ref=read(ALLCLASS/'allclass_reference.json');edge_checkpoint=Path(edge_ref['checkpoint'])
    class_best=read(CLASS_SAFE/'runs/classsafe_002/best_pipeline.json');class_checkpoint=Path(class_best['checkpoint'])
    dependencies=[Path(__file__),FINAL_RESULT,UPDATED/'consensus.py',UPDATED/'calibration.py',PREPARED,OFFICIAL,
                  checkpoint,edge_checkpoint,class_checkpoint,CLASS_SAFE/'classsafe.py',ALLCLASS/'edge_model.py',
                  ALLCLASS/'native_core.py',ALLCLASS/'weights/resnet18-f37072fd.pth']
    spec=dict(validation_AP=final['metrics']['AP'],validation_metrics=final['metrics'],track='SeaDronesSee Object Detection v2',
              detector_checkpoint=str(checkpoint),detector_settings=detail['settings'],edge_checkpoint=str(edge_checkpoint),
              classsafe_checkpoint=str(class_checkpoint),fusion={str(k):v for k,v in FUSION.items()},
              consensus={str(k):v for k,v in CONSENSUS.items()},test_manifest_sha256=sha(PREPARED),official_metadata_sha256=sha(OFFICIAL),
              hashes={str(p.resolve()):sha(p) for p in dependencies})
    path=RUN/'spec.json'
    if path.exists() and read(path)!=spec:raise RuntimeError('Frozen inputs changed. Use a new run directory.')
    save(path,spec);return spec

def verify_spec(spec):
    for path,digest in spec['hashes'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen input changed: '+path)
    if sha(PREPARED)!=spec['test_manifest_sha256'] or sha(OFFICIAL)!=spec['official_metadata_sha256']:
        raise RuntimeError('Test manifest or official metadata changed.')

def load_detector(spec):
    for name in ('worker','recovery','inference_tools','fixed_fusion','data_tools'):
        module=sys.modules.get(name)
        if module and not Path(module.__file__).resolve().is_relative_to(DETAIL):raise RuntimeError('Restart kernel: conflicting detector modules loaded.')
    sys.path.insert(0,str(DETAIL))
    import torch,worker,recovery
    from inference_tools import infer_at_scale,_translate_tile
    from data_tools import tile_windows
    settings=copy.deepcopy(spec['detector_settings']);settings.update(root=str(DETAIL),init_checkpoint=spec['detector_checkpoint'])
    worker.seed_all(46);torch.set_num_threads(8);torch.backends.cudnn.benchmark=False
    recovery.SESSION_STOP=None;recovery.DEADLINE=None
    model,_=worker.create_model(settings,checkpoint=spec['detector_checkpoint']);model.requires_grad_(False);model.eval()
    return dict(model=model,worker=worker,infer=infer_at_scale,translate=_translate_tile,windows=tile_windows,settings=settings)

def infer_sources(models,im):
    import cv2,torch
    bgr=cv2.imread(im['file_name'])
    if bgr is None or bgr.shape[:2]!=(im['height'],im['width']):raise RuntimeError('Bad test image dimensions')
    m=models;s=m['settings'];det=m['model']
    with torch.inference_mode():
        sources={'full':m['worker'].prediction_rows(m['infer'](det,bgr,s['eval_scale']),im['id'])}
        for index,(left,top,right,bottom) in enumerate(m['windows'](im['width'],im['height'],s['tile_size'],s['tile_overlap'])):
            rows=m['worker'].prediction_rows(m['infer'](det,bgr[top:bottom,left:right],s['eval_scale']),im['id'])
            sources[f'tile{index}']=m['translate'](rows,im['id'],left,top,right,bottom,im['width'],im['height'])
    for rows in sources.values():sub.validate_rows(rows,[im])
    return sources

def detector_stage(spec):
    images=read(PREPARED)['images'];cache=RUN/'raw_views';cache.mkdir(exist_ok=True)
    missing=[im for im in images if not (cache/f"{im['id']}.json").exists()]
    if not missing:log(stage='raw_views_complete',checked=len(images),total=len(images),new=0);return
    models=load_detector(spec);started=time.monotonic();new=0
    try:
        for index,im in enumerate(images,1):
            stop();path=cache/f"{im['id']}.json"
            if path.exists():
                item=read(path);assert item['image_id']==im['id'] and item['input_sha256']==im['sha256']
            else:
                if sha(im['file_name'])!=im['sha256']:raise RuntimeError('Test image bytes changed')
                tick=time.perf_counter();sources=infer_sources(models,im);seconds=time.perf_counter()-tick
                save(path,dict(image_id=im['id'],input_sha256=im['sha256'],sources=sources,seconds=seconds));new+=1
            if index%10==0 or index==len(images):
                elapsed=time.monotonic()-started;remaining=(len(images)-index)*elapsed/max(new,1)
                log(stage='raw_test_inference',checked=index,total=len(images),new=new,elapsed_seconds=elapsed,remaining_estimate_seconds=remaining)
    finally:
        del models;gc.collect()
        import torch;torch.cuda.empty_cache()

def load_refiners(spec):
    for name in ('native_core','edge_model'):
        module=sys.modules.get(name)
        if module and Path(module.__file__).resolve().parent!=ALLCLASS:raise RuntimeError('Restart kernel: conflicting refiner modules loaded.')
    sys.path.insert(0,str(ALLCLASS));sys.path.insert(0,str(CLASS_SAFE))
    import torch,native_core as native,classsafe as cs
    from edge_model import EdgeAlignedRefiner
    edge_state=torch.load(spec['edge_checkpoint'],map_location='cpu');edge_s=edge_state['settings']
    edge=EdgeAlignedRefiner(edge_s['boundary_checkpoint']).cuda().eval();edge.load_state_dict(edge_state['state_dict']);edge.requires_grad_(False)
    cs_s=read(CLASS_SAFE/'runs/classsafe_002/settings.json');state=torch.load(spec['classsafe_checkpoint'],map_location='cpu')
    classsafe=cs.Model(cs_s).cuda().eval();classsafe.load_state_dict(state['state_dict']);classsafe.requires_grad_(False)
    encoder=cs.fast.Encoder().cuda().eval();encoder.requires_grad_(False)
    return dict(torch=torch,native=native,cs=cs,edge=edge,edge_s=edge_s,classsafe=classsafe,cs_s=cs_s,encoder=encoder)

def fuse_sources(sources,image_id,models):
    # Same linear Soft-NMS implementation used by the measured validation pipeline.
    if str(FASTSEARCH) not in sys.path:sys.path.insert(0,str(FASTSEARCH))
    import fusion_probe as fp
    rows=[]
    for category in range(1,6):
        weight,iou=FUSION[category];rows.extend(fp.fuse_class(sources,image_id,category,weight,iou))
    return rows

def apply_refiners(rows,bgr,im,m):
    import cv2,numpy as np
    torch,native,cs=m['torch'],m['native'],m['cs'];rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        indices=[i for i,r in enumerate(rows) if native.eligible(r,m['edge_s'])]
        if indices:
            deltas=[];quality=[]
            for offset in range(0,len(indices),96):
                part=indices[offset:offset+96];patches=np.stack([native.crop_rgb(rgb,native.xyxy(rows[i]),m['edge_s']) for i in part])
                classes=torch.tensor([native.TARGETS.index(rows[i]['category_id']) for i in part],device='cuda')
                with torch.cuda.amp.autocast():d,q=m['edge'](native.tensor_crops(patches),classes)
                deltas.extend(d.float().cpu().numpy());quality.extend(q.float().sigmoid().cpu().numpy())
            rows=native.apply_deltas(rows,indices,deltas,quality,im['width'],im['height'],m['edge_s'])
        indices=[i for i,r in enumerate(rows) if cs.fast.eligible(r,cs.fast.settings())]
        if indices:
            boxes=[native.xyxy(rows[i]) for i in indices];features=cs.fast.descriptors(m['encoder'],rgb,boxes,cs.fast.settings())
            classes=torch.tensor([rows[i]['category_id']-1 for i in indices],device='cuda')
            delta=m['classsafe'](features.cuda(),classes).cpu();updated=native.decode(torch.tensor(np.stack(boxes)),delta).numpy()
            for k,i in enumerate(indices):
                box=native.clip_box(updated[k],im['width'],im['height'])
                if min(box[2:]-box[:2])>=1 and torch.count_nonzero(delta[k]):rows[i]['bbox']=[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])]
    return rows

def postprocess(sources,bgr,im,m):
    sys.path.insert(0,str(UPDATED));import consensus,calibration
    fused=apply_refiners(fuse_sources(sources,im['id'],m),bgr,im,m)
    output=[]
    for category in range(1,6):
        rows=[r for r in fused if r['category_id']==category]
        raw={name:[r for r in values if r['category_id']==category] for name,values in sources.items()}
        stats=consensus.describe(rows,raw);rows=consensus.transform(rows,stats,CONSENSUS[category],im)
        if category==4:
            rows=calibration.transform(rows,dict(kind='geometry',dx=0,dy=0,sx=1.02,sy=1.02,pixel=0),im)
        output.extend(rows)
    sub.validate_rows(output,[im]);return output

def postprocess_stage(spec):
    import cv2
    images=read(PREPARED)['images'];cache=RUN/'predictions_by_image';cache.mkdir(exist_ok=True);models=load_refiners(spec)
    started=time.monotonic();new=0
    try:
        for index,im in enumerate(images,1):
            stop();dest=cache/f"{im['id']}.json";raw=read(RUN/'raw_views'/f"{im['id']}.json")
            if dest.exists():
                item=read(dest);assert item['image_id']==im['id'] and item['input_sha256']==im['sha256'];sub.validate_rows(item['rows'],[im])
            else:
                bgr=cv2.imread(im['file_name']);tick=time.perf_counter();rows=postprocess(raw['sources'],bgr,im,models)
                save(dest,dict(image_id=im['id'],input_sha256=im['sha256'],rows=rows,seconds=time.perf_counter()-tick));new+=1
            if index%25==0 or index==len(images):log(stage='updated_postprocess',checked=index,total=len(images),new=new,elapsed_seconds=time.monotonic()-started)
    finally:
        del models;gc.collect()
        import torch;torch.cuda.empty_cache()

def export(spec):
    images=read(PREPARED)['images'];all_rows=[];empty=[];times=[]
    for im in images:
        item=read(RUN/'predictions_by_image'/f"{im['id']}.json");sub.validate_rows(item['rows'],[im]);all_rows.extend(item['rows']);times.append(item['seconds'])
        if not item['rows']:empty.append(im['id'])
    sub.validate_rows(all_rows,images);save(OUTPUT,all_rows)
    copied=RUN/'macvi_seadronessee_odv2_predictions.json';save(copied,all_rows)
    report=dict(upload_file=str(OUTPUT),sha256=sha(OUTPUT),images_processed=len(images),detections=len(all_rows),
                images_without_detections=len(empty),empty_image_ids=empty,schema='COCO results list; original-pixel [x,y,width,height]; categories 1..5',
                validation_AP=spec['validation_AP'],test_AP=None,test_AP_status='Withheld ground truth; available only after MaCVi upload',
                hardware='NVIDIA RTX 4000 Ada Generation',pipeline='fine class fusion + EdgeAligned + ClassSafe + multi-view consensus + appliance calibration',
                track=spec['track'],uploaded=False)
    save(RUN/'submission_report.json',report);log(stage='complete',**report);return report

def run():
    import torch,mmcv,mmdet
    spec=freeze();verify_spec(spec)
    assert torch.cuda.is_available() and torch.__version__.startswith('2.0.1') and mmcv.__version__=='1.7.2' and mmdet.__version__=='2.28.2'
    if (RUN/'submission_report.json').exists():return export(spec)
    try:
        detector_stage(spec);verify_spec(spec);postprocess_stage(spec);verify_spec(spec);return export(spec)
    except KeyboardInterrupt:
        log(stage='paused',message='Completed per-image caches are safe. Remove STOP and rerun.');return dict(paused=True)

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
