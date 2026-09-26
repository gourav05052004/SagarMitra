"""Resumable export of the verified 61.5735 pipeline; no training or subprocesses."""
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
LAB = WORK / 'Maritime_CBNet63_LabFinal_backend'
OLD = WORK / 'Maritime_MaCVi_611625_backend/runs/submission_611625_001'
TEST = WORK / 'Maritime_CBNet63_TestSubmission_backend'
QUALITY = WORK / 'Maritime_CBNet62_QualityEnsemble_backend/runs/quality_001'
EVIDENCE = WORK / 'Maritime_CBNet62_Evidence_backend'
PIXEL = WORK / 'Maritime_CBNet62_PixelBoundary_backend'
RUN = ROOT / 'runs/submission_615735_001'
OUTPUT = WORK / 'macvi_latest_61_5735_odv2.json'
for directory in (LAB, TEST, EVIDENCE, PIXEL, WORK/'Maritime_CBNet6215_Fast_backend'):
    sys.path.insert(0, str(directory))
import evaluate_complement as e
import submission as sub
import numpy as np
import torch


def module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def clean(rows):
    return [{k: r[k] for k in ('image_id', 'category_id', 'bbox', 'score')} for r in rows]


def status(stage, **values):
    e.log(RUN, stage, **values)


def freeze():
    """Freeze current verified winner and hashes, including reusable test shards."""
    import notebook_validation as nv
    path, record = nv.select_best()
    if record.get('kind') != 'tiny_tile_refinement':
        raise RuntimeError('The verified winner has changed. This exporter must be updated for its new recipe.')
    plan = e.read(Path(record['verification']).parent/'plan.json')
    previous = plan['previous']
    evidence = e.read(previous['verification'])
    reference = e.read(LAB/'retained_pipeline.json')
    assert evidence['baseline_prediction_sha256'] == reference['prediction_sha256']
    old_spec = e.read(OLD/'spec.json')
    for name, digest in old_spec['hashes'].items():
        assert e.sha(name) == digest, 'Old export provenance changed: '+name
    report = e.read(OLD/'submission_report.json')
    assert e.sha(report['upload_file']) == report['sha256']
    doc = e.read(TEST/'prepared_test.json')
    official = e.read(TEST/'instances_test_nogt.json')
    assert len(doc['images']) == len(official['images']) == 3750
    expected = {im['id']: im for im in official['images']}
    assert len(expected) == 3750 and {im['id'] for im in doc['images']} == set(expected)
    shards = {}
    for im in doc['images']:
        assert (im['width'], im['height']) == (expected[im['id']]['width'], expected[im['id']]['height'])
        shards[str(im['id'])] = {}
        for kind in ('raw_views', 'predictions_by_image'):
            source = OLD/kind/f"{im['id']}.json"
            item = e.read(source)
            assert item['image_id'] == im['id'] and item['input_sha256'] == im['sha256']
            shards[str(im['id'])][kind] = e.sha(source)
    relation = e.read(WORK/'Maritime_CBNet62_RelationRank_backend/runs/relation_001/best_pipeline.json')
    assert relation['frozen_blends'] == {'1': 1., '3': 1., '4': 1.}
    paths = [Path(__file__), path, Path(record['predictions']), LAB/'retained_pipeline.json',
             Path(evidence['expert_checkpoint']), Path(reference['pixel_checkpoint']),
             Path(relation['frozen_checkpoint']), QUALITY/'normalization.pt',
             LAB/'camera_quality/geometry_portable.json', LAB/'camera_quality/portable_verification.json',
             TEST/'prepared_test.json', TEST/'instances_test_nogt.json',
             LAB/'evaluate_complement.py', LAB/'tiny_tile_refinement.py', LAB/'probe_tiles.py',
             LAB/'regression_detail_extension.py', LAB/'portable_tree.py',
             EVIDENCE/'ranking_core.py', PIXEL/'pixel_core.py', PIXEL/'pixel_run.py',
             WORK/'Maritime_CBNet62_PrecisionBox_backend/box_core.py',
             WORK/'Maritime_CBNet6215_Fast_backend/fast.py',
             WORK/'Maritime_CBNet6215_AllClassContext_backend/native_core.py',
             WORK/'Maritime_CBNet6215_AllClassContext_backend/weights/resnet18-f37072fd.pth']
    assert e.sha(evidence['expert_checkpoint']) == evidence['expert_checkpoint_sha256']
    assert e.sha(reference['pixel_checkpoint']) == reference['pixel_checkpoint_sha256']
    assert e.sha(relation['frozen_checkpoint']) == relation['frozen_checkpoint_sha256']
    spec = dict(record=record, validation_AP=record['metrics']['AP'], reference=reference,
                evidence=evidence, relation=relation, shards=shards,
                hashes={str(p.resolve()): e.sha(p) for p in paths},
                test_manifest=str(TEST/'prepared_test.json'), old_report=report,
                full_batch=2, tile_batch=1, tile_size=1536, imgsz=1280, half=False)
    spec['signature'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    target = RUN/'spec.json'
    if target.exists():
        assert e.read(target) == spec, 'Frozen inputs changed; do not mix resumable runs.'
    else:
        e.save(target, spec)
    print(f"Selected verified full-validation AP: {spec['validation_AP']:.6f}")
    print('Reusable CBNet test images: 3750/3750. Fresh CBNet inference is skipped.')
    print('Expert checkpoint:', evidence['expert_checkpoint'])
    print('Upload file after completion:', OUTPUT)
    return spec


def verify(spec):
    for path, digest in spec['hashes'].items():
        assert e.sha(path) == digest, 'Frozen dependency changed: '+path


def load_old(im, spec):
    results = []
    for kind in ('predictions_by_image', 'raw_views'):
        path = OLD/kind/f"{im['id']}.json"
        assert e.sha(path) == spec['shards'][str(im['id'])][kind]
        item = e.read(path)
        assert item['input_sha256'] == im['sha256'] and item['image_id'] == im['id']
        results.append(item['rows'] if kind == 'predictions_by_image' else item['sources'])
    return results


class Refiner:
    def __init__(self, spec, gpu=True):
        import ranking_core
        import pixel_core
        from portable_tree import predict
        self.core, self.pixel, self.tree_predict = ranking_core, pixel_core, predict
        self.spec = spec
        self.pixel_run = module('submission_pixel_profiles', PIXEL/'pixel_run.py')
        self.norm = torch.load(QUALITY/'normalization.pt', map_location='cpu')
        state = torch.load(spec['relation']['frozen_checkpoint'], map_location='cpu')
        w = state['model']
        self.scorer = ranking_core.ScoringHead(w['meta_mean'], w['meta_std'], state['settings']['score_cap']).eval()
        self.scorer.load_state_dict(w)
        state = torch.load(spec['reference']['pixel_checkpoint'], map_location='cpu')
        self.model = pixel_core.PrecisionHead(state['model']['mean'], state['model']['std']).eval()
        self.model.load_state_dict(state['model'])
        self.tree = LAB/'camera_quality/geometry_portable.json'
        tree_record = e.read(self.tree)
        assert tree_record['columns'] == list(range(128,197))
        assert tree_record['source_sha256'] == spec['reference']['tree_model_sha256']
        assert e.sha(self.tree) == e.read(LAB/'camera_quality/portable_verification.json')['model_sha256']
        self.encoder = None
        if gpu:
            import fast
            self.fast = fast
            self.encoder = fast.Encoder().cuda().eval()

    @torch.inference_mode()
    def apply(self, rows, sources, im, bgr, visual=None, metadata=None):
        import cv2
        core = self.core
        indices = core.eligible(rows, .001, 1024)
        categories = torch.tensor([rows[j]['category_id'] for j in indices], dtype=torch.long)
        if visual is None:
            visual = self.fast.descriptors(self.encoder, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                core.boxes(rows)[indices], dict(crop_size=96, context=2., feature_batch=64))
        if metadata is None:
            metadata = torch.from_numpy(core.metadata(rows, sources, im, indices))
        scores = torch.tensor([rows[j]['score'] for j in indices], dtype=torch.float32)
        deltas = []
        for start in range(0, len(indices), 512):
            sl = slice(start,start+512)
            deltas.extend(self.scorer(visual[sl], metadata[sl], categories[sl], scores[sl])[1].numpy())
        original = core.apply_scores(rows, indices, np.asarray(deltas), {'1':1., '3':1., '4':1.})
        adjusted_scores = torch.tensor([original[j]['score'] for j in indices], dtype=torch.float32)
        # The verified geometry tree never uses the first 128 visual columns.
        features = torch.cat([torch.zeros(len(indices),128),
            ((metadata.float()-self.norm['mean'])/self.norm['std']).clamp(-8,8),
            torch.logit(adjusted_scores.clamp(1e-5,1-1e-5))[:,None]], 1)
        delta = self.tree_predict(self.tree, features.numpy(), categories.numpy())
        output = core.apply_scores(original, indices, delta, self.spec['reference']['tree_strengths'])
        chosen = [j for j in indices if original[j]['score'] >= .05]
        boxes = torch.tensor([original[j]['bbox'] for j in chosen], dtype=torch.float32).reshape(-1,4)
        boxes[:,2:] += boxes[:,:2]
        profiles = torch.from_numpy(self.pixel_run.extract_profiles(self.pixel_run.image_fields(bgr), boxes.numpy()))
        cats = torch.tensor([original[j]['category_id'] for j in chosen], dtype=torch.long)
        parts=[]
        for start in range(0,len(chosen),256):
            sl=slice(start,start+256)
            parts.append(self.model(profiles[sl], torch.zeros(len(profiles[sl]),68), cats[sl], boxes[sl]).numpy())
        values=np.concatenate(parts) if parts else np.empty((0,4),np.float32)
        localized = self.pixel.apply_boxes(original, chosen, values,
            self.spec['reference']['pixel_strengths'], im['width'], im['height'])
        for row, loc in zip(output, localized):
            row['bbox'] = loc['bbox']
        return clean(output)


def parity(spec, gpu=False, count=3):
    """Check the new export integration against recorded validation detections."""
    import cv2
    from collections import defaultdict
    doc=e.read(e.DOC)
    expected=defaultdict(list)
    for row in e.read(spec['reference']['predictions']): expected[row['image_id']].append(row)
    parent=EVIDENCE/'runs/ranking_001/cache/val_features'
    model=Refiner(spec,gpu=gpu)
    worst=0.
    try:
        for im in doc['images'][:count]:
            shard=torch.load(parent/f"{im['id']}.pt",map_location='cpu')
            if gpu:
                view=WORK/'Maritime_CBNet61_DetailHead_backend/runs/detail_001/validation_00200/cache/tiles_c6f725606c2aa9b9/views'/f"{im['id']}.json"
                sources=e.read(view)['sources']
            else: sources={}
            actual=model.apply(shard['rows'],sources,im,cv2.imread(im['file_name']),
                None if gpu else shard['visual'], None if gpu else shard['meta'])
            target=expected[im['id']]
            assert len(actual)==len(target)
            for a,b in zip(actual,target):
                assert (a['image_id'],a['category_id'])==(b['image_id'],b['category_id'])
                box_error=max(abs(x-y) for x,y in zip(a['bbox'],b['bbox']))
                score_error=abs(a['score']-b['score'])
                assert box_error<.01 and score_error<1e-4, (im['id'],box_error,score_error)
                worst=max(worst,box_error,score_error)
        print('Refinement parity passed:',count,'images; max difference',worst,flush=True)
        e.save(RUN/('fresh_parity.json' if gpu else 'cached_parity.json'),dict(images=count,max_difference=worst,signature=spec['signature']))
    finally:
        del model;gc.collect()
        if gpu: torch.cuda.empty_cache()


def wait_for_gpu(wait=True):
    import psutil
    while True:
        busy=[]
        for name in ('complement_medium','medium_delivery_verification'):
            path=LAB/name/'progress.json'
            if not path.exists(): continue
            state=e.read(path)
            try:
                proc=psutil.Process(state['pid'])
                if proc.pid!=os.getpid() and proc.is_running() and proc.create_time()<=path.stat().st_mtime+1:
                    busy.append(name)
            except (KeyError,psutil.NoSuchProcess): pass
        if not busy: return
        if not wait: raise RuntimeError('GPU jobs still active: '+', '.join(busy))
        status('waiting_for_gpu',jobs=busy,note='Existing work is untouched. Interrupt to stop waiting.')
        time.sleep(15)


def cache_read(stage, im, spec):
    path=RUN/stage/f"{im['id']}.json"
    if not path.exists(): return None
    item=e.read(path)
    assert item['signature']==spec['signature'] and item['input_sha256']==im['sha256'] and item['image_id']==im['id']
    sub.validate_rows(item['rows'],[im])
    return item['rows']


def cache_save(stage, im, rows, spec):
    rows=clean(rows);sub.validate_rows(rows,[im])
    e.save(RUN/stage/f"{im['id']}.json",dict(signature=spec['signature'],image_id=im['id'],input_sha256=im['sha256'],rows=rows))


def tick(stage, n, total, new, started):
    if n==1 or n%10==0 or n==total:
        elapsed=time.monotonic()-started
        status(stage,images=n,total=total,new_images=new,elapsed_seconds=round(elapsed,1),
            remaining_stage_seconds=round((total-n)*elapsed/new) if new else None)


def run(spec, wait=True):
    """All work runs in this Jupyter kernel; completed images resume safely."""
    import cv2
    import psutil
    verify(spec)
    assert torch.__version__.startswith('2.0.1') and torch.cuda.is_available(), 'Choose CBNet Diagnostics kernel.'
    lock=RUN/'RUNNING.json'
    if lock.exists():
        previous=e.read(lock)
        try:
            p=psutil.Process(previous['pid'])
            if abs(p.create_time()-previous['created'])<1: raise RuntimeError('This export is already running.')
        except psutil.NoSuchProcess: pass
        lock.unlink()
    with lock.open('x') as stream: json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),stream)
    try:
        wait_for_gpu(wait)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=True
        images=e.read(spec['test_manifest'])['images']
        for n,im in enumerate(images,1):
            assert e.sha(im['file_name'])==im['sha256'], 'Test image bytes changed'
            if n%500==0: print('Test files checked:',n,'/3750',flush=True)
        if not (RUN/'fresh_parity.json').exists(): parity(spec,gpu=True)
        else: assert e.read(RUN/'fresh_parity.json')['signature']==spec['signature']
        started=time.monotonic();new=0;model=None
        try:
            for n,im in enumerate(images,1):
                if cache_read('retained_branch',im,spec) is None:
                    if model is None: model=Refiner(spec,gpu=True)
                    rows,sources=load_old(im,spec)
                    out=model.apply(rows,sources,im,cv2.imread(im['file_name']))
                    cache_save('retained_branch',im,out,spec);new+=1
                tick('retained_refinement',n,len(images),new,started)
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
        import complement_detector as d
        import regression_detail_extension
        import probe_tiles as tiling
        import tiny_tile_refinement as tiny
        model=None
        try:
            started=time.monotonic();new=0
            for offset in range(0,len(images),2):
                part=images[offset:offset+2]
                missing=[im for im in part if cache_read('expert_full',im,spec) is None]
                if missing:
                    if model is None: model=d.YOLO(spec['evidence']['expert_checkpoint'])
                    assert {str(k):v for k,v in model.names.items()}==e.read(LAB/'complement_detector/dataset_manifest.json')['category_mapping']
                    results=model.predict([im['file_name'] for im in part],device='0',imgsz=1280,conf=.001,iou=.7,max_det=100,half=False,augment=False,verbose=False,save=False)
                    for im,pred in zip(part,results):
                        if im not in missing: continue
                        rows=[]
                        for box,score,cat in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),pred.boxes.cls.cpu().numpy()):
                            x0,y0,x1,y1=map(float,box)
                            x0,x1=np.clip([x0,x1],0,im['width']).tolist();y0,y1=np.clip([y0,y1],0,im['height']).tolist()
                            if x1>x0 and y1>y0: rows.append(dict(image_id=im['id'],category_id=int(cat)+1,bbox=[x0,y0,x1-x0,y1-y0],score=float(score)))
                        cache_save('expert_full',im,rows,spec);new+=1
                tick('expert_full',min(offset+2,len(images)),len(images),new,started)
            started=time.monotonic();new=0
            for n,im in enumerate(images,1):
                if cache_read('final',im,spec) is not None:
                    tick('expert_tiles_and_export',n,len(images),new,started);continue
                full=cache_read('expert_full',im,spec)
                base=cache_read('retained_branch',im,spec)
                fused=e.fuse(base,full,spec['evidence']['policy'])
                tiles=cache_read('expert_tiles',im,spec)
                if tiles is None:
                    if model is None: model=d.YOLO(spec['evidence']['expert_checkpoint'])
                    image=cv2.imread(im['file_name']);h,w=image.shape[:2]
                    assert (h,w)==(im['height'],im['width'])
                    raw=[]
                    for top in tiling.starts(h):
                        for left in tiling.starts(w):
                            right,bottom=min(w,left+1536),min(h,top+1536)
                            crop=image[top:bottom,left:right]
                            pred=model.predict(crop,device='0',imgsz=1280,conf=.001,iou=.7,max_det=100,half=False,augment=False,verbose=False,save=False)[0]
                            for box,score,cat in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),pred.boxes.cls.cpu().numpy()):
                                if int(cat)+1 not in (1,4): continue
                                x0,y0,x1,y1=map(float,box)
                                if ((left>0 and x0<=2) or (top>0 and y0<=2) or (right<w and x1>=crop.shape[1]-2) or (bottom<h and y1>=crop.shape[0]-2)): continue
                                if x1>x0 and y1>y0: raw.append(dict(image_id=im['id'],category_id=int(cat)+1,bbox=[left+x0,top+y0,x1-x0,y1-y0],score=float(score)))
                    # Suppression is class independent; other classes never enter tiny correction.
                    tiles=tiling.suppress([r for r in full if r['category_id'] in (1,4)]+raw,'soft')
                    cache_save('expert_tiles',im,tiles,spec)
                final,_=tiny.refine(fused,tiles,dict(images=[im]))
                cache_save('final',im,final,spec);new+=1
                tick('expert_tiles_and_export',n,len(images),new,started)
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
        return export(spec)
    except KeyboardInterrupt:
        status('paused',note='Completed images saved. Rerun this cell to resume.')
        return dict(paused=True)
    finally:
        lock.unlink(missing_ok=True)


def export(spec):
    verify(spec)
    images=e.read(spec['test_manifest'])['images'];rows=[];empty=[]
    for im in images:
        part=cache_read('final',im,spec)
        if part is None: raise RuntimeError(f"Incomplete export: missing image {im['id']}; rerun generation.")
        rows.extend(part)
        if not part: empty.append(im['id'])
    sub.validate_rows(rows,images)
    # Plain COCO results array: upload this file, not spec.json or the report.
    temp=OUTPUT.with_suffix('.json.partial')
    temp.write_text(json.dumps(rows,separators=(',',':'),allow_nan=False),encoding='utf-8')
    os.replace(temp,OUTPUT)
    report=dict(upload_file=str(OUTPUT),sha256=e.sha(OUTPUT),images=3750,detections=len(rows),
        validation_AP=spec['validation_AP'],test_AP=None,empty_image_ids=empty,
        note='Test AP is computed by MaCVi using withheld labels. No score guarantee.',signature=spec['signature'])
    e.save(RUN/'submission_report.json',report)
    status('completed',**report)
    return report
