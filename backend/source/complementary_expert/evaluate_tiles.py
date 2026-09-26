"""Fresh tiled complement evaluation after the training/full-image worker exits."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import cv2
import psutil
import complement_detector as d
import evaluate_complement as e
import probe_tiles as tiling


def evaluate(full_folder):
    full_folder=Path(full_folder)
    source=e.read(full_folder/'manifest.json');result=e.read(full_folder/'result.json')
    assert e.sha(result['predictions'])==result['prediction_sha256']
    checkpoint=full_folder/'checkpoint.pt';digest=e.sha(checkpoint)
    assert digest==source['settings']['checkpoint_sha256']
    assert e.sha(e.DOC)==source['settings']['annotation_sha256']
    doc=e.read(e.DOC);assert len(doc['images'])==1547
    settings=dict(checkpoint_sha256=digest,annotation_sha256=e.sha(e.DOC),
        full_prediction_sha256=result['prediction_sha256'],tile_size=1536,stride_fraction=.75,
        imgsz=1280,edge_margin=2,conf=.001,iou=.7,max_det=100,fusion='linear_soft_nms_0.6',
        code_sha256=e.sha(__file__),tiling_code_sha256=e.sha(tiling.__file__),
        evaluator_code_sha256=e.sha(e.__file__))
    signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    folder=e.ROOT/'complement_tile_evaluations'/signature[:16];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():assert e.read(folder/'manifest.json')['settings']==settings
    else:e.save(folder/'manifest.json',dict(settings=settings,signature=signature,full_source=str(full_folder)))
    snapshot=folder/'checkpoint.pt'
    if not snapshot.exists():shutil.copy2(checkpoint,snapshot)
    assert e.sha(snapshot)==digest
    full=defaultdict(list)
    for row in e.read(result['predictions']):full[row['image_id']].append(row)
    d.torch.set_num_threads(4);model=d.YOLO(str(snapshot))
    assert {str(k):v for k,v in model.names.items()}==e.read(d.ROOT/'dataset_manifest.json')['category_mapping']
    output=[];start=time.monotonic();fresh=0
    for number,im in enumerate(doc['images'],1):
        stat=Path(im['file_name']).stat();stamp=[stat.st_size,stat.st_mtime_ns]
        source_cache=e.read(full_folder/'images'/f"{im['id']}.json")
        assert source_cache['image_stamp']==stamp
        cache=folder/'images'/f"{im['id']}.json"
        if cache.exists():
            saved=e.read(cache);assert saved['signature']==signature and saved['image_stamp']==stamp
            rows=saved['rows']
        else:
            image=cv2.imread(im['file_name']);assert image is not None
            height,width=image.shape[:2];assert (height,width)==(im['height'],im['width'])
            tiles=[]
            for top in tiling.starts(height):
                for left in tiling.starts(width):
                    right=min(width,left+1536);bottom=min(height,top+1536)
                    crop=image[top:bottom,left:right]
                    prediction=model.predict(crop,device='0',imgsz=1280,conf=.001,iou=.7,max_det=100,
                        half=False,augment=False,verbose=False,save=False)[0]
                    assert prediction.orig_shape==crop.shape[:2]
                    for box,confidence,category in zip(prediction.boxes.xyxy.cpu().numpy(),
                            prediction.boxes.conf.cpu().numpy(),prediction.boxes.cls.cpu().numpy()):
                        x0,y0,x1,y1=map(float,box)
                        if ((left>0 and x0<=2) or (top>0 and y0<=2) or
                            (right<width and x1>=crop.shape[1]-2) or
                            (bottom<height and y1>=crop.shape[0]-2)):continue
                        if x1>x0 and y1>y0:tiles.append(dict(image_id=im['id'],category_id=int(category)+1,
                            bbox=[left+x0,top+y0,x1-x0,y1-y0],score=float(confidence)))
            rows=tiling.suppress(full[im['id']]+tiles,'soft')
            e.save(cache,dict(signature=signature,image_stamp=stamp,rows=rows));fresh+=1
        output.extend(rows)
        if number==1 or number%25==0 or number==len(doc['images']):
            e.log(folder,'tiled_inference',images=number,total=1547,new_images=fresh,seconds=time.monotonic()-start)
    e.validate_rows(output,doc);e.save(folder/'predictions.json',output)
    metrics=e.score(output,doc,[im['id'] for im in doc['images']])
    e.save(folder/'result.json',dict(metrics=metrics,predictions=str(folder/'predictions.json'),
        prediction_sha256=e.sha(folder/'predictions.json'),checkpoint_sha256=digest))
    record=e.evaluate_fusion(folder)
    record['tiled_expert']=True
    e.save(folder/'fusion_result.json',record)
    return folder,record


def wait_then_evaluate(pid,created):
    status=e.ROOT/'complement_tile_completion';deadline=time.monotonic()+3.5*3600
    while True:
        try:
            process=psutil.Process(pid)
            alive=abs(process.create_time()-created)<.01 and process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:alive=False
        if not alive:break
        e.log(status,'waiting_for_full_evaluation',worker_pid=pid,worker_create_time=created)
        if time.monotonic()>=deadline:
            e.log(status,'wait_limit_reached',note='Existing processes left untouched.')
            return
        time.sleep(30)
    previous=e.read(e.ROOT/'complement_completion/progress.json')
    assert previous['stage']=='completed', 'Full-image worker did not complete; inspect it before tiled inference.'
    folder,record=evaluate(previous['evaluation_folder'])
    retained=e.read(e.ROOT/'complement_completion/best_verified_pipeline.json')
    use=record['accepted'] and record['candidate']['AP']>retained['metrics']['AP']
    if use:
        retained=dict(metrics=record['candidate'],predictions=record['predictions'],
            prediction_sha256=record['prediction_sha256'],changed=True,target_reached=record['target_reached'],
            verification=str(folder/'fusion_result.json'))
    assert e.sha(retained['predictions'])==retained['prediction_sha256']
    e.save(status/'best_verified_pipeline.json',retained)
    e.log(status,'completed',candidate_AP=record['candidate']['AP'],best_verified_AP=retained['metrics']['AP'],
          selected_tiled=use,evaluation_folder=str(folder),target_reached=retained['metrics']['AP']>=63)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path);parser.add_argument('--pid',type=int)
    parser.add_argument('--created',type=float);args=parser.parse_args()
    try:
        if args.folder:evaluate(args.folder)
        else:
            assert args.pid is not None and args.created is not None
            wait_then_evaluate(args.pid,args.created)
    except Exception as exc:
        e.log(e.ROOT/'complement_tile_completion','failed',error=repr(exc));raise
