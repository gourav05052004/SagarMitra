"""Tiled complementary detector probe; no training or original-data writes."""
import argparse
from collections import defaultdict
import copy
import hashlib
from pathlib import Path
import time
import cv2
import numpy as np
import complement_detector as d
import evaluate_complement as e


def starts(length, size=1536):
    if length<=size:return [0]
    positions=list(range(0,length-size+1,int(size*.75)))
    if positions[-1]!=length-size:positions.append(length-size)
    return positions


def suppress(rows, method='soft', threshold=.6):
    grouped=defaultdict(list)
    for row in rows:grouped[row['category_id']].append(copy.deepcopy(row))
    output=[]
    for category,parts in grouped.items():
        while parts and sum(r['category_id']==category for r in output)<100:
            best=max(range(len(parts)),key=lambda i:parts[i]['score'])
            chosen=parts.pop(best);output.append(chosen)
            overlaps=e.overlap(chosen['bbox'],[r['bbox'] for r in parts])
            remaining=[]
            for row,iou in zip(parts,overlaps):
                if iou>threshold:row['score']*=0. if method=='hard' else 1-float(iou)
                if row['score']>=.001:remaining.append(row)
            parts=remaining
    return output


def run(checkpoint,count=64):
    d.torch.set_num_threads(4)
    doc=e.read(e.DOC);digest=e.sha(checkpoint)
    images=sorted(doc['images'],key=lambda im:hashlib.sha256(f"complement-screen-92463:{im['id']}".encode()).hexdigest())[:count]
    folder=e.ROOT/'complement_probes'/f'{digest[:12]}_{count}_tiles'
    folder.mkdir(parents=True,exist_ok=True)
    settings=dict(checkpoint_sha256=digest,tile_size=1536,stride_fraction=.75,imgsz=1280,
        conf=.001,iou=.7,max_det=100,edge_margin=2,code_sha256=e.sha(__file__),annotation_sha256=e.sha(e.DOC))
    if (folder/'manifest.json').exists():assert e.read(folder/'manifest.json')==settings
    else:e.save(folder/'manifest.json',settings)
    full_cache=e.ROOT/'complement_probes'/f'{digest[:12]}_128/images'
    model=d.YOLO(str(checkpoint));start=time.monotonic();combined={'hard':[],'soft':[]};full=[]
    for number,im in enumerate(images,1):
        cache=folder/'images'/f"{im['id']}.json"
        stat=Path(im['file_name']).stat();stamp=[stat.st_size,stat.st_mtime_ns]
        original=e.read(full_cache/f"{im['id']}.json")
        assert original['checkpoint_sha256']==digest and original['image_stamp']==stamp
        full.extend(original['rows'])
        if cache.exists():
            cached=e.read(cache);assert cached['image_stamp']==stamp;rows=cached['rows']
        else:
            image=cv2.imread(im['file_name']);assert image is not None
            height,width=image.shape[:2];assert (height,width)==(im['height'],im['width'])
            rows=[]
            for top in starts(height):
                for left in starts(width):
                    right=min(width,left+1536);bottom=min(height,top+1536)
                    crop=image[top:bottom,left:right]
                    result=model.predict(crop,device='cpu',imgsz=1280,conf=.001,iou=.7,max_det=100,
                        half=False,augment=False,verbose=False,save=False)[0]
                    for box,score,category in zip(result.boxes.xyxy.numpy(),result.boxes.conf.numpy(),result.boxes.cls.numpy()):
                        x0,y0,x1,y1=map(float,box)
                        # Discard truncated boxes at interior crop edges; overlapping
                        # views and the full image remain available for these objects.
                        if ((left>0 and x0<=2) or (top>0 and y0<=2) or
                            (right<width and x1>=crop.shape[1]-2) or
                            (bottom<height and y1>=crop.shape[0]-2)):continue
                        if x1>x0 and y1>y0:
                            rows.append(dict(image_id=im['id'],category_id=int(category)+1,
                                bbox=[left+x0,top+y0,x1-x0,y1-y0],score=float(score)))
            e.save(cache,dict(image_stamp=stamp,rows=rows))
        for method in combined:combined[method].extend(suppress(original['rows']+rows,method))
        if number==1 or number%5==0 or number==len(images):
            e.log(folder,'cpu_tile_inference',images=number,total=len(images),seconds=time.monotonic()-start)
    assert e.sha(checkpoint)==digest
    ids=[im['id'] for im in images];selected=set(ids)
    reference=e.read(e.ROOT/'retained_pipeline.json')
    assert e.sha(reference['predictions'])==reference['prediction_sha256']
    base=[r for r in e.read(reference['predictions']) if r['image_id'] in selected]
    report=dict(full_detector=e.score(full,doc,ids),base=e.score(base,doc,ids),tiled={},fusion={})
    for method,rows in combined.items():
        e.validate_rows(rows,doc);e.save(folder/f'{method}_predictions.json',rows)
        report['tiled'][method]=e.score(rows,doc,ids)
        for mode,score,add in [('geometry',0.,False),('consensus',.15,False),('recall',.15,True)]:
            policy=dict(name=f'swimmer_jetski_{mode}',categories=[1,3],box_weight=.4,score_weight=score,add_missing=add)
            report['fusion'][f'{method}_{mode}']=e.score(e.fuse(base,rows,policy),doc,ids)
    report.update(checkpoint_sha256=digest,image_ids=ids,
        note='Early CPU subset probe, not the full validation benchmark. Original dataset and training untouched.')
    e.save(folder/'result.json',report)
    e.log(folder,'completed',base_AP=report['base']['AP'],full_detector_AP=report['full_detector']['AP'],
        tiled_AP={k:v['AP'] for k,v in report['tiled'].items()},fusion_AP={k:v['AP'] for k,v in report['fusion'].items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--images',type=int,default=64)
    args=parser.parse_args();run(args.checkpoint,args.images)
