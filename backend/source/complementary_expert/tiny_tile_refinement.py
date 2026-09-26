"""Fixed tiny-object coordinate correction; no confidence or category changes."""
import copy
import hashlib
import math
from pathlib import Path
import evaluate_complement as e

ROOT=e.ROOT/'tiny_tile_refinement'


def refine(base, tiles, doc):
    images={im['id']:im for im in doc['images']}
    indices=[]
    for i,row in enumerate(base):
        im=images[row['image_id']]
        effective_size=math.sqrt(row['bbox'][2]*row['bbox'][3])*1280/max(im['width'],im['height'])
        if row['category_id'] in (1,4) and effective_size<16:
            indices.append(i)
    indices.sort(key=lambda i:(base[i]['image_id'],base[i]['category_id'],-base[i]['score']))
    policy=dict(name='tiny_tile_geometry',categories=[1,4],box_weight=.4,score_weight=0.,add_missing=False)
    corrected=e.fuse([base[i] for i in indices],tiles,policy)
    assert len(corrected)==len(indices)
    result=copy.deepcopy(base); changed=0
    for i,row in zip(indices,corrected):
        assert all(row[k]==base[i][k] for k in ('image_id','category_id','score'))
        changed+=row['bbox']!=base[i]['bbox']
        result[i]['bbox']=row['bbox']
    return result,dict(eligible=len(indices),changed=changed,policy=policy)


def run():
    previous=e.read(e.ROOT/'regression_detail_completion/best_verified_pipeline.json')
    tile_folder=e.ROOT/'complement_tile_evaluations/8efc6e4704633c90'
    tiled=e.read(tile_folder/'result.json')
    assert e.sha(previous['predictions'])==previous['prediction_sha256']
    assert e.sha(tiled['predictions'])==tiled['prediction_sha256']
    doc=e.read(e.DOC); ids=[im['id'] for im in doc['images']]
    screen=sorted(ids,key=lambda iid:hashlib.sha256(f'complement-screen-92463:{iid}'.encode()).hexdigest())[:256]
    remainder=[iid for iid in ids if iid not in set(screen)]
    e.save(ROOT/'plan.json',dict(previous=previous,tiled=tiled,screen_ids=screen,
        categories=[1,4],effective_size_below=16,reference_long_side=1280,
        code_sha256=e.sha(__file__),note='One fixed inference-only rule; size is derived from predicted boxes, never ground truth.'))
    base=e.read(previous['predictions'])
    rows,details=refine(base,e.read(tiled['predictions']),doc)
    e.validate_rows(rows,doc)
    subset=e.score(rows,doc,screen); reference=e.score(base,doc,screen)
    report=dict(screen=subset,screen_reference=reference,details=details,accepted=False)
    e.log(ROOT,'screen',AP=subset['AP'],reference_AP=reference['AP'],**details)
    if subset['AP']>reference['AP']:
        full=e.score(rows,doc,ids); rem=e.score(rows,doc,remainder); old=e.score(base,doc,remainder)
        e.save(ROOT/'predictions.json',rows)
        report.update(metrics=full,remainder=rem,remainder_reference=old,
            accepted=full['AP']>previous['metrics']['AP'] and rem['AP']>=old['AP'],
            predictions=str(ROOT/'predictions.json'),prediction_sha256=e.sha(ROOT/'predictions.json'))
    e.save(ROOT/'result.json',report)
    e.log(ROOT,'completed',**report)


if __name__=='__main__':run()
