"""Resumable fresh inference and unchanged COCO evaluation of the new detector.

This file never changes a training checkpoint or the retained CBNet pipeline.
Fusion settings are screened on a fixed 256-image development subset, then the
selected setting is checked on the remaining 1,291 and all 1,547 images.
The entire validation set has been used historically; the remainder is not an
independent test set. Only the MaCVi server can supply hidden-test AP.
"""
import argparse
from collections import defaultdict
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent / 'Maritime_CBNet61_DetailHead_backend/prepared/val.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def log(folder, stage, **values):
    row = dict(stage=stage, time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), **values)
    print(json.dumps(row), flush=True)
    save(folder / 'progress.json', row)


def validate_rows(rows, doc):
    images = {im['id']: im for im in doc['images']}
    categories = {cat['id'] for cat in doc['categories']}
    for row in rows:
        assert row['image_id'] in images and row['category_id'] in categories
        assert len(row['bbox']) == 4 and all(math.isfinite(v) for v in row['bbox'])
        assert row['bbox'][2] > 0 and row['bbox'][3] > 0
        assert math.isfinite(row['score']) and 0 <= row['score'] <= 1


def score(rows, doc, ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ids = sorted(ids)
    selected = set(ids)
    subset = [r for r in rows if r['image_id'] in selected]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(doc)
        gt.createIndex()
        if subset:
            dt = gt.loadRes(subset)
        else:
            dt = COCO()
            dt.dataset = dict(images=doc['images'], categories=doc['categories'], annotations=[])
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = ids
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval['precision']
    per_class = {}
    for index, cid in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, 2]
        values = values[values > -1]
        per_class[str(cid)] = float(values.mean() * 100) if len(values) else None
    return dict(AP=float(evaluator.stats[0] * 100), AP50=float(evaluator.stats[1] * 100),
                AP75=float(evaluator.stats[2] * 100), AP_small=float(evaluator.stats[3] * 100),
                AR100=float(evaluator.stats[8] * 100), per_class_AP=per_class,
                images=len(ids), metric='COCO bbox AP50:95', maxDets=100)


def infer(checkpoint, device='0', batch=2):
    # Import only here: CPU fusion/evaluation does not initialize CUDA or YOLO.
    import complement_detector as detector
    import torch
    import ultralytics
    torch.set_num_threads(4)
    checkpoint = Path(checkpoint).resolve()
    digest = sha(checkpoint)
    doc = read(DOC)
    dataset = read(ROOT / 'complement_detector/dataset_manifest.json')
    assert sha(DOC) == dataset['source_hashes']['val']
    assert len(doc['images']) == 1547
    settings = dict(checkpoint_sha256=digest, annotation_sha256=sha(DOC), imgsz=1280,
                    conf=.001, iou=.7, max_det=100, half=False, augment=False,
                    batch=batch, device=str(device), torch_version=torch.__version__,
                    ultralytics_version=ultralytics.__version__, fixed_batch_context=True,
                    code_sha256=sha(__file__), category_mapping=dataset['category_mapping'])
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    folder = ROOT / 'complement_evaluations' / signature[:16]
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        assert read(manifest)['settings'] == settings
    else:
        save(manifest, dict(settings=settings, source_checkpoint=str(checkpoint), signature=signature))
    snapshot = folder / 'checkpoint.pt'
    if not snapshot.exists():
        temp = folder / 'checkpoint.pt.tmp'
        shutil.copy2(checkpoint, temp)
        assert sha(temp) == digest == sha(checkpoint), 'Checkpoint changed while copying; retry after save completes.'
        os.replace(temp, snapshot)
    assert sha(snapshot) == digest
    model = detector.YOLO(str(snapshot))
    assert {str(k): v for k, v in model.names.items()} == dataset['category_mapping']
    start = time.monotonic()
    by_image = {}
    entries = []
    for im in doc['images']:
        path = folder / 'images' / f"{im['id']}.json"
        stamp = Path(im['file_name']).stat()
        identity = [stamp.st_size, stamp.st_mtime_ns]
        if path.exists():
            cached = read(path)
            assert cached['signature'] == signature and cached['image_stamp'] == identity
            assert cached['image_id'] == im['id']
            by_image[im['id']] = cached['rows']
        entries.append((im, identity))
    complete = len(by_image)
    log(folder, 'inference_start', reused_images=complete, pending_images=len(entries)-complete, checkpoint_sha256=digest)
    for offset in range(0, len(entries), batch):
        # Preserve original batch membership when resuming: Ultralytics padding
        # can differ for equal-shape versus mixed-shape image batches. If a
        # partially saved batch exists, only that batch is inferred again.
        part = entries[offset:offset + batch]
        missing = {im['id'] for im, _ in part if im['id'] not in by_image}
        if not missing:
            continue
        results = model.predict([im['file_name'] for im, _ in part], imgsz=1280, conf=.001,
                                iou=.7, max_det=100, half=False, augment=False,
                                device=device, verbose=False, save=False)
        assert len(results) == len(part)
        for (im, stamp), result in zip(part, results):
            assert result.orig_shape == (im['height'], im['width'])
            if im['id'] not in missing:
                continue
            rows = []
            for box, confidence, category in zip(result.boxes.xyxy.cpu().numpy(),
                                                result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                x0, y0, x1, y1 = map(float, box)
                x0, x1 = np.clip([x0, x1], 0, im['width']).tolist()
                y0, y1 = np.clip([y0, y1], 0, im['height']).tolist()
                if x1 > x0 and y1 > y0:
                    rows.append(dict(image_id=im['id'], category_id=int(category) + 1,
                                     bbox=[x0, y0, x1 - x0, y1 - y0], score=float(confidence)))
            save(folder / 'images' / f"{im['id']}.json",
                 dict(signature=signature, image_id=im['id'], image_stamp=stamp, rows=rows))
            by_image[im['id']] = rows
        complete += len(missing)
        if offset // batch % max(1, 25 // batch) == 0 or complete == 1547:
            log(folder, 'inference', images=complete, total=1547, elapsed_seconds=time.monotonic() - start)
    assert complete == len(doc['images']) == len(by_image)
    output = [row for im in doc['images'] for row in by_image[im['id']]]
    validate_rows(output, doc)
    save(folder / 'predictions.json', output)
    log(folder, 'coco_evaluation', images=1547)
    metrics = score(output, doc, [im['id'] for im in doc['images']])
    save(folder / 'result.json', dict(metrics=metrics, checkpoint_sha256=digest,
                                     predictions=str(folder / 'predictions.json'),
                                     prediction_sha256=sha(folder / 'predictions.json')))
    log(folder, 'inference_complete', metrics=metrics)
    return folder


def overlap(one, many):
    if len(many) == 0:
        return np.zeros(0)
    many = np.asarray(many, dtype=np.float64)
    one = np.asarray(one, dtype=np.float64)
    low = np.maximum(one[:2], many[:, :2])
    high = np.minimum(one[:2] + one[2:], many[:, :2] + many[:, 2:])
    intersection = np.maximum(0, high - low).prod(axis=1)
    union = one[2] * one[3] + many[:, 2] * many[:, 3] - intersection
    return intersection / np.maximum(union, 1e-12)


def fuse(base, expert, policy):
    if policy['name'] == 'baseline':
        return copy.deepcopy(base)
    if policy['name'] == 'expert_alone':
        return copy.deepcopy(expert)
    if policy['name'] == 'expert_swimmer_jetski':
        return copy.deepcopy([r for r in base if r['category_id'] not in (1, 3)] +
                             [r for r in expert if r['category_id'] in (1, 3)])
    grouped = defaultdict(list)
    for r in expert:
        grouped[r['image_id'], r['category_id']].append(r)
    anchors = defaultdict(list)
    for r in base:
        anchors[r['image_id'], r['category_id']].append(r)
    result = []
    for key in sorted(anchors.keys() | grouped.keys()):
        original = anchors[key]
        if key[1] not in policy['categories']:
            result.extend(copy.deepcopy(original))
            continue
        candidates = sorted(grouped[key], key=lambda r: r['score'], reverse=True)
        used = set()
        boxes = [r['bbox'] for r in candidates]
        for anchor in sorted(original, key=lambda r: r['score'], reverse=True):
            row = copy.deepcopy(anchor)
            ious = overlap(anchor['bbox'], boxes)
            eligible = [i for i in range(len(candidates)) if i not in used and
                        ious[i] >= .55 and candidates[i]['score'] >= .10]
            if eligible:
                index = max(eligible, key=lambda i: ious[i] * candidates[i]['score'])
                used.add(index)
                partner = candidates[index]
                fraction = policy['box_weight'] * partner['score'] / max(.001, anchor['score'] + partner['score'])
                row['bbox'] = ((1 - fraction) * np.asarray(anchor['bbox']) +
                               fraction * np.asarray(partner['bbox'])).tolist()
                row['score'] = float(anchor['score'] + policy['score_weight'] * partner['score'] * (1 - anchor['score']))
            result.append(row)
        if policy['add_missing']:
            # Candidate must be isolated from both strong and weak base boxes;
            # otherwise adding it could turn one object into two detections.
            base_boxes = [r['bbox'] for r in original]
            for index, candidate in enumerate(candidates):
                if index in used or candidate['score'] < .20:
                    continue
                ious = overlap(candidate['bbox'], base_boxes)
                if len(ious) and ious.max() >= .30:
                    continue
                row = copy.deepcopy(candidate)
                row['score'] *= .5
                result.append(row)
                base_boxes.append(row['bbox'])
    # COCO's maxDets is applied per category by the unchanged evaluator. Do not
    # silently truncate the original pipeline's output across all categories.
    return result


def evaluate_fusion(folder):
    folder = Path(folder)
    manifest = read(folder / 'manifest.json')
    assert manifest['settings']['annotation_sha256'] == sha(DOC)
    doc = read(DOC)
    reference = read(ROOT / 'retained_pipeline.json')
    assert sha(reference['predictions']) == reference['prediction_sha256']
    expert_record = read(folder / 'result.json')
    assert sha(expert_record['predictions']) == expert_record['prediction_sha256']
    base, expert = read(reference['predictions']), read(expert_record['predictions'])
    validate_rows(base, doc)
    validate_rows(expert, doc)
    ids = [im['id'] for im in doc['images']]
    assert len(ids) == len(set(ids)) == 1547
    ordered = sorted(ids, key=lambda iid: hashlib.sha256(f'complement-screen-92463:{iid}'.encode()).hexdigest())
    screen, remainder = ordered[:256], ordered[256:]
    policies = [dict(name='baseline'), dict(name='expert_alone'), dict(name='expert_swimmer_jetski')]
    for cats, label in [([1, 3], 'swimmer_jetski'), ([1, 2, 3, 4, 5], 'all')]:
        for mode, box, score_weight, add in [('geometry', .4, 0., False),
                                           ('consensus', .4, .15, False), ('recall', .4, .15, True)]:
            policies.append(dict(name=f'{label}_{mode}', categories=cats, box_weight=box,
                                 score_weight=score_weight, add_missing=add))
    save(folder / 'fusion_plan.json', dict(policies=policies, screen_ids=screen,
        remainder_ids=remainder, baseline_sha256=reference['prediction_sha256'],
        expert_sha256=expert_record['prediction_sha256'], fusion_code_sha256=sha(__file__)))
    reports = []
    best_rows, best_policy, best_ap = base, policies[0], -1.
    for policy in policies:
        rows = fuse(base, expert, policy)
        metrics = score(rows, doc, screen)
        reports.append(dict(policy=policy, metrics=metrics))
        log(folder, 'fusion_screen', policy=policy['name'], metrics=metrics)
        if metrics['AP'] > best_ap:
            best_rows, best_policy, best_ap = rows, policy, metrics['AP']
        save(folder / 'fusion_screens.json', reports)
    # Only the screen winner receives a full and remainder check, including the
    # unchanged baseline if none of the complementary settings improved it.
    full_base = score(base, doc, ids)
    assert abs(full_base['AP'] - reference['metrics']['AP']) < 1e-7
    full = score(best_rows, doc, ids)
    remaining_base = score(base, doc, remainder)
    remaining = score(best_rows, doc, remainder)
    retained = full['AP'] > full_base['AP'] and remaining['AP'] >= remaining_base['AP']
    save(folder / 'fusion_predictions.json', best_rows)
    record = dict(policy=best_policy, baseline=full_base, candidate=full,
                  remainder_baseline=remaining_base, remainder_candidate=remaining,
                  accepted=retained, target_reached=retained and full['AP'] >= 63,
                  predictions=str(folder / 'fusion_predictions.json'),
                  prediction_sha256=sha(folder / 'fusion_predictions.json'),
                  note='Development validation score; historical validation tuning applies. Hidden test unknown.')
    record.update(baseline_predictions=reference['predictions'],
                  baseline_prediction_sha256=reference['prediction_sha256'],
                  expert_checkpoint=str(folder / 'checkpoint.pt'),
                  expert_checkpoint_sha256=manifest['settings']['checkpoint_sha256'],
                  expert_metrics=expert_record['metrics'],
                  settings=str(folder / 'fusion_plan.json'))
    save(folder / 'fusion_result.json', record)
    log(folder, 'fusion_complete', **record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['infer', 'fusion', 'both'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=2)
    args = parser.parse_args()
    assert args.batch >= 1
    folder = args.folder
    if args.action in ('infer', 'both'):
        assert args.checkpoint is not None
        folder = infer(args.checkpoint, args.device, args.batch)
    if args.action in ('fusion', 'both'):
        assert folder is not None
        evaluate_fusion(folder)
