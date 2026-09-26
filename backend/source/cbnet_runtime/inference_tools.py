"""Interrupt-safe inference experiments called directly by Jupyter cells."""
import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector
from mmdet.core import multiclass_nms

import diagnostic_tools as diagnostics
import worker
from data_tools import read_json, save_json, tile_windows


def key_for(checkpoint, value):
    payload = dict(checkpoint=worker.sha256(Path(checkpoint)), value=value)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def nms_variants():
    result = {}
    for threshold in (.4, .5, .6, .7):
        result[f'hard_{threshold:.1f}'] = dict(type='nms', iou_threshold=threshold)
        result[f'linear_{threshold:.1f}'] = dict(type='soft_nms', iou_threshold=threshold,
                                                method='linear', min_score=.001)
    for sigma in (.3, .5, .7):
        result[f'gaussian_{sigma:.1f}'] = dict(type='soft_nms', iou_threshold=.5,
                                               sigma=sigma, method='gaussian', min_score=.001)
    return result


def suppress(raw, image_id, nms_cfg, max_per_img=300):
    boxes = torch.from_numpy(raw['boxes'].copy()).float()
    scores = torch.from_numpy(raw['scores'].copy()).float()
    detections, labels = multiclass_nms(boxes, scores, .001, dict(nms_cfg),
                                        max_num=max_per_img)
    rows = []
    for box, label in zip(detections.numpy(), labels.numpy()):
        x1, y1, x2, y2, score = map(float, box)
        if x2 > x1 and y2 > y1:
            rows.append(dict(image_id=int(image_id), category_id=int(label)+1,
                             bbox=[x1, y1, x2-x1, y2-y1], score=score))
    return rows


def nms_sweep(model, checkpoint, settings, output_root):
    import recovery
    document = read_json(Path(settings['root'])/'prepared/val.json')
    images = document['images']
    variants = nms_variants()
    key = key_for(checkpoint, dict(kind='raw_nms', eval_scale=settings['eval_scale']))
    output = Path(output_root)/f'nms_{key}'
    raw = output/'raw'
    output.mkdir(parents=True, exist_ok=True)
    diagnostics.collect_raw(model, images, raw)
    results = {}
    for name, config in variants.items():
        recovery.check_session()
        metrics_file = output/f'{name}_metrics.json'
        if metrics_file.exists():
            results[name] = read_json(metrics_file)
            continue
        rows = []
        for index, image in enumerate(images, 1):
            if index % 100 == 0:
                recovery.check_session()
            rows.extend(suppress(diagnostics.load_raw(raw, image['id']), image['id'], config))
            if index % 250 == 0:
                print(f'{name}: {index}/{len(images)}', flush=True)
        metrics = diagnostics.score(document, rows, [image['id'] for image in images])
        metrics['nms_cfg'] = config
        save_json(metrics_file, metrics)
        results[name] = metrics
        print(f'{name}: AP {metrics["AP"]:.3f}; AP75 {metrics["AP75"]:.3f}', flush=True)
    best_name = max(results, key=lambda name: results[name]['AP'])
    summary = dict(best=best_name, best_metrics=results[best_name], all=results,
                   checkpoint=str(checkpoint), output=str(output))
    save_json(output/'summary.json', summary)
    return summary


def _translate_tile(rows, image_id, left, top, right, bottom, image_width, image_height,
                    border_margin=4):
    translated = []
    tile_width, tile_height = right-left, bottom-top
    for row in rows:
        x, y, width, height = row['bbox']
        x2, y2 = x+width, y+height
        touches_internal = ((left > 0 and x <= border_margin) or
            (top > 0 and y <= border_margin) or
            (right < image_width and x2 >= tile_width-border_margin) or
            (bottom < image_height and y2 >= tile_height-border_margin))
        if touches_internal:
            continue
        x, y = max(0., x), max(0., y)
        x2, y2 = min(float(tile_width), x2), min(float(tile_height), y2)
        if x2 > x and y2 > y:
            translated.append(dict(image_id=int(image_id), category_id=row['category_id'],
                bbox=[x+left, y+top, x2-x, y2-y], score=row['score']))
    return translated


def infer_at_scale(model, image, scale):
    """Isolate each view from inference_detector's pipeline replacement."""
    original = model.cfg
    try:
        model.cfg = copy.deepcopy(original)
        model.cfg.data.test.pipeline[1]['img_scale'] = tuple(scale)
        model.eval()
        return inference_detector(model, image)
    finally:
        model.cfg = original


def collect_views(model, checkpoint, settings, mode, output_root):
    if mode not in ('tta', 'tiles'):
        raise ValueError(mode)
    # CBFPN deliberately returns both backbone feature pyramids while training,
    # but MMDetection's simple_test path requires only the lead pyramid.
    model.eval()
    document = read_json(Path(settings['root'])/'prepared/val.json')
    images = document['images']
    specification = dict(kind=mode, inference_version=2, eval_scale=settings['eval_scale'],
        tta_scale=settings.get('tta_scale'), tile_size=settings.get('tile_size'),
        tile_overlap=settings.get('tile_overlap', .25), nms_cfg=settings.get('nms_cfg'))
    key = key_for(checkpoint, specification)
    output = Path(output_root)/f'{mode}_{key}'
    cache = output/'views'
    cache.mkdir(parents=True, exist_ok=True)
    original_config = model.cfg
    started = time.monotonic()
    try:
        for index, image in enumerate(images, 1):
            import recovery
            recovery.check_session()
            destination = cache/f'{image["id"]}.json'
            if destination.exists():
                continue
            bgr = cv2.imread(image['file_name'])
            if bgr is None:
                raise OSError(image['file_name'])
            sources = {}
            if mode == 'tta':
                scales = [tuple(settings['eval_scale']), tuple(settings['tta_scale'])]
                for scale_index, scale in enumerate(scales):
                    for flipped in (False, True):
                        name = f'scale{scale_index}_flip{int(flipped)}'
                        view = np.ascontiguousarray(bgr[:, ::-1]) if flipped else bgr
                        rows = worker.prediction_rows(infer_at_scale(model, view, scale), image['id'])
                        if flipped:
                            for row in rows:
                                row['bbox'][0] = image['width']-row['bbox'][0]-row['bbox'][2]
                        sources[name] = rows
            else:
                sources['full'] = worker.prediction_rows(
                    infer_at_scale(model, bgr, settings['eval_scale']), image['id'])
                windows = tile_windows(image['width'], image['height'],
                                       settings['tile_size'], settings.get('tile_overlap', .25))
                for tile_index, (left, top, right, bottom) in enumerate(windows):
                    tile = bgr[top:bottom, left:right]
                    rows = worker.prediction_rows(infer_at_scale(model, tile, settings['eval_scale']), image['id'])
                    sources[f'tile{tile_index}'] = _translate_tile(
                        rows, image['id'], left, top, right, bottom,
                        image['width'], image['height'])
            save_json(destination, dict(image_id=image['id'], sources=sources))
            if index % 25 == 0 or index == len(images):
                print(f'{mode}: {index}/{len(images)}; {time.monotonic()-started:.0f}s', flush=True)
    finally:
        model.cfg = original_config
    save_json(output/'manifest.json', dict(checkpoint=str(checkpoint), specification=specification))
    return output


def _iou(box, boxes):
    if not boxes:
        return np.empty(0)
    boxes = np.asarray(boxes, np.float32)
    left = np.maximum(box[0], boxes[:, 0]); top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2]); bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(right-left, 0)*np.maximum(bottom-top, 0)
    area = max(0, box[2]-box[0])*max(0, box[3]-box[1])
    other = np.maximum(boxes[:, 2]-boxes[:, 0], 0)*np.maximum(boxes[:, 3]-boxes[:, 1], 0)
    return intersection/np.maximum(area+other-intersection, 1e-8)


def weighted_box_fusion(rows, image_id, threshold=.55, max_per_img=300):
    output = []
    by_class = defaultdict(list)
    for row in rows:
        x, y, width, height = row['bbox']
        by_class[row['category_id']].append(([x, y, x+width, y+height], row['score']))
    for category, candidates in by_class.items():
        clusters = []
        for box, score in sorted(candidates, key=lambda item:-item[1]):
            fused = [cluster['box'] for cluster in clusters]
            overlaps = _iou(box, fused)
            match = int(overlaps.argmax()) if len(overlaps) and overlaps.max() >= threshold else -1
            if match < 0:
                clusters.append(dict(box=np.asarray(box, np.float64), weighted=np.asarray(box)*score,
                                     weight=float(score), scores=[float(score)]))
            else:
                cluster = clusters[match]
                cluster['weighted'] += np.asarray(box)*score
                cluster['weight'] += float(score)
                cluster['scores'].append(float(score))
                cluster['box'] = cluster['weighted']/max(cluster['weight'], 1e-8)
        for cluster in clusters:
            x1, y1, x2, y2 = cluster['box']
            score = float(np.mean(cluster['scores']))
            if x2 > x1 and y2 > y1:
                output.append(dict(image_id=int(image_id), category_id=int(category),
                    bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=score))
    return sorted(output, key=lambda row:-row['score'])[:max_per_img]


def fuse_rows(sources, image_id, method, nms_cfg=None, wbf_iou=.55):
    rows = [row for source in sources.values() for row in source]
    if method == 'wbf':
        return weighted_box_fusion(rows, image_id, wbf_iou)
    config = nms_cfg or dict(type='soft_nms', iou_threshold=.5,
                             method='linear', min_score=.001)
    if not rows:
        return []
    boxes = np.asarray([[row['bbox'][0], row['bbox'][1], row['bbox'][0]+row['bbox'][2],
                         row['bbox'][1]+row['bbox'][3]] for row in rows], np.float32)
    scores = np.asarray([row['score'] for row in rows], np.float32)
    labels = np.asarray([row['category_id']-1 for row in rows], np.int64)
    # multiclass_nms accepts class-wise matrices; build sparse scores to keep classes independent.
    matrix = np.zeros((len(rows), 6), np.float32)
    matrix[np.arange(len(rows)), labels] = scores
    dets, kept_labels = multiclass_nms(torch.from_numpy(boxes), torch.from_numpy(matrix),
                                       .001, dict(config), max_num=300)
    fused = []
    for box, label in zip(dets.numpy(), kept_labels.numpy()):
        x1, y1, x2, y2, score = map(float, box)
        fused.append(dict(image_id=int(image_id), category_id=int(label)+1,
                          bbox=[x1, y1, x2-x1, y2-y1], score=score))
    return fused


def evaluate_cached_views(settings, cache_output, method='soft_nms', wbf_iou=.55):
    cache_output = Path(cache_output)
    result_dir = cache_output/f'fusion_{method}_{wbf_iou:.2f}'
    metrics_path = result_dir/'metrics.json'
    if metrics_path.exists():
        return read_json(metrics_path)
    document = read_json(Path(settings['root'])/'prepared/val.json')
    rows = []
    for index, image in enumerate(document['images'], 1):
        cached = read_json(cache_output/'views'/f'{image["id"]}.json')
        rows.extend(fuse_rows(cached['sources'], image['id'], method,
                              settings.get('nms_cfg'), wbf_iou))
        if index % 250 == 0:
            print(f'fusion {method}: {index}/{len(document["images"])}', flush=True)
    metrics = diagnostics.score(document, rows, [image['id'] for image in document['images']])
    metrics.update(fusion=method, wbf_iou=wbf_iou, source=str(cache_output))
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir/'predictions.json', rows)
    save_json(metrics_path, metrics)
    return metrics


def evaluate_tta_source_ablation(settings, cache_output):
    """Score cached TTA views alone and in conservative pairs; no GPU inference."""
    cache_output = Path(cache_output)
    result_dir = cache_output/'source_ablation'
    result_dir.mkdir(parents=True, exist_ok=True)
    document = read_json(Path(settings['root'])/'prepared/val.json')
    linear = dict(type='soft_nms', iou_threshold=.6, method='linear', min_score=.001)
    hard = dict(type='nms', iou_threshold=.6)
    plans = {
        'base': (['scale0_flip0'], None),
        'base_flip': (['scale0_flip1'], None),
        'large': (['scale1_flip0'], None),
        'large_flip': (['scale1_flip1'], None),
        'base_pair_linear': (['scale0_flip0','scale0_flip1'], linear),
        'base_pair_hard': (['scale0_flip0','scale0_flip1'], hard),
        'large_pair_linear': (['scale1_flip0','scale1_flip1'], linear),
        'large_pair_hard': (['scale1_flip0','scale1_flip1'], hard),
        'scales_unflipped_linear': (['scale0_flip0','scale1_flip0'], linear),
        'scales_unflipped_hard': (['scale0_flip0','scale1_flip0'], hard),
        'all_hard': (['scale0_flip0','scale0_flip1','scale1_flip0','scale1_flip1'], hard),
    }
    results = {}
    for name, (source_names, config) in plans.items():
        metrics_path = result_dir/f'{name}_metrics.json'
        if metrics_path.exists():
            results[name] = read_json(metrics_path)
            continue
        rows = []
        for index, image in enumerate(document['images'], 1):
            cached = read_json(cache_output/'views'/f'{image["id"]}.json')['sources']
            selected = {key:cached[key] for key in source_names}
            if config is None:
                image_rows = sorted(selected[source_names[0]],
                                    key=lambda row:-row['score'])[:300]
            else:
                image_rows = fuse_rows(selected, image['id'], 'nms', config)
            rows.extend(image_rows)
            if index % 500 == 0:
                print(f'{name}: {index}/{len(document["images"])}', flush=True)
        metrics = diagnostics.score(document, rows,
                                    [image['id'] for image in document['images']])
        metrics.update(view_sources=source_names, fusion=('single' if config is None else config))
        save_json(metrics_path, metrics)
        results[name] = metrics
        print(f'{name}: AP {metrics["AP"]:.3f}; AP75 {metrics["AP75"]:.3f}; '
              f'AP_small {metrics["AP_small"]:.3f}', flush=True)
    best_name = max(results, key=lambda key:results[key]['AP'])
    summary = dict(best=best_name, best_metrics=results[best_name], all=results,
                   cache=str(cache_output))
    save_json(result_dir/'summary.json', summary)
    return summary


def collect_single_model(model, checkpoint, settings, output_root, member_name):
    # Required for CBNet/CBFPN inference; training mode returns nested features.
    model.eval()
    document = read_json(Path(settings['root'])/'prepared/val.json')
    specification = dict(kind='ensemble_member', member=member_name,
                         eval_scale=settings['eval_scale'], nms_cfg=settings.get('nms_cfg'))
    key = key_for(checkpoint, specification)
    output = Path(output_root)/f'{member_name}_{key}'
    cache = output/'views'
    cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, image in enumerate(document['images'], 1):
        destination = cache/f'{image["id"]}.json'
        if not destination.exists():
            rows = worker.prediction_rows(inference_detector(model, image['file_name']), image['id'])
            save_json(destination, dict(image_id=image['id'], sources={member_name:rows}))
        if index % 50 == 0 or index == len(document['images']):
            print(f'{member_name}: {index}/{len(document["images"])}; '
                  f'{time.monotonic()-started:.0f}s', flush=True)
    save_json(output/'manifest.json', dict(checkpoint=str(checkpoint), specification=specification))
    return output


def evaluate_ensemble_caches(settings, cache_outputs, output_root, method='wbf', wbf_iou=.55):
    cache_outputs = [Path(path) for path in cache_outputs]
    identity = hashlib.sha256('|'.join(str(path) for path in cache_outputs).encode()).hexdigest()[:16]
    result_dir = Path(output_root)/f'ensemble_{identity}_{method}_{wbf_iou:.2f}'
    metrics_path = result_dir/'metrics.json'
    if metrics_path.exists():
        return read_json(metrics_path)
    document = read_json(Path(settings['root'])/'prepared/val.json')
    rows = []
    for index, image in enumerate(document['images'], 1):
        sources = {}
        for cache in cache_outputs:
            cached = read_json(cache/'views'/f'{image["id"]}.json')
            sources.update(cached['sources'])
        rows.extend(fuse_rows(sources, image['id'], method, settings.get('nms_cfg'), wbf_iou))
        if index % 250 == 0:
            print(f'ensemble fusion: {index}/{len(document["images"])}', flush=True)
    metrics = diagnostics.score(document, rows, [image['id'] for image in document['images']])
    metrics.update(fusion=method, wbf_iou=wbf_iou,
                   members=[str(path) for path in cache_outputs])
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir/'predictions.json', rows)
    save_json(metrics_path, metrics)
    return metrics
