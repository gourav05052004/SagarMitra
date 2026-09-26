"""Read-only source integration and resumable train/validation feature caches."""
import collections
import contextlib
import copy
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import ranking_core as core

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
DETAIL = WORK/'Maritime_CBNet61_DetailHead_backend'
COMMON = WORK/'Maritime_CBNet6215_AllClassContext_backend'
CONSENSUS = WORK/'Maritime_CBNet615_Consensus_backend'
EXPORTER = WORK/'Maritime_MaCVi_611625_backend'
SOURCE_ROWS = CONSENSUS/'runs/calibration_001/predictions.json'
SOURCE_RESULT = CONSENSUS/'runs/calibration_001/result.json'
SOURCE_SPEC = EXPORTER/'runs/submission_611625_001/spec.json'
VIEW_CACHE = DETAIL/'runs/detail_001/validation_00200/cache/tiles_c6f725606c2aa9b9'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def store(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def defaults(name='ranking_001'):
    if not name.replace('_', '').isalnum():
        raise ValueError('Use letters, digits and underscores in run_name.')
    return dict(run_name=name, run_dir=str(ROOT/'runs'/name), seed=62215,
                train_images=768, calibration_images=256, max_updates=2400,
                eval_every=200, save_every=25, batch_size=384, lr=1e-4,
                min_lr=6.25e-6, warmup=50, patience=4, plateau_every=1,
                min_gain=.02, target=62.15, head_device='cpu', threads=6,
                score_min=.001, max_box_side=1024, score_cap=2., feature_batch=64,
                blend_grid=[0., .25, .5, 1.], classes=list(core.TARGETS), version=1)


def log(s, **values):
    values['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    root = Path(s['run_dir']); save(root/'progress.json', values)
    with (root/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(values, allow_nan=False)+'\n')
    print(json.dumps(values), flush=True)


def stop(s):
    if (Path(s['run_dir'])/'STOP').exists():
        raise KeyboardInterrupt('STOP found. Remove it and rerun the cell to resume.')


@contextlib.contextmanager
def run_lock(s):
    import psutil
    root = Path(s['run_dir']); root.mkdir(parents=True, exist_ok=True)
    lock = root/'RUNNING.json'
    if lock.exists():
        item = read(lock)
        if psutil.pid_exists(item['pid']):
            try:
                same = abs(psutil.Process(item['pid']).create_time()-item['created']) < 1
            except psutil.NoSuchProcess:
                same = False
            if same:
                raise RuntimeError('This run already has an active cell. Interrupt it before rerunning.')
        lock.unlink()
    # Exclusive creation also prevents two notebooks starting the same run together.
    with lock.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), f)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def recipe():
    name = 'evidence_frozen_exporter'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXPORTER/'export_updated.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def evaluator():
    name = 'evidence_coco_evaluator'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, CONSENSUS/'consensus.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module; spec.loader.exec_module(module)
    return sys.modules[name]


def score(rows, doc, ids, categories=None):
    return evaluator().evaluate(rows, doc, ids, categories)


def video_group(im):
    src = im.get('source', {})
    if not src.get('folder_name'):
        raise ValueError('Missing source video grouping; do not split neighboring frames randomly.')
    return '|'.join(str(src.get(k, '')) for k in ('drone', 'folder_name', 'video'))


def select_images(pool, annotations, count, rng):
    bycat = collections.defaultdict(set)
    for a in annotations:
        if not a.get('iscrowd', 0):
            bycat[a['category_id']].add(a['image_id'])
    chosen = set(); available = {im['id'] for im in pool}
    # Cover rare target classes first; include ordinary/background images too.
    for cat in (4, 3, 1):
        candidates = np.array(sorted((bycat[cat] & available)-chosen), dtype=np.int64)
        rng.shuffle(candidates); chosen.update(map(int, candidates[:count//4]))
    rest = np.array(sorted(available-chosen), dtype=np.int64); rng.shuffle(rest)
    chosen.update(map(int, rest[:max(0, count-len(chosen))]))
    return sorted([im for im in pool if im['id'] in chosen], key=lambda im: im['id'])


def training_split(doc, s):
    groups = sorted({video_group(im) for im in doc['images']})
    rng = np.random.default_rng(s['seed'])
    for _ in range(1000):
        shuffled = list(groups); rng.shuffle(shuffled)
        hold = set(shuffled[:max(2, round(len(groups)*.25))])
        fit = [im for im in doc['images'] if video_group(im) not in hold]
        tune = [im for im in doc['images'] if video_group(im) in hold]
        fit = select_images(fit, doc['annotations'], s['train_images'], rng)
        tune = select_images(tune, doc['annotations'], s['calibration_images'], rng)
        if len(fit) < s['train_images'] or len(tune) < s['calibration_images']:
            continue
        counts = []
        for ims in (fit, tune):
            ids = {im['id'] for im in ims}
            c = collections.Counter(a['category_id'] for a in doc['annotations']
                                    if a['image_id'] in ids and not a.get('iscrowd', 0))
            counts.append(c)
        if all(c[t] >= 20 for c in counts for t in core.TARGETS):
            assert not ({video_group(im) for im in fit} & {video_group(im) for im in tune})
            return dict(fit_ids=[im['id'] for im in fit], tune_ids=[im['id'] for im in tune],
                        fit_groups=sorted({video_group(im) for im in fit}),
                        tune_groups=sorted({video_group(im) for im in tune}),
                        annotation_counts=[dict(c) for c in counts],
                        note='Both partitions come only from official training images. '
                             'Held-out source groups calibrate the head; the detector has previously seen train data.')
    raise RuntimeError('Cannot construct source-group-disjoint training/calibration partitions.')


def preflight(s):
    import mmcv, mmdet
    if not (torch.__version__.startswith('2.0.1') and mmcv.__version__ == '1.7.2'
            and mmdet.__version__ == '2.28.2'):
        raise RuntimeError('Select CBNet HPC (CUDA): torch 2.0.1 / mmcv 1.7.2 / mmdet 2.28.2.')
    if s['classes'] != list(core.TARGETS) or s['eval_every'] <= 0 or s['max_updates'] % s['eval_every']:
        raise ValueError('Use classes [1,3,4] and a budget divisible by eval_every.')
    if not (0 < s['min_lr'] <= s['lr'] and 0 < s['patience'] and s['head_device'] in ('cpu', 'cuda')):
        raise ValueError('Invalid optimizer configuration.')
    if (s['plateau_every'] < 1 or s['save_every'] < 1 or s['batch_size'] < 3
            or not s['blend_grid'] or 0. not in s['blend_grid']
            or any(not 0 <= v <= 1 for v in s['blend_grid'])):
        raise ValueError('Invalid checkpoint, batch or blend configuration.')
    torch.set_num_threads(s['threads'])
    spec = read(SOURCE_SPEC)
    # Verify the existing immutable recipe; this only reads previous experiment files.
    for path, digest in spec['hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen successful source changed: '+path)
    train = read(DETAIL/'prepared/train.json'); val = read(DETAIL/'prepared/val.json')
    assert len(train['images']) == 8930 and len(val['images']) == 1547
    train_names = {str(Path(im['file_name']).resolve()).casefold() for im in train['images']}
    val_names = {str(Path(im['file_name']).resolve()).casefold() for im in val['images']}
    assert not train_names.intersection(val_names), 'Train/validation image leakage.'
    manifest = read(VIEW_CACHE/'manifest.json')
    assert Path(manifest['checkpoint']).resolve() == Path(spec['detector_checkpoint']).resolve()
    for key in ('eval_scale', 'tile_size', 'tile_overlap', 'nms_cfg'):
        assert manifest['specification'][key] == spec['detector_settings'][key], key
    files = [SOURCE_ROWS, SOURCE_RESULT, SOURCE_SPEC, VIEW_CACHE/'manifest.json',
             DETAIL/'prepared/train.json', DETAIL/'prepared/val.json',
             ROOT/'ranking_core.py', ROOT/'pipeline.py', ROOT/'experiment.py']
    identity = dict(settings=s, files={str(p): sha(p) for p in files})
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(s['run_dir']); root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        record = read(root/'manifest.json')
        if record['signature'] != signature:
            raise RuntimeError('Settings/code/source changed. Choose a new run_name to keep caches honest.')
        print('Verified saved source AP:', record['baseline']['AP'], flush=True)
        return record
    rows = read(SOURCE_ROWS); baseline = score(rows, val, [im['id'] for im in val['images']])
    expected = read(SOURCE_RESULT)['metrics']['AP']
    if abs(baseline['AP']-expected) > 1e-6:
        raise RuntimeError('Saved full validation predictions failed independent reproduction.')
    split = training_split(train, s)
    record = dict(signature=signature, identity=identity, baseline=baseline, split=split,
                  detector_checkpoint=spec['detector_checkpoint'],
                  cached_baseline=True, metric='COCO bbox AP50:95, all1547, maxDets100')
    save(root/'manifest.json', record); save(root/'settings.json', s)
    log(s, stage='verified', baseline_full1547=baseline['AP'],
        train_head_images=len(split['fit_ids']), calibration_train_images=len(split['tune_ids']),
        validation_images=1547)
    return record


def ensure_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('The one-time cache preparation needs the CBNet CUDA kernel.')
    free, _ = torch.cuda.mem_get_info()
    if free < 7*1024**3:
        raise RuntimeError('Less than 7 GiB GPU memory is free. Stop your other training notebook first; '
                           'this notebook does not terminate other processes.')


def image_stamp(im):
    p = Path(im['file_name']); st = p.stat()
    return dict(path=str(p.resolve()), size=st.st_size, modified_ns=st.st_mtime_ns)


def cache_root(s):
    return Path(s['run_dir'])/'cache'


def baseline_by_image():
    by = collections.defaultdict(list)
    for row in read(SOURCE_ROWS):
        by[row['image_id']].append(row)
    return by


def source_views(iid):
    item = read(VIEW_CACHE/'views'/f'{iid}.json')
    return item['sources']


def _cached_item(path, signature, im):
    if not path.exists():
        return None
    item = read(path)
    if item['signature'] != signature or item['image'] != image_stamp(im):
        raise RuntimeError('Stale per-image cache: '+str(path))
    return item


def _prepare(s, record):
    import cv2
    ensure_gpu(); spec = read(SOURCE_SPEC); exp = recipe()
    train = read(DETAIL/'prepared/train.json'); val = read(DETAIL/'prepared/val.json')
    wanted = set(record['split']['fit_ids']+record['split']['tune_ids'])
    ims = sorted([im for im in train['images'] if im['id'] in wanted], key=lambda im: im['id'])
    sig = record['signature']; cache = cache_root(s)
    started = time.monotonic()
    pending = [im for im in ims if not (cache/'train_raw'/f"{im['id']}.json").exists()]
    if pending:
        models = exp.load_detector(spec)
        try:
            for n, im in enumerate(pending, 1):
                stop(s); rows = exp.infer_sources(models, im)
                save(cache/'train_raw'/f"{im['id']}.json",
                     dict(signature=sig, image=image_stamp(im), sources=rows))
                elapsed = time.monotonic()-started
                if n % 10 == 0 or n == len(pending):
                    log(s, stage='cache_train_detector', completed=n, remaining=len(pending)-n,
                        total=len(ims), seconds=round(elapsed, 1),
                        remaining_seconds_estimate=round(elapsed/n*(len(pending)-n), 1))
        finally:
            del models; gc.collect(); torch.cuda.empty_cache()
    # Validate existing shards even if no new detector inference was needed.
    for im in ims:
        _cached_item(cache/'train_raw'/f"{im['id']}.json", sig, im)
    pending = [im for im in ims if not (cache/'train_final'/f"{im['id']}.json").exists()]
    if pending:
        refiners = exp.load_refiners(spec); started = time.monotonic()
        try:
            for n, im in enumerate(pending, 1):
                stop(s); raw = _cached_item(cache/'train_raw'/f"{im['id']}.json", sig, im)
                rows = exp.postprocess(raw['sources'], cv2.imread(im['file_name']), im, refiners)
                save(cache/'train_final'/f"{im['id']}.json",
                     dict(signature=sig, image=image_stamp(im), rows=rows))
                if n % 10 == 0 or n == len(pending):
                    log(s, stage='cache_train_final_pipeline', completed=n, total=len(pending),
                        seconds=round(time.monotonic()-started, 1))
        finally:
            del refiners; gc.collect(); torch.cuda.empty_cache()
    for im in ims:
        _cached_item(cache/'train_final'/f"{im['id']}.json", sig, im)
    # Reuse the already verified validation detections; extract final-box appearance once.
    # Train descriptors use exactly the same final pipeline as validation descriptors.
    fast_dir = WORK/'Maritime_CBNet6215_Fast_backend'
    if str(fast_dir) not in sys.path:
        sys.path.insert(0, str(fast_dir))
    import fast
    encoder = fast.Encoder().cuda().eval()
    feature_s = dict(crop_size=96, context=2., feature_batch=s['feature_batch'])
    annotations = collections.defaultdict(list)
    for a in train['annotations']:
        annotations[a['image_id']].append(a)
    val_rows = baseline_by_image(); started = time.monotonic()
    try:
        for split, images in (('train', ims), ('val', val['images'])):
            new = 0
            for n, im in enumerate(images, 1):
                stop(s); path = cache/(split+'_features')/f"{im['id']}.pt"
                if path.exists():
                    item = torch.load(path, map_location='cpu')
                    if item['signature'] != sig or item['image'] != image_stamp(im):
                        raise RuntimeError('Stale feature cache: '+str(path))
                else:
                    if split == 'train':
                        rows = _cached_item(cache/'train_final'/f"{im['id']}.json", sig, im)['rows']
                        sources = _cached_item(cache/'train_raw'/f"{im['id']}.json", sig, im)['sources']
                    else:
                        rows = val_rows[im['id']]; sources = source_views(im['id'])
                    indices = core.eligible(rows, s['score_min'], s['max_box_side'])
                    bgr = cv2.imread(im['file_name'])
                    if bgr is None or bgr.shape[:2] != (im['height'], im['width']):
                        raise RuntimeError('Image dimensions changed: '+im['file_name'])
                    visual = fast.descriptors(encoder, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                              core.boxes(rows)[indices], feature_s)
                    item = dict(signature=sig, image=image_stamp(im), indices=indices, rows=rows,
                                visual=visual, meta=torch.from_numpy(core.metadata(rows, sources, im, indices)),
                                category=torch.tensor([rows[j]['category_id'] for j in indices], dtype=torch.long),
                                scores=torch.tensor([rows[j]['score'] for j in indices], dtype=torch.float32))
                    if split == 'train':
                        item['target'] = torch.from_numpy(core.training_targets(rows, annotations[im['id']], indices))
                    # Validation features deliberately contain no labels/targets.
                    store(path, item); new += 1
                if split == 'val' and 'target' in item:
                    raise RuntimeError('Validation labels must never enter feature tensors.')
                if n % 25 == 0 or n == len(images):
                    log(s, stage='cache_'+split+'_features', completed=n, total=len(images), new=new,
                        seconds=round(time.monotonic()-started, 1))
    finally:
        del encoder; gc.collect(); torch.cuda.empty_cache()
    result = dict(signature=sig, training_images=len(ims), validation_images=len(val['images']),
                  note='Detector and visual encoder are now cached; head updates and validation need no detector inference.')
    save(cache/'complete.json', result); log(s, stage='cache_ready', **result)
    return result


def prepare(s):
    record = preflight(s)
    with run_lock(s):
        stop(s)
        completed = cache_root(s)/'complete.json'
        if completed.exists():
            result = read(completed)
            if result['signature'] != record['signature']:
                raise RuntimeError('Changed cache signature.')
            # Check all shard identities without allocating CUDA memory.
            train = read(DETAIL/'prepared/train.json'); val = read(DETAIL/'prepared/val.json')
            wanted = set(record['split']['fit_ids']+record['split']['tune_ids'])
            for split, images in (('train', [im for im in train['images'] if im['id'] in wanted]),
                                  ('val', val['images'])):
                for im in images:
                    stop(s)
                    path = cache_root(s)/(split+'_features')/f"{im['id']}.pt"
                    item = torch.load(path, map_location='cpu')
                    if item['signature'] != record['signature'] or item['image'] != image_stamp(im):
                        raise RuntimeError('Stale completed cache: '+str(path))
            print('Complete cache verified; no GPU inference needed.', flush=True)
            return result
        return _prepare(s, record)


def fresh_identity_check(s, count=3):
    """User-run GPU check of the existing source pipeline, before head training."""
    import cv2
    record = preflight(s); ensure_gpu()
    with run_lock(s):
        spec = read(SOURCE_SPEC); exp = recipe()
        doc = read(DETAIL/'prepared/val.json'); base = baseline_by_image()
        rng = np.random.default_rng(s['seed'])
        chosen = [doc['images'][j] for j in rng.choice(len(doc['images']), count, replace=False)]
        raw = {}; model = exp.load_detector(spec)
        try:
            for im in chosen:
                stop(s); raw[im['id']] = exp.infer_sources(model, im)
        finally:
            del model; gc.collect(); torch.cuda.empty_cache()
        refiners = exp.load_refiners(spec); report = []
        try:
            for im in chosen:
                rows = exp.postprocess(raw[im['id']], cv2.imread(im['file_name']), im, refiners)
                expected = base[im['id']]
                # The deterministic recipe must preserve ordering and values as well as AP.
                if len(rows) != len(expected) or any(a['category_id'] != b['category_id'] for a, b in zip(rows, expected)):
                    raise RuntimeError('Fresh source differs from cached source. Investigate before training.')
                box_error = float(np.max(np.abs(core.boxes(rows)-core.boxes(expected)), initial=0))
                score_error = max((abs(a['score']-b['score']) for a, b in zip(rows, expected)), default=0.)
                if box_error > .02 or score_error > 1e-4:
                    raise RuntimeError(f'Fresh/cached mismatch for {im["id"]}: box={box_error}, score={score_error}')
                report.append(dict(image_id=im['id'], maximum_box_error=box_error, maximum_score_error=score_error))
        finally:
            del refiners; gc.collect(); torch.cuda.empty_cache()
        result = dict(signature=record['signature'], images=report, passed=True,
                      note='Small pipeline identity smoke check; not a fresh full1547 evaluation.')
        save(Path(s['run_dir'])/'fresh_identity.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return result
