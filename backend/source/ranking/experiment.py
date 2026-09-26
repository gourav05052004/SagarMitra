"""Train the added scoring head; measure raw and deployed full1547 AP every 200 updates."""
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import pipeline as p
import ranking_core as core


def load_shards(s, record, split, ids):
    result = []
    for iid in ids:
        path = p.cache_root(s)/(split+'_features')/f'{iid}.pt'
        if not path.exists():
            raise FileNotFoundError(f'{path}: run the cache preparation cell first.')
        shard = torch.load(path, map_location='cpu')
        if shard['signature'] != record['signature']:
            raise RuntimeError('Feature cache belongs to a different experiment.')
        if split == 'val' and 'target' in shard:
            raise RuntimeError('Validation target leakage.')
        result.append((iid, shard))
    return result


def load_data(s, record):
    complete = p.read(p.cache_root(s)/'complete.json')
    if complete['signature'] != record['signature']:
        raise RuntimeError('Incomplete or stale feature cache.')
    split = record['split']; fitting = load_shards(s, record, 'train', split['fit_ids'])
    tune = load_shards(s, record, 'train', split['tune_ids'])
    val_doc = p.read(p.DETAIL/'prepared/val.json'); train_doc = p.read(p.DETAIL/'prepared/train.json')
    val = load_shards(s, record, 'val', [im['id'] for im in val_doc['images']])
    keys = ('visual', 'meta', 'category', 'scores', 'target')
    fit = {key: torch.cat([item[key] for _, item in fitting], 0) for key in keys}
    valid = (fit['target'] >= 0).any(1)
    fit = {key: value[valid] for key, value in fit.items()}
    for cat in core.TARGETS:
        mask = fit['category'] == cat
        if mask.sum() < 100 or (fit['target'][mask] > 0).any(1).sum() < 20:
            raise RuntimeError(f'Too few usable training detections for category {cat}. Increase training images in a NEW run.')
    summary = {str(cat): dict(examples=int((fit['category'] == cat).sum()),
                              unique_positive=int((fit['target'][fit['category'] == cat] > 0).any(1).sum()))
               for cat in core.TARGETS}
    p.save(Path(s['run_dir'])/'training_examples.json', summary)
    print('Training examples:', summary, flush=True)
    return fit, tune, val, train_doc, val_doc


def baseline_rows(shards):
    return [r for _, shard in shards for r in shard['rows']]


@torch.inference_mode()
def predict_deltas(model, shards, s, label):
    device = next(model.parameters()).device
    model.eval(); predictions = {}
    for n, (iid, shard) in enumerate(shards, 1):
        p.stop(s); pieces = []
        for first in range(0, len(shard['indices']), 512):
            last = first+512
            _, delta = model(*(shard[key][first:last].to(device) for key in ('visual', 'meta', 'category', 'scores')))
            pieces.append(delta.cpu().numpy())
        predictions[iid] = np.concatenate(pieces) if pieces else np.empty(0, np.float32)
        if n % 400 == 0:
            print(f'{label}: scored {n}/{len(shards)} cached images', flush=True)
    return predictions


def render(shards, deltas, blends):
    return [r for iid, shard in shards
            for r in core.apply_scores(shard['rows'], shard['indices'], deltas[iid], blends)]


def calibrate(model, shards, doc, baseline, s):
    """All blend selection is on held-out TRAIN source groups, never val/test."""
    ids = [iid for iid, _ in shards]
    deltas = predict_deltas(model, shards, s, 'train calibration')
    measured = {}
    for strength in s['blend_grid']:
        p.stop(s)
        blend = {str(cat): float(strength) for cat in core.TARGETS}
        rows = render(shards, deltas, blend)
        metrics = p.score(rows, doc, ids, list(core.TARGETS))
        measured[str(strength)] = metrics
    blends = {}
    for cat in core.TARGETS:
        # A tie keeps identity or the smallest change, not the most aggressive head.
        strength = max(s['blend_grid'], key=lambda v: (measured[str(v)]['per_class_AP'][str(cat)], -v))
        improvement = measured[str(strength)]['per_class_AP'][str(cat)]-baseline['per_class_AP'][str(cat)]
        blends[str(cat)] = float(strength) if improvement >= .05 else 0.
    chosen = p.score(render(shards, deltas, blends), doc, ids)
    return dict(blends=blends, metrics=chosen, grid=measured,
                source='held-out official TRAIN source groups; not validation images')


def evaluate(model, tune, val, train_doc, val_doc, tune_base, s, update):
    started = time.monotonic()
    p.log(s, stage='evaluate', update=update, message='Train-only calibration, then full1547 raw and deployed AP.')
    cal = calibrate(model, tune, train_doc, tune_base, s)
    deltas = predict_deltas(model, val, s, 'FULL1547')
    ids = [iid for iid, _ in val]
    raw_blends = {str(cat): 1. for cat in core.TARGETS}
    raw_rows = render(val, deltas, raw_blends)
    raw = p.score(raw_rows, val_doc, ids)
    chosen_rows = render(val, deltas, cal['blends'])
    chosen = raw if cal['blends'] == raw_blends else p.score(chosen_rows, val_doc, ids)
    # Unmodified categories must retain exactly their original detections/scores.
    original = [r for r in baseline_rows(val) if r['category_id'] in (2, 5)]
    protected = [r for r in chosen_rows if r['category_id'] in (2, 5)]
    assert original == protected, 'Protected categories changed.'
    record = dict(update=update, calibration=cal, raw_candidate_full1547=raw,
                  deployed_candidate_full1547=chosen, seconds=time.monotonic()-started)
    p.save(Path(s['run_dir'])/'evaluations'/f'update_{update:05d}.json', record)
    return record, chosen_rows


def state_dict_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_model(fit, s):
    model = core.ScoringHead(fit['meta'].mean(0), fit['meta'].std(0, unbiased=False), s['score_cap'])
    return model.to(s['head_device'])


def make_optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01, foreach=False)


def pack(model, optimizer, state, s, signature):
    return dict(model=state_dict_cpu(model), optimizer=optimizer.state_dict(), state=copy.deepcopy(state),
                settings=s, signature=signature, torch_rng=torch.get_rng_state())


def sample_batch(fit, pools, s, update):
    rng = np.random.default_rng(s['seed']+update*104729)
    n = s['batch_size']; order = []
    for j, cat in enumerate(core.TARGETS):
        count = n//len(core.TARGETS)+(j < n % len(core.TARGETS))
        order.extend(rng.choice(pools[cat], count, replace=True).tolist())
    rng.shuffle(order); idx = torch.tensor(order)
    return {k: v[idx].to(s['head_device']) for k, v in fit.items()}


def _run(s, record):
    root = Path(s['run_dir']); signature = record['signature']
    if (root/'result.json').exists():
        result = p.read(root/'result.json')
        assert result['signature'] == signature
        print('Completed run; use a new run_name for a different experiment.', flush=True)
        print(json.dumps(result, indent=2), flush=True)
        return result
    fit, tune, val, train_doc, val_doc = load_data(s, record)
    tune_base = p.score(baseline_rows(tune), train_doc, [iid for iid, _ in tune])
    p.save(root/'train_calibration_baseline.json', tune_base)
    torch.manual_seed(s['seed']); model = make_model(fit, s)
    optimizer = make_optimizer(model, s['lr'])
    original_ap = record['baseline']['AP']
    state = dict(update=0, phase='train', stale=0, lr=s['lr'], best_tune=tune_base['AP'],
                 best_full=original_ap, history=[], elapsed_seconds=0., reason=None)
    if (root/'last.pt').exists():
        saved = torch.load(root/'last.pt', map_location='cpu')
        assert saved['signature'] == signature
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']; torch.set_rng_state(saved['torch_rng'])
        print('Resuming scoring-head update', state['update'], 'phase', state['phase'], flush=True)
    else:
        identity_blends = {str(cat): 0. for cat in core.TARGETS}
        p.store(root/'best_tune.pt', dict(model=state_dict_cpu(model), update=0, signature=signature))
        p.save(root/'best_pipeline.json', dict(AP=original_ap, update=0, signature=signature,
               scoring_checkpoint=None, blends=identity_blends, predictions=str(p.SOURCE_ROWS),
               source_spec=str(p.SOURCE_SPEC), measured_full1547=True, target_reached=original_ap >= s['target']))
        p.store(root/'last.pt', pack(model, optimizer, state, s, signature))
    pools = {cat: torch.where(fit['category'] == cat)[0].numpy() for cat in core.TARGETS}
    clock = time.monotonic(); previous_elapsed = state['elapsed_seconds']

    def snapshot():
        state['elapsed_seconds'] = previous_elapsed+time.monotonic()-clock
        p.store(root/'last.pt', pack(model, optimizer, state, s, signature))

    try:
        while not state['reason']:
            p.stop(s)
            if state['phase'] == 'evaluate':
                update = state['update']; cp = root/'checkpoints'/f'update_{update:05d}.pt'
                p.store(cp, dict(model=state_dict_cpu(model), settings=s, signature=signature, update=update))
                metrics, rows = evaluate(model, tune, val, train_doc, val_doc, tune_base, s, update)
                full = metrics['deployed_candidate_full1547']['AP']
                prior_best = state['best_full']
                # Save every strict full improvement, even if smaller than the scheduler threshold.
                accepted = full > prior_best+1e-8
                if accepted:
                    state['best_full'] = full
                    prediction_path = root/'evaluations'/f'update_{update:05d}_predictions.json'
                    p.save(prediction_path, rows)
                    p.save(root/'best_pipeline.json', dict(AP=full, update=update, signature=signature,
                        scoring_checkpoint=str(cp), scoring_checkpoint_sha256=p.sha(cp),
                        blends=metrics['calibration']['blends'], predictions=str(prediction_path),
                        predictions_sha256=p.sha(prediction_path), source_spec=str(p.SOURCE_SPEC),
                        metrics=metrics['deployed_candidate_full1547'], measured_full1547=True,
                        target_reached=full >= s['target']))
                tune_ap = metrics['calibration']['metrics']['AP']
                tune_improved = tune_ap > state['best_tune']+s['min_gain']
                if tune_improved:
                    state['best_tune'] = tune_ap; state['stale'] = 0
                    p.store(root/'best_tune.pt', dict(model=state_dict_cpu(model), update=update, signature=signature))
                else:
                    state['stale'] += 1
                    if state['stale'] % s['plateau_every'] == 0:
                        state['lr'] = max(s['min_lr'], state['lr']*.5)
                        # Roll back to train-calibration-best weights; never train on validation scores.
                        saved = torch.load(root/'best_tune.pt', map_location='cpu')
                        model.load_state_dict(saved['model']); optimizer = make_optimizer(model, state['lr'])
                history = dict(update=update, raw_full1547_AP=metrics['raw_candidate_full1547']['AP'],
                    candidate_full1547_AP=full, gain_vs_start=full-original_ap,
                    gain_vs_previous_best=full-prior_best, best_retained_full1547_AP=state['best_full'],
                    accepted=accepted, train_calibration_AP=tune_ap, blends=metrics['calibration']['blends'],
                    next_lr=state['lr'], patience=f"{state['stale']}/{s['patience']}",
                    target_reached=state['best_full'] >= s['target'], validation_seconds=metrics['seconds'])
                state['history'] = [r for r in state['history'] if r['update'] != update]+[history]
                p.save(root/'history.json', state['history']); p.log(s, stage='validation_complete', **history)
                print(f"UPDATE {update}: raw FULL1547={history['raw_full1547_AP']:.4f}; "
                      f"calibrated FULL1547={full:.4f}; best={state['best_full']:.4f}; "
                      f"gain from start={full-original_ap:+.4f}; next LR={state['lr']:.2g}; "
                      f"target {s['target']}: {'REACHED' if history['target_reached'] else 'NOT YET'}", flush=True)
                state['phase'] = 'train'
                if state['best_full'] >= s['target']:
                    state['reason'] = 'target_measured_on_full_validation'
                elif state['stale'] >= s['patience']:
                    state['reason'] = 'train_calibration_plateau'
                elif update >= s['max_updates']:
                    state['reason'] = 'update_budget_complete'
                snapshot()
                continue
            update = state['update']+1
            if update > s['max_updates']:
                state['reason'] = 'update_budget_complete'; break
            model.train(); batch = sample_batch(fit, pools, s, update)
            lr = state['lr']*min(1., update/max(s['warmup'], 1))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True); loss = core.objective(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss: checkpoint retained; do not continue blindly.')
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            if not torch.isfinite(grad):
                raise FloatingPointError('Nonfinite gradient: update was not applied.')
            optimizer.step(); state['update'] = update
            if update % s['eval_every'] == 0:
                state['phase'] = 'evaluate'
            if update == 1 or update % s['save_every'] == 0 or state['phase'] == 'evaluate':
                snapshot(); p.log(s, stage='head_training', update=update, max_updates=s['max_updates'],
                    loss=float(loss.detach()), lr=lr, best_retained_full1547_AP=state['best_full'],
                    seconds=round(state['elapsed_seconds'], 1), pending_validation=state['phase'] == 'evaluate')
    except KeyboardInterrupt:
        # Use the last atomic safe-point. Interrupts can occur mid-optimizer step;
        # never overwrite it with potentially partially updated parameters here.
        p.log(s, stage='paused', message='Rerun this cell to resume the last saved update (at most 24 lost).')
        return dict(paused=True, checkpoint=str(root/'last.pt'))
    finally:
        del model, optimizer; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = p.read(root/'best_pipeline.json')
    result = dict(signature=signature, baseline_full1547_AP=original_ap,
                  best_full1547_AP=best['AP'], gain=best['AP']-original_ap,
                  target=s['target'], target_reached=best['AP'] >= s['target'],
                  reason=state['reason'], updates=state['update'], best_pipeline=best,
                  test_AP=None, note='Validation model selection, not a hidden-test/leaderboard score.')
    p.save(root/'result.json', result); p.log(s, stage='complete', **result)
    return result


def run(s):
    record = p.preflight(s)
    identity = Path(s['run_dir'])/'fresh_identity.json'
    if not identity.exists() or p.read(identity)['signature'] != record['signature']:
        raise RuntimeError('Run the fresh source-identity check cell before training.')
    with p.run_lock(s):
        return _run(s, record)


def verify_best(s):
    """Independent CPU COCO evaluation of the retained complete-pipeline JSON."""
    record = p.preflight(s); best = p.read(Path(s['run_dir'])/'best_pipeline.json')
    if best['signature'] != record['signature']:
        raise RuntimeError('Best artifact belongs to another run.')
    if best.get('scoring_checkpoint') and p.sha(best['scoring_checkpoint']) != best['scoring_checkpoint_sha256']:
        raise RuntimeError('Best scoring checkpoint changed.')
    if best.get('predictions_sha256') and p.sha(best['predictions']) != best['predictions_sha256']:
        raise RuntimeError('Retained predictions changed.')
    doc = p.read(p.DETAIL/'prepared/val.json')
    metrics = p.score(p.read(best['predictions']), doc, [im['id'] for im in doc['images']])
    if abs(metrics['AP']-best['AP']) > 1e-8:
        raise RuntimeError('Independent reproduction did not match the recorded AP.')
    result = dict(metrics=metrics, gain=metrics['AP']-record['baseline']['AP'],
                  target_reached=metrics['AP'] >= s['target'], predictions=best['predictions'],
                  scoring_checkpoint=best.get('scoring_checkpoint'),
                  note='This JSON contains validation image IDs. Do not upload it as a test submission.')
    p.save(Path(s['run_dir'])/'independent_verification.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def load_best_head(s, device='cpu'):
    """Load the retained head for later deployment on new, unlabeled images."""
    best = p.read(Path(s['run_dir'])/'best_pipeline.json')
    if best['scoring_checkpoint'] is None:
        return None, best['blends']
    if p.sha(best['scoring_checkpoint']) != best['scoring_checkpoint_sha256']:
        raise RuntimeError('Scoring checkpoint changed.')
    cp = torch.load(best['scoring_checkpoint'], map_location='cpu')
    weights = cp['model']
    model = core.ScoringHead(weights['meta_mean'], weights['meta_std'], cp['settings']['score_cap']).to(device)
    model.load_state_dict(weights); model.eval(); model.requires_grad_(False)
    return model, best['blends']


@torch.inference_mode()
def apply_to_detections(model, blends, encoder, rows, sources, bgr, image, s):
    """Add-on after the unchanged exporter's postprocess(); no GT is accepted."""
    if model is None or not any(blends.values()):
        return copy.deepcopy(rows)
    import cv2
    import fast  # Loaded by pipeline.recipe().load_refiners(), as in the existing exporter.
    indices = core.eligible(rows, s['score_min'], s['max_box_side'])
    if not indices:
        return copy.deepcopy(rows)
    features = fast.descriptors(encoder, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), core.boxes(rows)[indices],
                               dict(crop_size=96, context=2., feature_batch=s['feature_batch']))
    metadata = torch.from_numpy(core.metadata(rows, sources, image, indices))
    categories = torch.tensor([rows[j]['category_id'] for j in indices], dtype=torch.long)
    scores = torch.tensor([rows[j]['score'] for j in indices])
    device = next(model.parameters()).device
    _, delta = model(features.to(device), metadata.to(device), categories.to(device), scores.to(device))
    return core.apply_scores(rows, indices, delta.cpu().numpy(), blends)
