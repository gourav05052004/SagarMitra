"""Budgeted, resumable CBNet fine-tuning and full COCO evaluation."""
import argparse
import copy
import functools
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from mmcv.parallel import collate, scatter
from mmcv.runner import build_optimizer
from mmdet.datasets import build_dataset
from mmdet.apis import inference_detector
import worker
from recovery_config import make_config
from data_tools import read_json, save_json

worker.make_config = make_config
ROOT = Path(__file__).resolve().parent
DEADLINE = None
SESSION_STOP = None


class OvernightStop(RuntimeError):
    pass


def check_session():
    if SESSION_STOP and Path(SESSION_STOP).exists():
        raise OvernightStop('User requested overnight stop')
    if DEADLINE is not None and time.time() >= DEADLINE:
        raise OvernightStop('Overnight wall-clock budget reached')


def fingerprint(s):
    content = dict(settings=s, code={})
    for name in ('recovery.py', 'recovery_config.py', 'worker.py', 'model_config.py',
                 'data_tools.py', 'checkpoint_io.py', 'advanced_runtime.py',
                 'inference_tools.py', 'quality_head.py', 'overnight.py',
                 'deformable_head.py', 'refinement_head.py', 'architecture_runtime.py', 'experiments.py',
                 'train_policy.py', 'overnight_runtime.py', 'fixed_fusion.py'):
        content['code'][name] = worker.sha256(ROOT/name)
    for path in sorted((ROOT/'cbnet_port').glob('*.py')):
        content['code'][str(path.relative_to(ROOT))] = worker.sha256(path)
    content['annotations'] = {split: worker.sha256(ROOT/f'prepared/{split}.json') for split in ('train','val')}
    if s.get('train_annotation'):
        content['train_annotation'] = worker.sha256(Path(s['train_annotation']))
    content['initial_checkpoint'] = worker.sha256(Path(s['init_checkpoint']))
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def train_mode(model, s):
    model.train()
    if s.get('refinement_only') or s.get('train_scope'):
        model.eval()
        for head in model.roi_head.bbox_head:
            if hasattr(head, 'reg_refine'):
                head.reg_refine.train()
    if s.get('freeze_bn_stats', False):
        # Preserve learned affine parameters; freeze only moving statistics.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def evaluate(model, s, directory, mode='single'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    coco = read_json(ROOT/'prepared/val.json')
    assert len(coco['images']) == 1547
    model.eval()
    rows = []
    start = time.perf_counter()
    with torch.no_grad():
        for i, im in enumerate(coco['images']):
            if i % 25 == 0:
                check_session()
            if mode == 'single':
                rows.extend(worker.prediction_rows(inference_detector(model, im['file_name']), im['id']))
            else:
                rows.extend(worker.extra_inference(model, im, s, mode))
            if i % 50 == 0:
                save_json(directory/'progress.json', dict(stage='validation', image=i+1, total=1547,
                          seconds=time.perf_counter()-start, scale=s['eval_scale'], mode=mode))
                print(f'Validation {mode} {s["eval_scale"]}: {i+1}/1547', flush=True)
    metrics = worker.coco_metrics(ROOT/'prepared/val.json', rows, [im['id'] for im in coco['images']])
    metrics.update(seconds=time.perf_counter()-start, inference=mode, eval_scale=s['eval_scale'],
                   smoke_only=False, target_map=s['target_map'], target_reached=False)
    metrics['target_reached'] = metrics['map'] >= s['target_map']
    save_json(directory/'predictions.json', rows)
    save_json(directory/'metrics.json', metrics)
    save_json(directory/'progress.json', dict(stage='complete', **metrics))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def preflight(s):
    """Actual optimizer/gradient check at max training rectangle; discarded."""
    model, cfg = worker.create_model(s, checkpoint=s['init_checkpoint'])
    dataset = build_dataset(cfg.data.train)
    raw = dataset[0]
    batch = scatter(collate([raw], samples_per_gpu=1), [0])[0]
    scale = max(s['train_scales'], key=lambda p:p[0]*p[1])
    h, w = math.ceil(min(scale)/32)*32, math.ceil(max(scale)/32)*32
    batch['img'] = torch.zeros((1,3,h,w), device='cuda')
    batch['img_metas'][0].update(img_shape=(h,w,3), pad_shape=(h,w,3), batch_input_shape=(h,w))
    batch['gt_bboxes'] = [torch.tensor([[16.,16.,32.,32.],[100.,100.,200.,180.]], device='cuda')]
    batch['gt_labels'] = [torch.tensor([0,3], device='cuda')]
    optimizer = build_optimizer(model, cfg.optimizer)
    scaler = torch.cuda.amp.GradScaler(init_scale=128.)
    train_mode(model, s)
    torch.cuda.reset_peak_memory_stats()
    before = model.backbone.cb_modules[0].patch_embed.proj.weight.detach().clone()
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss, _ = worker.parse_loss(model, model(**batch, return_loss=True))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),35.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradients in recovery preflight')
        from train_policy import trainable
        for name, parameter in model.named_parameters():
            if not trainable(name,s['train_scope']):
                assert parameter.grad is None, 'Frozen parameter received gradients'
            elif parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), 'Nonfinite trainable gradient'
        if s['train_scope'] in ('neck','spatial'):
            assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.neck.parameters()), 'Missing neck gradients'
        if s.get('deformable_roi'):
            offset_grads=[p.grad for name,p in model.named_parameters() if '.reg_refine.' in name]
            assert offset_grads and all(g is not None and torch.isfinite(g).all() for g in offset_grads)
            assert any(g.abs().max()>0 for g in offset_grads), 'No deformable offset gradients'
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    delta = float((model.backbone.cb_modules[0].patch_embed.proj.weight-before).abs().max())
    assert delta==0, 'Frozen backbone changed'
    # Probe training on a real augmented image, including empty crops if sampled.
    real = scatter(collate([dataset[min(5,len(dataset)-1)]], samples_per_gpu=1),[0])[0]
    with torch.no_grad(), torch.cuda.amp.autocast():
        real_loss,_ = worker.parse_loss(model, model(**real,return_loss=True))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved()/2**30
    assert peak < torch.cuda.get_device_properties(0).total_memory/2**30*.90, 'Insufficient memory headroom'
    out = dict(passed=True, signature=fingerprint(s), peak_reserved_gib=peak,
               max_scale=scale, losses=losses, real_image_loss=float(real_loss),
               backbone_max_parameter_change=delta, backbone_frozen=True,train_scope=s['train_scope'])
    save_json(Path(s['run_dir'])/'preflight.json',out)
    print(json.dumps(out,indent=2),flush=True)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy'])
    torch.set_rng_state(r['torch']); torch.cuda.set_rng_state_all(r['cuda'])


def train(s):
    run = Path(s['run_dir']); run.mkdir(parents=True,exist_ok=True)
    sig = fingerprint(s)
    pre = read_json(run/'preflight.json')
    assert pre['passed'] and pre['signature']==sig, 'Run preflight for these exact settings'
    path = run/'last.pth'
    model,cfg = worker.create_model(s, checkpoint=path if path.exists() else s['init_checkpoint'])
    cfg.dump(str(run/'resolved_config.py'))
    dataset = build_dataset(cfg.data.train)
    optimizer = build_optimizer(model,cfg.optimizer)
    lr_multipliers = [group['lr']/float(cfg.optimizer.lr) for group in optimizer.param_groups]
    scaler = torch.cuda.amp.GradScaler(init_scale=128.)
    update=offset=pass_index=overflows=stale=0
    history=[]; best=s['baseline_map']; elapsed_prior=0.
    if path.exists():
        state=torch.load(path,map_location='cpu')
        assert state['signature']==sig, 'Resume settings or code changed; create another stage'
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value) and key!='step': values[key]=value.cuda()
        update,offset,pass_index=(state[k] for k in ('update','offset','pass_index'))
        best,history,stale,overflows=(state[k] for k in ('best','history','stale','overflows'))
        elapsed_prior=state['elapsed_seconds']; restore_rng(state['rng']); del state
    elif not (run/'best.pth').exists():
        # Baseline remains eligible if every recovery checkpoint is worse.
        worker.atomic_torch(dict(state_dict=worker.state_cpu(model),metrics=dict(map=best),
                                 update=0,settings=s),run/'best.pth')
    start=time.perf_counter(); loss_sum=0.; seen=0
    last_evaluated = history[-1]['update'] if history else 0
    def elapsed(): return elapsed_prior+time.perf_counter()-start
    def save():
        worker.atomic_torch(dict(state_dict=worker.state_cpu(model),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),update=update,offset=offset,pass_index=pass_index,
            best=best,history=history,stale=stale,overflows=overflows,rng=rng_state(),
            elapsed_seconds=elapsed(),signature=sig,settings=s),path)
        save_json(run/'history.json',history)
    def validate():
        nonlocal best,stale,last_evaluated
        metrics=evaluate(model,s,run/f'update_{update:05d}')
        metrics.update(update=update,train_loss=loss_sum/max(seen,1),lr=optimizer.param_groups[0]['lr'])
        improvement=metrics['map']-best
        if improvement>0:
            best=metrics['map']
            worker.atomic_torch(dict(state_dict=worker.state_cpu(model),metrics=metrics,
                update=update,settings=s),run/'best.pth')
        stale=0 if improvement>=s['min_delta'] else stale+1
        history.append(metrics); last_evaluated=update
        save_json(run/'history.json',history)
        print(f'FULL-IMAGE validation update {update}: AP={metrics["map"]:.4f}; '
              f'target {s["target_map"]:.2f}: {"PASS" if metrics["map"]>=s["target_map"] else "FAIL"}. '
              'Fixed tile fusion is evaluated separately after candidate selection.',flush=True)
        train_mode(model,s)
    train_mode(model,s); optimizer.zero_grad(set_to_none=True)
    reason='update_budget'
    if (run/'STOP').exists():
        save()
        save_json(run/'result.json',dict(reason='user_stop',best_map=best,
            target_map=s['target_map'],target_reached=best>=s['target_map'],updates=update,
            elapsed_hours=elapsed()/3600,checkpoint=str(run/'best.pth')))
        return
    while update<s['max_updates']:
        generator=torch.Generator().manual_seed(s['seed']+pass_index)
        order=torch.randperm(len(dataset),generator=generator).tolist()
        loader=DataLoader(dataset,batch_size=1,sampler=order[offset:],num_workers=0,
                          collate_fn=functools.partial(collate,samples_per_gpu=1),
                          generator=torch.Generator().manual_seed(s['seed']+pass_index))
        group_count=0
        for raw in loader:
            if group_count==0:
                group_size=min(s['accumulate'],len(dataset)-offset)
                t=min(1.,update/max(1,s['max_updates']-1))
                minimum=float(s.get('min_lr_ratio', .1))
                lr=s['learning_rate']*(minimum+(1-minimum)*.5*(1+math.cos(math.pi*t)))
                lr*=min(1.,(update+1)/s['warmup_updates'])
                for group,multiplier in zip(optimizer.param_groups,lr_multipliers):
                    group['lr']=lr*multiplier
            batch=scatter(raw,[0])[0]
            if s.get('trace_samples',False):
                meta=batch['img_metas'][0]
                trace=dict(filename=meta['filename'],flip=meta.get('flip'),shape=list(batch['img'].shape),
                    torch_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    cuda_rng=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                    numpy_rng=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest())
                with (run/'sample_trace.jsonl').open('a') as f: f.write(json.dumps(trace)+'\n')
            with torch.cuda.amp.autocast(): loss,_=worker.parse_loss(model,model(**batch,return_loss=True))
            scaler.scale(loss/group_size).backward()
            loss_sum+=float(loss.detach()); seen+=1; offset+=1; group_count+=1
            if group_count<group_size: continue
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),35.)
            if torch.isfinite(norm):
                scaler.step(optimizer); scaler.update()
            else:
                overflows+=1; scaler.update(new_scale=max(1.,scaler.get_scale()/2))
            optimizer.zero_grad(set_to_none=True); update+=1; group_count=0
            if offset==len(dataset): pass_index+=1; offset=0
            if update%10==0 or update==1:
                progress=dict(stage='train',update=update,max_updates=s['max_updates'],best_map=best,
                    loss=loss_sum/seen,lr=lr,elapsed_hours=elapsed()/3600,overflows=overflows)
                save_json(run/'progress.json',progress); print(json.dumps(progress),flush=True)
            stop=(run/'STOP').exists() or bool(SESSION_STOP and Path(SESSION_STOP).exists())
            time_up=(elapsed()>=s['max_hours']*3600 or
                     (DEADLINE is not None and time.time()>=DEADLINE))
            if update%s['eval_every']==0 and not stop and not time_up:
                save()  # Preserve completed updates even if validation is interrupted.
                validate(); save()
                time_up=(elapsed()>=s['max_hours']*3600 or (DEADLINE is not None and time.time()>=DEADLINE))
            elif update%s['save_every']==0 or stop or time_up or update>=s['max_updates']:
                save()
            if overflows>10:
                save()
                raise FloatingPointError('Repeated gradient overflows; recovery checkpoint saved')
            if stop: reason='user_stop'
            elif best>=s['target_map']: reason='target_reached'
            elif stale>=s['patience']: reason='validation_plateau'
            elif time_up: reason='time_budget'
            else: reason='update_budget'
            if stop or best>=s['target_map'] or stale>=s['patience'] or time_up or update>=s['max_updates']:
                break
        if reason!='update_budget' or update>=s['max_updates']: break
    if update>last_evaluated and reason not in ('user_stop','time_budget'):
        validate()
    save()
    result=dict(reason=reason,best_map=best,target_map=s['target_map'],target_reached=best>=s['target_map'],
                updates=update,elapsed_hours=elapsed()/3600,checkpoint=str(run/'best.pth'))
    save_json(run/'result.json',result); save_json(run/'progress.json',dict(stage='stopped',**result))
    print(json.dumps(result,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['preflight','train','evaluate'])
    ap.add_argument('--settings',required=True)
    ap.add_argument('--output'); ap.add_argument('--mode',default='single',choices=['single','tta','tiles'])
    a=ap.parse_args(); s=read_json(a.settings)
    worker.seed_all(s['seed']); torch.set_num_threads(8); torch.backends.cudnn.benchmark=False
    if a.command=='preflight': preflight(s)
    elif a.command=='train': train(s)
    else:
        model,_=worker.create_model(s,checkpoint=s['init_checkpoint'])
        evaluate(model,s,a.output,a.mode)


if __name__=='__main__': main()
