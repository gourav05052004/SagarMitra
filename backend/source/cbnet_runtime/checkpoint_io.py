"""Checkpoint integrity helpers; no environment installer or process launcher."""
import hashlib
from pathlib import Path

MODEL_FILE = 'cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compatible_state(state, target_keys, stages, allow_quality_init=False, allow_deform_init=False, allow_detail_init=False):
    """Strict transfer, except an intentional stage removal or new quality layer."""
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    excluded = [key for key in state if key.startswith('roi_head.bbox_head.2.')]
    if stages == 2:
        state = {key: value for key, value in state.items() if key not in excluded}
    else:
        excluded = []
    missing = set(target_keys) - set(state)
    detail_missing={key for key in missing if key.startswith('roi_head.detail_')
                    and key.endswith(('.weight','.bias'))}
    if allow_detail_init:
        missing -= detail_missing
    else:
        detail_missing=set()
    deform_missing={key for key in missing if key.startswith('roi_head.bbox_head.')
                    and '.reg_refine.' in key
                    and key.endswith(('.weight','.bias'))}
    if allow_deform_init:
        missing-=deform_missing
    else:
        deform_missing=set()
    quality_missing = {key for key in missing if '.fc_quality.' in key}
    if allow_quality_init:
        missing -= quality_missing
    elif quality_missing:
        quality_missing = set()
    unexpected = set(state) - set(target_keys)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: missing={sorted(missing)[:12]}, '
                           f'unexpected={sorted(unexpected)[:12]}')
    return state, excluded, sorted(quality_missing | deform_missing | detail_missing)
