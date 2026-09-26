"""Build a deployment bundle by copying, never moving, verified source artifacts."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
WORK = HERE.parent

FILES = {
    # Primary detector and its runtime definitions.
    'models/cbnet_primary_update_00200.pth': 'Maritime_CBNet61_DetailHead_backend/runs/detail_001/checkpoints/update_00200.pth',
    # Native-image refinement lineage required by the constructors.
    'models/native_refiner_update_00800.pth': 'Maritime_CBNet62_NativeRefiner_backend/runs/native_001/checkpoints/update_00800.pth',
    'models/boundary_distribution_update_00400.pth': 'Maritime_CBNet62_BoundaryDistribution_backend/runs/boundary_001/checkpoints/update_00400.pth',
    'models/adaptive_boundary_update_00600.pth': 'Maritime_CBNet62_AdaptiveBoundary_backend/runs/adaptive_001/checkpoints/update_00600.pth',
    'models/edge_aligned_update_01000.pth': 'Maritime_CBNet62_EdgeAligned_backend/runs/edge_001/checkpoints/update_01000.pth',
    'models/classsafe_update_00050.pt': 'Maritime_CBNet6215_ClassSafe_backend/runs/classsafe_002/checkpoints/update_00050.pt',
    'models/fast_residual_update_00400.pt': 'Maritime_CBNet6215_Fast_backend/runs/fast_001/checkpoints/update_00400.pt',
    'models/ranking_update_00200.pt': 'Maritime_CBNet62_Evidence_backend/runs/ranking_001/checkpoints/update_00200.pt',
    'models/pixel_boundary_update_00600.pt': 'Maritime_CBNet62_PixelBoundary_backend/runs/pixel_001/checkpoints/update_00600.pt',
    'models/resnet18-f37072fd.pth': 'Maritime_CBNet6215_AllClassContext_backend/weights/resnet18-f37072fd.pth',
    'models/geometry_portable.json': 'Maritime_CBNet63_LabFinal_backend/camera_quality/geometry_portable.json',
    'models/scoring_normalization.pt': 'Maritime_CBNet62_QualityEnsemble_backend/runs/quality_001/normalization.pt',
    # Complementary expert actually used by the final result.
    'models/yolo11s_regression_detail_epoch10.pt': 'Maritime_CBNet63_LabFinal_backend/complement_evaluations/7b0e2316d69875dd/checkpoint.pt',
    # Frozen provenance and application rules.
    'records/best_verified_pipeline.json': 'Maritime_CBNet63_LabFinal_backend/tiny_tile_refinement/best_verified_pipeline.json',
    'records/final_result.json': 'Maritime_CBNet63_LabFinal_backend/tiny_tile_refinement/result.json',
    'records/final_plan.json': 'Maritime_CBNet63_LabFinal_backend/tiny_tile_refinement/plan.json',
    'records/retained_pipeline.json': 'Maritime_CBNet63_LabFinal_backend/retained_pipeline.json',
    'records/allclass_reference.json': 'Maritime_CBNet6215_AllClassContext_backend/allclass_reference.json',
    'records/native_reference.json': 'Maritime_CBNet6215_AllClassContext_backend/reference.json',
    'records/boundary_reference.json': 'Maritime_CBNet6215_AllClassContext_backend/boundary_reference.json',
    'records/edge_reference.json': 'Maritime_CBNet6215_AllClassContext_backend/edge_reference.json',
    'records/classsafe_best.json': 'Maritime_CBNet6215_ClassSafe_backend/runs/classsafe_002/best_pipeline.json',
    'records/ranking_best.json': 'Maritime_CBNet62_Evidence_backend/runs/ranking_001/best_pipeline.json',
    'records/relation_best.json': 'Maritime_CBNet62_RelationRank_backend/runs/relation_001/best_pipeline.json',
    'records/pixel_manifest.json': 'Maritime_CBNet62_PixelBoundary_backend/runs/pixel_001/manifest.json',
    'records/edge_settings.json': 'Maritime_CBNet62_EdgeAligned_backend/runs/edge_001/settings.json',
    'records/expert_args.yaml': 'Maritime_CBNet63_LabFinal_backend/regression_detail/runs/detail_001/args.yaml',
    'records/expert_initialization.json': 'Maritime_CBNet63_LabFinal_backend/regression_detail/initialized.json',
    'records/cbnet_starting_model.json': 'Maritime_CBNet6215_AllClassContext_backend/starting_model.json',
    # Existing integration serves as a readable reference for the future app.
    'reference/export_latest.py': 'Maritime_MaCVi_LatestFast_backend/export_latest.py',
    'reference/export_updated.py': 'Maritime_MaCVi_611625_backend/export_updated.py',
}

SOURCE_DIRS = {
    'source/cbnet_runtime': 'Maritime_CBNet61_DetailHead_backend',
    'source/allclass_refiners': 'Maritime_CBNet6215_AllClassContext_backend',
    'source/classsafe': 'Maritime_CBNet6215_ClassSafe_backend',
    'source/fast_features': 'Maritime_CBNet6215_Fast_backend',
    'source/ranking': 'Maritime_CBNet62_Evidence_backend',
    'source/pixel_boundary': 'Maritime_CBNet62_PixelBoundary_backend',
    'source/complementary_expert': 'Maritime_CBNet63_LabFinal_backend',
}

SOURCE_NAMES = {
    'source/cbnet_runtime': {
        'worker.py','model_config.py','recovery_config.py','recovery.py','checkpoint_io.py',
        'detail_roi.py','refinement_head.py','deformable_head.py','inference_tools.py',
        'data_tools.py','train_policy.py','quality_head.py','regression_quality.py'
    },
    'source/allclass_refiners': {
        'native_core.py','boundary_model.py','adaptive_model.py','edge_model.py',
        'allclass_core.py','allclass_data.py'
    },
    'source/classsafe': {'classsafe.py'},
    'source/fast_features': {'fast.py'},
    'source/ranking': {'ranking_core.py','experiment.py','pipeline.py'},
    'source/pixel_boundary': {'pixel_core.py','pixel_run.py'},
    'source/complementary_expert': {
        'complement_detector.py','regression_detail_extension.py','evaluate_complement.py',
        'evaluate_tiles.py','probe_tiles.py','tiny_tile_refinement.py','portable_tree.py',
        'cached_retained_branch.py'
    },
}

# pixel_core resolves this path relative to its parent at import time.
FILES['source/Maritime_CBNet62_PrecisionBox_backend/box_core.py'] = \
    'Maritime_CBNet62_PrecisionBox_backend/box_core.py'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_file(relative, source):
    source = WORK / source
    if not source.is_file():
        raise FileNotFoundError(source)
    target = HERE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha(target) == sha(source):
        return source, target
    temporary = target.with_suffix(target.suffix + '.partial')
    shutil.copy2(source, temporary)
    if sha(temporary) != sha(source):
        temporary.unlink(missing_ok=True)
        raise RuntimeError('Copy verification failed: ' + str(source))
    temporary.replace(target)
    return source, target


def main():
    copied = []
    for relative, source in FILES.items():
        copied.append(copy_file(relative, source))
    # The CBNet port is a package; retain its complete Python implementation.
    cb_source = WORK/'Maritime_CBNet61_DetailHead_backend/cbnet_port'
    for source in cb_source.rglob('*.py'):
        relative = Path('source/cbnet_runtime/cbnet_port')/source.relative_to(cb_source)
        copied.append(copy_file(relative, source.relative_to(WORK)))
    for target_dir, source_dir in SOURCE_DIRS.items():
        source_dir = WORK/source_dir
        for name in sorted(SOURCE_NAMES[target_dir]):
            source = source_dir/name
            if source.exists():
                copied.append(copy_file(Path(target_dir)/name, source.relative_to(WORK)))
    entries=[]
    for original,bundled in copied:
        entries.append(dict(
            bundled_path=str(bundled.relative_to(HERE)).replace('\\','/'),
            original_path=str(original.resolve()),
            bytes=bundled.stat().st_size,
            sha256=sha(bundled)))
    entries.sort(key=lambda item:item['bundled_path'])
    manifest=dict(
        bundle='Maritime best verified app model',
        validation_AP=61.57352337147663,
        metric='COCO bbox AP50:95', validation_images=1547, maxDets=100,
        category_mapping={'1':'swimmer','2':'boat','3':'jetski','4':'life saving appliance','5':'buoy'},
        note='This score belongs to the complete staged pipeline, not one checkpoint. Hidden-test AP is server-evaluated.',
        files=entries)
    path=HERE/'bundle_manifest.json'
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(entries),bytes=sum(x['bytes'] for x in entries),manifest=str(path)),indent=2))


if __name__ == '__main__':
    main()
