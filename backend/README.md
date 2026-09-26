# Best verified maritime model — application bundle

This folder contains only the files required to integrate the highest-scoring verified pipeline into an application. Its measured score is **61.573523 COCO bbox AP50:95 on 1,547 validation images**. Hidden-test performance must be obtained from MaCVi.

## Important: this is a pipeline, not one checkpoint

The 61.5735 result cannot be reproduced by loading only the CBNet checkpoint or only the YOLO checkpoint. The deployed sequence is:

1. CBNetV2–Swin-S Cascade R-CNN full-image and tiled detection.
2. Native edge/class-safe box refinement, scoring, tree correction and pixel-boundary correction.
3. Full-image regression-detail YOLO11s expert fusion.
4. Tiled expert correction for tiny swimmers and life-saving appliances.

`pipeline_config.json` gives the fixed application order and inference parameters. `bundle_manifest.json` records the SHA256, byte size and original location of every copied artifact.

The `native_refiner`, `boundary_distribution`, `adaptive_boundary`, `fast_residual`, and ResNet files are retained dependencies of the selected refiners' constructors. They are lineage dependencies of the one best pipeline, not alternative models offered for selection.

## Folder layout

- `models/`: all checkpoints and fixed tree data needed by the final pipeline.
- `source/`: model definitions needed to deserialize and execute the custom architectures.
- `records/`: frozen validation records and settings establishing which checkpoints and policies were selected.
- `reference/`: the existing complete export integration. It is included as implementation guidance; its workspace paths must be replaced by bundle-relative paths in the app.
- `runtime/`: portable inference helpers that avoid importing training controllers.
- `pipeline_config.json`: concise, path-independent inference recipe.
- `verify_bundle.py`: integrity verification for all packaged files.

## Verify before app development

Run in PowerShell:

```powershell
& 'C:\Users\HPCLAB\MaritimeTraining\CBNet_HPC\.venv\Scripts\python.exe' verify_bundle.py
```

This does not perform inference. It confirms that every copied model and source file matches the packaged manifest.

## Application contract

Input is one BGR/RGB image at its original resolution. Output is a JSON-compatible list of detections:

```json
{"image_id": 1, "category_id": 1, "bbox": [x, y, width, height], "score": 0.9}
```

Coordinates are in original-image pixels. Category IDs are 1–5 and must retain the mapping in `pipeline_config.json`. For a visual app, convert these records to labeled overlays after all fusion and correction stages complete.

The application environment should initially match Python 3.11.13, PyTorch 2.0.1+cu118, MMCV 1.7.2, MMDetection 2.28.2 and Ultralytics 8.3.0. Porting to a newer stack should be treated as a separate conversion and parity-validation task.

This is a complete **model-assets bundle**, not yet a web server or mobile package. The app layer still needs a persistent model loader, a single-image inference function, image/video UI, and overlays. Start from `reference/export_latest.py`, replace its workspace constants with paths from `pipeline_config.json`, and use `runtime/pixel_profiles.py` for the pixel head rather than importing its training controller.

Do not rename a single checkpoint as “the 61.5735 model.” The score belongs to the complete staged pipeline in this folder.
