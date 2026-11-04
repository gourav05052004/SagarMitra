import os
import io
import csv
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import boto3
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse

# Import pipeline and database
from pipeline_engine import MaritimePipeline, draw_detections
from database import init_db, save_run, get_records, get_total_runs, get_stats, get_all_runs_for_csv, get_detections_for_run

# ── Initialize FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(
    title="Maritime Object Detection API",
    description="Real-time maritime object detection with persistent metadata logging.",
    version="2.0.0",
)

# ── Configuration ──────────────────────────────────────────────────────────────
S3_BUCKET_NAME = "maritime-capstone-app-images-2026"

# ── Clients & Engine ───────────────────────────────────────────────────────────
s3_client = boto3.client('s3')

# Initialize SQLite metadata store
init_db()

# Load the comprehensive maritime pipeline engine
try:
    engine = MaritimePipeline()
    print("Pipeline Engine initialized successfully!")
except Exception as e:
    print(f"Failed to initialize pipeline engine: {e}")
    engine = None


# ── /detect ────────────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    min_confidence: float = Query(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Filter out detections below this confidence threshold (default 0.50 / 50%)"
    )
):
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "Pipeline engine not loaded"})

    try:
        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        image_height, image_width = img.shape[:2]

        # ── Run inference and measure wall-clock time ──────────────────────────
        t_start = time.perf_counter()
        raw_detections = engine.predict(img)
        inference_time_s = time.perf_counter() - t_start

        # ── Filter detections to 50%+ confidence (or custom min_confidence) ────
        final_detections = [
            det for det in raw_detections
            if float(det.get("score", 0.0)) >= min_confidence
        ]

        # Draw bounding boxes (only for high-confidence detections)
        annotated_img = draw_detections(img, final_detections, engine.config)

        # Encode the annotated image to bytes (JPEG format)
        success, encoded_img = cv2.imencode(".jpg", annotated_img)
        if not success:
            return JSONResponse(status_code=500, content={"error": "Failed to encode the processed image"})
        image_bytes = encoded_img.tobytes()

        # Generate unique filename and upload to S3
        run_id = str(uuid.uuid4())
        filename = f"{run_id}.jpg"

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=image_bytes,
            ContentType='image/jpeg',
            ACL='public-read'
        )

        image_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # ── Persist metadata to SQLite ─────────────────────────────────────────
        save_run(
            run_id=run_id,
            timestamp=timestamp,
            filename=filename,
            image_url=image_url,
            image_width=image_width,
            image_height=image_height,
            detections=final_detections,
            device=engine.device,
            inference_time_s=round(inference_time_s, 4),
            cbnet_checkpoint=engine.config.get("primary_detector", {}).get("checkpoint", ""),
            yolo_checkpoint=engine.config.get("complementary_expert", {}).get("checkpoint", ""),
        )

        return {
            "success": True,
            "run_id": run_id,
            "image_url": image_url,
            "filename": filename,
            "detections": final_detections,
            "image_width": image_width,
            "image_height": image_height,
            "inference_time_s": round(inference_time_s, 4),
            "message": "Image processed, uploaded, and metadata logged successfully.",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── /records ───────────────────────────────────────────────────────────────────
@app.get("/records")
def list_records(
    limit: int = Query(default=50, ge=1, le=500, description="Max rows to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
):
    """
    Paginated list of all detection runs stored in the SQLite metadata store.
    Each record includes per-class counts, avg confidence, image dimensions,
    inference time, and device info — ready for research paper tables.
    """
    records = get_records(limit=limit, offset=offset)
    total = get_total_runs()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": records,
    }


# ── /records/{run_id}/detections ───────────────────────────────────────────────
@app.get("/records/{run_id}/detections")
def run_detections(run_id: str):
    """Return individual per-object detections for a specific run."""
    dets = get_detections_for_run(run_id)
    if not dets:
        return JSONResponse(status_code=404, content={"error": "Run not found or no detections recorded."})
    return {"run_id": run_id, "detections": dets}


# ── /stats ─────────────────────────────────────────────────────────────────────
@app.get("/stats")
def detection_stats():
    """
    Aggregate statistics across ALL detection runs.
    Returns total object counts per class, overall average confidence,
    average inference time, and per-class score distributions —
    directly usable in research paper results sections.
    """
    return get_stats()


# ── /export/csv ────────────────────────────────────────────────────────────────
@app.get("/export/csv")
def export_csv():
    """
    Download all detection-run metadata as a CSV file.
    Useful for importing into Excel / LaTeX tables for your research paper.
    """
    runs = get_all_runs_for_csv()
    if not runs:
        return JSONResponse(status_code=404, content={"error": "No records found."})

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=runs[0].keys())
    writer.writeheader()
    writer.writerows(runs)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=maritime_detections.csv"},
    )


# ── /history (legacy S3 listing) ──────────────────────────────────────────────
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

@app.get("/history")
def get_history(limit: int = 20):
    """Legacy endpoint: list images from S3 (no detailed metadata)."""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        if 'Contents' not in response:
            return {"history": []}

        # Filter out non-image files (e.g. .zip, .sh, .txt, scripts, etc.)
        image_objects = [
            obj for obj in response['Contents']
            if obj['Key'].lower().endswith(VALID_IMAGE_EXTENSIONS)
        ]

        objects = sorted(image_objects, key=lambda x: x['LastModified'], reverse=True)

        history = []
        for obj in objects[:limit]:
            filename = obj['Key']
            image_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"
            history.append({
                "filename": filename,
                "image_url": image_url,
                "timestamp": obj['LastModified'].isoformat()
            })

        return {"history": history}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── / ──────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Maritime Object Detection API is running",
        "endpoints": {
            "POST /detect": "Run detection and log metadata",
            "GET /records": "Paginated detection run metadata (SQLite)",
            "GET /records/{run_id}/detections": "Per-object detections for a run",
            "GET /stats": "Aggregate statistics across all runs",
            "GET /export/csv": "Download all metadata as CSV",
            "GET /history": "Legacy S3 image listing",
        }
    }
