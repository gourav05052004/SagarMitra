import os
import uuid
import cv2
import numpy as np
import boto3
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Import our new pipeline engine
from pipeline_engine import MaritimePipeline, draw_detections

# Initialize FastAPI App
app = FastAPI(title="Maritime Object Detection API")

# Configuration
S3_BUCKET_NAME = "maritime-capstone-app-images-2026"

# Initialize S3 Client (it uses credentials from `aws configure`)
s3_client = boto3.client('s3')

# Load the comprehensive maritime pipeline engine
try:
    engine = MaritimePipeline()
    print("Pipeline Engine initialized successfully!")
except Exception as e:
    print(f"Failed to initialize pipeline engine: {e}")
    engine = None

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "Pipeline engine not loaded"})

    try:
        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        # Run inference through the pipeline
        final_detections = engine.predict(img)
        
        # Draw bounding boxes using our custom utility
        annotated_img = draw_detections(img, final_detections, engine.config)

        # Encode the annotated image to bytes (JPEG format)
        success, encoded_img = cv2.imencode(".jpg", annotated_img)
        if not success:
            return JSONResponse(status_code=500, content={"error": "Failed to encode the processed image"})
        image_bytes = encoded_img.tobytes()

        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"

        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=image_bytes,
            ContentType='image/jpeg',
            ACL='public-read'
        )

        # Construct the permanently public URL
        image_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"

        return {
            "success": True,
            "image_url": image_url,
            "filename": filename,
            "detections": final_detections,
            "message": "Image processed and uploaded successfully."
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history")
def get_history(limit: int = 20):
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        if 'Contents' not in response:
            return {"history": []}
        
        # Sort objects by LastModified descending
        objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        
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

@app.get("/")
def root():
    return {"message": "Maritime Object Detection API is running"}
