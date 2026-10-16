#!/bin/bash
set -e
echo "=== Maritime Backend Hot-Update: SQLite Metadata Store ==="

# Download the code update zip from S3
cd /opt/ml_backend
aws s3 cp s3://maritime-capstone-app-images-2026/backend_code_update.zip . --region ap-south-1
unzip -o backend_code_update.zip

# Ensure the host-side /data dir exists for SQLite persistence
mkdir -p /data

# Rebuild the Docker image (picks up new Dockerfile with /data VOLUME declaration)
echo "Rebuilding Docker image..."
docker build -t maritime-backend .

# Stop and remove the old container (ignore error if not running)
docker stop maritime-api 2>/dev/null || true
docker rm   maritime-api 2>/dev/null || true

# Start updated container with /data bind-mount so detections.db persists on host
echo "Starting updated container with /data volume mount..."
docker run -d --name maritime-api \
  -p 80:8000 \
  -v /data:/data \
  -e AWS_ACCESS_KEY_ID=$(aws configure get default.aws_access_key_id) \
  -e AWS_SECRET_ACCESS_KEY=$(aws configure get default.aws_secret_access_key) \
  -e AWS_DEFAULT_REGION=ap-south-1 \
  maritime-backend

echo ""
echo "=== Update complete! ==="
echo "SQLite metadata DB will live at /data/detections.db on the EC2 host."
docker ps | grep maritime-api
