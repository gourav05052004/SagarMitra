<#
.SYNOPSIS
Deploys the ML backend to an Amazon EC2 instance using AWS CLI.

.DESCRIPTION
This script archives the backend code, uploads it to an S3 bucket, and then uses 
AWS Systems Manager (SSM) to run a command on the target EC2 instance to pull, 
unzip, build the Docker container (which handles the strict cu118 / mmcv dependencies), 
and run the FastAPI server on port 80.

.EXAMPLE
.\deploy_to_ec2.ps1 -S3Bucket "maritime-capstone-app-images-2026" -InstanceId "i-0abcd1234efgh5678"
#>

param (
    [Parameter(Mandatory=$true)]
    [string]$S3Bucket,

    [Parameter(Mandatory=$true)]
    [string]$InstanceId,

    [string]$Region = "us-east-1",
    [string]$ZipName = "backend_bundle.zip"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " AWS EC2 Docker Backend Deployment Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Zip the backend directory (excluding the zip itself and the massive conda env if present)
Write-Host "`n[1/4] Archiving backend files..." -ForegroundColor Yellow
$DestinationPath = Join-Path $PWD $ZipName
if (Test-Path $DestinationPath) { Remove-Item $DestinationPath -Force }

# Use Compress-Archive to package everything
Compress-Archive -Path ".\*" -DestinationPath $DestinationPath -Force
Write-Host "Archive created at: $DestinationPath" -ForegroundColor Green

# 2. Upload to S3
Write-Host "`n[2/4] Uploading $ZipName to S3 bucket: $S3Bucket..." -ForegroundColor Yellow
aws s3 cp $DestinationPath s3://$S3Bucket/$ZipName --region $Region
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upload to S3. Ensure AWS CLI is configured and has correct permissions."
}
Write-Host "Upload successful." -ForegroundColor Green

# 3. Define the deployment commands to run on the EC2 instance
$DeployCommands = @"
#!/bin/bash
sudo apt-get update -y
sudo apt-get install -y docker.io unzip
sudo systemctl start docker
sudo systemctl enable docker
sudo mkdir -p /opt/ml_backend
sudo chown -R ubuntu:ubuntu /opt/ml_backend
cd /opt/ml_backend
aws s3 cp s3://$S3Bucket/$ZipName . --region $Region
unzip -o $ZipName
echo 'Building Docker Image (this will take 10-15 mins for MMCV compilation)...'
sudo docker build -t maritime-backend .
echo 'Stopping old container if it exists...'
sudo docker stop maritime-api || true
sudo docker rm maritime-api || true
echo 'Starting new container on port 80...'
sudo docker run -d --name maritime-api -p 80:8000 -e AWS_ACCESS_KEY_ID=`$(aws configure get default.aws_access_key_id)` -e AWS_SECRET_ACCESS_KEY=`$(aws configure get default.aws_secret_access_key)` -e AWS_DEFAULT_REGION=$Region maritime-backend
echo 'Deployment complete. Server is running on port 80.'
"@

# Save setup.sh and upload to S3
$SetupScriptPath = Join-Path $PWD "setup.sh"
Set-Content -Path $SetupScriptPath -Value $DeployCommands
aws s3 cp $SetupScriptPath s3://$S3Bucket/setup.sh --region $Region

# 4. Execute the command via SSM
Write-Host "`n[3/4] Sending deployment command to EC2 instance: $InstanceId..." -ForegroundColor Yellow
$ssmOutput = aws ssm send-command `
    --instance-ids $InstanceId `
    --document-name "AWS-RunShellScript" `
    --parameters commands="aws s3 cp s3://$S3Bucket/setup.sh . --region $Region; sudo bash setup.sh" `
    --region $Region `
    --query "Command.CommandId" `
    --output text

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to send SSM command. Ensure the EC2 instance has the AmazonSSMManagedInstanceCore IAM role attached."
}

Write-Host "Command sent successfully. Command ID: $ssmOutput" -ForegroundColor Green

Write-Host "`n[4/4] Deployment process initiated." -ForegroundColor Cyan
Write-Host "You can check the execution status on your EC2 instance by running:"
Write-Host "aws ssm list-command-invocations --command-id $ssmOutput --details" -ForegroundColor Gray
