$ErrorActionPreference = "Continue"
$Region = "us-east-1"
$RandomSuffix = Get-Random -Minimum 1000 -Maximum 9999
$BucketName = "maritime-capstone-backend-$RandomSuffix"

Write-Host "Creating S3 Bucket: $BucketName"
aws s3 mb s3://$BucketName --region $Region

$RoleName = "MaritimeEC2Role-$RandomSuffix"
$AssumeRolePolicy = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

Write-Host "Creating IAM Role: $RoleName"
$null = aws iam create-role --role-name $RoleName --assume-role-policy-document $AssumeRolePolicy
$null = aws iam attach-role-policy --role-name $RoleName --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
$null = aws iam attach-role-policy --role-name $RoleName --policy-arn "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"

Write-Host "Creating Instance Profile: $RoleName"
$null = aws iam create-instance-profile --instance-profile-name $RoleName
$null = aws iam add-role-to-instance-profile --instance-profile-name $RoleName --role-name $RoleName

Write-Host "Waiting 15s for IAM propagation..."
Start-Sleep -Seconds 15

Write-Host "Creating Security Group..."
$VpcId = aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text
$SgName = "MaritimeBackendSG-$RandomSuffix"
$SgId = aws ec2 create-security-group --group-name $SgName --description "Security group for Maritime backend" --vpc-id $VpcId --query "GroupId" --output text
if ($SgId) {
    $null = aws ec2 authorize-security-group-ingress --group-id $SgId --protocol tcp --port 80 --cidr "0.0.0.0/0"
    $null = aws ec2 authorize-security-group-ingress --group-id $SgId --protocol tcp --port 22 --cidr "0.0.0.0/0"
}

Write-Host "Finding appropriate AMI..."
# Try Deep Learning AMI first
$AmiId = aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*" --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text

Write-Host "Launching EC2 Instance..."
# Try to launch g4dn.xlarge
$InstanceId = $null
try {
    $InstanceId = aws ec2 run-instances --image-id $AmiId --instance-type "g4dn.xlarge" --security-group-ids $SgId --iam-instance-profile Name=$RoleName --query "Instances[0].InstanceId" --output text
} catch {
    Write-Host "Failed to launch g4dn.xlarge. Falling back to t3.xlarge..."
}

if (-not $InstanceId) {
    $AmiIdUbuntu = aws ec2 describe-images --owners 099720109477 --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text
    $InstanceId = aws ec2 run-instances --image-id $AmiIdUbuntu --instance-type "t3.xlarge" --security-group-ids $SgId --iam-instance-profile Name=$RoleName --query "Instances[0].InstanceId" --output text
}

Write-Host ""
Write-Host "INFRASTRUCTURE CREATED SUCCESSFULLY"
Write-Host "----------------------------------"
Write-Host "S3Bucket: $BucketName"
Write-Host "InstanceId: $InstanceId"
Write-Host "----------------------------------"
