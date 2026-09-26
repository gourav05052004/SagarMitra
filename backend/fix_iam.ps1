$ErrorActionPreference = "Continue"
$RoleName = "MaritimeEC2Role-Fix"
$InstanceId = "i-0b9e17486685f4cc2"

Write-Host "Creating IAM Role..."
aws iam create-role --role-name $RoleName --assume-role-policy-document file://trust.json

Write-Host "Attaching policies..."
$null = aws iam attach-role-policy --role-name $RoleName --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
$null = aws iam attach-role-policy --role-name $RoleName --policy-arn "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"

Write-Host "Creating Instance Profile..."
$null = aws iam create-instance-profile --instance-profile-name $RoleName
$null = aws iam add-role-to-instance-profile --instance-profile-name $RoleName --role-name $RoleName

Write-Host "Waiting 15s for propagation..."
Start-Sleep -Seconds 15

Write-Host "Associating instance profile with EC2..."
aws ec2 associate-iam-instance-profile --instance-id $InstanceId --iam-instance-profile Name=$RoleName
