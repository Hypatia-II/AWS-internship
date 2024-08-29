#!/bin/sh

# Prior to running this code, it is necessary to have a running m6a.4xlarge EC2 instance with a standard EBS volume
# The EC2 instance should have IAM permissions for S3 and ECR

# To run this code, it is necessary to pass the following parameters:
# The name of the bucket where you stored the zip file with the requirements.txt and Dockerfile files
BUCKET=$1
# The name of the zip file
ZIPFILE=$2
# The EBS volume ID of the instance 
VOL_ID=$3
# The region where the docker image will be stored
REGION=$4
# The account ID
ACCOUNT_ID=$5


# Update the packages on your instance
sudo yum update -y
# Install Docker
sudo yum install docker -y
# Start the Docker Service
sudo service docker start
#Add the ec2-user to the docker group so you 
# can execute Docker commands without using sudo
sudo usermod -a -G docker ec2-user

# Copy files from S3 and unzip them
# The zip folder contains the requirements.txt and Dockerfile files
# aws s3 cp <uri_address> .
aws s3 cp s3://{$BUCKET}/{$ZIPFILE}  .

# unzip build_docker_image.zip
unzip $ZIPFILE

# Add ECR credentials to download the container to extend
# This ECR does not corresponds to your personal one, it is the one where the docker image to extend is stored
aws ecr get-login-password —region us-east-1 | sudo docker login —username AWS —password-stdin 	763104351884.dkr.ecr.us-east-1.amazonaws.com

# Command to increase the size of the EBS volume (iops 100/3000)
# aws ec2 modify-volume --volume-type gp2 --iops 100 --size 25 --volume-id vol-<volume_id>
aws ec2 modify-volume --volume-type gp2 --iops 100 --size 25 --volume-id vol-{$VOL_ID}

# Command to extend the partition number 1 of the nvme0n1 volume
sudo growpart /dev/nvme0n1 1 
# To extend the file system on each volume
sudo yum install xfsprogs
sudo xfs_growfs –d /

docker build -t extend-container-tts:latest .

# Once the image is created we push it to ECR
# aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin {$ACCOUNT_ID}.dkr.ecr.{$REGION}.amazonaws.com


# docker tag <image_id> <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<my-repository>:<tag>
## Piece of advice: do not use underscores in the name of the repository, it causes troubles later
docker tag extend-container-tts:latest {$ACCOUNT_ID}.dkr.ecr.{$REGION}.amazonaws.com/extend-container-tts:latest

# docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<my-repository>:<tag>
docker push {$ACCOUNT_ID}.dkr.ecr.{$REGION}.amazonaws.com/extend-container-tts:latest

# how to run the script:
# . path/TO/script.sh param1 param2 param3 ...