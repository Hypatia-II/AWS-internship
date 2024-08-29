# Création d’une instance EC2 m6a.4xlarge avec EBS standard
#Ajout des permissions S3 et ECR
### Possible en ligne de commande ? 



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
aws s3 cp s3://benitoin-buckettest/build_docker_image.zip  .
unzip build_docker_image.zip
# Add ECR credentials to download the container to extend
# This ECR does not corresponds to your personal one, it is the one where the docker image to extend is stored
aws ecr get-login-password —region us-east-1 | sudo docker login —username AWS —password-stdin 	763104351884.dkr.ecr.us-east-1.amazonaws.com
# Build the docker image
docker build -t extend-container-tts .
## Here an error should occur if the size of the EBS volume is too small
## Hence we increase the volume and run this command again

# Command to increase the size of the EBS volume (iops 100/3000)
# aws ec2 modify-volume --volume-type gp2 --iops 100 --size 25 --volume-id vol-<volume_id>
aws ec2 modify-volume --volume-type gp2 --iops 100 --size 25 --volume-id vol-0968b20094390707e

df -hT
lsblk
# Command to extend the partition number 1 of the nvme0n1 volume
sudo growpart /dev/nvme0n1 1 
lsblk
df -hT
# To extend the file system on each volume
# sudo yum install xfsprogs
sudo xfs_growfs –d /

docker build -t extend-container-tts .

# Once the image is created we push it to ECR
# aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 082780074557.dkr.ecr.eu-west-1.amazonaws.com
# To determine the image id we want to push to ECR
docker images
# docker tag <image_id> <aws_account_id>.dkr.ecr.<region>.amazonaws.com/my-repository:tag
## Piece of advice: do not use underscores in the name of the repository, it causes troubles later
docker tag e9ae3c220b23 082780074557.dkr.ecr.eu-west-1.amazonaws.com/extend-container-tts:latest
docker push 082780074557.dkr.ecr.eu-west-1.amazonaws.com/extend-container-tts:latest