# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# RUN git clone https://github.com/mozilla/TTS
# Il ne faut pas faire de git clone dans le dockerfile 
# 1- Pour des raisons de sécurité, si le git est privé, il faut une clé et si l'on oublie de retirer la clé de l'image docker, fuite de données
# 2- Si le code git évolue, mon container ne devient plus immuable

COPY requirements.txt /opt/ml/code/requirements.txt
# Make sure the copy worked
# RUN ls /opt/ml/code
RUN pip install -r /opt/ml/code/requirements.txt
RUN rm -rf /opt/ml/code/requirements.txt
RUN pip install numpy --upgrade 
RUN apt-get -y update
RUN apt-get --yes install libsndfile1-dev
RUN apt-get --yes install espeak-ng