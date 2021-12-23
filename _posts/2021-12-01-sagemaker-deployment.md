---
layout: post
title: "Sagemaker Deployment"
date: 2021-12-01
category: Cloud-Computing
image: 
excerpt: Deploy custom model in Sagemaker
katex: True
---

### Write custom Container

We create a folder '/opt/program' inside the container where we store the server files:
* serve: starts the server API
* predictor.py: defines Flask REST API

When Sagemaker runs the container it starts the CMD "serve".

The file
```console 
predictor.py
``` 
loads the pickle model and implements a REST API with two methods:
* GET /ping
* POST /invocations

which Sagemaker expects.

The pickled model can be copied directly to the container to a folder of choice. Or it can be stored in a S3 bucket and passed on to Sagemaker as an artifact. Sagemaker then extracts the tar.gz file from S3 and copies it to the folder '/opt/ml/model'. Therefore, if we pass the model as an artifact, the predictor module needs to unpickle the file at '/opt/ml/model'.

The Dockerfile is the basic structure:
```Dockerfile
FROM ubuntu:latest

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         python3-pip\
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

#Install python libraries
COPY requirements.txt /opt/program/
RUN python3 -m pip install /opt/prorgam/requirements.txt && \
        rm -rf /root/.cache

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

#copy model to /opt/ml/model or other folder
COPY model.pkl /opt/ml/model/
# Set up the program in the image
COPY model-files /opt/program
WORKDIR /opt/program
RUN chmod +x serve

CMD [ "serve" ]
```

We can run the container locally and test the API:
```console
#build model
docker build -t sagemaker-model .
docker run -p 8080:8080 sagemaker-model:latest 
```

Now we can access the API at 127.0.0.1:8080:
```console
curl --location --request POST 'http://localhost:8080/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{"data": [[1,2],[3,4],[3,3],[10,1],[7,8]]}'
```

### Sagemaker Deployment

First we need to push our docker image to our AWS ECR repository. Assuming that we have already created a repository with name "aws_account_id".dkr.ecr."region".amazonaws.com/sagemaker-model, we tag the docker image with the same repository URI, that is,

```console
docker tag sagemaker-model:latest "aws_account_id".dkr.ecr."region".amazonaws.com/sagemaker-model:latest
```
and then push
```console
docker push "aws_account_id".dkr.ecr."region".amazonaws.com/model-sagemaker:latest
```

Now that we have uploaded the docker image we can go to Sagemaker section and create a new model.