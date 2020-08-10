## Ubuntu Dockerfile


This repository contains **Dockerfile** of [Ubuntu](http://www.ubuntu.com/) for [Docker](https://www.docker.com/)'s [automated build](https://registry.hub.docker.com/u/dockerfile/ubuntu/) published to the public [Docker Hub Registry](https://registry.hub.docker.com/).


### Base Docker Image

* [tensorflow:1.14.0-gpu-py3-jupyter](https://hub.docker.com/r/tensorflow/tensorflow/)


### Installation

1. Install [Docker](https://www.docker.com/).

2. Download [automated build](https://registry.hub.docker.com/u/dockerfile/ubuntu/) from public [Docker Hub Registry](https://hub.docker.com/layers/tensorflow/tensorflow/1.14.0-gpu-py3-jupyter/images/sha256-a55c6041a788e4ca58304d8f850fa5cd70f99e1c37228bba8794e8b6a9c45ac1): `docker pull tensorflow/tensorflow:1.14.0-gpu-py3-jupyter`

   (alternatively, you can build an image from Dockerfile: `docker build -t harbor.gemini.com:30003/test/tensorflow_classroom:v2 . --no-cache`)


### Usage

    docker run -it -e account=edward -e PASSWORD=password -p 8088:8888 -p 8022:22 harbor.gemini.com:30003/test/tensorflow_classroom:v2
    
    
### Test
    
 1. use browser connect the website
 
```
IP:8088

You can use the "password" to login JupyterNoteBook.
```
    
 2. use ssh to connect the pods
```
ssh root@IP -p 8022
```
