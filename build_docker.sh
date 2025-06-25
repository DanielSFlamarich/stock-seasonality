#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "Usage: build_docker.sh container_tag    "
  else
    echo $1
    sudo docker build -t $1:latest

    docker run -v /var/git/dataanalytics-jupyter/container_apps/.aws:/root/.aws:ro  -p 8504:8504 -e AWS_PROFILE=default  $1:latest
fi
