FROM ubuntu:22.04

WORKDIR /solution

# dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
        build-essential git python3 python3-pip wget \
        ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0

COPY requirements.txt requirements.txt
RUN pip3 install -U pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN gdown 1ALUL9HRmv7vrTYN50_ORwWX9YuVGBqfI
RUN mkdir weights
RUN mv llama-2-7b-chat.Q4_K_M.gguf weights/llama-2-7b-chat.Q4_K_M.gguf
COPY . .


CMD /bin/sh -c "python3 api.py"