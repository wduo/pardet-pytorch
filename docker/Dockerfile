ARG PYTORCH="1.4"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

LABEL maintainer="wduo <wduo2017@163.com>"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Install necessary libraries(Ubuntu 18.04)
COPY ./tsinghua_source/ /tsinghua_source/
RUN mv /tsinghua_source/etc/apt/sources.list /etc/apt/ \
  && mkdir /root/.pip/ && mv /tsinghua_source/root/.pip/pip.conf /root/.pip/

# For OpenCV
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN pip install addict yapf opencv-python
RUN pip install future tensorboard

WORKDIR /pardet-pytorch/
