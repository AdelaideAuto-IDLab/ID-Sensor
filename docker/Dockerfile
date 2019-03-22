FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y wget zlib1g-dev libssl-dev curl libsqlite3-dev
RUN wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz && tar -xvf Python-3.6.8.tgz
RUN cd Python-3.6.8 && ./configure && make && make install
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.0.1-cp36-cp36m-linux_x86_64.whl
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt
RUN ln -fs /usr/local/bin/python3 /usr/local/bin/python && rm Python-3.6.8.tgz && rm -R Python-3.6.8
WORKDIR /

ENV LANG C.UTF-8