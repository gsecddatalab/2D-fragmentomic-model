ARG BASE_IMAGE=tensorflow/tensorflow:2.17.0-gpu

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /module

COPY ./requirements.txt ./

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install -r requirements.txt

COPY ./config /module/config
COPY ./src /module/src
COPY ./pyproject.toml /module/pyproject.toml

RUN pip install -e .
