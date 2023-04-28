FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime@sha256:fabb5a665a05b8ee0ac76f0d943acc40039e13536e11a44d3dc47625a266e759

RUN apt update -yq && \
    apt install -y build-essential vim && \
    pip install -U pip setuptools wheel && \
    pip install -U so-vits-svc-fork

ENTRYPOINT [ "bash" ]
