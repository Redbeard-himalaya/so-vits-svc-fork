FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt update -yq && \
    apt install -y build-essential && \
    pip install -U pip setuptools wheel && \
    pip install -U so-vits-svc-fork

ENTRYPOINT [ "bash" ]
