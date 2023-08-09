FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY . .

RUN apt update -yq && \
    apt install -y build-essential vim && \
    pip install -U pip setuptools wheel && \
    pip install . && \
    rm -rf /workspace

ENTRYPOINT [ "bash" ]
