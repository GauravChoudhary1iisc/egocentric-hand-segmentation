FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONHASHSEED=42
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY common.py annotate.py ./
COPY weights/ weights/
COPY outputs/config.json outputs/config.json
ENTRYPOINT ["python", "annotate.py"]
# docker build -t egohos-hands .
# docker run --rm --gpus all -v $PWD/sample50:/in:ro -v $PWD/docker_out:/out egohos-hands --input /in --output /out
