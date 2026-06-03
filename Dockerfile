FROM ghcr.io/andrewembry312-hub/elabs-server/whisper-runpod:latest
LABEL maintainer="E-Labs AI Studio" description="Whisper large-v3 RunPod worker handler"
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY handler.py /workspace/handler.py
COPY requirements-runpod.txt /workspace/requirements-runpod.txt

CMD ["python", "-u", "handler.py"]
