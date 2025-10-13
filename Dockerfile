# Dockerfile
FROM python:3.11.13

ENV PYTHONUNBUFFERED=1

# System deps: ffmpeg for decoding, libsndfile for soundfile, libgomp for onnx runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] numpy python-multipart

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
 --index-url https://download.pytorch.org/whl/cu124

RUN wget https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250715+cuda12.4.torch2.6.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl

RUN pip install ./k2-1.24.4.dev20250715+cuda12.4.torch2.6.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 

RUN pip install git+https://github.com/lhotse-speech/lhotse

RUN pip install kaldifst kaldilm kaldialign sentencepiece tensorboard

RUN pip install jiwer==3.0.3
# Copy model files into the image (or mount at runtime)
# Ensure you have tokens.txt + either (encoder/decoder/joiner) or paraformer.onnx

RUN wget https://huggingface.co/zzasdf/viet_iter3_pseudo_label/resolve/main/exp/pretrained.pt
COPY app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
