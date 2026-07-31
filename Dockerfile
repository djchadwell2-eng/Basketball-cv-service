# RunPod Serverless image -- Phase 1 proof (an already-set-up clip, TEST1).
# Built by .github/workflows/build-serverless.yml -> ghcr.io.
# Same torch/CUDA versions already proven on the training pod.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# opencv-python/easyocr need these at runtime even headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod

# .dockerignore trims this down to code + TEST1's caches + its video, not
# every clip's multi-GB debug overlays.
COPY . .

CMD ["python", "-u", "serverless_handler.py"]
