# RunPod Serverless image -- Phase 1 proof (an already-set-up clip, TEST1).
# Built by .github/workflows/build-serverless.yml -> ghcr.io.
# Same torch/CUDA versions already proven on the training pod.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# opencv-python/easyocr need these at runtime even headless; openssh-server
# is for debug Pod deploys only (see docker-entrypoint.sh) -- the base image
# has no SSH access at all otherwise, unlike RunPod's own official templates.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 openssh-server \
    && mkdir -p /var/run/sshd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod

# Bake EasyOCR's models in at build time (phase2/ocr_reader.py: en, CPU) --
# without this it silently downloads them from the internet on every cold
# start, which is slow and a bad fit for a job meant to finish in minutes.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

# .dockerignore trims this down to code + TEST1's caches + its video, not
# every clip's multi-GB debug overlays.
COPY . .

# WHICH COMMIT IS THIS? RunPod reuses warm workers between jobs, and pointing
# the template at a new image does not evict one that is already running. A job
# then lands on yesterday's code and fails in ways that look like today's bug:
# one 44-minute run was spent chasing a ModuleNotFoundError that had already
# been fixed, on a worker that had never seen the fix. Every response now
# carries the sha it actually ran, so a stale worker is obvious in one line
# instead of an afternoon.
ARG GIT_SHA=unknown
RUN echo "$GIT_SHA" > /app/IMAGE_SHA

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "-u", "serverless_handler.py"]
