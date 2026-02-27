# RunPod Serverless HunyuanVideo 1.5 i2v Worker
# Pure Diffusers pipeline — same pattern as agntc-labs/sdxl-worker
#
# Model: HunyuanVideo-1.5-Diffusers-720p_i2v (~54GB, FP16 transformer)
# Models loaded from RunPod network volume at /runpod-volume/hunyuan-1.5-i2v
# GPU target: AMPERE_48 (A6000, 48GB VRAM) — FP16 fits without offloading
# Fallback: ADA_24 (RTX 4090, 24GB) — with CPU offload + VAE tiling

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy handler + download script ──────────────────────────────────
COPY handler.py .
COPY download_models.py .

CMD ["python3", "-u", "handler.py"]
