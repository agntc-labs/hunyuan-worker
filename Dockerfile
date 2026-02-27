# RunPod Serverless HunyuanVideo 1.5 i2v Worker
# Pure Diffusers pipeline — same pattern as agntc-labs/sdxl-worker
#
# Model: HunyuanVideo-1.5-Diffusers-720p_i2v (~54GB, FP16 transformer)
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

# ── Download models from HuggingFace (~54GB) ─────────────────────────
# Pre-baked into image so cold start doesn't download 54GB
RUN python3 -c "
from huggingface_hub import snapshot_download
import os

print('Downloading HunyuanVideo 1.5 720p i2v (Diffusers format)...')
snapshot_download(
    'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v',
    local_dir='/models/hunyuan-1.5-i2v',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.md', '*.txt', '.gitattributes'],
)
print('Model download complete.')

# Verify
for root, dirs, files in os.walk('/models/hunyuan-1.5-i2v'):
    for f in files:
        fpath = os.path.join(root, f)
        sz = os.path.getsize(fpath)
        if sz > 100_000_000:  # >100MB
            print(f'  {fpath}: {sz / 1e9:.1f} GB')
"

# ── Copy handler ─────────────────────────────────────────────────────
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
