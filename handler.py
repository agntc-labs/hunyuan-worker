"""RunPod Serverless HunyuanVideo 1.5 i2v Worker — image-to-video generation on CUDA.

Pure Diffusers pipeline (same pattern as agntc-labs/sdxl-worker):
  - HunyuanVideo-1.5-Diffusers-720p_i2v (8.3B params, BF16)
  - SigLIP vision encoder for image conditioning
  - Qwen2.5-VL-7B + ByT5 text encoders
  - FlowMatch Euler scheduler with ClassifierFreeGuidance guider
  - CPU offload + VAE tiling for VRAM efficiency

Input schema:
  {
    "input": {
      "prompt": str,                    # Motion/scene description
      "image_b64": str,                 # Base64-encoded source image (PNG/JPEG)
      "num_frames": int (default 121),  # 61=~2.5s, 121=~5s, 193=~8s at 24fps
      "num_inference_steps": int (default 30),  # 30-50 recommended
      "guidance_scale": float (default 6.0),   # Applied via guider, not pipe kwarg
      "seed": int | null,              # Random if null
      "width": int (default 1280),     # Output width
      "height": int (default 720),     # Output height
      "fps": int (default 24),         # Output video FPS
    }
  }

Output:
  {
    "video_b64": str,       # Base64-encoded MP4 video
    "num_frames": int,
    "width": int, "height": int,
    "fps": int,
    "seed": int,
    "gen_time": float,
    "model": "hunyuan-1.5-i2v-720p"
  }
"""

import os
import io
import sys
import time
import base64
import random
import logging
import threading

import runpod

log = logging.getLogger("hunyuan-worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

# Model paths — check network volume first, then local
_VOLUME_MODEL_PATH = "/runpod-volume/hunyuan-1.5-i2v"
_LOCAL_MODEL_PATH = "/models/hunyuan-1.5-i2v"


def _resolve_model_path():
    """Find model weights: network volume > local > download from HuggingFace."""
    # Check network volume (preferred — pre-loaded, instant)
    if os.path.exists(os.path.join(_VOLUME_MODEL_PATH, "model_index.json")):
        log.info("Models found on network volume: %s", _VOLUME_MODEL_PATH)
        return _VOLUME_MODEL_PATH

    # Check local path (from baked Docker image)
    if os.path.exists(os.path.join(_LOCAL_MODEL_PATH, "model_index.json")):
        log.info("Models found locally: %s", _LOCAL_MODEL_PATH)
        return _LOCAL_MODEL_PATH

    # Last resort: download from HuggingFace (slow — ~54GB)
    log.warning("No pre-loaded models found. Downloading from HuggingFace (~54GB)...")
    try:
        from huggingface_hub import snapshot_download
        target = _VOLUME_MODEL_PATH if os.path.exists("/runpod-volume") else _LOCAL_MODEL_PATH
        t0 = time.time()
        snapshot_download(
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
            local_dir=target,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        log.info("Model download complete in %.0fs -> %s", time.time() - t0, target)
        return target
    except Exception as e:
        log.error("Failed to download models: %s", e)
        return _LOCAL_MODEL_PATH


MODEL_PATH = None  # Set at startup

# ── Global pipeline state ────────────────────────────────────────────
_pipe = None
_pipe_lock = threading.Lock()
_gen_count = 0
_torch = None
_Image = None


def _load_pipeline():
    """Load the HunyuanVideo 1.5 i2v pipeline. Called once at container startup."""
    global _pipe, _torch, _Image

    import torch
    from diffusers import HunyuanVideo15ImageToVideoPipeline
    from PIL import Image

    _torch = torch
    _Image = Image

    t0 = time.time()
    log.info("Loading HunyuanVideo 1.5 i2v pipeline...")

    # Detect available VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        log.info("GPU: %s, VRAM: %.1f GB", torch.cuda.get_device_name(0), vram_gb)
    else:
        vram_gb = 0
        log.error("No CUDA GPU detected!")

    # Load model in bfloat16 (more numerically stable than float16 for this model)
    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )

    # Always use CPU offload — model is too large for even 48GB GPUs during generation
    pipe.enable_model_cpu_offload()
    log.info("CPU offload enabled (%.1f GB VRAM available)", vram_gb)

    # Enable VAE tiling to reduce peak VRAM during decode
    pipe.vae.enable_tiling()
    log.info("VAE tiling enabled")

    # Try to set optimal attention backend (requires `kernels` package)
    for backend in ("flash_hub", "sage_hub"):
        try:
            pipe.transformer.set_attention_backend(backend)
            log.info("Attention backend: %s", backend)
            break
        except Exception as e:
            log.info("Attention backend %s not available: %s", backend, e)

    t_load = time.time() - t0
    _pipe = pipe
    log.info("Pipeline ready in %.1fs", t_load)


def _generate(params):
    """Generate a video from an image. Returns dict with video_b64 and metadata."""
    global _gen_count

    prompt = params.get("prompt", "gentle natural movement, subtle animation")
    image_b64 = params.get("image_b64", "")
    num_frames = params.get("num_frames", 121)
    steps = params.get("num_inference_steps", 30)
    guidance = params.get("guidance_scale", 6.0)
    seed = params.get("seed")
    width = params.get("width", 1280)
    height = params.get("height", 720)
    fps = params.get("fps", 24)

    if not image_b64:
        return {"error": "image_b64 is required"}

    # Decode input image
    try:
        img_bytes = base64.b64decode(image_b64)
        image = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
        log.info("Input image: %dx%d (%d bytes)", image.width, image.height, len(img_bytes))
    except Exception as e:
        return {"error": "Failed to decode image: %s" % e}

    # Seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = _torch.Generator(device="cpu").manual_seed(seed)

    with _pipe_lock:
        _gen_count += 1
        gen_id = _gen_count

        log.info(
            "[gen #%d] %dx%d, %d frames, %d steps, cfg=%.1f, seed=%d — %s",
            gen_id, width, height, num_frames, steps, guidance, seed, prompt[:100],
        )

        # Set guidance scale via guider (NOT a pipe kwarg for HunyuanVideo 1.5)
        if hasattr(_pipe, 'guider') and _pipe.guider is not None:
            try:
                _pipe.guider = _pipe.guider.new(guidance_scale=guidance)
            except Exception:
                log.info("Could not update guider guidance_scale, using default")

        t0 = time.time()

        try:
            result = _pipe(
                prompt=prompt,
                image=image,
                generator=generator,
                num_frames=num_frames,
                num_inference_steps=steps,
                width=width,
                height=height,
            )
            frames = result.frames[0]
        except Exception as e:
            t_gen = time.time() - t0
            log.error("[gen #%d] Failed in %.1fs: %s", gen_id, t_gen, e)
            return {"error": "Generation failed: %s" % e}

        t_gen = time.time() - t0

    # Encode video to MP4
    try:
        import imageio
        import numpy as np
        buf = io.BytesIO()
        writer = imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264",
                                     quality=8, pixelformat="yuv420p")
        for frame in frames:
            if hasattr(frame, 'numpy'):
                frame_np = frame.numpy()
            elif isinstance(frame, _Image.Image):
                frame_np = np.array(frame)
            else:
                frame_np = np.array(frame)
            writer.append_data(frame_np)
        writer.close()
        video_bytes = buf.getvalue()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        log.info("[gen #%d] Video encoded: %d bytes (%.1f MB)", gen_id, len(video_bytes),
                 len(video_bytes) / 1e6)
    except Exception as e:
        log.error("[gen #%d] Video encoding failed: %s", gen_id, e)
        return {"error": "Video encoding failed: %s" % e}

    log.info("[gen #%d] Done in %.1fs (%d frames)", gen_id, t_gen, num_frames)

    return {
        "video_b64": video_b64,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "seed": seed,
        "gen_time": round(t_gen, 1),
        "model": "hunyuan-1.5-i2v-720p",
    }


def handler(job):
    """RunPod serverless handler."""
    job_input = job.get("input", {})

    if _pipe is None:
        return {"error": "Pipeline still loading, please retry"}

    try:
        return _generate(job_input)
    except Exception as e:
        log.error("Handler error: %s", e, exc_info=True)
        return {"error": str(e)}


# ── Startup ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Initializing HunyuanVideo 1.5 i2v worker (CUDA)...")
    try:
        MODEL_PATH = _resolve_model_path()
        log.info("Using model path: %s", MODEL_PATH)
        _load_pipeline()
        log.info("Starting RunPod serverless worker...")
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        log.error("FATAL: Startup failed: %s", e, exc_info=True)
        sys.exit(1)
