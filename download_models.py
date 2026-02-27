#!/usr/bin/env python3
"""Download HunyuanVideo 1.5 i2v models to a RunPod network volume.

Run this on a temporary RunPod Pod with the network volume mounted:
    python3 download_models.py

Downloads to /runpod-volume/hunyuan-1.5-i2v/ (~54GB).
"""

import os
import sys
import time

def main():
    target_dir = "/runpod-volume/hunyuan-1.5-i2v"

    # Check if already downloaded
    if os.path.exists(os.path.join(target_dir, "model_index.json")):
        print(f"Models already exist at {target_dir}")
        # Show size
        total = 0
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        print(f"Total size: {total / 1e9:.1f} GB")
        return

    print(f"Downloading HunyuanVideo 1.5 720p i2v to {target_dir}...")
    t0 = time.time()

    from huggingface_hub import snapshot_download

    snapshot_download(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )

    elapsed = time.time() - t0
    print(f"Download complete in {elapsed:.0f}s")

    # Verify
    total = 0
    big_files = []
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            fpath = os.path.join(root, f)
            sz = os.path.getsize(fpath)
            total += sz
            if sz > 100_000_000:
                big_files.append((fpath, sz))

    print(f"Total: {total / 1e9:.1f} GB across {len(big_files)} large files")
    for fpath, sz in sorted(big_files, key=lambda x: -x[1])[:10]:
        print(f"  {fpath}: {sz / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
