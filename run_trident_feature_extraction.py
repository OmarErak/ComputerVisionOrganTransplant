#!/usr/bin/env python3
"""
Run TRIDENT feature extraction on a batch of WSIs.

Assumptions:
  - All Python dependencies are already installed via requirements.txt.
  - TRIDENT repository is already cloned (/TRIDENT).
  - TRIDENT is installed with: pip install -e /path/to/TRIDENT


"""

import os
import argparse
import subprocess
import sys
from typing import Optional

try:
    from huggingface_hub import login as hf_login
except ImportError:
    hf_login = None


def run_cmd(cmd: str) -> None:
    """Run a shell command with error checking and live output."""
    print(f"\n[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def maybe_hf_login(hf_token: Optional[str]) -> None:
    """Log in to HuggingFace if a token is provided."""
    if hf_token is None:
        print("[INFO] Skipping Hugging Face login (no token provided).")
        return

    if hf_login is None:
        print("[WARN] huggingface_hub not installed; cannot log in.")
        return

    print("[INFO] Logging in to Hugging Face Hub...")
    hf_login(token=hf_token)
    print("[INFO] Hugging Face login done.")


def run_trident(
    trident_dir: str,
    wsi_dir: str,
    job_dir: str,
    patch_encoder: str = "conch_v15",
    patch_size: int = 512,
    gpu: int = 0,
    task: str = "all",
) -> None:
    """Run TRIDENT's batch feature extraction."""
    if not os.path.isdir(trident_dir):
        print(f"[ERROR] TRIDENT directory not found: {trident_dir}")
        print("        Please clone the repo and/or update --trident_dir.")
        sys.exit(1)

    os.makedirs(job_dir, exist_ok=True)

    cmd = (
        f"cd {trident_dir} && "
        f"python run_batch_of_slides.py "
        f"--task {task} "
        f"--wsi_dir \"{wsi_dir}\" "
        f"--job_dir \"{job_dir}\" "
        f"--patch_encoder {patch_encoder} "
        f"--patch_size {patch_size} "
        f"--gpu {gpu}"
    )

    print("\n[INFO] Starting TRIDENT feature extraction...")
    print(f"[INFO] TRIDENT : {trident_dir}")
    print(f"[INFO] WSI_DIR : {wsi_dir}")
    print(f"[INFO] JOB_DIR : {job_dir}")
    print(f"[INFO] Encoder : {patch_encoder}")
    print(f"[INFO] Patch   : {patch_size}")
    print(f"[INFO] GPU     : {gpu}")
    run_cmd(cmd)
    print("\n[INFO] TRIDENT feature extraction finished successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TRIDENT feature extraction with CONCH v1.5.")

    # Paths
    parser.add_argument(
        "--trident_dir",
        type=str,
        default="/content/TRIDENT",
        help="Path where the TRIDENT repo is located.",
    )
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=True,
        help="Directory containing .svs WSIs.",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Output directory for TRIDENT feature files.",
    )

    # Optional HF login
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face access token (optional). If not provided, login is skipped.",
    )

    # TRIDENT options
    parser.add_argument(
        "--patch_encoder",
        type=str,
        default="conch_v15",
        help="Patch encoder to use (e.g., conch_v15, titan, etc.).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Patch size in pixels.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="TRIDENT task argument (default: all).",
    )

    # Misc
    parser.add_argument(
        "--show_gpu",
        action="store_true",
        help="If set, run nvidia-smi before starting.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Optional HF login
    maybe_hf_login(args.hf_token)

    # Optional GPU info
    if args.show_gpu:
        print("[INFO] GPU info (nvidia-smi):")
        run_cmd("nvidia-smi")

    # Run TRIDENT
    run_trident(
        trident_dir=args.trident_dir,
        wsi_dir=args.wsi_dir,
        job_dir=args.job_dir,
        patch_encoder=args.patch_encoder,
        patch_size=args.patch_size,
        gpu=args.gpu,
        task=args.task,
    )


if __name__ == "__main__":
    main()
