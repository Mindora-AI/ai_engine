#!/usr/bin/env python3
"""
MWIZERWA AI — Model Downloader Script
Downloads and verifies required model checkpoints from Hugging Face Hub:
1. LLM: AfriqueQwen3.5-4B-Instruct-v1 GGUF (approx. 2.86 GB)
2. TTS: KinyaFlex TTS MB-iSTFT-VITS2 Checkpoint (approx. 1.03 GB)
3. ASR: Cached automatically on first use via Hugging Face Transformers pipeline.

Supports:
- Automatic directory creation
- Skip existing valid files
- Resumable downloads via huggingface_hub
- Pre-download disk space checks
- SHA-256 cryptographic verification
- --check / --verify-only dry run mode
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Resolve Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.model_paths import get_models_root  # noqa: E402


def build_model_specs(models_root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Build download specs using centralized models root (local or Google Drive)."""
    root = Path(models_root) if models_root else get_models_root()
    models = {
        "llm": {
            "name": "AfriqueQwen3.5-4B-Instruct GGUF",
            "repo_id": "mradermacher/AfriqueQwen3.5-4B-Instruct-v1-i1-GGUF",
            "filename": "AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf",
            "dest_dir": root / "afriqueqwen",
            "expected_filename": "AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf",
            "expected_sha256": "86c212f91f44a66993d17be425eabdb85da63724d63e0c4430e0624aedf367f6",
            "approx_size_bytes": 3_066_385_984,
            "verify_sha256": True,
        },
        "tts": {
            "name": "KinyaFlex TTS Base Checkpoint",
            "repo_id": "C4IR-RW/kinya-flex-tts",
            "filename": "kinya_flex_tts_base_trained.pt",
            "dest_dir": root / "kinya-flex-tts",
            "expected_filename": "kinya_flex_tts_base_trained.pt",
            "expected_sha256": "e7fcd3d65046405a327a8d1ec73e7b66209eec903ba5219aaf6afc2a4ba8b041",
            "approx_size_bytes": 1_111_271_099,
            "verify_sha256": True,
        },
    }
    return models


MODELS = build_model_specs()


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file using buffered streaming."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_model_status(key: str, spec: Dict[str, Any], verify_hash: bool = True) -> Dict[str, Any]:
    """Inspects whether a model exists on disk and verifies its integrity."""
    dest_path = spec["dest_dir"] / spec["expected_filename"]
    exists = dest_path.is_file()
    size = dest_path.stat().st_size if exists else 0
    hash_matched = None
    actual_hash = None

    if exists:
        if verify_hash and spec.get("expected_sha256"):
            actual_hash = compute_sha256(dest_path)
            hash_matched = (actual_hash == spec["expected_sha256"])

    return {
        "key": key,
        "name": spec["name"],
        "path": dest_path,
        "exists": exists,
        "size_bytes": size,
        "size_mb": round(size / (1024 * 1024), 2),
        "hash_matched": hash_matched,
        "actual_hash": actual_hash,
        "expected_hash": spec.get("expected_sha256"),
    }


def download_model(key: str, spec: Dict[str, Any], force: bool = False) -> bool:
    """Downloads a single model file from Hugging Face Hub with validation."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[Error] 'huggingface_hub' is required. Install via: pip install huggingface_hub", file=sys.stderr)
        return False

    dest_dir = spec["dest_dir"]
    expected_filename = spec["expected_filename"]
    dest_path = dest_dir / expected_filename
    required_bytes = spec["approx_size_bytes"]

    # 1. Check if already present and valid
    if dest_path.is_file() and not force:
        status = check_model_status(key, spec, verify_hash=True)
        if status["hash_matched"] is True:
            print(f"[Already Present] {spec['name']}: {dest_path} (SHA-256 verified)")
            return True
        elif status["hash_matched"] is False:
            print(f"[Checksum Mismatch] {spec['name']} exists but checksum does not match expected hash.")
            print(f"  Expected: {spec['expected_sha256']}")
            print(f"  Actual:   {status['actual_hash']}")
            print("  Re-downloading...")
        else:
            print(f"[Already Present] {spec['name']}: {dest_path} ({status['size_mb']} MB)")
            return True

    # 2. Check disk space
    dest_dir.mkdir(parents=True, exist_ok=True)
    disk_free = shutil.disk_usage(dest_dir).free
    safety_margin = 500 * 1024 * 1024  # 500 MB buffer
    if disk_free < (required_bytes + safety_margin):
        print(
            f"[Error] Insufficient disk space on {dest_dir}. "
            f"Need {required_bytes / (1024**3):.2f} GB, but only {disk_free / (1024**3):.2f} GB free.",
            file=sys.stderr,
        )
        return False

    # 3. Perform download
    print(f"\n[Downloading] {spec['name']} from {spec['repo_id']}...")
    print(f"  Target File: {spec['filename']}")
    print(f"  Destination: {dest_path}")
    print(f"  Approx Size: {required_bytes / (1024**3):.2f} GB")

    try:
        downloaded_file = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=str(dest_dir),
        )

        # Handle filename difference if any
        downloaded_path = Path(downloaded_file)
        if downloaded_path.name != expected_filename and downloaded_path.is_file():
            if dest_path.exists():
                dest_path.unlink()
            downloaded_path.rename(dest_path)

        # 4. Verify checksum
        if spec.get("expected_sha256"):
            print("  Verifying downloaded file SHA-256...")
            actual_sha = compute_sha256(dest_path)
            if actual_sha != spec["expected_sha256"]:
                print(f"[Error] Checksum verification failed for {dest_path}!", file=sys.stderr)
                print(f"  Expected: {spec['expected_sha256']}", file=sys.stderr)
                print(f"  Actual:   {actual_sha}", file=sys.stderr)
                return False
            print("  [OK] SHA-256 hash verified successfully.")

        print(f"[Success] {spec['name']} downloaded and verified successfully.\n")
        return True

    except Exception as e:
        print(f"[Error] Failed to download {spec['name']}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="MWIZERWA AI — Model Checkpoint Downloader & Integrity Verifier"
    )
    parser.add_argument(
        "--check",
        "--verify-only",
        dest="verify_only",
        action="store_true",
        help="Check presence and cryptographic integrity of models without downloading.",
    )
    parser.add_argument(
        "--model",
        choices=["all", "llm", "tts"],
        default="all",
        help="Which checkpoint to download or check (default: all). EN/FR TTS uses Pocket TTS via pip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local file exists.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Override model storage root (or set MWIZERWA_MODELS_DIR).",
    )
    args = parser.parse_args()

    global MODELS
    if args.models_dir:
        os.environ["MWIZERWA_MODELS_DIR"] = str(Path(args.models_dir).expanduser())
    MODELS = build_model_specs()
    print(f"Models root: {get_models_root()}\n")

    target_keys = ["llm", "tts"] if args.model == "all" else [args.model]

    print("=" * 70)
    print("MWIZERWA AI — Model Checkpoint Status")
    print("=" * 70)

    all_ok = True
    for key in target_keys:
        spec = MODELS[key]
        status = check_model_status(key, spec, verify_hash=True)

        print(f"\nModel: {status['name']}")
        print(f"  Location:  {status['path']}")
        print(f"  Present:   {'YES' if status['exists'] else 'NO'}")
        if status["exists"]:
            print(f"  Size:      {status['size_mb']} MB")
            if status["hash_matched"] is True:
                print("  Checksum:  MATCH (SHA-256 verified)")
            elif status["hash_matched"] is False:
                print(f"  Checksum:  MISMATCH (Expected: {status['expected_hash']}, Got: {status['actual_hash']})")
                all_ok = False
            else:
                print(f"  SHA-256:   {status['actual_hash']} (No trusted hash configured)")
        else:
            print("  Status:    MISSING")
            all_ok = False

    print("\n" + "=" * 70)

    if args.verify_only:
        if all_ok:
            print("[Check Result] All requested model checkpoints are present and verified.")
            sys.exit(0)
        else:
            print("[Check Result] One or more checkpoints are missing or invalid.")
            print("  KinyaFlex / LLM: python scripts/download_models.py")
            sys.exit(1)

    # Perform downloads if not verify-only
    success = True
    for key in target_keys:
        ok = download_model(key, MODELS[key], force=args.force)
        if not ok:
            success = False

    if success:
        print("[All Done] Model checkpoints are ready for inference.")
        sys.exit(0)
    else:
        print("[Failure] One or more models could not be downloaded.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
