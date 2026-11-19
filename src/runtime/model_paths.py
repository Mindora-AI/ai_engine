"""
Central path resolution for MWIZERWA model and output storage.

Environment variables (optional):
  MWIZERWA_PROJECT_ROOT  — repository root (default: auto-detected)
  MWIZERWA_MODELS_DIR    — persistent model store (default: <project>/models)
  MWIZERWA_CACHE_DIR     — Hugging Face / auxiliary cache (default: <project>/cache or HF_HOME)
  MWIZERWA_OUTPUT_DIR    — generated audio/logs (default: <project>/outputs)
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    env = os.environ.get("MWIZERWA_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent.parent


def get_models_root() -> Path:
    env = os.environ.get("MWIZERWA_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return get_project_root() / "models"


def get_output_root() -> Path:
    env = os.environ.get("MWIZERWA_OUTPUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return get_project_root() / "outputs"


def get_cache_root() -> Path:
    env = os.environ.get("MWIZERWA_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    hf = os.environ.get("HF_HOME")
    if hf:
        return Path(hf).expanduser().resolve()
    return get_project_root() / "cache"


def resolve_llm_gguf_path() -> Path:
    return (
        get_models_root()
        / "afriqueqwen"
        / "AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf"
    )


def resolve_kinya_flex_checkpoint() -> Path:
    return get_models_root() / "kinya-flex-tts" / "kinya_flex_tts_base_trained.pt"


def resolve_piper_voices_root() -> Path:
    return get_models_root() / "piper"


def drive_layout_docs() -> dict:
    """Recommended Google Drive layout under MyDrive/MWIZERWA/."""
    return {
        "root": "MyDrive/MWIZERWA",
        "models": "MyDrive/MWIZERWA/models",
        "cache": "MyDrive/MWIZERWA/cache",
        "outputs": "MyDrive/MWIZERWA/outputs",
        "logs": "MyDrive/MWIZERWA/logs",
        "repo": "MyDrive/MWIZERWA/ai_engine",
    }