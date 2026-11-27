"""Inexpensive /health backend checks without loading models."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.asr.kinya_health_asr import KinyaHealthASREngine, MODEL_ID as KINYA_ASR_MODEL_ID
from src.asr.whisper_asr import DEFAULT_WHISPER_MODEL, WhisperASREngine
from src.runtime.model_paths import resolve_llm_gguf_path, resolve_kinya_flex_checkpoint
from src.tts.tts_engine import DEFAULT_MODEL_PATH
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.service import MWIZERWAApiService


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _hf_whisper_small_cached() -> bool:
    hf = os.environ.get("HF_HOME")
    roots = []
    if hf:
        roots.append(Path(hf).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    marker = "models--Systran--faster-whisper-small"
    for root in roots:
        if (root / marker).is_dir():
            return True
    return False


def _tts_artifact_status(service: "MWIZERWAApiService") -> Dict[str, Any]:
    kinya_ckpt = resolve_kinya_flex_checkpoint().is_file()
    pocket_pkg = _module_available("pocket_tts")
    loaded = service._tts is not None
    by_lang: Dict[str, Any] = {
        "rw": {"artifact_present": kinya_ckpt or DEFAULT_MODEL_PATH.is_file()},
        "en": {"artifact_present": pocket_pkg},
        "fr": {"artifact_present": pocket_pkg},
    }
    return {
        "dependency_available": pocket_pkg,
        "artifact_present": kinya_ckpt or pocket_pkg,
        "loaded": loaded,
        "languages": by_lang,
    }


def build_backend_readiness(service: "MWIZERWAApiService") -> Dict[str, Any]:
    llm_gguf = resolve_llm_gguf_path()
    llm_loaded = service._llm is not None and service.engine_initialized

    transformers_ok = _module_available("transformers")
    kinya_loaded = KinyaHealthASREngine._shared_pipeline is not None

    whisper_dep = _module_available("faster_whisper")
    whisper_loaded = WhisperASREngine._shared_model is not None

    return {
        "llm": {
            "dependency_available": _module_available("llama_cpp"),
            "artifact_present": llm_gguf.is_file(),
            "loaded": llm_loaded,
            "detail": None if llm_gguf.is_file() else f"GGUF missing at {llm_gguf}",
        },
        "asr_kinyarwanda": {
            "dependency_available": transformers_ok,
            "artifact_present": None,
            "loaded": kinya_loaded,
            "model_id": KINYA_ASR_MODEL_ID,
            "detail": None if transformers_ok else "transformers not installed",
        },
        "asr_whisper": {
            "dependency_available": whisper_dep,
            "artifact_present": _hf_whisper_small_cached() if whisper_dep else False,
            "loaded": whisper_loaded,
            "configured_model": DEFAULT_WHISPER_MODEL,
            "detail": None if whisper_dep else "faster-whisper not installed (optional EN/FR)",
        },
        "tts": _tts_artifact_status(service),
    }