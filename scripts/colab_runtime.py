#!/usr/bin/env python3
"""
MWIZERWA — Colab runtime helpers (hardware, Drive layout, llama-cpp install).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.model_paths import (  # noqa: E402
    drive_layout_docs,
    get_cache_root,
    get_models_root,
    get_output_root,
    resolve_kinya_flex_checkpoint,
    resolve_llm_gguf_path,
    resolve_piper_voices_root,
)

DEFAULT_COLAB_PROJECT = Path("/content/ai_engine")
DEFAULT_REPO_URL = "https://github.com/Mindora-AI/ai_engine.git"
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/MWIZERWA")
LLAMA_CPP_INDEX = "https://abetlen.github.io/llama-cpp-python/whl"
PINNED_DEEPKIN_COMMIT = "c28db6bfb6b80666583717c87e8edaa2a8f384d5"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def init_project_root(project_dir: Path | str) -> Path:
    """Set MWIZERWA_PROJECT_ROOT and sys.path."""
    project = Path(project_dir).resolve()
    os.environ["MWIZERWA_PROJECT_ROOT"] = str(project)
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    return project


def clone_mwizerwa_if_needed(
    project_dir: Path | str = DEFAULT_COLAB_PROJECT,
    repo_url: Optional[str] = None,
) -> Path:
    """
    Clone Mindora-AI/ai_engine into project_dir when not already a checkout.
    Does not delete or overwrite a non-empty directory that is not a valid repo.
    """
    project = Path(project_dir)
    marker = project / "run_assistant.py"
    url = repo_url or os.environ.get("MWIZERWA_REPO_URL", DEFAULT_REPO_URL)

    if marker.is_file():
        print(f"[Git] Using existing MWIZERWA checkout: {project}")
        return init_project_root(project)

    if project.exists() and any(project.iterdir()):
        raise RuntimeError(
            f"{project} exists but is not a MWIZERWA repository (missing run_assistant.py). "
            "Rename or remove it manually, or set a different path."
        )

    project.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Git] Cloning {url} -> {project}")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(project)],
        check=True,
    )
    if not marker.is_file():
        raise RuntimeError(f"Clone completed but {marker} is missing — verify repository URL/access.")
    return init_project_root(project)


def require_colab_scripts(project_root: Path | str) -> None:
    """Ensure Colab helper scripts exist (must be in the ai_engine repository)."""
    root = Path(project_root)
    expected = (
        root / "scripts" / "colab_runtime.py",
        root / "scripts" / "colab_validate.py",
        root / "scripts" / "colab_gradio.py",
        root / "requirements-colab.txt",
    )
    missing = [str(p.relative_to(root)) for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Colab deployment files missing from checkout:\n  "
            + "\n  ".join(missing)
            + "\nPull the latest Mindora-AI/ai_engine (main) and re-run cell 1."
        )


def inspect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "colab": is_colab(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / (1024**3), 2)
        info["disk_free_gb"] = round(shutil.disk_usage("/").free / (1024**3), 2)
    except Exception as e:
        info["psutil_error"] = str(e)

    try:
        import torch

        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_mem_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception as e:
        info["torch_error"] = str(e)

    return info


def detect_cuda_wheel_tag(hardware: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Derive abetlen wheel tag (e.g. cu124) from torch.version.cuda when available.
    Returns None when CUDA is not detectable.
    """
    hw = hardware or inspect_hardware()
    if not hw.get("cuda_available"):
        return None
    cuda_ver = hw.get("torch_cuda_version")
    if not cuda_ver:
        return None
    parts = cuda_ver.split(".")
    if len(parts) < 2:
        return None
    major, minor = parts[0], parts[1]
    return f"cu{major}{minor}"


def llama_supports_gpu_offload() -> bool:
    try:
        from llama_cpp.llama_cpp import llama_supports_gpu_offload

        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


def install_llama_cpp_for_colab() -> Dict[str, Any]:
    """
    Install llama-cpp-python with a CUDA wheel when possible; otherwise CPU with explicit notice.
    Replaces any CPU build installed via requirements.txt.
    """
    hw = inspect_hardware()
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    tags: List[str] = []
    detected = detect_cuda_wheel_tag(hw)
    if detected:
        tags.append(detected)
    for fallback in ("cu124", "cu123", "cu122", "cu121"):
        if fallback not in tags:
            tags.append(fallback)

    if not hw.get("cuda_available"):
        print("[llama-cpp] No CUDA device — installing CPU build.")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python>=0.2.70"],
        )
        return {
            "backend": "cpu",
            "wheel_tag": None,
            "gpu_offload_supported": False,
        }

    for tag in tags:
        index = f"{LLAMA_CPP_INDEX}/{tag}"
        print(f"[llama-cpp] Trying CUDA wheel index: {index}")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "llama-cpp-python>=0.2.70",
                "--extra-index-url",
                index,
                "--force-reinstall",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue
        # Restore dependencies llama-cpp needs
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "diskcache", "jinja2", "numpy"],
            check=False,
        )
        if llama_supports_gpu_offload():
            print(f"[llama-cpp] CUDA offload available (wheel tag {tag}).")
            return {
                "backend": "cuda",
                "wheel_tag": tag,
                "gpu_offload_supported": True,
            }
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print(
        "[llama-cpp] WARNING: No compatible CUDA wheel found. "
        "Installing CPU build; set MWIZERWA_LLM_N_GPU_LAYERS=0."
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python>=0.2.70"],
    )
    return {
        "backend": "cpu",
        "wheel_tag": None,
        "gpu_offload_supported": False,
    }


def suggest_llm_gpu_layers(
    hardware: Optional[Dict[str, Any]] = None,
    gpu_offload_supported: Optional[bool] = None,
) -> int:
    """Conservative AfriqueQwen Q4_K_M layer offload; 0 unless CUDA offload is verified."""
    hw = hardware or inspect_hardware()
    if gpu_offload_supported is None:
        gpu_offload_supported = llama_supports_gpu_offload()
    if not hw.get("cuda_available") or not gpu_offload_supported:
        return 0
    mem = float(hw.get("gpu_total_mem_gb") or 0)
    if mem >= 20:
        return 32
    if mem >= 14:
        return 24
    if mem >= 10:
        return 16
    return 8


def apply_llm_gpu_layers(
    hardware: Optional[Dict[str, Any]] = None,
    llama_install: Optional[Dict[str, Any]] = None,
) -> int:
    layers = suggest_llm_gpu_layers(
        hardware,
        (llama_install or {}).get("gpu_offload_supported"),
    )
    os.environ["MWIZERWA_LLM_N_GPU_LAYERS"] = str(layers)
    print(f"[Config] MWIZERWA_LLM_N_GPU_LAYERS={layers}")
    if layers == 0 and hardware and hardware.get("cuda_available"):
        print("[Config] LLM will run on CPU (no verified llama-cpp GPU offload).")
    return layers


def mount_google_drive(mount_point: str = "/content/drive") -> Path:
    if not is_colab():
        raise RuntimeError("Google Drive mount is only available in Colab.")
    from google.colab import drive

    drive.mount(mount_point, force_remount=False)
    root = Path(mount_point) / "MyDrive" / "MWIZERWA"
    for folder in ("models", "cache", "outputs", "logs"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    print(f"[Drive] MWIZERWA root: {root}")
    return root


def configure_drive_persistence(
    drive_root: Path | str = DEFAULT_DRIVE_ROOT,
    project_root: Optional[Path | str] = None,
) -> Dict[str, str]:
    """Point model/output/cache env vars at Drive."""
    drive_root = Path(drive_root)
    drive_parent = drive_root.parent
    if not drive_parent.is_dir():
        raise RuntimeError(
            f"Google Drive is not mounted at {drive_parent}. "
            "Run drive.mount('/content/drive') and complete authentication first."
        )

    models = drive_root / "models"
    outputs = drive_root / "outputs"
    cache = drive_root / "cache"
    logs = drive_root / "logs"
    for p in (models, outputs, cache, logs):
        p.mkdir(parents=True, exist_ok=True)

    os.environ["MWIZERWA_MODELS_DIR"] = str(models)
    os.environ["MWIZERWA_OUTPUT_DIR"] = str(outputs)
    os.environ["MWIZERWA_CACHE_DIR"] = str(cache)
    hf_home = cache / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home)

    if project_root:
        init_project_root(project_root)

    paths = {
        "MWIZERWA_PROJECT_ROOT": os.environ.get("MWIZERWA_PROJECT_ROOT", ""),
        "MWIZERWA_MODELS_DIR": str(models),
        "MWIZERWA_CACHE_DIR": str(cache),
        "MWIZERWA_OUTPUT_DIR": str(outputs),
        "HF_HOME": str(hf_home),
    }
    print("[Drive] Expected model tree (files may still be MISSING until download):")
    for sub in (
        "models/afriqueqwen",
        "models/kinya-flex-tts",
        "models/piper/en_US-ljspeech-medium",
        "models/piper/fr_FR-siwis-medium",
    ):
        full = drive_root / sub
        print(f"  {sub}  exists={full.exists()}")
    for key, val in paths.items():
        if not val:
            continue
        print(f"[Config] {key}={val} (exists={Path(val).exists()})")
    return paths


def report_model_asset_status() -> Dict[str, bool]:
    """Print expected loader paths; return presence map (dirs may exist without files)."""
    checks = {
        "llm_gguf": resolve_llm_gguf_path(),
        "kinya_flex": resolve_kinya_flex_checkpoint(),
        "piper_en": resolve_piper_voices_root()
        / "en_US-ljspeech-medium"
        / "en_US-ljspeech-medium.onnx",
        "piper_fr": resolve_piper_voices_root()
        / "fr_FR-siwis-medium"
        / "fr_FR-siwis-medium.onnx",
    }
    status: Dict[str, bool] = {}
    print("[Models] Asset status (file must exist before inference):")
    for name, path in checks.items():
        ok = path.is_file()
        status[name] = ok
        print(f"  {'OK' if ok else 'MISSING':7} {name}: {path}")
    print(f"[Models] models root: {get_models_root()}")
    print(f"[Models] cache root: {get_cache_root()}")
    print(f"[Models] outputs root: {get_output_root()}")
    return status


def install_python_dependencies(project_root: Path) -> None:
    """
    Install Colab requirements without replacing Colab's CUDA-enabled PyTorch.
    llama-cpp-python is installed separately via install_llama_cpp_for_colab().
    """
    project_root = Path(project_root)
    req = project_root / "requirements-colab.txt"
    if not req.is_file():
        raise FileNotFoundError(f"Missing {req}")
    try:
        import torch

        print(f"[Deps] Keeping existing torch {torch.__version__} (cuda={torch.cuda.is_available()})")
    except ImportError:
        print("[Deps] torch not found — Colab GPU runtimes should include it; continuing.")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "pip install failed:\n"
            + (result.stderr or result.stdout or "no output")
        )
    print("[Deps] requirements-colab.txt installed (torch/llama-cpp excluded from this step).")


def install_deepkin(project_root: Path, pinned_commit: str = PINNED_DEEPKIN_COMMIT) -> None:
    """Clone/install DeepKIN-AgAI required by KinyaFlex."""
    project_root = Path(project_root)
    dest = project_root / "ac-ai-models"
    if not (dest / "DeepKIN-AgAI").exists():
        if dest.exists() and any(dest.iterdir()):
            raise RuntimeError(
                f"{dest} exists but DeepKIN-AgAI is missing — resolve manually to avoid data loss."
            )
        subprocess.check_call(
            ["git", "clone", "https://github.com/c4ir-rw/ac-ai-models.git", str(dest)],
            cwd=project_root,
        )
        subprocess.check_call(["git", "checkout", pinned_commit], cwd=dest)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(dest / "DeepKIN-AgAI")],
        cwd=project_root,
    )


def verify_critical_imports() -> None:
    errors: List[str] = []
    for label, stmt in (
        ("torch", "import torch"),
        ("transformers", "import transformers"),
        ("llama_cpp", "from llama_cpp import Llama"),
        ("gradio", "import gradio"),
        ("deepkin", "import deepkin"),
    ):
        try:
            exec(stmt, {})
        except Exception as e:
            errors.append(f"{label}: {e}")
    if errors:
        raise RuntimeError("Import verification failed:\n  " + "\n  ".join(errors))
    print("[Imports] Critical packages OK.")


def main():
    hw = inspect_hardware()
    print(json.dumps(hw, indent=2))
    print(f"Active models root: {get_models_root()}")


if __name__ == "__main__":
    main()