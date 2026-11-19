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
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional, Tuple

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
# Gradio 4.x oauth.py imports huggingface_hub.HfFolder (removed in hub 1.0+).
PINNED_GRADIO_VERSION = "5.50.0"
MIN_GRADIO_VERSION = (5, 7, 1)


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
        root / "constraints-colab.txt",
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


def _installed_version(dist_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _pip(
    args: List[str],
    *,
    step: str,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run pip with the active notebook interpreter; show output on failure."""
    cmd = [sys.executable, "-m", "pip", *args]
    print(f"[pip] {step}: {' '.join(args)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        raise RuntimeError(
            f"pip failed ({step}, exit {result.returncode}):\n"
            + (result.stderr or result.stdout or "no output")
        )
    return result


def _parse_version_tuple(version: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for piece in version.split(".")[:3]:
        piece = "".join(c for c in piece if c.isdigit())
        if piece:
            parts.append(int(piece))
    return tuple(parts) if parts else (0,)


def huggingface_hub_major_version() -> Optional[int]:
    v = _installed_version("huggingface-hub") or _installed_version("huggingface_hub")
    if not v:
        return None
    return _parse_version_tuple(v)[0]


def gradio_dist_version() -> Optional[str]:
    return _installed_version("gradio")


def _hf_folder_available() -> bool:
    try:
        import huggingface_hub as hub

        return hasattr(hub, "HfFolder")
    except Exception:
        return False


def _import_gradio_in_subprocess() -> Tuple[bool, str]:
    """Fresh import test (avoids stale sys.modules after pip)."""
    script = (
        "import gradio; import huggingface_hub as h; "
        "print(gradio.__version__, h.__version__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, (result.stdout or "").strip()
    return False, (result.stderr or result.stdout or "import failed").strip()


def ensure_gradio_hub_compatibility(force: bool = False) -> Dict[str, str]:
    """
    Gradio 4.x (oauth.py) does: from huggingface_hub import HfFolder — fails on hub 1.x.
    Colab often ships gradio 4.x, then transformers upgrades hub to 1.33+.
    """
    gver = gradio_dist_version()
    hub_major = huggingface_hub_major_version() or 0
    needs_fix = force
    if gver:
        needs_fix = needs_fix or _parse_version_tuple(gver) < MIN_GRADIO_VERSION
    else:
        needs_fix = True
    if hub_major >= 1 and not needs_fix and gver:
        needs_fix = _parse_version_tuple(gver) < MIN_GRADIO_VERSION

    if not needs_fix:
        ok, detail = _import_gradio_in_subprocess()
        if ok:
            print(f"[Deps] Gradio OK (subprocess): {detail}")
            return {"gradio": gver or "unknown", "action": "unchanged"}

    print(
        "[Deps] Upgrading Gradio (hub 1.x removed HfFolder; Gradio 4.x cannot import). "
        f"Current gradio={gver}, huggingface_hub major={hub_major}."
    )
    _pip(["uninstall", "-y", "gradio"], step="uninstall old gradio")
    _pip(
        ["install", f"gradio=={PINNED_GRADIO_VERSION}"],
        step="install pinned gradio",
    )

    gver = gradio_dist_version()
    ok, detail = _import_gradio_in_subprocess()
    if not ok:
        raise RuntimeError(
            "Gradio still cannot import after upgrade to "
            f"{PINNED_GRADIO_VERSION}. "
            f"Subprocess error: {detail}"
        )
    print(f"[Deps] Verified gradio import: {detail}")
    return {"gradio": gver or PINNED_GRADIO_VERSION, "action": "upgraded"}


def install_python_dependencies(project_root: Path) -> None:
    """
    Deterministic Colab stack:
    1) requirements-colab-packages.txt (transformers/hub — no torch)
    2) Gradio 5.x last (never leave gradio 4 + hub 1.x)
    """
    project_root = Path(project_root)
    req = project_root / "requirements-colab.txt"
    constraints = project_root / "constraints-colab.txt"
    if not req.is_file():
        raise FileNotFoundError(f"Missing {req}")
    try:
        import torch

        print(f"[Deps] Keeping existing torch {torch.__version__} (cuda={torch.cuda.is_available()})")
    except ImportError:
        print("[Deps] torch not found — Colab GPU runtimes should include it; continuing.")

    pip_args = ["install", "-r", str(req)]
    if constraints.is_file():
        pip_args.extend(["-c", str(constraints)])

    _pip(pip_args, step="requirements-colab", cwd=project_root)
    print("[Deps] Base packages installed (Gradio installed next).")
    ensure_gradio_hub_compatibility(force=True)


def deepkin_importable() -> bool:
    try:
        import deepkin  # noqa: F401

        return True
    except ImportError:
        return False


def diagnose_colab_dependencies() -> Dict[str, Any]:
    """Print versions relevant to Gradio/DeepKIN failures (for Colab debug cells)."""
    info: Dict[str, Any] = {
        "python": sys.executable,
        "huggingface_hub": _installed_version("huggingface-hub"),
        "hf_folder_available": _hf_folder_available(),
        "gradio_dist": gradio_dist_version(),
        "deepkin_importable": deepkin_importable(),
    }
    ok, detail = _import_gradio_in_subprocess()
    info["gradio_subprocess_import"] = ok
    info["gradio_subprocess_detail"] = detail
    print(json.dumps(info, indent=2))
    return info


def install_deepkin(
    project_root: Path,
    pinned_commit: str = PINNED_DEEPKIN_COMMIT,
) -> Dict[str, Any]:
    """
    Clone/install DeepKIN-AgAI (import name: deepkin) for KinyaFlex TTS.
    Required for Kinyarwanda synthesis — not optional on Colab.
    """
    project_root = Path(project_root)
    dest = project_root / "ac-ai-models"
    pkg_dir = dest / "DeepKIN-AgAI"
    repo_url = "https://github.com/c4ir-rw/ac-ai-models.git"

    if deepkin_importable():
        print("[DeepKIN] deepkin already importable — skipping reinstall.")
        return {"status": "already_installed", "package_dir": str(pkg_dir)}

    if dest.exists() and not pkg_dir.is_dir():
        if any(dest.iterdir()):
            raise RuntimeError(
                f"{dest} exists but {pkg_dir.name} is missing. "
                "Remove or fix ac-ai-models manually, then re-run this cell."
            )

    if not dest.is_dir():
        print(f"[DeepKIN] Cloning {repo_url}")
        clone = subprocess.run(
            ["git", "clone", repo_url, str(dest)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            raise RuntimeError(
                "git clone ac-ai-models failed:\n"
                + (clone.stderr or clone.stdout or "no output")
            )

    print(f"[DeepKIN] Checking out pinned commit {pinned_commit[:12]}…")
    checkout = subprocess.run(
        ["git", "checkout", pinned_commit],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    if checkout.returncode != 0:
        raise RuntimeError(
            "git checkout DeepKIN pin failed (need full ac-ai-models history):\n"
            + (checkout.stderr or checkout.stdout)
        )

    if not pkg_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {pkg_dir} after checkout — verify commit {pinned_commit} on ac-ai-models."
        )

    print(f"[DeepKIN] pip install -e {pkg_dir} (--no-deps, same interpreter)")
    pip = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(pkg_dir),
            "--no-deps",
        ],
        capture_output=True,
        text=True,
    )
    if pip.returncode != 0:
        raise RuntimeError(
            "DeepKIN editable install failed:\n" + (pip.stderr or pip.stdout or "no output")
        )

    mono = pkg_dir / "monotonic_align"
    if (mono / "setup.py").is_file():
        build = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=mono,
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            print(
                "[DeepKIN] WARNING: monotonic_align extension build failed; "
                "KinyaFlex may still run on CPU:"
            )
            print(build.stderr or build.stdout)

    import importlib

    importlib.invalidate_caches()
    verify = subprocess.run(
        [
            sys.executable,
            "-c",
            "import deepkin; from deepkin.models.flex_tts import FlexKinyaTTS; "
            "print('deepkin', deepkin.__file__)",
        ],
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        raise RuntimeError(
            "DeepKIN install finished but subprocess import failed:\n"
            + (verify.stderr or verify.stdout)
            + f"\nInterpreter: {sys.executable}"
        )
    print(f"[DeepKIN] Verified: {verify.stdout.strip()}")
    return {"status": "installed", "package_dir": str(pkg_dir), "commit": pinned_commit}


@dataclass(frozen=True)
class ImportSpec:
    package_name: str
    import_stmt: str
    required: bool
    hint: str = ""


CRITICAL_IMPORTS: Tuple[ImportSpec, ...] = (
    ImportSpec("torch", "import torch", True, "Colab GPU runtime should include torch."),
    ImportSpec("transformers", "import transformers", True, "Required for ASR."),
    ImportSpec(
        "llama-cpp-python",
        "from llama_cpp import Llama",
        True,
        "Run install_llama_cpp_for_colab() in the previous cell.",
    ),
    ImportSpec(
        "gradio",
        "import gradio",
        True,
        "pip install 'gradio>=5.7.1,<6' (fixes HfFolder vs huggingface_hub 1.x).",
    ),
    ImportSpec(
        "deepkin",
        "import deepkin",
        True,
        "Run install_deepkin(PROJECT); KinyaFlex requires DeepKIN-AgAI.",
    ),
    ImportSpec(
        "huggingface_hub",
        "import huggingface_hub",
        True,
        "Pulled in by transformers / model download.",
    ),
    ImportSpec(
        "onnxruntime",
        "import onnxruntime",
        True,
        "Required for Piper TTS.",
    ),
)


def verify_critical_imports() -> List[Dict[str, Any]]:
    """
    Verify imports required for Colab. Returns per-package results; raises if any required import fails.
    """
    print(
        f"[Imports] Interpreter: {sys.executable} | "
        f"hub={_installed_version('huggingface-hub')} | "
        f"gradio(dist)={gradio_dist_version()} | HfFolder={_hf_folder_available()}"
    )
    if huggingface_hub_major_version() and huggingface_hub_major_version() >= 1:
        gver = gradio_dist_version()
        if not gver or _parse_version_tuple(gver) < MIN_GRADIO_VERSION:
            ensure_gradio_hub_compatibility(force=True)

    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    for spec in CRITICAL_IMPORTS:
        row: Dict[str, Any] = {
            "package": spec.package_name,
            "import": spec.import_stmt,
            "required": spec.required,
            "ok": False,
            "version": _installed_version(spec.package_name),
            "error": None,
        }
        try:
            if spec.package_name == "gradio":
                ok, detail = _import_gradio_in_subprocess()
                if not ok:
                    raise ImportError(detail)
                row["ok"] = True
                row["version"] = detail.split()[0] if detail else gradio_dist_version()
            elif spec.package_name == "deepkin":
                import importlib

                importlib.invalidate_caches()
                exec(spec.import_stmt, {})
                row["ok"] = True
                mod = __import__("deepkin")
                row["version"] = getattr(mod, "__version__", _installed_version("deepkin"))
            else:
                exec(spec.import_stmt, {})
                row["ok"] = True
                mod_name = spec.import_stmt.split()[-1]
                try:
                    mod = __import__(mod_name)
                    row["version"] = getattr(mod, "__version__", row["version"])
                except Exception:
                    pass
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            if spec.required:
                failures.append(
                    f"{spec.package_name} ({spec.import_stmt}): {row['error']}"
                    + (f" — {spec.hint}" if spec.hint else "")
                )

        status = "OK" if row["ok"] else "FAIL"
        ver = row.get("version") or "unknown"
        req = "required" if spec.required else "optional"
        print(f"[Import] {status:4} {spec.package_name:20} v={ver:12} ({req})")
        if row["error"]:
            print(f"         {row['error']}")
            if spec.hint:
                print(f"         → {spec.hint}")

        results.append(row)

    if failures:
        raise RuntimeError(
            "Import verification failed:\n  " + "\n  ".join(failures)
        )

    print("[Imports] All required packages verified.")
    return results


def main():
    hw = inspect_hardware()
    print(json.dumps(hw, indent=2))
    print(f"Active models root: {get_models_root()}")


if __name__ == "__main__":
    main()