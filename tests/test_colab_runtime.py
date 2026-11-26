"""Unit tests for Colab runtime helpers (no Colab GPU required)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import colab_runtime  # noqa: E402
from src.runtime.model_paths import get_cache_root, get_models_root, get_output_root  # noqa: E402


class TestGpuLayers:
    def test_zero_layers_without_cuda(self):
        hw = {"cuda_available": False, "gpu_total_mem_gb": 0}
        assert colab_runtime.suggest_llm_gpu_layers(hw, gpu_offload_supported=True) == 0

    def test_zero_layers_without_offload_support(self):
        hw = {"cuda_available": True, "gpu_total_mem_gb": 16}
        assert colab_runtime.suggest_llm_gpu_layers(hw, gpu_offload_supported=False) == 0

    def test_conservative_layers_with_offload(self):
        hw = {"cuda_available": True, "gpu_total_mem_gb": 15.5}
        assert colab_runtime.suggest_llm_gpu_layers(hw, gpu_offload_supported=True) == 24


class TestCudaWheelTag:
    def test_detect_from_torch_version(self):
        hw = {"cuda_available": True, "torch_cuda_version": "12.4.1"}
        assert colab_runtime.detect_cuda_wheel_tag(hw) == "cu124"

    def test_none_when_no_cuda(self):
        assert colab_runtime.detect_cuda_wheel_tag({"cuda_available": False}) is None


class TestDrivePersistence:
    def test_configure_sets_env_vars(self, tmp_path, monkeypatch):
        drive_root = tmp_path / "MWIZERWA"
        project = tmp_path / "ai_engine"
        project.mkdir()
        (project / "run_assistant.py").write_text("# stub", encoding="utf-8")

        monkeypatch.chdir(project)
        paths = colab_runtime.configure_drive_persistence(drive_root, project)

        assert os.environ["MWIZERWA_MODELS_DIR"] == str(drive_root / "models")
        assert os.environ["MWIZERWA_CACHE_DIR"] == str(drive_root / "cache")
        assert os.environ["MWIZERWA_OUTPUT_DIR"] == str(drive_root / "outputs")
        assert "MWIZERWA_MODELS_DIR" in paths
        assert get_models_root() == drive_root / "models"
        assert get_cache_root() == drive_root / "cache"
        assert get_output_root() == drive_root / "outputs"

    def test_raises_when_drive_parent_missing(self, tmp_path):
        drive_root = tmp_path / "not_mounted" / "MWIZERWA"
        with pytest.raises(RuntimeError, match="Google Drive is not mounted"):
            colab_runtime.configure_drive_persistence(drive_root)


class TestColabScripts:
    def test_require_colab_scripts_ok(self):
        colab_runtime.require_colab_scripts(PROJECT_ROOT)

    def test_require_colab_scripts_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Colab deployment files missing"):
            colab_runtime.require_colab_scripts(tmp_path)


class TestHardware:
    def test_inspect_hardware_has_keys(self):
        hw = colab_runtime.inspect_hardware()
        assert "platform" in hw
        assert "python" in hw
        assert "colab" in hw


class TestApplyGpuLayers:
    def test_sets_env(self, monkeypatch):
        monkeypatch.delenv("MWIZERWA_LLM_N_GPU_LAYERS", raising=False)
        hw = {"cuda_available": False}
        llama_info = {"gpu_offload_supported": False}
        layers = colab_runtime.apply_llm_gpu_layers(hw, llama_info)
        assert layers == 0
        assert os.environ["MWIZERWA_LLM_N_GPU_LAYERS"] == "0"


class TestVersionParsing:
    def test_parse_version_tuple(self):
        assert colab_runtime._parse_version_tuple("5.50.0") >= (5, 7, 1)
        assert colab_runtime._parse_version_tuple("4.44.0") < (5, 7, 1)

    def test_import_gradio_subprocess(self):
        if importlib.util.find_spec("gradio") is None:
            pytest.skip(
                "gradio is a Colab/runtime dependency; not required for local API work"
            )
        ok, detail = colab_runtime._import_gradio_in_subprocess()
        assert ok, detail


class TestCriticalImports:
    def test_specs_include_gradio_and_deepkin(self):
        names = {s.package_name for s in colab_runtime.CRITICAL_IMPORTS}
        assert "gradio" in names
        assert "deepkin" in names

    def test_deepkin_importable_local(self):
        """DeepKIN is optional locally; required in Colab KinyaFlex training flows."""
        assert isinstance(colab_runtime.deepkin_importable(), bool)
        if not colab_runtime.deepkin_importable():
            pytest.skip("deepkin not installed in this environment")


class TestInstallDeepkinIdempotent:
    def test_skips_when_already_installed(self, tmp_path):
        if not colab_runtime.deepkin_importable():
            pytest.skip("deepkin not installed in this environment")
        project = tmp_path / "proj"
        project.mkdir()
        (project / "run_assistant.py").write_text("", encoding="utf-8")
        info = colab_runtime.install_deepkin(project)
        assert info["status"] == "already_installed"