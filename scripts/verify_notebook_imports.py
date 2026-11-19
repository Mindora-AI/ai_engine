#!/usr/bin/env python3
"""Static check: every scripts.colab_* symbol used by the Colab notebook exists."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_ROOT / "notebooks" / "MWIZERWA_Colab.ipynb"

EXPECTED_RUNTIME_SYMBOLS = {
    "configure_drive_persistence",
    "report_model_asset_status",
    "inspect_hardware",
    "install_python_dependencies",
    "install_llama_cpp_for_colab",
    "apply_llm_gpu_layers",
    "llama_supports_gpu_offload",
    "install_deepkin",
    "verify_critical_imports",
    "require_colab_scripts",
    "clone_mwizerwa_if_needed",
}

SCRIPTS = (
    "colab_runtime.py",
    "colab_validate.py",
    "colab_gradio.py",
    "download_models.py",
    "verify_notebook_imports.py",
)
COLAB_ROOT_FILES = ("constraints-colab.txt", "requirements-colab.txt")


def _load_colab_runtime_symbols() -> Set[str]:
    path = PROJECT_ROOT / "scripts" / "colab_runtime.py"
    spec = importlib.util.spec_from_file_location("colab_runtime", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {name for name in dir(mod) if not name.startswith("_")}


def _notebook_imports_from_scripts() -> Set[str]:
    import json

    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    found: Set[str] = set()
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        lines = [
            line
            for line in src.splitlines()
            if line.strip() and not line.lstrip().startswith("!")
        ]
        py_src = "\n".join(lines)
        if not py_src.strip():
            continue
        try:
            tree = ast.parse(py_src)
        except SyntaxError as e:
            raise SyntaxError(f"Invalid notebook code cell:\n{src[:200]}...") from e
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "scripts.colab_runtime":
                for alias in node.names:
                    found.add(alias.name)
    return found


def verify_scripts_on_disk() -> None:
    missing = [s for s in SCRIPTS if not (PROJECT_ROOT / "scripts" / s).is_file()]
    missing += [f for f in COLAB_ROOT_FILES if not (PROJECT_ROOT / f).is_file()]
    if missing:
        raise FileNotFoundError("Missing Colab files: " + ", ".join(missing))


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    verify_scripts_on_disk()
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(NOTEBOOK)

    runtime_symbols = _load_colab_runtime_symbols()
    notebook_wants = _notebook_imports_from_scripts()
    unknown = notebook_wants - runtime_symbols
    if unknown:
        raise RuntimeError(
            "Notebook imports undefined colab_runtime symbols: " + ", ".join(sorted(unknown))
        )

    for sym in EXPECTED_RUNTIME_SYMBOLS:
        if sym not in runtime_symbols:
            raise RuntimeError(f"Expected colab_runtime.{sym} is not defined")

    print("[OK] Notebook imports resolve against scripts/colab_runtime.py")
    print(f"[OK] scripts present: {', '.join(SCRIPTS)}")
    print(f"[OK] notebook imports: {', '.join(sorted(notebook_wants))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())