#!/usr/bin/env python3
"""
AfriqueQwen + Whisper Small co-residency (separate subprocess per phase for clean RSS).
Requires one Python env with llama-cpp-python and faster-whisper (see benchmarks/asr_en_fr/README.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "scripts" / "coreidency_worker.py"


def _run_phase(python: Path, mode: str) -> dict:
    proc = subprocess.run(
        [str(python), str(WORKER), str(PROJECT_ROOT), mode],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        return {
            "mode": mode,
            "error": (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip(),
        }
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"mode": mode, "error": "worker produced no JSON", "stdout": proc.stdout[:2000]}
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=Path,
        default=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.python.is_file():
        print(f"Python not found: {args.python}", file=sys.stderr)
        return 1
    report = {
        "python": str(args.python),
        "gguf": str(
            PROJECT_ROOT
            / "models"
            / "afriqueqwen"
            / "AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf"
        ),
        "phases": {},
    }
    for mode in ("llm_only", "whisper_only", "both_sequential"):
        print(f"Phase {mode}...", flush=True)
        report["phases"][mode] = _run_phase(args.python, mode)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())