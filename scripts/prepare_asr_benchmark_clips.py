#!/usr/bin/env python3
"""Generate synthetic EN/FR WAV clips listed in benchmarks/asr_en_fr/manifest.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST = PROJECT_ROOT / "benchmarks" / "asr_en_fr" / "manifest.json"


def main() -> int:
    if not MANIFEST.is_file():
        print(f"Missing manifest: {MANIFEST}")
        return 1

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    from src.tts.piper_engine import PiperTTSEngine

    engine = PiperTTSEngine()
    root = MANIFEST.parent
    created = 0
    for clip in data.get("clips") or []:
        rel = clip.get("file")
        if not rel:
            continue
        out = root / rel
        if out.is_file():
            print(f"Skip existing: {out}")
            continue
        lang = clip["language"]
        text = clip["reference_text"]
        status = engine.check_voice_files(lang)
        if not status.get("present"):
            print(f"Missing Piper voice for {lang}: {status.get('instructions')}")
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        engine.synthesize_speech(text, language=lang, output_path=out)
        print(f"Wrote {out}")
        created += 1

    print(f"Done. Created {created} clip(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())