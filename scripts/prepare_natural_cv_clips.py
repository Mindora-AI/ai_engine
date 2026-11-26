#!/usr/bin/env python3
"""
Fetch a small Common Voice validation subset (EN/FR) with provenance metadata.
Requires: pip install datasets soundfile (one-off; not in main requirements).

Dataset: mozilla-foundation/common_voice_17_0 (CC0-1.0 per Mozilla Common Voice).
Requires Hugging Face access (accept dataset terms; set HF_TOKEN if rate-limited).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "benchmarks" / "asr_en_fr" / "manifest.json"
CLIPS_ROOT = MANIFEST.parent / "clips" / "natural"

DATASET_ID = "mozilla-foundation/common_voice_17_0"
DATASET_VERSION = "17.0"
LICENSE = "CC0-1.0"
SOURCE_LABEL = "mozilla-common-voice-17.0-validation"


def _load_existing_manifest() -> dict:
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"description": "", "clip_schema": {}, "clips": []}


def fetch_language(
    lang: str,
    *,
    max_clips: int,
    max_per_speaker: int,
    min_duration: float,
    max_duration: float,
) -> list[dict]:
    from datasets import load_dataset
    import numpy as np
    import soundfile as sf

    ds = load_dataset(DATASET_ID, lang, split="validation", streaming=True)
    speaker_counts: dict[str, int] = defaultdict(int)
    entries: list[dict] = []
    CLIPS_ROOT.mkdir(parents=True, exist_ok=True)

    for row in ds:
        if len(entries) >= max_clips:
            break
        if row.get("up_votes", 0) < row.get("down_votes", 0):
            continue
        duration = float(row.get("duration", 0) or 0)
        if duration < min_duration or duration > max_duration:
            continue
        client = str(row.get("client_id") or "unknown")
        if speaker_counts[client] >= max_per_speaker:
            continue
        text = (row.get("sentence") or "").strip()
        if not text:
            continue
        clip_id = f"{lang}_nat_cv16_{len(entries)+1:02d}"
        wav_path = CLIPS_ROOT / f"{clip_id}.wav"
        array = row["audio"]["array"]
        sr = int(row["audio"]["sampling_rate"])
        sf.write(str(wav_path), np.asarray(array), sr)
        rel = f"clips/natural/{clip_id}.wav"
        entries.append(
            {
                "id": clip_id,
                "language": lang,
                "sample_type": "natural",
                "source": SOURCE_LABEL,
                "license": LICENSE,
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "file": rel,
                "reference_text": text,
                "speaker_id": client,
                "duration_s": round(duration, 2),
                "notes": f"Common Voice {lang} validation; speaker anonymized client_id.",
            }
        )
        speaker_counts[client] += 1

    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=25, help="Per language")
    parser.add_argument("--max-per-speaker", type=int, default=3)
    parser.add_argument("--min-duration", type=float, default=2.5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--languages", nargs="+", default=["en", "fr"])
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError as e:
        print("Missing dependency:", e, file=sys.stderr)
        print("Install: pip install datasets soundfile", file=sys.stderr)
        return 1

    manifest = _load_existing_manifest()
    keep = [c for c in manifest.get("clips") or [] if c.get("sample_type") != "natural"]
    new_clips: list[dict] = []
    for lang in args.languages:
        print(f"Fetching {lang} (up to {args.max_clips})...")
        new_clips.extend(
            fetch_language(
                lang,
                max_clips=args.max_clips,
                max_per_speaker=args.max_per_speaker,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
        )
    manifest["clips"] = keep + new_clips
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(new_clips)} natural clip(s) to manifest ({MANIFEST})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())