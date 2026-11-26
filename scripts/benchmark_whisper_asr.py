#!/usr/bin/env python3
"""
Benchmark faster-whisper for English/French on local manifest clips.
Writes JSON results only from measured runs (no fabricated numbers).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "asr_en_fr" / "manifest.json"


def _audio_duration_s(path: Path) -> Optional[float]:
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else None
    except Exception:
        return None


def _process_rss_mb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def _wer(reference: str, hypothesis: str) -> Optional[float]:
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()
    if not ref:
        return None
    try:
        import jiwer

        return round(jiwer.wer(ref, hyp), 4)
    except Exception:
        return None


def _transcribe_clip(
    engine,
    path: Path,
    language: Optional[str],
    *,
    beam_size: int = 5,
) -> Dict[str, Any]:
    rss_before = _process_rss_mb()
    result_holder: Dict[str, Any] = {}

    def _run():
        result_holder["result"] = engine.transcribe_with_info(
            path, language=language, beam_size=beam_size
        )

    try:
        import psutil

        proc = psutil.Process()
        proc.cpu_percent(None)
        t0 = time.perf_counter()
        _run()
        elapsed = time.perf_counter() - t0
        cpu = round(proc.cpu_percent(None), 1)
    except Exception:
        t0 = time.perf_counter()
        _run()
        elapsed = time.perf_counter() - t0
        cpu = None

    result = result_holder["result"]
    rss_after = _process_rss_mb()
    duration = _audio_duration_s(path) or result.duration_s
    rtf = round(elapsed / duration, 3) if duration and duration > 0 else None

    return {
        "latency_s": round(elapsed, 3),
        "rtf": rtf,
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "cpu_percent": cpu,
        "transcription": result.text,
        "detected_language": result.language,
        "language_probability": (
            round(result.language_probability, 4)
            if result.language_probability is not None
            else None
        ),
    }


def _co_residency(llm_prompt: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    gguf = (
        PROJECT_ROOT
        / "models"
        / "afriqueqwen"
        / "AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf"
    )
    if not gguf.is_file():
        out["skip"] = f"GGUF not found at {gguf}"
        return out

    try:
        from src.asr.whisper_asr import WhisperASREngine
        from src.llm.afrique_qwen import AfriqueQwenLLM
        from src.llm.prompts import SYSTEM_PROMPT

        whisper = WhisperASREngine()
        out["rss_after_whisper_load_mb"] = _process_rss_mb()
        t_load = time.perf_counter()
        whisper.load_model()
        out["whisper_load_s"] = round(time.perf_counter() - t_load, 3)
        out["rss_after_whisper_loaded_mb"] = _process_rss_mb()

        llm = AfriqueQwenLLM()
        t_llm = time.perf_counter()
        llm.load_model()
        out["llm_load_s"] = round(time.perf_counter() - t_llm, 3)
        out["rss_after_llm_loaded_mb"] = _process_rss_mb()

        t_inf = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": llm_prompt},
        ]
        _ = llm.generate_completion(messages)
        out["llm_infer_s"] = round(time.perf_counter() - t_inf, 3)
        out["rss_after_llm_infer_mb"] = _process_rss_mb()

        probe = (
            PROJECT_ROOT
            / "benchmarks"
            / "asr_en_fr"
            / "clips"
            / "en_syn_short_01.wav"
        )
        if probe.is_file():
            t_asr = time.perf_counter()
            _ = whisper.transcribe_with_info(probe, language="en")
            out["whisper_infer_s"] = round(time.perf_counter() - t_asr, 3)
            out["rss_after_whisper_infer_mb"] = _process_rss_mb()
        else:
            out["whisper_infer_skip"] = f"missing probe clip: {probe}"
    except Exception as e:
        out["error"] = type(e).__name__ + ": " + str(e)
    return out


def _summarize_by_language(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-language / sample_type WER and latency (no blended averages across langs)."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for clip in clips:
        if clip.get("skip"):
            continue
        key = f"{clip.get('language')}_{clip.get('sample_type', 'unknown')}"
        buckets.setdefault(key, []).append(clip)

    summary: Dict[str, Any] = {}
    for key, items in sorted(buckets.items()):
        wers = [c["wer_forced"] for c in items if c.get("wer_forced") is not None]
        rts = [c["forced"]["rtf"] for c in items if c.get("forced", {}).get("rtf") is not None]
        lats = [c["forced"]["latency_s"] for c in items if c.get("forced", {}).get("latency_s") is not None]
        lang, sample_type = key.rsplit("_", 1)
        summary[key] = {
            "language": lang,
            "sample_type": sample_type,
            "clip_count": len(items),
            "wer_forced_mean": round(sum(wers) / len(wers), 4) if wers else None,
            "wer_forced_per_clip": {c["id"]: c.get("wer_forced") for c in items},
            "rtf_forced_mean": round(sum(rts) / len(rts), 3) if rts else None,
            "latency_forced_mean_s": round(sum(lats) / len(lats), 3) if lats else None,
        }
    return summary


def run_benchmark(
    manifest: Path,
    *,
    model_size: Optional[str] = None,
    include_auto_detect: bool = True,
    co_residency: bool = False,
    beam_size: int = 5,
) -> Dict[str, Any]:
    from src.asr.whisper_asr import DEFAULT_WHISPER_MODEL, WhisperASREngine

    if not manifest.is_file():
        return {"error": f"Manifest not found: {manifest}"}

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    root = manifest.parent
    engine = WhisperASREngine(model_size=model_size or DEFAULT_WHISPER_MODEL)

    t_load = time.perf_counter()
    engine.load_model()
    load_s = round(time.perf_counter() - t_load, 3)
    t_warm = time.perf_counter()
    engine.load_model()
    load_warm = round(time.perf_counter() - t_warm, 3)

    results: List[Dict[str, Any]] = []
    for clip in payload.get("clips") or []:
        wav = root / clip["file"]
        if not wav.is_file():
            results.append(
                {
                    "id": clip.get("id"),
                    "skip": f"missing audio: {wav}",
                }
            )
            continue

        forced = _transcribe_clip(engine, wav, clip["language"], beam_size=beam_size)
        forced["mode"] = "forced_language"
        forced["requested_language"] = clip["language"]

        entry = {
            "id": clip.get("id"),
            "language": clip.get("language"),
            "sample_type": clip.get("sample_type"),
            "reference_text": clip.get("reference_text"),
            "audio_path": str(wav),
            "forced": forced,
            "wer_forced": _wer(clip.get("reference_text", ""), forced.get("transcription", "")),
        }

        if include_auto_detect:
            auto = _transcribe_clip(engine, wav, None, beam_size=beam_size)
            auto["mode"] = "auto_detect"
            entry["auto_detect"] = auto
            entry["wer_auto"] = _wer(
                clip.get("reference_text", ""), auto.get("transcription", "")
            )
        results.append(entry)

    out: Dict[str, Any] = {
        "manifest": str(manifest),
        "model": engine.model_size,
        "device": engine.device,
        "compute_type": engine.compute_type,
        "beam_size": beam_size,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "whisper_load_cold_s": load_s,
        "whisper_load_warm_s": load_warm,
        "rss_after_load_mb": _process_rss_mb(),
        "clips": results,
        "summary_by_language_and_sample_type": _summarize_by_language(results),
    }
    if co_residency:
        out["co_residency"] = _co_residency(
            "Hello, I am feeling a bit stressed with my work today."
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper EN/FR ASR")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--model",
        default=None,
        help="Whisper model id (e.g. small, medium). Default: MWIZERWA_WHISPER_MODEL or small.",
    )
    parser.add_argument("--output", "-o", type=Path, help="Write JSON results here")
    parser.add_argument("--no-auto-detect", action="store_true")
    parser.add_argument("--co-residency", action="store_true")
    parser.add_argument("--beam-size", type=int, default=5, help="Whisper beam size (default 5)")
    args = parser.parse_args()

    data = run_benchmark(
        args.manifest,
        model_size=args.model,
        include_auto_detect=not args.no_auto_detect,
        co_residency=args.co_residency,
        beam_size=args.beam_size,
    )
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()