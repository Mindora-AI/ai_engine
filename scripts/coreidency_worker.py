"""Subprocess worker for LLM/Whisper RSS benchmarks (one mode per invocation)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def rss_mb() -> float:
    import psutil

    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def run(mode: str, root: Path) -> dict:
    out: dict = {}
    if mode == "llm_only":
        from src.llm.afrique_qwen import AfriqueQwenLLM
        from src.llm.prompts import SYSTEM_PROMPT

        out["rss_baseline_mb"] = rss_mb()
        llm = AfriqueQwenLLM()
        t0 = time.perf_counter()
        llm.load_model()
        out["llm_load_s"] = round(time.perf_counter() - t0, 3)
        out["rss_after_llm_load_mb"] = rss_mb()
        import psutil

        proc = psutil.Process()
        proc.cpu_percent(None)
        t1 = time.perf_counter()
        llm.generate_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Hello, I am feeling a bit stressed."},
            ]
        )
        out["llm_infer_s"] = round(time.perf_counter() - t1, 3)
        out["rss_after_llm_infer_mb"] = rss_mb()
        out["cpu_percent_llm_infer"] = round(proc.cpu_percent(None), 1)
        out["llm_config"] = {
            "model_path": str(llm.model_path),
            "n_ctx": llm.n_ctx,
            "n_threads": llm.n_threads,
            "n_gpu_layers": llm.n_gpu_layers,
            "max_tokens": llm.max_tokens,
        }
        return out

    if mode == "whisper_only":
        from src.asr.whisper_asr import DEFAULT_WHISPER_MODEL, WhisperASREngine

        out["rss_baseline_mb"] = rss_mb()
        w = WhisperASREngine()
        out["whisper_config"] = {
            "model": w.model_size,
            "device": w.device,
            "compute_type": w.compute_type,
            "default_model": DEFAULT_WHISPER_MODEL,
        }
        t0 = time.perf_counter()
        w.load_model()
        out["whisper_load_s"] = round(time.perf_counter() - t0, 3)
        out["rss_after_whisper_load_mb"] = rss_mb()
        probe = root / "benchmarks" / "asr_en_fr" / "clips" / "en_syn_short_01.wav"
        if probe.is_file():
            import psutil

            proc = psutil.Process()
            proc.cpu_percent(None)
            t1 = time.perf_counter()
            w.transcribe_with_info(probe, language="en", beam_size=5)
            out["whisper_infer_s"] = round(time.perf_counter() - t1, 3)
            out["rss_after_whisper_infer_mb"] = rss_mb()
            out["cpu_percent_whisper_infer"] = round(proc.cpu_percent(None), 1)
        else:
            out["whisper_infer_skip"] = str(probe)
        return out

    if mode == "both_sequential":
        from src.asr.whisper_asr import WhisperASREngine
        from src.llm.afrique_qwen import AfriqueQwenLLM
        from src.llm.prompts import SYSTEM_PROMPT

        out["rss_baseline_mb"] = rss_mb()
        w = WhisperASREngine()
        t0 = time.perf_counter()
        w.load_model()
        out["whisper_load_s"] = round(time.perf_counter() - t0, 3)
        out["rss_after_whisper_only_mb"] = rss_mb()
        llm = AfriqueQwenLLM()
        t1 = time.perf_counter()
        llm.load_model()
        out["llm_load_s"] = round(time.perf_counter() - t1, 3)
        out["rss_both_loaded_mb"] = rss_mb()
        probe = root / "benchmarks" / "asr_en_fr" / "clips" / "en_syn_short_01.wav"
        if probe.is_file():
            import psutil

            proc = psutil.Process()
            proc.cpu_percent(None)
            t2 = time.perf_counter()
            w.transcribe_with_info(probe, language="en", beam_size=5)
            out["whisper_infer_s"] = round(time.perf_counter() - t2, 3)
            out["rss_after_whisper_infer_mb"] = rss_mb()
            out["cpu_percent_whisper_infer"] = round(proc.cpu_percent(None), 1)
        t3 = time.perf_counter()
        proc = psutil.Process()
        proc.cpu_percent(None)
        llm.generate_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Hello, I am feeling a bit stressed."},
            ]
        )
        out["llm_infer_s"] = round(time.perf_counter() - t3, 3)
        out["rss_after_llm_infer_mb"] = rss_mb()
        out["cpu_percent_llm_infer"] = round(proc.cpu_percent(None), 1)
        out["peak_rss_mb"] = max(
            out.get("rss_both_loaded_mb", 0),
            out.get("rss_after_whisper_infer_mb", 0),
            out.get("rss_after_llm_infer_mb", 0),
        )
        out["llm_config"] = {
            "n_ctx": llm.n_ctx,
            "n_threads": llm.n_threads,
            "n_gpu_layers": llm.n_gpu_layers,
        }
        out["inference_mode"] = "sequential"
        return out

    return {"error": f"unknown mode {mode}"}


if __name__ == "__main__":
    root = Path(sys.argv[1])
    mode = sys.argv[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    print(json.dumps(run(mode, root)))