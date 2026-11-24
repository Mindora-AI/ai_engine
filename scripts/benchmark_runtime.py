#!/usr/bin/env python3
"""
MWIZERWA end-to-end runtime benchmark (real models when present).

Writes JSON to stdout or --output path. Skips stages when dependencies or files are missing.
Does not invent numbers — each stage records skip_reason when unavailable.

Example:
  python scripts/benchmark_runtime.py --output benchmark_validation_results.json
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

TURN_PROMPTS = [
    ("rw", "Muraho neza, amakuru ki? Nshobora kubona ubufasha?"),
    ("en", "Hello, I am feeling a bit stressed with my work today."),
    ("fr", "Bonjour, je me sens un peu deborde en ce moment."),
]

TTS_PHRASES = {
    "rw": "Muraho, ndi hano ngo ngufashe.",
    "en": "Hello, I am here to listen and support you.",
    "fr": "Bonjour, je suis la pour vous ecouter.",
}


def _timed(load_fn) -> float:
    t0 = time.perf_counter()
    load_fn()
    return time.perf_counter() - t0


def _try_psutil() -> Optional[Dict[str, float]]:
    try:
        import psutil

        proc = psutil.Process()
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "process_ram_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
            "system_ram_used_pct": round(mem.percent, 1),
            "swap_used_gb": round(swap.used / (1024 ** 3), 2),
        }
    except Exception:
        return None


def profile_llm_ttft(llm, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Structured JSON contract generation profile via streaming TTFT."""
    from src.llm.afrique_qwen import STRUCTURED_RESPONSE_SCHEMA

    model = llm.load_model()
    prompt_tokens = llm.count_tokens(messages)
    effective_max = llm._effective_max_tokens(messages, prompt_tokens=prompt_tokens)
    response_fmt = {"type": "json_object", "schema": STRUCTURED_RESPONSE_SCHEMA}
    t0 = time.perf_counter()
    first_token: Optional[float] = None
    chunks: List[str] = []
    stream = model.create_chat_completion(
        messages=messages,
        temperature=llm.temperature,
        top_p=llm.top_p,
        max_tokens=effective_max,
        stream=True,
        response_format=response_fmt,
    )
    for chunk in stream:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        if content:
            if first_token is None:
                first_token = time.perf_counter()
            chunks.append(content)
    t1 = time.perf_counter()
    raw = "".join(chunks)
    gen_tokens = llm.count_text_tokens(raw) if raw else 0
    ttft = (first_token - t0) if first_token else None
    gen_dur = (t1 - first_token) if first_token else 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "effective_max_tokens": effective_max,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "tokens_gen": gen_tokens,
        "gen_duration_s": round(gen_dur, 3),
        "tps": round(gen_tokens / gen_dur, 2) if gen_dur > 0 and gen_tokens else 0.0,
    }


def run_benchmark(asr_audio: Optional[Path] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    t0 = time.perf_counter()
    try:
        from src.llm.afrique_qwen import AfriqueQwenLLM
        from src.llm.conversation import ConversationManager
        from src.conversation.engine import ConversationEngine
        from src.conversation.session import TurnRequest
        from src.tts.tts_engine import TTSEngine
        from src.tts.piper_engine import PiperTTSEngine
        from src.tts.kinya_flex_engine import KinyaFlexTTSEngine
        from src.asr.asr_engine import ASREngine
        from src.llm.prompts import SYSTEM_PROMPT
    except Exception as e:
        out["import_error"] = str(e)
        out["import_time_s"] = round(time.perf_counter() - t0, 3)
        return out

    out["import_time_s"] = round(time.perf_counter() - t0, 3)

    gguf = PROJECT_ROOT / "models" / "afriqueqwen" / "AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf"
    if not gguf.is_file():
        out["llm_skip"] = f"GGUF not found at {gguf}"
        return out

    try:
        llm = AfriqueQwenLLM()
        out["llm_load_cold_s"] = round(_timed(llm.load_model), 3)
        out["llm_load_warm_s"] = round(_timed(llm.load_model), 3)
    except Exception as e:
        out["llm_load_error"] = type(e).__name__ + ": " + str(e)
        return out

    manager = ConversationManager(llm=llm)
    engine = ConversationEngine(manager=manager)

    turns: List[Dict[str, Any]] = []
    for lang, prompt in TURN_PROMPTS:
        t_start = time.perf_counter()
        try:
            result = engine.process_turn(TurnRequest(user_text=prompt, language=lang))
            turns.append(
                {
                    "lang": lang,
                    "prompt": prompt,
                    "response": result.response_text[:500],
                    "intent": (
                        result.metadata.get("interaction_intent")
                        if result.metadata
                        else None
                    ),
                    "contract_valid": bool(result.response_text),
                    "latency_s": round(time.perf_counter() - t_start, 3),
                }
            )
        except Exception as e:
            turns.append(
                {
                    "lang": lang,
                    "prompt": prompt,
                    "error": type(e).__name__ + ": " + str(e),
                    "latency_s": round(time.perf_counter() - t_start, 3),
                }
            )
    out["conversational_turns"] = turns

    profile_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TURN_PROMPTS[0][1]},
    ]
    try:
        out["llm_profile"] = profile_llm_ttft(llm, profile_messages)
    except Exception as e:
        out["llm_profile_error"] = type(e).__name__ + ": " + str(e)

    piper = PiperTTSEngine()
    for lang in ("en", "fr"):
        try:
            out[f"piper_{lang}_load_cold_s"] = round(_timed(lambda l=lang: piper.load_voice(l)), 3)
            out[f"piper_{lang}_load_warm_s"] = round(_timed(lambda l=lang: piper.load_voice(l)), 3)
        except Exception as e:
            out[f"piper_{lang}_skip"] = str(e)

    kinya = KinyaFlexTTSEngine()
    try:
        out["kinya_flex_load_cold_s"] = round(_timed(kinya.load_model), 3)
        out["kinya_flex_load_warm_s"] = round(_timed(kinya.load_model), 3)
    except Exception as e:
        out["kinya_flex_skip"] = str(e)

    tts = TTSEngine()
    tts_bm: Dict[str, Dict[str, float]] = {}
    for lang, phrase in TTS_PHRASES.items():
        try:
            if not tts.check_models(lang).get("present"):
                tts_bm[lang] = {"skip": "model missing"}
                continue
            t1 = time.perf_counter()
            tts.synthesize_speech(phrase, language=lang, check_lang=False)
            pass1 = time.perf_counter() - t1
            t2 = time.perf_counter()
            tts.synthesize_speech(phrase, language=lang, check_lang=False)
            pass2 = time.perf_counter() - t2
            tts_bm[lang] = {"pass_1_s": round(pass1, 3), "pass_2_s": round(pass2, 3)}
        except Exception as e:
            tts_bm[lang] = {"error": str(e)}
    out["tts_benchmarks"] = tts_bm

    if asr_audio and asr_audio.is_file():
        try:
            asr = ASREngine()
            out["asr_load_time_s"] = round(_timed(asr.load_model), 3)
            t_asr = time.perf_counter()
            text = asr.transcribe_audio(asr_audio)
            out["asr_transcription_s"] = round(time.perf_counter() - t_asr, 3)
            out["asr_sample_text_len"] = len(text)
        except Exception as e:
            out["asr_error"] = str(e)
    else:
        out["asr_skip"] = "no --asr-audio sample provided"

    mem = _try_psutil()
    if mem:
        out["memory"] = mem

    cfg = {
        "n_ctx": llm.n_ctx,
        "max_tokens": llm.max_tokens,
        "n_threads": llm.n_threads,
        "n_gpu_layers": llm.n_gpu_layers,
    }
    out["llm_config"] = cfg
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MWIZERWA runtime benchmark")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write JSON results to this file (also prints summary to stdout)",
    )
    parser.add_argument(
        "--asr-audio",
        type=Path,
        help="Optional WAV path for ASR timing",
    )
    args = parser.parse_args()
    results = run_benchmark(asr_audio=args.asr_audio)
    text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()