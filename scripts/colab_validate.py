#!/usr/bin/env python3
"""
MWIZERWA — Colab/local validation (real inference when models are present).
"""

from __future__ import annotations

import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversation.engine import ConversationEngine  # noqa: E402
from src.conversation.session import TurnRequest  # noqa: E402
from src.runtime.model_paths import (  # noqa: E402
    get_cache_root,
    get_models_root,
    get_output_root,
    resolve_kinya_flex_checkpoint,
    resolve_llm_gguf_path,
)
import importlib.util

REQUIRED_ASSET_TESTS = frozenset(
    {
        "runtime_paths_configured",
        "llm_file_present",
        "kinya_checkpoint_present",
        "pocket_tts_present",
    }
)
REQUIRED_INFERENCE_TESTS = frozenset(
    {
        "llm_turn_rw",
        "llm_turn_en",
        "llm_turn_fr",
        "pipeline_metadata_rw",
        "pipeline_metadata_en",
        "pipeline_metadata_fr",
        "tts_rw",
        "tts_en",
        "tts_fr",
        "tts_response_routing_rw",
        "tts_response_routing_en",
        "tts_response_routing_fr",
        "gradio_ui_build",
    }
)
OPTIONAL_TESTS = frozenset({"asr_test_wav", "voice_to_voice"})


def _wav_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 500:
        return False
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() > 0
    except wave.Error:
        return False


def _paths_configured() -> Tuple[bool, str]:
    required = (
        "MWIZERWA_PROJECT_ROOT",
        "MWIZERWA_MODELS_DIR",
        "MWIZERWA_OUTPUT_DIR",
    )
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        return False, f"missing env: {', '.join(missing)}"
    models = get_models_root()
    if not models.is_dir():
        return False, f"models root not a directory: {models}"
    cache = get_cache_root()
    outputs = get_output_root()
    return True, f"models={models}; cache={cache}; outputs={outputs}"


def run_validation() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    turn_results: Dict[str, Any] = {}

    def record(name: str, ok: bool, detail: str = "", seconds: float | None = None):
        row = {"test": name, "ok": ok, "detail": detail, "required": name not in OPTIONAL_TESTS}
        if seconds is not None:
            row["seconds"] = round(seconds, 3)
        results.append(row)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" ({seconds:.2f}s)" if seconds else ""))
        if detail:
            print(f"       {detail}")

    ok_paths, path_detail = _paths_configured()
    record("runtime_paths_configured", ok_paths, path_detail)

    llm_path = resolve_llm_gguf_path()
    kinya_path = resolve_kinya_flex_checkpoint()
    pocket_ok = importlib.util.find_spec("pocket_tts") is not None
    record("llm_file_present", llm_path.is_file(), str(llm_path))
    record("kinya_checkpoint_present", kinya_path.is_file(), str(kinya_path))
    record(
        "pocket_tts_present",
        pocket_ok,
        "pip install pocket-tts; weights download on first synthesis",
    )

    assets_ok = all(r["ok"] for r in results if r["test"] in REQUIRED_ASSET_TESTS)
    if not assets_ok:
        print("\n[Validation] Required model files or paths missing — fix Drive/env, then:")
        print("  python scripts/download_models.py --check")
        return results

    try:
        engine = ConversationEngine()
        prompts = [
            ("rw", "Muraho, ndumva mpayijwe n'ubwoba."),
            ("en", "Hello, I feel anxious about my exams."),
            ("fr", "Bonjour, je me sens très stressé."),
        ]
        for lang, user_text in prompts:
            t0 = time.perf_counter()
            result = engine.process_turn(TurnRequest(user_text=user_text))
            dt = time.perf_counter() - t0
            ok = bool(result.response_text.strip())
            record(f"llm_turn_{lang}", ok, result.response_text[:80], dt)
            turn_results[lang] = result
            meta_ok = bool(
                result.metadata.get("interaction_intent")
                or result.metadata.get("parsed_response")
                or result.metadata.get("safety_assessment")
            )
            record(
                f"pipeline_metadata_{lang}",
                meta_ok,
                f"lang={result.language}; keys={list(result.metadata.keys())[:6]}",
            )
    except Exception as e:
        record("llm_conversation", False, str(e))

    try:
        from src.tts.tts_engine import TTSEngine

        tts = TTSEngine()
        out_dir = get_output_root() / "tts_responses"
        out_dir.mkdir(parents=True, exist_ok=True)
        samples = [
            ("rw", "Muraho neza, umeze ute?"),
            ("en", "Hello, this is an English synthesis validation."),
            ("fr", "Bonjour, ceci est un test en français."),
        ]
        for lang, text in samples:
            t0 = time.perf_counter()
            try:
                wav = tts.synthesize_speech(
                    text,
                    output_path=out_dir / f"validate_{lang}.wav",
                    language=lang,
                    check_lang=False,
                )
                dt = time.perf_counter() - t0
                ok = _wav_ok(Path(wav)) if wav else False
                record(f"tts_{lang}", ok, str(wav), dt)
            except Exception as e:
                record(f"tts_{lang}", False, str(e))

        for lang in ("rw", "en", "fr"):
            tr = turn_results.get(lang)
            if not tr or not tr.response_text.strip():
                record(f"tts_response_routing_{lang}", False, "no assistant response from LLM turn")
                continue
            t0 = time.perf_counter()
            try:
                wav = tts.synthesize_speech(
                    tr.response_text,
                    output_path=out_dir / f"validate_assistant_{lang}.wav",
                    language=tr.language,
                    check_lang=False,
                )
                dt = time.perf_counter() - t0
                ok = _wav_ok(Path(wav)) if wav else False
                record(
                    f"tts_response_routing_{lang}",
                    ok,
                    f"response_lang={tr.language}; wav={wav}",
                    dt,
                )
            except Exception as e:
                record(f"tts_response_routing_{lang}", False, str(e))
    except Exception as e:
        record("tts_engine", False, str(e))

    test_wav = PROJECT_ROOT / "test.wav"
    if test_wav.is_file():
        try:
            from src.asr.asr_engine import ASREngine

            asr = ASREngine()
            t0 = time.perf_counter()
            text = asr.transcribe_audio(test_wav)
            record(
                "asr_test_wav",
                bool(text.strip()),
                text[:120],
                time.perf_counter() - t0,
            )
            if text.strip():
                t0 = time.perf_counter()
                engine = ConversationEngine()
                turn = engine.process_turn(TurnRequest(user_text=text.strip()))
                from src.tts.tts_engine import TTSEngine

                v2v_audio = TTSEngine().synthesize_speech(
                    turn.response_text,
                    language=turn.language,
                    check_lang=False,
                )
                record(
                    "voice_to_voice",
                    _wav_ok(Path(v2v_audio)) if v2v_audio else False,
                    turn.response_text[:80],
                    time.perf_counter() - t0,
                )
        except Exception as e:
            record("asr_test_wav", False, str(e))
            record("voice_to_voice", False, str(e))
    else:
        record("asr_test_wav", False, "optional — test.wav not in project root")

    try:
        from scripts.colab_gradio import build_ui

        t0 = time.perf_counter()
        build_ui()
        record("gradio_ui_build", True, "ConversationEngine + TTSEngine + ASREngine wired", time.perf_counter() - t0)
    except Exception as e:
        record("gradio_ui_build", False, str(e))

    required = REQUIRED_ASSET_TESTS | REQUIRED_INFERENCE_TESTS
    failed = [r for r in results if r["test"] in required and not r["ok"]]
    passed = sum(1 for r in results if r["ok"])
    print(f"\nSummary: {passed}/{len(results)} checks passed")
    if failed:
        print("Required failures:", ", ".join(r["test"] for r in failed))
    return results


def validation_passed(results: List[Dict[str, Any]]) -> bool:
    required = REQUIRED_ASSET_TESTS | REQUIRED_INFERENCE_TESTS
    return all(r["ok"] for r in results if r["test"] in required)


if __name__ == "__main__":
    out = run_validation()
    sys.exit(0 if validation_passed(out) else 1)