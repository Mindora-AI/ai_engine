"""
Multilingual ASR facade.

Routing (language resolved before backend selection):
  - rw  -> Kinyarwanda Health ASR (ElvisTata2024/Kinyarwanda-Health-ASR)
  - en/fr -> faster-whisper (optional dependency)
"""

from pathlib import Path
from typing import Optional, Union

from src.asr.exceptions import UnsupportedASRLanguageError
from src.asr.kinya_health_asr import KinyaHealthASREngine, MODEL_ID
from src.asr.whisper_asr import WhisperASREngine

SUPPORTED_ASR_LANGUAGES = frozenset({"rw", "en", "fr"})


def resolve_asr_language(language: Optional[str], *, default: str = "rw") -> str:
    """
    Normalize explicit language for ASR routing.
    None uses `default` for backward-compatible Kinyarwanda-only callers.
    """
    if language is None:
        return default
    lang = language.strip().lower()
    if not lang:
        return default
    if lang not in SUPPORTED_ASR_LANGUAGES:
        raise UnsupportedASRLanguageError(
            f"Unsupported ASR language '{language}'. Supported: rw, en, fr."
        )
    return lang


class ASREngine:
    """
    Stable entry point for transcription. Selects backend by explicit language.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        whisper_engine: Optional[WhisperASREngine] = None,
        kinya_engine: Optional[KinyaHealthASREngine] = None,
    ):
        self._kinya = kinya_engine or KinyaHealthASREngine(model_id=model_id)
        self._whisper = whisper_engine

    def _whisper_backend(self) -> WhisperASREngine:
        if self._whisper is None:
            self._whisper = WhisperASREngine()
        return self._whisper

    def transcribe_audio(
        self,
        audio_input: Union[str, Path],
        *,
        language: Optional[str] = None,
    ) -> str:
        lang = resolve_asr_language(language)
        if lang == "rw":
            return self._kinya.transcribe_audio(audio_input)
        return self._whisper_backend().transcribe_audio(audio_input, language=lang)