"""English/French ASR via faster-whisper (optional dependency)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from src.asr.exceptions import WhisperASRNotAvailableError

DEFAULT_WHISPER_MODEL = os.environ.get("MWIZERWA_WHISPER_MODEL", "small")
DEFAULT_WHISPER_DEVICE = os.environ.get("MWIZERWA_WHISPER_DEVICE", "cpu")
DEFAULT_WHISPER_COMPUTE_TYPE = os.environ.get("MWIZERWA_WHISPER_COMPUTE_TYPE", "int8")

SUPPORTED_WHISPER_LANGUAGES = frozenset({"en", "fr"})


@dataclass(frozen=True)
class WhisperTranscriptionResult:
    text: str
    language: Optional[str]
    language_probability: Optional[float]
    duration_s: Optional[float]


class WhisperASREngine:
    """Lazy singleton faster-whisper model for en/fr transcription."""

    _shared_model: Any = None
    _shared_key: Optional[tuple] = None

    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL,
        device: str = DEFAULT_WHISPER_DEVICE,
        compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE,
        cpu_threads: Optional[int] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._model = WhisperASREngine._shared_model

    def _cache_key(self) -> tuple:
        return (self.model_size, self.device, self.compute_type, self.cpu_threads)

    def load_model(self):
        if self._model is not None:
            return self._model

        key = self._cache_key()
        if WhisperASREngine._shared_model is not None and WhisperASREngine._shared_key == key:
            self._model = WhisperASREngine._shared_model
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise WhisperASRNotAvailableError(
                "faster-whisper is required for English/French ASR. "
                "Install optional requirements: pip install -r requirements-asr-whisper.txt"
            ) from e

        threads = self.cpu_threads
        if threads is None:
            cpu = os.cpu_count() or 4
            threads = max(1, min(8, cpu - 1))

        print(
            f"Loading faster-whisper '{self.model_size}' "
            f"(device={self.device}, compute_type={self.compute_type}, threads={threads})..."
        )
        model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=threads,
        )
        self._model = model
        WhisperASREngine._shared_model = model
        WhisperASREngine._shared_key = key
        print("faster-whisper model loaded.")
        return model

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        if language is None:
            return None
        lang = language.strip().lower()
        if lang in SUPPORTED_WHISPER_LANGUAGES:
            return lang
        return lang

    def transcribe_with_info(
        self,
        audio_input: Union[str, Path],
        *,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> WhisperTranscriptionResult:
        if not audio_input:
            return WhisperTranscriptionResult("", None, None, None)

        audio_path = Path(audio_input)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path.resolve()}")

        lang = self._normalize_language(language)
        if lang is not None and lang not in SUPPORTED_WHISPER_LANGUAGES:
            raise ValueError(
                f"Whisper backend does not support language '{language}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_WHISPER_LANGUAGES))}"
            )

        model = self.load_model()
        segments, info = model.transcribe(
            str(audio_path.resolve()),
            language=lang,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
        )
        parts = [seg.text.strip() for seg in segments if (seg.text or "").strip()]
        text = " ".join(parts).strip()
        return WhisperTranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration_s=getattr(info, "duration", None),
        )

    def transcribe_audio(
        self,
        audio_input: Union[str, Path],
        *,
        language: Optional[str] = None,
    ) -> str:
        return self.transcribe_with_info(audio_input, language=language).text