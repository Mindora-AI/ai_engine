"""
Public TTS entry point for MWIZERWA.
TTSEngine is a thin facade over TTSRouter (KinyaFlex + Pocket TTS).
"""

from pathlib import Path

from src.tts.kinya_flex_engine import DEFAULT_MODEL_PATH, DEFAULT_SAMPLE_RATE
from src.tts.tts_router import (
    DEFAULT_OUTPUT_DIR,
    LATEST_RESPONSE_FILENAME,
    TTSRouter,
    TTSSynthesisError,
    UnsupportedTTSLanguageError,
    default_output_path,
    latest_response_path,
    resolve_response_language,
)

# Re-export paths and helpers for scripts/tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

__all__ = [
    "TTSEngine",
    "TTSRouter",
    "resolve_response_language",
    "UnsupportedTTSLanguageError",
    "TTSSynthesisError",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SAMPLE_RATE",
    "LATEST_RESPONSE_FILENAME",
    "latest_response_path",
    "default_output_path",
]


class TTSEngine(TTSRouter):
    """
    Multilingual TTS facade:
    - Kinyarwanda (rw) -> KinyaFlex (speaker 0)
    - English (en) / French (fr) -> Pocket TTS
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        from src.tts.kinya_flex_engine import KinyaFlexTTSEngine

        kinya = KinyaFlexTTSEngine(
            model_path=model_path,
            sample_rate=sample_rate,
        )
        super().__init__(kinya_engine=kinya)
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate