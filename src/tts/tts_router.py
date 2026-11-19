from pathlib import Path
from typing import Optional, Union

from src.tts.kinya_flex_engine import KinyaFlexTTSEngine
from src.tts.piper_engine import PiperTTSEngine
from src.utils.language_detector import LanguageDetectionResult, detect_language

from src.runtime.model_paths import get_output_root, get_project_root

PROJECT_ROOT = get_project_root()
DEFAULT_OUTPUT_DIR = get_output_root() / "tts_responses"
LATEST_RESPONSE_FILENAME = "latest_response.wav"

SUPPORTED_TTS_LANGUAGES = frozenset({"rw", "en", "fr"})


class UnsupportedTTSLanguageError(Exception):
    """Raised when the response language cannot be routed safely."""


class TTSSynthesisError(Exception):
    """Raised when synthesis fails after routing."""


def resolve_response_language(
    text: str,
    language: Optional[str] = None,
) -> str:
    """
    Resolve the ISO code (rw/en/fr) for the final assistant response text.
    An explicit language override must be a supported code.
    """
    if language is not None:
        lang = language.strip().lower()
        if lang in SUPPORTED_TTS_LANGUAGES:
            return lang
        raise UnsupportedTTSLanguageError(
            f"Unsupported explicit response language '{language}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_TTS_LANGUAGES))}"
        )

    result: LanguageDetectionResult = detect_language(text)
    if not result.is_reliable or result.language not in SUPPORTED_TTS_LANGUAGES:
        detected = result.language if result else None
        raise UnsupportedTTSLanguageError(
            "Cannot route TTS: response language is uncertain or unsupported "
            f"(detected={detected!r}, reliable={getattr(result, 'is_reliable', False)}). "
            "Refusing to send audio to the wrong engine."
        )
    return result.language


def default_output_path(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Stable session output path (KinyaFlex and Piper)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / LATEST_RESPONSE_FILENAME


def latest_response_path(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    return default_output_path(output_dir)


class TTSRouter:
    """
    Routes assistant response text to KinyaFlex (rw) or Piper (en/fr).
    Backend engines are loaded lazily and cached.
    """

    def __init__(
        self,
        default_speaker: int = 0,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        kinya_engine: Optional[KinyaFlexTTSEngine] = None,
        piper_engine: Optional[PiperTTSEngine] = None,
    ):
        self.default_speaker = default_speaker
        self.output_dir = Path(output_dir)
        self._kinya = kinya_engine or KinyaFlexTTSEngine(default_speaker=default_speaker)
        self._piper = piper_engine or PiperTTSEngine()

    def check_models(self, language: str) -> dict:
        if language == "rw":
            path = self._kinya.model_path
            return {
                "language": "rw",
                "engine": "kinya-flex",
                "present": path.is_file(),
                "paths": [path],
                "instructions": "python scripts/download_models.py --model tts",
            }
        if language in ("en", "fr"):
            status = self._piper.check_voice_files(language)
            status["engine"] = "piper"
            return status
        return {
            "language": language,
            "engine": None,
            "present": False,
            "instructions": "Unsupported TTS language.",
        }

    def synthesize_speech(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker_id: Optional[int] = None,
        language: Optional[str] = None,
        check_lang: bool = True,
    ) -> Optional[str]:
        """
        Synthesize assistant response audio.
        Returns WAV path, or None if text is empty.
        Raises UnsupportedTTSLanguageError / FileNotFoundError / TTSSynthesisError.
        """
        if not text or not text.strip():
            return None

        if not check_lang:
            if language not in SUPPORTED_TTS_LANGUAGES:
                raise UnsupportedTTSLanguageError(
                    "check_lang=False requires an explicit supported language."
                )
            target_lang = language
        else:
            try:
                target_lang = resolve_response_language(text, language=language)
            except UnsupportedTTSLanguageError:
                raise

        if output_path is None:
            output_file = latest_response_path(self.output_dir)
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_file = Path(output_path)

        engine_name = "KinyaFlex" if target_lang == "rw" else "Piper"
        print(f"[TTS Router] Routing to {engine_name} for language '{target_lang}'...")

        try:
            if target_lang == "rw":
                sid = self.default_speaker if speaker_id is None else speaker_id
                result = self._kinya.synthesize_speech(
                    text,
                    output_path=output_file,
                    speaker_id=sid,
                    check_lang=False,
                )
            else:
                result = self._piper.synthesize_speech(
                    text,
                    language=target_lang,
                    output_path=output_file,
                )
            return result
        except (FileNotFoundError, ImportError, ValueError) as e:
            raise TTSSynthesisError(
                f"{engine_name} synthesis failed for language '{target_lang}': {e}"
            ) from e