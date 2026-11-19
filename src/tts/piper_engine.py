import wave
from pathlib import Path
from typing import Dict, Optional, Union

from src.runtime.model_paths import resolve_piper_voices_root

PIPER_VOICES_ROOT = resolve_piper_voices_root()

PIPER_VOICE_SPECS: Dict[str, Dict[str, str]] = {
    "en": {
        "voice_id": "en_US-ljspeech-medium",
        "onnx": "en_US-ljspeech-medium.onnx",
        "config": "en_US-ljspeech-medium.onnx.json",
    },
    "fr": {
        "voice_id": "fr_FR-siwis-medium",
        "onnx": "fr_FR-siwis-medium.onnx",
        "config": "fr_FR-siwis-medium.onnx.json",
    },
}


class PiperTTSEngine:
    """
    Adapter for Piper TTS (English and French voices).
    Loads each voice ONNX model once and caches PiperVoice instances.
    """

    def __init__(self, voices_root: Path = PIPER_VOICES_ROOT):
        self.voices_root = Path(voices_root)
        self._voices: Dict[str, object] = {}

    def voice_dir(self, language: str) -> Path:
        spec = PIPER_VOICE_SPECS.get(language)
        if spec is None:
            raise ValueError(f"No Piper voice configured for language '{language}'.")
        return self.voices_root / spec["voice_id"]

    def model_paths(self, language: str) -> tuple[Path, Path]:
        spec = PIPER_VOICE_SPECS[language]
        voice_dir = self.voice_dir(language)
        onnx_path = voice_dir / spec["onnx"]
        config_path = voice_dir / spec["config"]
        return onnx_path, config_path

    def check_voice_files(self, language: str) -> Dict[str, object]:
        """Returns presence info and setup instructions for a Piper voice."""
        spec = PIPER_VOICE_SPECS.get(language)
        if spec is None:
            return {
                "language": language,
                "configured": False,
                "present": False,
                "paths": [],
                "instructions": f"Piper is not configured for language '{language}'.",
            }

        onnx_path, config_path = self.model_paths(language)
        paths = [onnx_path, config_path]
        present = all(p.is_file() for p in paths)
        return {
            "language": language,
            "voice_id": spec["voice_id"],
            "configured": True,
            "present": present,
            "paths": paths,
            "instructions": (
                f"Download Piper voice '{spec['voice_id']}' with:\n"
                f"  python scripts/download_models.py --model piper-{language}"
            ),
        }

    def load_voice(self, language: str):
        if language in self._voices:
            return self._voices[language]

        status = self.check_voice_files(language)
        if not status["present"]:
            missing = [str(p) for p in status["paths"] if not Path(p).is_file()]
            raise FileNotFoundError(
                f"Piper voice files missing for {language} ({status.get('voice_id')}):\n"
                + "\n".join(f"  - {m}" for m in missing)
                + f"\n{status['instructions']}"
            )

        try:
            from piper import PiperVoice
        except ImportError as e:
            raise ImportError(
                "piper-tts package is required. Install with: pip install piper-tts onnxruntime"
            ) from e

        onnx_path, config_path = self.model_paths(language)
        print(f"Loading Piper voice for {language} from: {onnx_path}")
        voice = PiperVoice.load(str(onnx_path), config_path=str(config_path))
        self._voices[language] = voice
        print(f"Piper voice loaded for {language}.")
        return voice

    def synthesize_speech(
        self,
        text: str,
        language: str,
        output_path: Union[str, Path],
    ) -> str:
        if language not in PIPER_VOICE_SPECS:
            raise ValueError(f"Piper does not support language '{language}'.")

        if not text or not text.strip():
            raise ValueError("Piper synthesis requires non-empty text.")

        voice = self.load_voice(language)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_file), "wb") as wav_file:
            voice.synthesize_wav(text.strip(), wav_file)

        print(f"[Piper TTS] Synthesized audio saved to: {output_file.resolve()}")
        return str(output_file.resolve())

    @property
    def sample_rate(self) -> Optional[int]:
        """Sample rate from the first loaded voice config, if any."""
        if not self._voices:
            return 22050
        voice = next(iter(self._voices.values()))
        return getattr(getattr(voice, "config", None), "sample_rate", 22050)