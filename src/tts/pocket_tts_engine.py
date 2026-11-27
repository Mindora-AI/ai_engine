"""Kyutai Pocket TTS adapter for English and French synthesis."""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

# Pocket TTS language config keys (package defaults).
POCKET_LANGUAGE: Dict[str, str] = {
    "en": "english",
    "fr": "french",
}

EN_VOICE = "anna"
FR_VOICE = "estelle"
DEFAULT_FR_VOICE = os.environ.get("MWIZERWA_POCKET_FR_VOICE", FR_VOICE).strip().lower()


class PocketTTSEngine:
    """
    Adapter for Pocket TTS (English and French).
    Loads one TTSModel per language and caches voice states.
    """

    def __init__(self) -> None:
        self._models: Dict[str, object] = {}
        self._voice_states: Dict[tuple[str, str], object] = {}

    @staticmethod
    def _voice_for_language(language: str) -> str:
        if language == "en":
            return EN_VOICE
        if language == "fr":
            return DEFAULT_FR_VOICE
        raise ValueError(f"No Pocket TTS voice configured for language '{language}'.")

    @staticmethod
    def pocket_language(language: str) -> str:
        pocket = POCKET_LANGUAGE.get(language)
        if pocket is None:
            raise ValueError(f"Pocket TTS does not support language '{language}'.")
        return pocket

    def check_voice_files(self, language: str) -> Dict[str, object]:
        """Returns dependency / setup info (weights load from Hugging Face on first use)."""
        if language not in POCKET_LANGUAGE:
            return {
                "language": language,
                "configured": False,
                "present": False,
                "paths": [],
                "instructions": f"Pocket TTS is not configured for language '{language}'.",
            }
        try:
            import importlib.util

            present = importlib.util.find_spec("pocket_tts") is not None
        except ImportError:
            present = False
        voice = self._voice_for_language(language)
        return {
            "language": language,
            "voice_id": voice,
            "engine": "pocket-tts",
            "configured": True,
            "present": present,
            "paths": [],
            "instructions": (
                "Install Pocket TTS: pip install pocket-tts\n"
                "First synthesis downloads kyutai/pocket-tts weights from Hugging Face."
            ),
        }

    def load_model(self, language: str):
        if language in self._models:
            return self._models[language]

        status = self.check_voice_files(language)
        if not status["present"]:
            raise ImportError(
                "pocket-tts package is required. Install with: pip install pocket-tts"
            )

        from pocket_tts import TTSModel

        pocket_lang = self.pocket_language(language)
        print(f"Loading Pocket TTS model for {language} (config={pocket_lang})...")
        model = TTSModel.load_model(language=pocket_lang)
        self._models[language] = model
        print(f"Pocket TTS model loaded for {language}.")
        return model

    def _voice_state(self, language: str, voice_name: str):
        key = (language, voice_name)
        if key in self._voice_states:
            return self._voice_states[key]

        model = self.load_model(language)
        print(f"Loading Pocket TTS voice '{voice_name}' for {language}...")
        state = model.get_state_for_audio_prompt(voice_name)
        self._voice_states[key] = state
        return state

    def synthesize_speech(
        self,
        text: str,
        language: str,
        output_path: Union[str, Path],
    ) -> str:
        if language not in POCKET_LANGUAGE:
            raise ValueError(f"Pocket TTS does not support language '{language}'.")

        if not text or not text.strip():
            raise ValueError("Pocket TTS synthesis requires non-empty text.")

        voice_name = self._voice_for_language(language)
        model = self.load_model(language)
        voice_state = self._voice_state(language, voice_name)

        audio = model.generate_audio(voice_state, text.strip())
        pcm = audio.detach().cpu().numpy()
        pcm = np.squeeze(pcm)
        pcm = np.clip(pcm, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = int(model.sample_rate)

        with wave.open(str(output_file), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())

        print(f"[Pocket TTS] Synthesized audio saved to: {output_file.resolve()}")
        return str(output_file.resolve())

    @property
    def sample_rate(self) -> Optional[int]:
        if not self._models:
            return 24000
        model = next(iter(self._models.values()))
        return int(getattr(model, "sample_rate", 24000))