import wave
from pathlib import Path
from typing import Union

from src.utils.language_detector import detect_language

from src.runtime.model_paths import resolve_kinya_flex_checkpoint

DEFAULT_MODEL_PATH = resolve_kinya_flex_checkpoint()
DEFAULT_SAMPLE_RATE = 24000

# KinyaFlex speaker 0 (female default) — fixed implementation detail.
KINYAFLEX_SPEAKER_ID = 0


class KinyaFlexTTSEngine:
    """
    Adapter for KinyaFlex TTS text-to-speech synthesis (Kinyarwanda only).
    Loads the checkpoint once and reuses it across synthesize calls.
    """

    _shared_model = None

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self._model = KinyaFlexTTSEngine._shared_model
        self._device = None

    def load_model(self):
        """Loads FlexKinyaTTS model if not already loaded."""
        if self._model is not None:
            return self._model

        if KinyaFlexTTSEngine._shared_model is not None:
            self._model = KinyaFlexTTSEngine._shared_model
            return self._model

        import torch

        if self._device is None:
            self._device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "KinyaFlex TTS model file not found at: "
                f"{self.model_path.resolve()}\n"
                "Download it with: python scripts/download_models.py --model tts"
            )

        try:
            from deepkin.models.flex_tts import FlexKinyaTTS
        except ImportError as e:
            raise ImportError(
                "deepkin package is required for KinyaFlex TTS. Ensure DeepKIN-AgAI is installed."
            ) from e

        print(
            f"Loading KinyaFlex TTS model from: {self.model_path} "
            f"(device={self._device})..."
        )
        model = FlexKinyaTTS.from_pretrained(
            self._device,
            str(self.model_path.resolve()),
        )
        model.eval()
        self._model = model
        KinyaFlexTTSEngine._shared_model = model
        print("KinyaFlex TTS model loaded successfully!")
        return self._model

    def synthesize_speech(
        self,
        text: str,
        output_path: Union[str, Path],
        check_lang: bool = True,
    ) -> str:
        """
        Synthesizes Kinyarwanda text into a WAV file at output_path.
        Raises FileNotFoundError / ImportError / ValueError on failure.
        """
        if not text or not text.strip():
            raise ValueError("KinyaFlex synthesis requires non-empty text.")

        if check_lang:
            lang_result = detect_language(text)
            if lang_result.language != "rw" or not lang_result.is_reliable:
                raise ValueError(
                    "KinyaFlex TTS only supports reliable Kinyarwanda (rw) text."
                )

        import numpy as np
        import torch

        model = self.load_model()

        from deepkin.data.kinya_norm import text_to_sequence
        from deepkin.modules.tts_commons import intersperse

        text_id_sequence = intersperse(
            text_to_sequence(text, norm=True),
            0,
        )

        if len(text_id_sequence) == 0:
            raise ValueError("Text converted to empty KinyaFlex sequence.")

        with torch.no_grad():
            audio_data = model(text_id_sequence, KINYAFLEX_SPEAKER_ID)

        audio = audio_data.detach().cpu().numpy()
        audio = np.squeeze(audio)
        audio = np.clip(audio, -1.0, 1.0)
        pcm_audio = (audio * 32767).astype(np.int16)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_file), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_audio.tobytes())

        print(f"[KinyaFlex TTS] Synthesized audio saved to: {output_file.resolve()}")
        return str(output_file.resolve())