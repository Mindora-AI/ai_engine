"""Kinyarwanda Health ASR via Hugging Face transformers (independent of faster-whisper)."""

from pathlib import Path
from typing import Union

MODEL_ID = "ElvisTata2024/Kinyarwanda-Health-ASR"


class KinyaHealthASREngine:
    """
    Adapter for Kinyarwanda Health ASR. Loads once and reuses the HF pipeline.
    """

    _shared_pipeline = None

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._pipeline = KinyaHealthASREngine._shared_pipeline

    def load_model(self):
        if self._pipeline is not None:
            return self._pipeline

        if KinyaHealthASREngine._shared_pipeline is not None:
            self._pipeline = KinyaHealthASREngine._shared_pipeline
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers is required for Kinyarwanda ASR. Please install transformers."
            ) from e

        import torch

        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading ASR model: '{self.model_id}' (device={device})...")
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device,
        )
        KinyaHealthASREngine._shared_pipeline = self._pipeline
        print("ASR model loaded successfully!")
        return self._pipeline

    def transcribe_audio(self, audio_input: Union[str, Path]) -> str:
        if not audio_input:
            return ""

        audio_path = Path(audio_input)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path.resolve()}")

        import torch

        pipe = self.load_model()
        with torch.inference_mode():
            result = pipe(str(audio_path.resolve()))

        if not result or not isinstance(result, dict):
            return ""

        return (result.get("text") or "").strip()