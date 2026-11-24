import os
from pathlib import Path
from typing import Union, Optional

MODEL_ID = "ElvisTata2024/Kinyarwanda-Health-ASR"


class ASREngine:
    """
    Adapter for Kinyarwanda Health ASR speech recognition model using HuggingFace pipeline.
    Loads model once and reuses instance across all instances for transcription.
    """

    _instance = None
    _shared_pipeline = None

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._pipeline = ASREngine._shared_pipeline

    def load_model(self):
        """Loads HuggingFace ASR pipeline if not already loaded."""
        if self._pipeline is not None:
            return self._pipeline

        if ASREngine._shared_pipeline is not None:
            self._pipeline = ASREngine._shared_pipeline
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers is required for ASR pipeline. Please install transformers."
            ) from e

        import torch

        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading ASR model: '{self.model_id}' (device={device})...")
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device,
        )
        ASREngine._shared_pipeline = self._pipeline
        print("ASR model loaded successfully!")
        return self._pipeline

    def transcribe_audio(self, audio_input: Union[str, Path]) -> str:
        """
        Transcribes audio input (file path) to text.
        Returns empty string if audio is missing, invalid, or transcription is empty.
        """
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

        text = result.get("text", "").strip()
        return text
