"""ASR routing: Kinyarwanda HF vs optional faster-whisper."""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.asr.asr_engine import ASREngine, resolve_asr_language
from src.asr.exceptions import UnsupportedASRLanguageError, WhisperASRNotAvailableError
from src.asr.kinya_health_asr import KinyaHealthASREngine
from src.asr.whisper_asr import (
    DEFAULT_WHISPER_MODEL,
    WhisperASREngine,
    WhisperTranscriptionResult,
)


class TestASRLanguageResolution(unittest.TestCase):
    def test_default_rw(self):
        self.assertEqual(resolve_asr_language(None), "rw")

    def test_explicit_codes(self):
        self.assertEqual(resolve_asr_language("EN"), "en")
        self.assertEqual(resolve_asr_language("fr"), "fr")

    def test_unsupported_raises(self):
        with self.assertRaises(UnsupportedASRLanguageError):
            resolve_asr_language("de")


class TestASREngineRouting(unittest.TestCase):
    def setUp(self):
        self.mock_kinya = MagicMock(spec=KinyaHealthASREngine)
        self.mock_kinya.transcribe_audio.return_value = "Muraho"
        self.mock_whisper = MagicMock(spec=WhisperASREngine)
        self.mock_whisper.transcribe_audio.return_value = "Hello"
        self.engine = ASREngine(
            kinya_engine=self.mock_kinya,
            whisper_engine=self.mock_whisper,
        )

    def test_rw_default_uses_kinya(self):
        text = self.engine.transcribe_audio("x.wav")
        self.assertEqual(text, "Muraho")
        self.mock_kinya.transcribe_audio.assert_called_once()
        self.mock_whisper.transcribe_audio.assert_not_called()

    def test_rw_explicit(self):
        self.engine.transcribe_audio("x.wav", language="rw")
        self.mock_kinya.transcribe_audio.assert_called_once()

    def test_en_routes_whisper(self):
        text = self.engine.transcribe_audio("x.wav", language="en")
        self.assertEqual(text, "Hello")
        self.mock_whisper.transcribe_audio.assert_called_once_with("x.wav", language="en")
        self.mock_kinya.transcribe_audio.assert_not_called()

    def test_fr_routes_whisper(self):
        self.engine.transcribe_audio("x.wav", language="fr")
        self.mock_whisper.transcribe_audio.assert_called_with("x.wav", language="fr")

    def test_unsupported_language(self):
        with self.assertRaises(UnsupportedASRLanguageError):
            self.engine.transcribe_audio("x.wav", language="es")


class TestKinyaHealthUnchanged(unittest.TestCase):
    def test_empty_input(self):
        engine = KinyaHealthASREngine()
        self.assertEqual(engine.transcribe_audio(""), "")

    def test_missing_file(self):
        engine = KinyaHealthASREngine()
        with self.assertRaises(FileNotFoundError):
            engine.transcribe_audio("nonexistent-mwizerwa-asr.wav")


class TestWhisperDefaultModel(unittest.TestCase):
    def test_default_model_is_small_when_env_unset(self):
        env = os.environ.copy()
        env.pop("MWIZERWA_WHISPER_MODEL", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                os.environ.get("MWIZERWA_WHISPER_MODEL", "small"),
                "small",
            )

    def test_engine_default_uses_small(self):
        engine = WhisperASREngine()
        self.assertEqual(engine.model_size, DEFAULT_WHISPER_MODEL)
        if "MWIZERWA_WHISPER_MODEL" not in os.environ:
            self.assertEqual(engine.model_size, "small")


class TestWhisperLazyAndReuse(unittest.TestCase):
    def tearDown(self):
        WhisperASREngine._shared_model = None
        WhisperASREngine._shared_key = None

    def test_lazy_load_and_reuse(self):
        fake = MagicMock()
        seg = MagicMock()
        seg.text = " Bonjour "
        fake.transcribe.return_value = ([seg], MagicMock(language="fr", language_probability=0.9, duration=1.0))

        engine = WhisperASREngine(model_size="small")
        engine._model = fake
        WhisperASREngine._shared_model = fake
        WhisperASREngine._shared_key = engine._cache_key()

        with patch.object(Path, "exists", return_value=True):
            t1 = engine.transcribe_audio("a.wav", language="fr")
            t2 = engine.transcribe_audio("a.wav", language="fr")
        self.assertEqual(t1, "Bonjour")
        self.assertEqual(t2, "Bonjour")
        fake.transcribe.assert_called()

    def test_missing_dependency(self):
        engine = WhisperASREngine()
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with self.assertRaises(WhisperASRNotAvailableError):
                engine.load_model()


class TestWhisperTranscriptionErrors(unittest.TestCase):
    def test_missing_audio_file(self):
        engine = WhisperASREngine()
        with self.assertRaises(FileNotFoundError):
            engine.transcribe_audio("missing-file.wav", language="en")

    def test_empty_segments(self):
        fake = MagicMock()
        fake.transcribe.return_value = ([], MagicMock(language="en", language_probability=0.5, duration=1.0))
        engine = WhisperASREngine()
        engine._model = fake
        with patch.object(Path, "exists", return_value=True):
            self.assertEqual(engine.transcribe_audio("a.wav", language="en"), "")


if __name__ == "__main__":
    unittest.main()