import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.tts.tts_engine import (
    TTSEngine,
    TTSRouter,
    UnsupportedTTSLanguageError,
    resolve_response_language,
)
from src.tts.tts_router import default_output_path, latest_response_path
from src.tts.kinya_flex_engine import KinyaFlexTTSEngine
from src.tts.piper_engine import PiperTTSEngine
from run_assistant import transcribe_and_respond


def _write_stub_wav(output_path) -> str:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * 10)
    return str(p.resolve())


class TestTTSRouting(unittest.TestCase):
    def test_resolve_response_language_rw(self):
        lang = resolve_response_language("Muraho neza, umeze ute?")
        self.assertEqual(lang, "rw")

    def test_resolve_response_language_en(self):
        lang = resolve_response_language("Hello, I am here to listen.")
        self.assertEqual(lang, "en")

    def test_resolve_response_language_fr(self):
        lang = resolve_response_language("Bonjour, comment allez-vous aujourd'hui?")
        self.assertEqual(lang, "fr")

    def test_explicit_language_override(self):
        self.assertEqual(resolve_response_language("any text", language="en"), "en")

    def test_unreliable_language_raises(self):
        with self.assertRaises(UnsupportedTTSLanguageError):
            resolve_response_language("123 456")

    @patch.object(
        KinyaFlexTTSEngine,
        "synthesize_speech",
        side_effect=lambda text, output_path, **kw: _write_stub_wav(output_path),
    )
    @patch.object(PiperTTSEngine, "synthesize_speech")
    def test_router_rw_to_kinya(self, mock_piper, mock_kinya):
        router = TTSRouter()
        out = router.synthesize_speech(
            "Muraho neza, umeze ute?",
            output_path=Path(tempfile.gettempdir()) / "rw_test.wav",
        )
        mock_kinya.assert_called_once()
        mock_piper.assert_not_called()
        self.assertTrue(Path(out).is_file())

    @patch.object(KinyaFlexTTSEngine, "synthesize_speech")
    @patch.object(
        PiperTTSEngine,
        "synthesize_speech",
        side_effect=lambda text, language, output_path: _write_stub_wav(output_path),
    )
    def test_router_en_to_piper(self, mock_piper, mock_kinya):
        router = TTSRouter()
        out = router.synthesize_speech(
            "Hello, thank you for sharing that with me.",
            output_path=Path(tempfile.gettempdir()) / "en_test.wav",
        )
        mock_piper.assert_called_once()
        call_kw = mock_piper.call_args.kwargs
        self.assertEqual(call_kw["language"], "en")
        mock_kinya.assert_not_called()
        self.assertTrue(Path(out).is_file())

    @patch.object(KinyaFlexTTSEngine, "synthesize_speech")
    @patch.object(
        PiperTTSEngine,
        "synthesize_speech",
        side_effect=lambda text, language, output_path: _write_stub_wav(output_path),
    )
    def test_router_fr_to_piper(self, mock_piper, mock_kinya):
        router = TTSRouter()
        router.synthesize_speech(
            "Je suis là pour vous écouter avec attention.",
            output_path=Path(tempfile.gettempdir()) / "fr_test.wav",
        )
        self.assertEqual(mock_piper.call_args.kwargs["language"], "fr")
        mock_kinya.assert_not_called()

    def test_language_switching_in_one_router_instance(self):
        router = TTSRouter()
        mock_kinya = MagicMock(
            side_effect=lambda text, output_path, **kw: _write_stub_wav(output_path)
        )
        mock_piper = MagicMock(
            side_effect=lambda text, language, output_path: _write_stub_wav(output_path)
        )
        router._kinya.synthesize_speech = mock_kinya
        router._piper.synthesize_speech = mock_piper

        out_dir = Path(tempfile.gettempdir())
        router.synthesize_speech("Muraho", output_path=out_dir / "a.wav")
        router.synthesize_speech("Hello there", output_path=out_dir / "b.wav")
        router.synthesize_speech("Bonjour", output_path=out_dir / "c.wav")

        self.assertEqual(mock_kinya.call_count, 1)
        self.assertEqual(mock_piper.call_count, 2)

    def test_missing_piper_model_raises_clear_error(self):
        piper = PiperTTSEngine(voices_root=Path(tempfile.gettempdir()) / "nonexistent_piper")
        with self.assertRaises(FileNotFoundError) as ctx:
            piper.synthesize_speech("Hello", language="en", output_path="x.wav")
        self.assertIn("download_models.py", str(ctx.exception))

    def test_default_output_path_is_latest_response(self):
        path = default_output_path()
        self.assertEqual(path.name, "latest_response.wav")

    def test_piper_writes_latest_response_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            unique = tmp_path / "response_en.wav"
            with wave.open(str(unique), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(b"\x00\x00" * 100)

            def _fake_piper(text, language, output_path):
                shutil.copy2(unique, output_path)
                return str(output_path)

            router = TTSRouter(output_dir=tmp_path)
            router._piper.synthesize_speech = _fake_piper

            with patch.object(KinyaFlexTTSEngine, "synthesize_speech") as mock_kinya:
                router.synthesize_speech("Hello there, how are you today?")
                mock_kinya.assert_not_called()
                latest = latest_response_path(tmp_path)
                self.assertTrue(latest.is_file())
                self.assertGreater(latest.stat().st_size, 0)

    def test_voice_to_voice_calls_synthesize_on_response(self):
        import json
        from src.llm.conversation import ConversationManager
        from src.llm.afrique_qwen import AfriqueQwenLLM
        from src.asr.asr_engine import ASREngine

        mock_tts = MagicMock(spec=TTSEngine)
        mock_tts.synthesize_speech.return_value = "out.wav"
        mock_asr = MagicMock(spec=ASREngine)
        mock_asr.transcribe_audio.return_value = "Hi"

        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "Hello friend",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        manager = ConversationManager(llm=mock_llm)

        transcribe_and_respond(manager, mock_asr, mock_tts, "x.wav")
        mock_tts.synthesize_speech.assert_called_once_with("Hello friend")


if __name__ == "__main__":
    unittest.main()