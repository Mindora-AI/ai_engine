"""FastAPI layer: chat, TTS, ASR, voice (mocked models)."""

from __future__ import annotations

import base64
import io
import json
import os
import unittest
import wave
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.service import MWIZERWAApiService
from src.tts.tts_engine import TTSSynthesisError


def _tiny_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


class TestMWIZERWAApi(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        contract = json.dumps(
            {
                "response_text": "Muraho neza!",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        self.mock_llm.generate_completion.return_value = contract

        self.service = MWIZERWAApiService()
        self.service._llm = self.mock_llm
        self.service._initialized = True

        self.mock_tts = MagicMock()
        self.mock_asr = MagicMock()
        self.mock_asr.transcribe_audio.return_value = "Muraho"

        def _fake_tts(text, output_path=None, **kwargs):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 800)
            return str(path)

        self.mock_tts.synthesize_speech.side_effect = _fake_tts
        self.mock_tts.check_models.return_value = {"present": True, "language": "rw"}

        self.service._tts = self.mock_tts
        self.service._asr = self.mock_asr

        self._service_patcher = patch(
            "src.api.app.get_api_service",
            return_value=self.service,
        )
        self._service_patcher.start()

        async def _noop_lifespan(app_):
            yield

        app.router.lifespan_context = _noop_lifespan
        self.client = TestClient(app)

    def tearDown(self):
        self._service_patcher.stop()

    def test_health_ready_when_initialized(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ready")
        self.assertTrue(data["engine_initialized"])

    def test_health_unavailable_when_not_initialized(self):
        self.service._initialized = False
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "unavailable")

    def test_chat_returns_structured_response(self):
        r = self.client.post("/chat", json={"user_text": "Muraho"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("session_id", data)
        self.assertEqual(data["response_text"], "Muraho neza!")
        self.assertEqual(data["mode"], "normal")
        self.assertEqual(data["metadata"].get("interaction_intent"), "LISTENING")

    def test_chat_rejects_empty_text(self):
        r = self.client.post("/chat", json={"user_text": "   "})
        self.assertEqual(r.status_code, 422)

    def test_chat_503_when_engine_down(self):
        self.service._initialized = False
        r = self.client.post("/chat", json={"user_text": "Hi"})
        self.assertEqual(r.status_code, 503)

    def test_session_isolation(self):
        r1 = self.client.post("/chat", json={"user_text": "Session A turn 1"})
        sid_a = r1.json()["session_id"]
        r2 = self.client.post(
            "/chat",
            json={"user_text": "Session B turn 1", "session_id": "client-b-fixed"},
        )
        sid_b = r2.json()["session_id"]
        self.assertEqual(sid_b, "client-b-fixed")
        self.assertNotEqual(sid_a, sid_b)
        self.assertEqual(self.service.session_count(), 2)

    def test_tts_success(self):
        r = self.client.post(
            "/tts",
            data={"text": "Muraho", "language": "rw"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("content-type"), "audio/wav")
        self.assertTrue(len(r.content) > 44)

    def test_tts_missing_model(self):
        self.mock_tts.check_models.return_value = {"present": False}
        r = self.client.post("/tts", data={"text": "Hello", "language": "en"})
        self.assertEqual(r.status_code, 503)

    def test_transcribe_success(self):
        r = self.client.post(
            "/transcribe",
            files={"audio": ("test.wav", _tiny_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["transcription"], "Muraho")

    def test_transcribe_oversized(self):
        with patch.dict(os.environ, {"MWIZERWA_MAX_UPLOAD_BYTES": "512"}):
            r = self.client.post(
                "/transcribe",
                files={"audio": ("big.wav", b"x" * 2048, "audio/wav")},
            )
        self.assertEqual(r.status_code, 413)

    def test_transcribe_invalid_extension(self):
        r = self.client.post(
            "/transcribe",
            files={"audio": ("evil.txt", b"not audio", "text/plain")},
        )
        self.assertEqual(r.status_code, 400)

    def test_voice_end_to_end(self):
        r = self.client.post(
            "/voice",
            files={"audio": ("u.wav", _tiny_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["transcription"], "Muraho")
        self.assertEqual(data["response_text"], "Muraho neza!")
        self.assertEqual(data["audio_status"], "success")
        self.assertEqual(data["asr_language"], "rw")
        self.assertTrue(data["audio_base64"])
        wav = base64.b64decode(data["audio_base64"])
        self.assertGreater(len(wav), 44)

    def test_text_then_voice_same_session(self):
        r1 = self.client.post("/chat", json={"user_text": "First text turn"})
        sid = r1.json()["session_id"]
        r2 = self.client.post(
            "/voice",
            data={"session_id": sid},
            files={"audio": ("u.wav", _tiny_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["session_id"], sid)

    def test_startup_loads_model_once(self):
        svc = MWIZERWAApiService()
        mock_llm = MagicMock()
        with patch("src.api.service.AfriqueQwenLLM", return_value=mock_llm):
            svc.startup()
        mock_llm.load_model.assert_called_once()
        self.assertTrue(svc.engine_initialized)

    def test_chat_rejects_unsupported_language(self):
        r = self.client.post(
            "/chat",
            json={"user_text": "Hello", "language": "es"},
        )
        self.assertEqual(r.status_code, 422)

    def test_chat_normalizes_language_hint(self):
        r = self.client.post(
            "/chat",
            json={"user_text": "Bonjour", "language": "FR"},
        )
        self.assertEqual(r.status_code, 200)

    def test_chat_rejects_blank_session_id(self):
        r = self.client.post(
            "/chat",
            json={"user_text": "Hi", "session_id": "   "},
        )
        self.assertEqual(r.status_code, 422)

    def test_chat_session_continuity(self):
        r1 = self.client.post("/chat", json={"user_text": "Turn one"})
        sid = r1.json()["session_id"]
        r2 = self.client.post(
            "/chat",
            json={"user_text": "Turn two", "session_id": sid},
        )
        self.assertEqual(r2.json()["session_id"], sid)
        self.assertEqual(self.service.session_count(), 1)

    def test_memory_consent_endpoint_returns_structured_response(self):
        with patch.dict(os.environ, {"MWIZERWA_ALLOW_DEV_MEMORY_CONSENT": "1"}):
            r = self.client.post(
                "/memory/consent",
                json={
                    "user_id": "server-user-1",
                    "proposal_id": "unknown-prop",
                    "decision": "declined",
                },
            )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["user_id"], "server-user-1")
        self.assertEqual(data["proposal_id"], "unknown-prop")
        self.assertFalse(data["summary_updated"])

    def test_chat_internal_error_is_generic(self):
        self.mock_llm.generate_completion.side_effect = RuntimeError("GPU failure")
        r = self.client.post("/chat", json={"user_text": "Muraho"})
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()["detail"], "Unable to process this message. Please try again.")

    def test_transcribe_defaults_language_rw(self):
        r = self.client.post(
            "/transcribe",
            files={"audio": ("test.wav", _tiny_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["language"], "rw")
        self.mock_asr.transcribe_audio.assert_called_with(ANY, language=None)

    def test_transcribe_passes_en_to_whisper_backend(self):
        r = self.client.post(
            "/transcribe",
            data={"language": "en"},
            files={"audio": ("test.wav", _tiny_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["language"], "en")
        self.mock_asr.transcribe_audio.assert_called_with(ANY, language="en")

    def test_transcribe_rejects_unsupported_language(self):
        r = self.client.post(
            "/transcribe",
            data={"language": "de"},
            files={"audio": ("test.wav", _tiny_wav_bytes(), "audio/wav")},
        )
        self.assertIn(r.status_code, (400, 422))

    def test_voice_language_propagates_to_asr_and_chat(self):
        r = self.client.post(
            "/voice",
            data={"language": "fr"},
            files={"audio": ("u.wav", _tiny_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["asr_language"], "fr")
        self.mock_asr.transcribe_audio.assert_called_with(ANY, language="fr")

    def test_voice_tts_failure_partial_success(self):
        self.mock_tts.synthesize_speech.side_effect = TTSSynthesisError("boom")
        before = self.mock_llm.generate_completion.call_count
        r = self.client.post(
            "/voice",
            files={"audio": ("u.wav", _tiny_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["transcription"], "Muraho")
        self.assertEqual(data["response_text"], "Muraho neza!")
        self.assertEqual(data["audio_status"], "failed")
        self.assertIsNone(data["audio_base64"])
        self.assertIn("Speech", data["audio_error"] or "")
        self.assertEqual(self.mock_llm.generate_completion.call_count, before + 1)

    def test_health_includes_backend_readiness(self):
        r = self.client.get("/health")
        data = r.json()
        self.assertIn("backends", data)
        self.assertIn("llm", data["backends"])
        self.assertIn("asr_whisper", data["backends"])
        self.assertIn("tts", data["backends"])
        self.assertTrue(data["backends"]["llm"]["loaded"])


if __name__ == "__main__":
    unittest.main()