"""FastAPI layer: health, chat, session isolation (mocked LLM)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.service import MWIZERWAApiService


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

        self._service_patcher = patch(
            "src.api.app.get_api_service",
            return_value=self.service,
        )
        self._service_patcher.start()

        # Skip real model load during TestClient lifespan
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

        engine_a = self.service._sessions[sid_a]
        engine_b = self.service._sessions[sid_b]
        self.assertIsNot(engine_a, engine_b)
        self.assertIs(engine_a.manager.llm, engine_b.manager.llm)

    def test_startup_loads_model_once(self):
        svc = MWIZERWAApiService()
        mock_llm = MagicMock()
        with patch("src.api.service.AfriqueQwenLLM", return_value=mock_llm):
            svc.startup()
        mock_llm.load_model.assert_called_once()
        self.assertTrue(svc.engine_initialized)


if __name__ == "__main__":
    unittest.main()