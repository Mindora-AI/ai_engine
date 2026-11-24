"""Regression tests for Stage 1 memory/summarization bug fixes."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.service import MWIZERWAApiService
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.conversation import ConversationManager, OversizedMessageError
from src.memory.consent import ConsentDecision, MockConsentGateway
from src.memory.coordinator import MemoryCoordinator
from src.memory.in_memory_repository import InMemoryMemoryRepository
from src.memory.types import MemoryProposal
from src.llm import rolling_summary as rolling_summary_mod


class TestAuxiliaryInferenceBudget(unittest.TestCase):
    def _bare_llm(self) -> AfriqueQwenLLM:
        llm = AfriqueQwenLLM.__new__(AfriqueQwenLLM)
        llm.n_ctx = 2048
        llm.max_tokens = 192
        llm.temperature = 0.7
        llm.top_p = 0.9
        llm.last_effective_max_tokens = None
        llm.last_finish_reason = None
        llm.count_tokens = MagicMock(return_value=100)
        llm.load_model = MagicMock()
        backend = MagicMock()
        llm.load_model.return_value = backend
        backend.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"summary_text":"ok"}'}}]
        }
        return llm

    def test_effective_max_tokens_completion_cap(self):
        llm = self._bare_llm()
        effective = llm._effective_max_tokens(
            [{"role": "user", "content": "hi"}],
            prompt_tokens=100,
            completion_cap=64,
        )
        self.assertEqual(effective, 64)

    def test_generate_auxiliary_json_uses_completion_cap_without_typeerror(self):
        llm = self._bare_llm()
        schema = {
            "type": "object",
            "properties": {"summary_text": {"type": "string"}},
            "required": ["summary_text"],
        }
        data = llm.generate_auxiliary_json(
            [{"role": "user", "content": "fold"}],
            schema=schema,
            max_tokens=128,
        )
        self.assertEqual(data["summary_text"], "ok")
        llm.load_model.return_value.create_chat_completion.assert_called_once()
        _, kwargs = llm.load_model.return_value.create_chat_completion.call_args
        self.assertLessEqual(kwargs["max_tokens"], 128)


class TestConsentSecurity(unittest.TestCase):
    def setUp(self):
        self.repo = InMemoryMemoryRepository()
        self.consent = MockConsentGateway()
        self.llm = MagicMock()
        self.llm.generate_auxiliary_json.return_value = {"summary_text": "merged"}
        self.coord = MemoryCoordinator(
            repository=self.repo, consent=self.consent, llm=self.llm
        )
        self.proposal = MemoryProposal(
            proposal_id="p1",
            content="User prefers morning chats",
            rationale="Stated preference",
        )
        self.consent.register_pending("alice", self.proposal)

    def test_accept_without_resubmitting_proposal_content(self):
        summary = self.coord.apply_consent_decision(
            "alice", "p1", ConsentDecision.ACCEPTED
        )
        self.assertEqual(summary, "merged")
        self.assertEqual(self.repo.get_summary("alice").summary_text, "merged")

    def test_cross_user_rejected(self):
        self.assertIsNone(
            self.coord.apply_consent_decision("bob", "p1", ConsentDecision.ACCEPTED)
        )
        self.assertIsNone(self.repo.get_summary("bob"))
        self.assertTrue(self.consent.fetch_pending("alice", "p1") is not None)

    def test_replay_rejected(self):
        self.coord.apply_consent_decision("alice", "p1", ConsentDecision.DECLINED)
        self.assertTrue(self.consent.is_resolved("p1"))
        self.assertIsNone(
            self.coord.apply_consent_decision("alice", "p1", ConsentDecision.ACCEPTED)
        )
        self.assertIsNone(self.repo.get_summary("alice"))

    def test_failed_merge_keeps_pending(self):
        self.consent.register_pending(
            "alice",
            MemoryProposal(
                proposal_id="p2",
                content="Another fact here",
                rationale="Because",
            ),
        )
        self.llm.generate_auxiliary_json.return_value = None
        self.assertIsNone(
            self.coord.apply_consent_decision("alice", "p2", ConsentDecision.ACCEPTED)
        )
        self.assertIsNotNone(self.consent.fetch_pending("alice", "p2"))

    def test_delete_invalidates_pending(self):
        self.coord.delete_user_memory("alice")
        self.assertIsNone(self.consent.fetch_pending("alice", "p1"))

    def test_failed_repository_update_keeps_pending(self):
        class FailingRepo(InMemoryMemoryRepository):
            def update_summary(self, user_id: str, summary_text: str):
                raise RuntimeError("storage unavailable")

        repo = FailingRepo()
        coord = MemoryCoordinator(
            repository=repo, consent=self.consent, llm=self.llm
        )
        self.consent.register_pending(
            "alice",
            MemoryProposal(
                proposal_id="p3",
                content="Stable preference",
                rationale="User said so",
            ),
        )
        with self.assertRaises(RuntimeError):
            coord.apply_consent_decision("alice", "p3", ConsentDecision.ACCEPTED)
        self.assertIsNotNone(self.consent.fetch_pending("alice", "p3"))


class TestRollingSummaryDataLoss(unittest.TestCase):
    def test_failed_merge_preserves_history(self):
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.n_ctx = 120
        mock_llm.max_tokens = 40

        def _count(msgs):
            return 500 if len(msgs) > 3 else 30

        mock_llm.count_tokens.side_effect = _count
        mock_llm.generate_auxiliary_json.return_value = None
        mgr = ConversationManager(
            llm=mock_llm,
            n_ctx=120,
            max_tokens=40,
            system_prompt="Short.",
        )
        mgr.verbatim_tail_turns = 1
        mgr.rolling_summary_enabled = True
        mgr.messages.extend(
            [
                {"role": "user", "content": "old-one"},
                {"role": "assistant", "content": "reply-one"},
                {"role": "user", "content": "recent"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        before = list(mgr.messages)
        with self.assertRaises(OversizedMessageError):
            mgr.generate_response("new user message here")
        self.assertEqual(mgr.messages, before)
        self.assertFalse(mgr.rolling_summary.strip())

    def test_oversized_rolling_summary_rejected(self):
        huge = "x" * (rolling_summary_mod.DEFAULT_ROLLING_SUMMARY_MAX_CHARS + 10)
        with patch.dict(
            os.environ,
            {"MWIZERWA_ROLLING_SUMMARY_MAX_CHARS": str(rolling_summary_mod.DEFAULT_ROLLING_SUMMARY_MAX_CHARS)},
        ):
            mock_llm = MagicMock()
            mock_llm.generate_auxiliary_json.return_value = {"summary_text": huge}
            from src.llm.rolling_summary import merge_turns_into_summary

            self.assertIsNone(
                merge_turns_into_summary(
                    mock_llm,
                    "",
                    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
                )
            )


class TestDevConsentEndpoint(unittest.TestCase):
    def setUp(self):
        self.service = MWIZERWAApiService()
        self.mock_llm = MagicMock()
        self.service._llm = self.mock_llm
        self.service._initialized = True
        self.service.memory_coordinator.set_llm(self.mock_llm)
        contract = json.dumps(
            {
                "response_text": "OK",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        self.mock_llm.generate_completion.return_value = contract
        self.mock_llm.generate_auxiliary_json.side_effect = [
            {
                "proposals": [
                    {
                        "content": "User likes evening chats",
                        "rationale": "preference",
                        "category": "preference",
                        "confidence": "medium",
                        "source_turn_snippet": "evening",
                    }
                ]
            },
            {"summary_text": "Evening preference"},
        ]
        self._patch = patch("src.api.app.get_api_service", return_value=self.service)
        self._patch.start()
        async def _noop(app_):
            yield
        app.router.lifespan_context = _noop
        self.client = TestClient(app)

    def tearDown(self):
        self._patch.stop()

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MWIZERWA_ALLOW_DEV_MEMORY_CONSENT", None)
            os.environ.pop("MWIZERWA_ENV", None)
            r = self.client.post(
                "/memory/consent",
                json={"user_id": "u1", "proposal_id": "x", "decision": "declined"},
            )
            self.assertEqual(r.status_code, 404)

    def test_http_acceptance_uses_registered_proposal(self):
        with patch.dict(os.environ, {"MWIZERWA_ALLOW_DEV_MEMORY_CONSENT": "1"}):
            chat = self.client.post(
                "/chat",
                json={"user_text": "I prefer evening chats", "user_id": "u1"},
            )
            pid = chat.json()["metadata"]["memory_proposals"][0]["proposal_id"]
            r = self.client.post(
                "/memory/consent",
                json={"user_id": "u1", "proposal_id": pid, "decision": "accepted"},
            )
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json()["summary_updated"])
            self.assertEqual(
                self.service.memory_coordinator.get_approved_summary_text("u1"),
                "Evening preference",
            )


if __name__ == "__main__":
    unittest.main()