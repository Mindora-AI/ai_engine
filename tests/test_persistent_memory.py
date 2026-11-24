import json
import unittest
from unittest.mock import MagicMock

from src.conversation.engine import ConversationEngine
from src.conversation.session import TurnRequest
from src.llm.conversation import ConversationManager
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.memory.consent import ConsentDecision, MockConsentGateway
from src.memory.coordinator import MemoryCoordinator
from src.memory.in_memory_repository import InMemoryMemoryRepository
from src.memory.types import MemoryProposal


class TestPersistentMemory(unittest.TestCase):
    def _engine_with_memory(self, llm):
        repo = InMemoryMemoryRepository()
        consent = MockConsentGateway()
        coordinator = MemoryCoordinator(repository=repo, consent=consent, llm=llm)
        manager = ConversationManager(llm=llm)
        return ConversationEngine(manager=manager, memory_coordinator=coordinator), coordinator, repo

    def test_proposal_requires_consent_before_storage(self):
        llm = MagicMock(spec=AfriqueQwenLLM)
        llm.n_ctx = 4096
        llm.max_tokens = 192
        llm.count_tokens.return_value = 50
        llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "Muraho!",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        llm.generate_auxiliary_json.return_value = {
            "proposals": [
                {
                    "content": "User prefers evening chats",
                    "rationale": "Stated preference",
                    "category": "preference",
                    "confidence": "medium",
                    "source_turn_snippet": "I prefer talking in the evening",
                }
            ]
        }
        engine, coordinator, repo = self._engine_with_memory(llm)
        result = engine.process_turn(
            TurnRequest(user_text="I prefer talking in the evening", user_id="user-a")
        )
        self.assertIn("memory_proposals", result.metadata)
        pid = result.metadata["memory_proposals"][0]["proposal_id"]
        self.assertIsNone(repo.get_summary("user-a"))

        coordinator.apply_consent_decision("user-a", pid, ConsentDecision.DECLINED)
        self.assertIsNone(repo.get_summary("user-a"))

        llm.generate_auxiliary_json.return_value = {
            "proposals": [
                {
                    "content": "User works from home",
                    "rationale": "Stable circumstance",
                    "category": "circumstance",
                    "confidence": "low",
                    "source_turn_snippet": "I work from home",
                }
            ]
        }
        result2 = engine.process_turn(
            TurnRequest(user_text="I work from home", user_id="user-a")
        )
        pid2 = result2.metadata["memory_proposals"][0]["proposal_id"]
        summary = coordinator.apply_consent_decision(
            "user-a", pid2, ConsentDecision.ACCEPTED
        )
        self.assertIsNotNone(summary)
        self.assertEqual(repo.get_summary("user-a").summary_text, summary)

    def test_user_isolation(self):
        repo = InMemoryMemoryRepository()
        repo.update_summary("u1", "Summary one")
        repo.update_summary("u2", "Summary two")
        self.assertNotEqual(
            repo.get_summary("u1").summary_text,
            repo.get_summary("u2").summary_text,
        )
        self.assertTrue(repo.delete_memory("u1"))
        self.assertIsNone(repo.get_summary("u1"))

    def test_approved_memory_injected_without_client_trust(self):
        llm = MagicMock(spec=AfriqueQwenLLM)
        llm.n_ctx = 4096
        llm.max_tokens = 192
        llm.count_tokens.return_value = 40
        llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "OK",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        llm.generate_auxiliary_json.return_value = {"proposals": []}
        engine, coordinator, repo = self._engine_with_memory(llm)
        repo.update_summary("trusted-user", "Works night shifts.")

        engine.process_turn(TurnRequest(user_text="Hello", user_id="trusted-user"))
        call_kwargs = llm.generate_completion.call_args
        system_msg = call_kwargs[0][0][0]["content"]
        self.assertIn("Approved Persistent Memory", system_msg)
        self.assertIn("night shifts", system_msg)

    def test_missing_memory_does_not_break_chat(self):
        llm = MagicMock(spec=AfriqueQwenLLM)
        llm.n_ctx = 4096
        llm.max_tokens = 192
        llm.count_tokens.return_value = 30
        llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "Hi",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        llm.generate_auxiliary_json.return_value = {"proposals": []}
        engine, _, _ = self._engine_with_memory(llm)
        result = engine.process_turn(TurnRequest(user_text="Hi"))
        self.assertEqual(result.response_text, "Hi")


if __name__ == "__main__":
    unittest.main()