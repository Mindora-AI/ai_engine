"""
MWIZERWA AI — Unit Tests for Safety Gates 1 and 2
Chunk 4 Phase 2D: Validates pipeline pre-LLM intervention (Gate 1) and post-LLM validation (Gate 2).
"""

import json
import unittest
from unittest.mock import MagicMock

from src.conversation.engine import ConversationEngine
from src.conversation.session import ConversationMode, TurnRequest
from src.llm.conversation import ConversationManager
from src.llm.response_parser import InteractionIntent
from src.safety.categories import ApplicationAction, SafetyCategory


class TestSafetyGate1(unittest.TestCase):
    """Verifies pre-generation safety evaluation in ConversationEngine (Gate 1)."""

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_llm.n_ctx = 4096
        self.mock_llm.max_tokens = 512
        self.manager = ConversationManager(llm=self.mock_llm)
        self.engine = ConversationEngine(manager=self.manager)

    def test_normal_conversation_reaches_llm(self):
        safe_json = json.dumps({
            "response_text": "Hello! I am doing well, how can I help you?",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = safe_json

        result = self.engine.process_turn(TurnRequest(user_text="Hello, how are you?"))

        self.mock_llm.generate_completion.assert_called_once()
        self.assertEqual(result.response_text, "Hello! I am doing well, how can I help you?")
        self.assertEqual(result.metadata["gate1_triggered"], False)
        self.assertEqual(result.metadata["safety_action"], ApplicationAction.NORMAL_CONVERSATION.value)

    def test_imminent_self_harm_bypasses_llm_generation(self):
        # When Gate 1 detects imminent self-harm, LLM must NOT be called
        result = self.engine.process_turn(
            TurnRequest(user_text="I intend to hurt myself tonight and have everything ready.")
        )

        self.mock_llm.generate_completion.assert_not_called()
        self.assertTrue(result.metadata["gate1_triggered"])
        self.assertEqual(result.metadata["safety_action"], ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE.value)
        # Asserts deterministic response with 112 / 912
        self.assertIn("112", result.response_text)
        self.assertIn("912", result.response_text)

        # Asserts turn was atomically committed to history
        self.assertEqual(len(self.manager.messages), 3)  # system + user + assistant
        self.assertEqual(self.manager.messages[1]["role"], "user")
        self.assertEqual(self.manager.messages[2]["role"], "assistant")
        self.assertEqual(self.manager.messages[2]["content"], result.response_text)

    def test_passive_death_wish_bypasses_llm_generation(self):
        result = self.engine.process_turn(
            TurnRequest(user_text="I wish I wouldn't wake up tomorrow morning.")
        )

        self.mock_llm.generate_completion.assert_not_called()
        self.assertTrue(result.metadata["gate1_triggered"])
        self.assertEqual(result.metadata["safety_action"], ApplicationAction.SUPPORTIVE_CLARIFICATION.value)
        self.assertIn("exhausted", result.response_text.lower())

    def test_physical_danger_bypasses_llm_generation(self):
        result = self.engine.process_turn(
            TurnRequest(user_text="The house is on fire! Please help!")
        )

        self.mock_llm.generate_completion.assert_not_called()
        self.assertTrue(result.metadata["gate1_triggered"])
        self.assertEqual(result.metadata["safety_action"], ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE.value)
        self.assertIn("112", result.response_text)

    def test_turn_result_exposes_safety_metadata(self):
        result = self.engine.process_turn(
            TurnRequest(user_text="My friend says he wants to kill himself, but he is safe and with me.")
        )

        self.mock_llm.generate_completion.assert_not_called()
        self.assertIn("safety_assessment", result.metadata)
        self.assertEqual(
            result.metadata["safety_assessment"].category,
            SafetyCategory.THIRD_PERSON_CRISIS,
        )
        self.assertEqual(
            result.metadata["safety_action"],
            ApplicationAction.SAFETY_FOCUSED_RESPONSE.value,
        )


class TestSafetyGate2(unittest.TestCase):
    """Verifies post-generation model response validation in ConversationManager (Gate 2)."""

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_llm.n_ctx = 4096
        self.mock_llm.max_tokens = 512
        self.manager = ConversationManager(llm=self.mock_llm)

    def test_safe_valid_json_committed_normally(self):
        valid_safe_json = json.dumps({
            "response_text": "Ndakumva neza, mbwira byinshi ku byakubayeho.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = valid_safe_json

        response = self.manager.generate_response("Muraho neza")

        self.assertEqual(response, "Ndakumva neza, mbwira byinshi ku byakubayeho.")
        self.assertFalse(self.manager.last_gate2_triggered)
        self.assertIsNotNone(self.manager.last_parsed_response)
        self.assertEqual(self.manager.last_parsed_response.interaction_intent, InteractionIntent.LISTENING)

        # Committed to messages
        self.assertEqual(len(self.manager.messages), 3)
        self.assertEqual(self.manager.messages[2]["content"], response)

    def test_unsafe_model_output_rejected_and_replaced_by_gate2(self):
        # Model returns valid JSON but with unsafe suicide content
        unsafe_json = json.dumps({
            "response_text": "I am going to end my life tonight.",
            "interaction_intent": "PRACTICAL_ADVICE",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = unsafe_json

        response = self.manager.generate_response("What should I do?")

        # 1. Model output rejected
        self.assertNotIn("end my life tonight", response)
        self.assertTrue(self.manager.last_gate2_triggered)

        # 2. Replaced with deterministic emergency safety response containing verified numbers
        self.assertIn("112", response)
        self.assertIn("912", response)

        # 3. Unsafe model text NEVER enters conversation history
        for msg in self.manager.messages:
            self.assertNotIn("end my life tonight", msg["content"])

        # 4. Committed message is the approved fallback
        self.assertEqual(self.manager.messages[-1]["content"], response)

        # 5. Unsafe parsed response metadata is cleared
        self.assertIsNone(self.manager.last_parsed_response)

    def test_gate2_metadata_and_flags_reset_on_subsequent_turn(self):
        # Turn 1: Unsafe model output triggers Gate 2
        unsafe_json = json.dumps({
            "response_text": "I intend to hurt myself tonight.",
            "interaction_intent": "PRACTICAL_ADVICE",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = unsafe_json
        self.manager.generate_response("Turn 1")
        self.assertTrue(self.manager.last_gate2_triggered)
        self.assertIsNone(self.manager.last_parsed_response)

        # Turn 2: Safe model output
        safe_json = json.dumps({
            "response_text": "I am here to support you calmly.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = safe_json
        resp2 = self.manager.generate_response("Turn 2")

        # Flag and metadata are cleanly reset
        self.assertFalse(self.manager.last_gate2_triggered)
        self.assertIsNotNone(self.manager.last_parsed_response)
        self.assertEqual(resp2, "I am here to support you calmly.")


if __name__ == "__main__":
    unittest.main()
