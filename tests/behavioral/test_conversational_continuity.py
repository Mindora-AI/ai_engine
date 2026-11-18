"""
MWIZERWA AI — Behavioral Test Suite: Conversational Continuity
Evaluates multi-turn context preservation, topic shifts, sliding-window token management,
interleaved deterministic exercise turns, and conversation reset behavior.
"""

import json
import unittest
from unittest.mock import MagicMock

from src.conversation.engine import ConversationEngine
from src.conversation.session import SessionState, TurnRequest
from src.llm.conversation import ConversationManager
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.response_parser import InteractionIntent
from src.exercises import ExerciseStatus


class TestConversationalContinuity(unittest.TestCase):
    """
    Tests continuity across multi-turn interactions:
    - Context preservation across consecutive turns.
    - Handling topic shifts without memory corruption.
    - Sliding-window context management under token constraints.
    - Interleaving deterministic exercise execution with LLM inference turns.
    - State reset behavior.
    """

    def test_01_multi_turn_context_preserved_in_manager_messages(self):
        """Turn 1 context is available in conversation history for Turn 2."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.side_effect = [
            json.dumps({
                "response_text": "I am so sorry about Max. He was clearly very special to you.",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }),
            json.dumps({
                "response_text": "Grief often disrupts sleep. Would a calm routine before bed help?",
                "interaction_intent": "PRACTICAL_ADVICE",
                "requires_clarification": False,
            }),
        ]
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # Turn 1
        res1 = engine.process_turn(TurnRequest(user_text="My dog Max passed away yesterday."))
        self.assertIn("Max", res1.response_text)

        # Turn 2
        res2 = engine.process_turn(TurnRequest(user_text="I couldn't sleep at all last night."))
        self.assertIn("sleep", res2.response_text)

        # Verify messages: system + (user1, asst1) + (user2, asst2)
        messages = engine.messages
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["content"], "My dog Max passed away yesterday.")
        self.assertIn("Max", messages[2]["content"])
        self.assertEqual(messages[3]["content"], "I couldn't sleep at all last night.")

    def test_02_topic_shift_retains_prior_history(self):
        """User shifts topic from sorrow to practical question; history retains both."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.side_effect = [
            json.dumps({
                "response_text": "C'est difficile de traverser une telle rupture.",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }),
            json.dumps({
                "response_text": "Pour la météo à Kigali, il fait généralement chaud en ce moment.",
                "interaction_intent": "PRACTICAL_ADVICE",
                "requires_clarification": False,
            }),
        ]
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        engine.process_turn(TurnRequest(user_text="Mon ami a rompu avec moi."))
        engine.process_turn(TurnRequest(user_text="Au fait, quel temps fait-il à Kigali ?"))

        self.assertEqual(len(engine.messages), 5)
        self.assertEqual(engine.messages[1]["content"], "Mon ami a rompu avec moi.")
        self.assertEqual(engine.messages[3]["content"], "Au fait, quel temps fait-il à Kigali ?")

    def test_03_token_budget_sliding_window_trims_oldest_turns_preserving_system_prompt(self):
        """Under tight context budget, oldest turn pairs are trimmed while preserving system prompt."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Response text.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        # Set max_turns=2 (retaining at most 2 turn pairs)
        manager = ConversationManager(llm=mock_llm, max_turns=2)
        engine = ConversationEngine(manager=manager)

        engine.process_turn(TurnRequest(user_text="Turn 1"))
        engine.process_turn(TurnRequest(user_text="Turn 2"))
        engine.process_turn(TurnRequest(user_text="Turn 3"))

        messages = engine.messages
        # System prompt always preserved at index 0
        self.assertEqual(messages[0]["role"], "system")
        # With max_turns=2, Turn 1 should have been trimmed; Turn 2 and 3 remain
        user_contents = [m["content"] for m in messages if m["role"] == "user"]
        self.assertNotIn("Turn 1", user_contents)
        self.assertIn("Turn 2", user_contents)
        self.assertIn("Turn 3", user_contents)

    def test_04_interleaved_exercise_and_llm_turns(self):
        """Explicit exercise run to completion, then user resumes normal LLM conversation."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "I am glad the breathing helped you center yourself. What would you like to discuss next?",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # 1. Start explicit breathing exercise (deterministic) with English language
        res1 = engine.process_turn(TurnRequest(user_text="Can we do a breathing exercise?", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)
        self.assertIn("Step 1 of 4", res1.response_text)

        # 2. Advance through all steps (4 steps: Step 1 -> Step 2 -> Step 3 -> Step 4 -> Complete)
        engine.process_turn(TurnRequest(user_text="next", language="en"))
        engine.process_turn(TurnRequest(user_text="next", language="en"))
        engine.process_turn(TurnRequest(user_text="next", language="en"))
        res_comp = engine.process_turn(TurnRequest(user_text="next", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.COMPLETED)

        # 3. Resume normal LLM conversation
        res_llm = engine.process_turn(TurnRequest(user_text="I feel much calmer now.", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IDLE)
        self.assertIn("breathing helped", res_llm.response_text)
        # Verify history includes both deterministic exercise turns and LLM turns
        self.assertGreater(len(engine.messages), 4)

    def test_05_reset_conversation_restores_clean_state(self):
        """reset_conversation clears history and resets active exercise state."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Hello there!",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        engine.process_turn(TurnRequest(user_text="Hello"))
        self.assertEqual(len(engine.messages), 3)

        engine.reset_conversation()
        self.assertEqual(len(engine.messages), 1)
        self.assertEqual(engine.messages[0]["role"], "system")
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IDLE)


if __name__ == "__main__":
    unittest.main()
