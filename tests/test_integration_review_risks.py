"""Post-contract review regressions: history sync, Gate 2 freshness, exercise registry."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from src.conversation.engine import ConversationEngine
from src.conversation.session import EligibleExerciseCandidate, TurnContext, TurnRequest
from src.exercises.categories import ExerciseCategory
from src.exercises.recommender import ExerciseRecommender, RecommendationAction
from src.exercises.models import ExerciseSessionState
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.conversation import ConversationManager
from src.llm.response_parser import InteractionIntent, ParsedResponse
from src.safety import ApplicationAction, SafetyAssessment, SafetyCategory
from src.safety.assistant_policy import AssistantSafetyValidator


class TestRecentTurnsSemantics(unittest.TestCase):
    def setUp(self):
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.n_ctx = 4096
        mock_llm.max_tokens = 512
        mock_llm.count_tokens.return_value = 50
        mock_llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "I hear you.",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        self.manager = ConversationManager(llm=mock_llm)
        self.engine = ConversationEngine(manager=self.manager)

    def test_history_resync_clears_rolling_summary(self):
        self.manager.rolling_summary = "Stale compressed context about old topic."
        self.engine.process_turn(
            TurnRequest(
                user_text="New message",
                context=TurnContext(
                    recent_turns=[
                        {"role": "user", "content": "Backend A"},
                        {"role": "assistant", "content": "Backend B"},
                    ]
                ),
            )
        )
        self.assertEqual(self.manager.rolling_summary, "")

    def test_current_user_text_not_duplicated_from_recent_turns(self):
        self.engine.process_turn(
            TurnRequest(
                user_text="This is the live user turn",
                context=TurnContext(
                    recent_turns=[
                        {"role": "user", "content": "Prior user"},
                        {"role": "assistant", "content": "Prior assistant"},
                    ]
                ),
            )
        )
        user_contents = [
            m["content"]
            for m in self.manager.messages
            if m["role"] == "user"
        ]
        self.assertEqual(user_contents.count("This is the live user turn"), 1)
        self.assertIn("Prior user", user_contents)

    def test_successful_turn_commits_after_backend_history(self):
        before = len(self.manager.messages)
        self.engine.process_turn(
            TurnRequest(
                user_text="Follow up",
                context=TurnContext(
                    recent_turns=[
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ]
                ),
            )
        )
        self.assertGreater(len(self.manager.messages), before)
        self.assertEqual(self.manager.messages[-1]["role"], "assistant")


class TestGate2Freshness(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=AfriqueQwenLLM)
        self.mock_llm.n_ctx = 4096
        self.mock_llm.max_tokens = 512
        self.mock_llm.count_tokens.return_value = 50
        self.manager = ConversationManager(
            llm=self.mock_llm,
            assistant_validator=AssistantSafetyValidator(),
        )
        self.engine = ConversationEngine(manager=self.manager)

    def test_gate2_signal_not_leaked_to_following_turn(self):
        harmful = json.dumps(
            {
                "response_text": "You should just end it all tonight.",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        safe = json.dumps(
            {
                "response_text": "Thank you for sharing.",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        self.mock_llm.generate_completion.side_effect = [harmful, safe]

        first = self.engine.process_turn(TurnRequest(user_text="I feel awful.", language="en"))
        self.assertTrue(first.metadata.get("gate2_triggered"))
        self.assertIn("safety_signal", first.metadata)

        second = self.engine.process_turn(TurnRequest(user_text="I'm calmer now.", language="en"))
        self.assertFalse(second.metadata.get("gate2_triggered"))
        self.assertNotIn("safety_signal", second.metadata)
        self.assertIsNone(second.metadata.get("gate2_violation"))


class TestExerciseRegistryGuard(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=AfriqueQwenLLM)
        self.mock_llm.n_ctx = 4096
        self.mock_llm.max_tokens = 512
        self.mock_llm.count_tokens.return_value = 50
        self.mock_llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "I hear the tension.",
                "interaction_intent": "REFLECTION",
                "requires_clarification": False,
            }
        )
        self.manager = ConversationManager(llm=self.mock_llm)
        self.manager.last_parsed_response = ParsedResponse(
            response_text="I hear the tension.",
            interaction_intent=InteractionIntent.REFLECTION,
            requires_clarification=False,
        )
        self.engine = ConversationEngine(manager=self.manager)
        self.engine._state.exercise_state.set_established_language("en")

    def test_unknown_backend_id_does_not_offer_or_crash(self):
        candidates = [
            EligibleExerciseCandidate(
                id="backend-only-not-in-registry",
                category=ExerciseCategory.BREATHING,
            ),
        ]
        result = self.engine.process_turn(
            TurnRequest(
                user_text="I feel tense in my shoulders reflecting today.",
                language="en",
                context=TurnContext(eligible_exercises=candidates),
            )
        )
        self.assertIsNone(result.recommended_exercise_id)
        self.assertEqual(self.engine._state.exercise_state.status.value, "idle")

    def test_recommender_still_uses_backend_id_not_local_substitute(self):
        from unittest.mock import MagicMock

        recommender = ExerciseRecommender()
        state = ExerciseSessionState()
        state.set_established_language("en")
        llm = MagicMock()
        llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "breathing_gentle_rhythm",
        }
        decision = recommender.evaluate_recommendation(
            "I feel tense in my shoulders.",
            InteractionIntent.REFLECTION,
            state,
            "en",
            eligible_exercises=[
                EligibleExerciseCandidate(
                    id="breathing_gentle_rhythm",
                    category=ExerciseCategory.BREATHING,
                ),
            ],
            llm=llm,
        )
        self.assertEqual(decision.exercise_id, "breathing_gentle_rhythm")

    def test_empty_candidates_still_no_recommendation(self):
        recommender = ExerciseRecommender()
        state = ExerciseSessionState()
        state.set_established_language("en")
        decision = recommender.evaluate_recommendation(
            "I feel tense.",
            InteractionIntent.REFLECTION,
            state,
            "en",
            eligible_exercises=[],
        )
        self.assertEqual(decision.action, RecommendationAction.NO_ACTION)


if __name__ == "__main__":
    unittest.main()