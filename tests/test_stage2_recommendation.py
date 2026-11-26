"""Stage 2: recommendation policy, context, suitability, Gate 2."""

import json
import unittest
from unittest.mock import MagicMock

from src.conversation.engine import ConversationEngine
from src.conversation.session import EligibleExerciseCandidate, TurnRequest
from src.exercises.categories import ExerciseCategory
from src.exercises.models import ExerciseSessionState
from src.exercises.recommendation_policy import unsolicited_offer_allowed
from src.exercises.recommender import ExerciseRecommender, RecommendationAction
from src.exercises import ExerciseStatus
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.conversation import ConversationManager
from src.llm.response_parser import InteractionIntent
from src.safety.categories import ApplicationAction, SafetyCategory


class TestGate1RecommendationPolicy(unittest.TestCase):
    def test_non_normal_action_blocks_unsolicited(self):
        self.assertFalse(
            unsolicited_offer_allowed(
                ApplicationAction.SAFETY_FOCUSED_RESPONSE,
                SafetyCategory.SAFE_CONVERSATIONAL,
            )
        )

    def test_ordinary_distress_blocks_even_on_normal_conversation(self):
        self.assertFalse(
            unsolicited_offer_allowed(
                ApplicationAction.NORMAL_CONVERSATION,
                SafetyCategory.ORDINARY_DISTRESS,
            )
        )

    def test_safe_normal_allows_policy_check_pass(self):
        self.assertTrue(
            unsolicited_offer_allowed(
                ApplicationAction.NORMAL_CONVERSATION,
                SafetyCategory.SAFE_CONVERSATIONAL,
            )
        )


class TestGate2SuppressesRecommendation(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=AfriqueQwenLLM)
        self.mock_llm.n_ctx = 4096
        self.mock_llm.max_tokens = 512
        self.mock_llm.count_tokens.return_value = 50
        self.manager = ConversationManager(llm=self.mock_llm)
        self.engine = ConversationEngine(manager=self.manager)
        self.engine._state.exercise_state.set_established_language("en")

    def test_gate2_no_recommendation_lifecycle_idle(self):
        self.mock_llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "I want to hurt myself right now.",
                "interaction_intent": "REFLECTION",
                "requires_clarification": False,
            }
        )
        self.mock_llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "breathing_gentle_rhythm",
        }
        res = self.engine.process_turn(
            TurnRequest(
                user_text="I want to hurt myself right now.",
                language="en",
            )
        )
        self.assertFalse(res.response_text.lower().startswith("i'd like to offer"))
        self.assertIsNone(res.recommended_exercise_id)
        self.assertEqual(
            self.engine.state.exercise_state.status,
            ExerciseStatus.IDLE,
        )


class TestContextAndSuitability(unittest.TestCase):
    def setUp(self):
        self.session = ExerciseSessionState()
        self.recommender = ExerciseRecommender()

    def test_reluctance_blocks_offer(self):
        ctx = {
            "recent_turns": [
                {"role": "user", "content": "I don't want an exercise today."},
                {"role": "assistant", "content": "I understand."},
            ],
            "rolling_summary": None,
        }
        d = self.recommender.evaluate_recommendation(
            "I still feel tense though.",
            InteractionIntent.REFLECTION,
            self.session,
            "en",
            conversation_context=ctx,
            safety_action=ApplicationAction.NORMAL_CONVERSATION,
            safety_category=SafetyCategory.SAFE_CONVERSATIONAL,
        )
        self.assertEqual(d.action, RecommendationAction.NO_ACTION)

    def test_purpose_summaries_influence_llm_selection(self):
        mock_llm = MagicMock()
        mock_llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "breathing_gentle_rhythm",
        }
        candidates = [
            EligibleExerciseCandidate(
                id="breathing_gentle_rhythm",
                category=ExerciseCategory.BREATHING,
                purpose_summary="calm breathing for anxiety",
            ),
            EligibleExerciseCandidate(
                id="grounding_five_senses",
                category=ExerciseCategory.GROUNDING,
                purpose_summary="sensory grounding when overwhelmed",
            ),
        ]
        d = self.recommender.evaluate_recommendation(
            "I feel anxious",
            InteractionIntent.REFLECTION,
            self.session,
            "en",
            eligible_exercises=candidates,
            llm=mock_llm,
            safety_action=ApplicationAction.NORMAL_CONVERSATION,
            safety_category=SafetyCategory.SAFE_CONVERSATIONAL,
        )
        self.assertEqual(d.exercise_id, "breathing_gentle_rhythm")
        user_content = mock_llm.generate_auxiliary_json.call_args[0][0][1]["content"]
        self.assertIn("calm breathing", user_content)

    def test_no_suitable_returns_no_action(self):
        mock_llm = MagicMock()
        mock_llm.generate_auxiliary_json.return_value = {"recommended_exercise_id": None}
        d = self.recommender.evaluate_recommendation(
            "Just chatting",
            InteractionIntent.LISTENING,
            self.session,
            "en",
            llm=mock_llm,
            safety_action=ApplicationAction.NORMAL_CONVERSATION,
            safety_category=SafetyCategory.SAFE_CONVERSATIONAL,
        )
        self.assertEqual(d.action, RecommendationAction.NO_ACTION)

    def test_invalid_model_id_fail_closed(self):
        mock_llm = MagicMock()
        mock_llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "not-in-list",
        }
        candidates = [
            EligibleExerciseCandidate(id="breathing_gentle_rhythm", category=ExerciseCategory.BREATHING),
        ]
        d = self.recommender.evaluate_recommendation(
            "tense",
            InteractionIntent.REFLECTION,
            self.session,
            "en",
            eligible_exercises=candidates,
            llm=mock_llm,
            safety_action=ApplicationAction.NORMAL_CONVERSATION,
            safety_category=SafetyCategory.SAFE_CONVERSATIONAL,
        )
        self.assertEqual(d.action, RecommendationAction.NO_ACTION)


class TestOrdinaryDistressNoUnsolicited(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=AfriqueQwenLLM)
        self.mock_llm.n_ctx = 4096
        self.mock_llm.max_tokens = 512
        self.mock_llm.count_tokens.return_value = 50
        self.manager = ConversationManager(llm=self.mock_llm)
        self.engine = ConversationEngine(manager=self.manager)
        self.engine._state.exercise_state.set_established_language("en")

    def test_ordinary_distress_suppresses_offer(self):
        # Phrase matched by evaluator _ordinary_distress (e.g. overwhelmed / hopeless everyday distress)
        self.mock_llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "That sounds really hard.",
                "interaction_intent": "REFLECTION",
                "requires_clarification": False,
            }
        )
        self.mock_llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "breathing_gentle_rhythm",
        }
        res = self.engine.process_turn(
            TurnRequest(
                user_text="I feel so overwhelmed and hopeless about everything lately, my chest is tight.",
                language="en",
            )
        )
        self.assertIsNone(res.recommended_exercise_id)
        self.assertEqual(
            self.engine.state.exercise_state.status,
            ExerciseStatus.IDLE,
        )


if __name__ == "__main__":
    unittest.main()