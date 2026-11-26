"""Stage 2 hardening: bounded suitability payload and fail-closed unsolicited path."""

import json
import unittest
from unittest.mock import MagicMock

from src.conversation.engine import ConversationEngine
from src.conversation.session import EligibleExerciseCandidate, TurnRequest
from src.llm.conversation import ConversationManager
from src.exercises.categories import ExerciseCategory
from src.exercises.models import ExerciseSessionState
from src.exercises.recommender import ExerciseRecommender, RecommendationAction
from src.exercises.suitability_selection import (
    MAX_SUITABILITY_PAYLOAD_CHARS,
    build_bounded_suitability_payload,
    select_exercise_candidate,
    suitability_payload_char_len,
)
from src.llm.response_parser import InteractionIntent
from src.safety.categories import ApplicationAction, SafetyCategory
from src.exercises.recommendation_policy import unsolicited_offer_allowed


class TestBoundedSuitabilityPayload(unittest.TestCase):
    def _candidate(self, cid: str, purpose: str = "") -> EligibleExerciseCandidate:
        return EligibleExerciseCandidate(
            id=cid,
            category=ExerciseCategory.BREATHING,
            purpose_summary=purpose or None,
        )

    def test_trims_old_turns_before_summary(self):
        huge = "x" * 2000
        turns = [
            {"role": "user", "content": huge},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "recent"},
        ]
        budget = 900
        payload = build_bounded_suitability_payload(
            user_text="current user message",
            interaction_intent=InteractionIntent.REFLECTION,
            language="en",
            candidates=[self._candidate("breathing_gentle_rhythm", "calm breath")],
            conversation_context={
                "recent_turns": turns,
                "rolling_summary": "summary " + ("s" * 500),
            },
            max_chars=budget,
        )
        self.assertIsNotNone(payload)
        self.assertLessEqual(suitability_payload_char_len(payload), budget)
        self.assertEqual(payload["user_message"], "current user message")
        self.assertEqual(payload["eligible_exercises"][0]["id"], "breathing_gentle_rhythm")
        self.assertEqual(payload["eligible_exercises"][0]["category"], "breathing")
        # Oldest huge turn should be removed before fitting.
        self.assertNotIn(huge, json.dumps(payload["recent_turns"]))

    def test_preserves_all_candidate_ids_under_truncation(self):
        candidates = [
            self._candidate("a-1", "p" * 300),
            self._candidate("b-2", "q" * 300),
        ]
        payload = build_bounded_suitability_payload(
            user_text="hi",
            interaction_intent=InteractionIntent.REFLECTION,
            language="en",
            candidates=candidates,
            conversation_context={"recent_turns": [], "rolling_summary": None},
            max_chars=600,
        )
        self.assertIsNotNone(payload)
        ids = {item["id"] for item in payload["eligible_exercises"]}
        self.assertEqual(ids, {"a-1", "b-2"})
        for item in payload["eligible_exercises"]:
            self.assertEqual(item["category"], "breathing")

    def test_oversized_user_message_returns_none(self):
        payload = build_bounded_suitability_payload(
            user_text="u" * (MAX_SUITABILITY_PAYLOAD_CHARS + 500),
            interaction_intent=InteractionIntent.REFLECTION,
            language="en",
            candidates=[self._candidate("only-id")],
            conversation_context={"recent_turns": [], "rolling_summary": None},
        )
        self.assertIsNone(payload)

    def test_impossible_payload_skips_auxiliary_call(self):
        llm = MagicMock()
        self.assertIsNone(
            build_bounded_suitability_payload(
                user_text="u" * 500,
                interaction_intent=InteractionIntent.REFLECTION,
                language="en",
                candidates=[self._candidate("id-only")],
                conversation_context={},
                max_chars=50,
            )
        )
        oversized = "u" * (MAX_SUITABILITY_PAYLOAD_CHARS + 100)
        selected = select_exercise_candidate(
            llm,
            user_text=oversized,
            interaction_intent=InteractionIntent.REFLECTION,
            language="en",
            candidates=[self._candidate("id-only")],
            conversation_context={},
        )
        self.assertIsNone(selected)
        llm.generate_auxiliary_json.assert_not_called()


class TestFailClosedUnsolicited(unittest.TestCase):
    def setUp(self):
        self.recommender = ExerciseRecommender()
        self.session = ExerciseSessionState()

    def test_gate1_and_missing_llm_unchanged(self):
        self.assertFalse(
            unsolicited_offer_allowed(
                ApplicationAction.SAFETY_FOCUSED_RESPONSE,
                SafetyCategory.SAFE_CONVERSATIONAL,
            )
        )
        d = self.recommender.evaluate_recommendation(
            "I feel tense.",
            InteractionIntent.REFLECTION,
            self.session,
            "en",
            llm=None,
        )
        self.assertEqual(d.action, RecommendationAction.NO_ACTION)

    def test_explicit_detection_unchanged(self):
        res = self.recommender.detect_explicit_request(
            "Can we do a breathing exercise please?"
        )
        self.assertTrue(res.is_explicit_request)

    def test_engine_explicit_still_starts_without_auxiliary(self):
        mock_llm = MagicMock()
        mock_llm.n_ctx = 4096
        mock_llm.max_tokens = 512
        engine = ConversationEngine(manager=ConversationManager(llm=mock_llm))
        result = engine.process_turn(
            TurnRequest(
                user_text="Can we do a breathing exercise please?",
                language="en",
            )
        )
        self.assertIn("Step 1", result.response_text)
        self.assertIsNone(result.recommended_exercise_id)
        mock_llm.generate_completion.assert_not_called()


if __name__ == "__main__":
    unittest.main()