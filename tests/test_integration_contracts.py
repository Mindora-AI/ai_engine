"""Backend ↔ AI integration contract regression tests."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.service import MWIZERWAApiService, _public_metadata
from src.conversation.context_integration import (
    TurnContextValidationError,
    resolve_approved_memory_summary,
    validate_recent_turns,
)
from src.conversation.engine import ConversationEngine
from src.conversation.safety_signal import build_safety_signal
from src.conversation.session import (
    EligibleExerciseCandidate,
    TurnContext,
    TurnRequest,
)
from src.exercises.categories import ExerciseCategory
from src.exercises.models import ExerciseSessionState
from src.exercises.recommender import ExerciseRecommender, RecommendationAction
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.response_parser import InteractionIntent
from src.memory.coordinator import MemoryCoordinator
from src.memory.in_memory_repository import InMemoryMemoryRepository
from src.safety import ApplicationAction, SafetyAssessment, SafetyCategory
from src.safety.assistant_policy import AssistantSafetyResult


class TestBackendContextMemory(unittest.TestCase):
    def test_backend_memory_overrides_internal_repo(self):
        repo = InMemoryMemoryRepository()
        repo.update_summary("u1", "Internal stale memory")
        coord = MemoryCoordinator(repository=repo)
        request = TurnRequest(
            user_text="Hi",
            user_id="u1",
            context=TurnContext(
                approved_memory_summary="Backend authoritative memory.",
                backend_supplied_approved_memory=True,
            ),
        )

        resolved = resolve_approved_memory_summary(
            request, lambda: coord.get_approved_summary_text("u1")
        )
        self.assertEqual(resolved, "Backend authoritative memory.")

    def test_explicit_empty_memory_does_not_fallback(self):
        repo = InMemoryMemoryRepository()
        repo.update_summary("u1", "Should not appear")
        request = TurnRequest(
            user_text="Hi",
            user_id="u1",
            context=TurnContext(
                approved_memory_summary="",
                backend_supplied_approved_memory=True,
            ),
        )
        resolved = resolve_approved_memory_summary(
            request, lambda: repo.get_summary("u1").summary_text
        )
        self.assertEqual(resolved, "")

    def test_omitted_context_uses_internal_fallback(self):
        repo = InMemoryMemoryRepository()
        repo.update_summary("u1", "Dev memory")
        request = TurnRequest(user_text="Hi", user_id="u1")
        resolved = resolve_approved_memory_summary(
            request, lambda: repo.get_summary("u1").summary_text
        )
        self.assertEqual(resolved, "Dev memory")


class TestRecentTurnsValidation(unittest.TestCase):
    def test_validate_roles_and_bounds(self):
        turns = validate_recent_turns(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        )
        self.assertEqual(len(turns), 2)

    def test_invalid_role_rejected(self):
        with self.assertRaises(TurnContextValidationError):
            validate_recent_turns([{"role": "system", "content": "nope"}])


class TestRecentTurnsApplication(unittest.TestCase):
    def setUp(self):
        self.manager = MagicMock()
        self.manager.messages = [{"role": "system", "content": "SYS"}]
        self.manager.system_prompt = "SYS"
        self.engine = ConversationEngine(manager=self.manager)

    def test_same_recent_turns_not_duplicated(self):
        ctx = TurnContext(
            recent_turns=[
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
            ]
        )
        req = TurnRequest(user_text=" ", context=ctx)
        self.engine._apply_backend_recent_turns(req)
        first_len = len(self.manager.messages)
        self.engine._apply_backend_recent_turns(req)
        self.assertEqual(len(self.manager.messages), first_len)


class TestExerciseCandidateConstraints(unittest.TestCase):
    def setUp(self):
        self.recommender = ExerciseRecommender()
        self.state = ExerciseSessionState()
        self.state.set_established_language("en")

    def test_empty_backend_list_produces_no_recommendation(self):
        decision = self.recommender.evaluate_recommendation(
            "I feel tense in my shoulders.",
            InteractionIntent.REFLECTION,
            self.state,
            "en",
            eligible_exercises=[],
        )
        self.assertEqual(decision.action, RecommendationAction.NO_ACTION)

    def test_selects_only_supplied_candidate_id(self):
        candidates = [
            EligibleExerciseCandidate(id="custom-breath-01", category=ExerciseCategory.BREATHING),
        ]
        llm = MagicMock()
        llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "custom-breath-01",
        }
        decision = self.recommender.evaluate_recommendation(
            "I feel tense in my shoulders.",
            InteractionIntent.REFLECTION,
            self.state,
            "en",
            eligible_exercises=candidates,
            llm=llm,
        )
        self.assertEqual(decision.action, RecommendationAction.OFFER_EXERCISE)
        self.assertEqual(decision.exercise_id, "custom-breath-01")

    def test_backend_mode_skips_local_cooldown(self):
        self.state.apply_cooldown(5)
        candidates = [
            EligibleExerciseCandidate(id="custom-breath-01", category=ExerciseCategory.BREATHING),
        ]
        llm = MagicMock()
        llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "custom-breath-01",
        }
        decision = self.recommender.evaluate_recommendation(
            "I feel tense in my shoulders.",
            InteractionIntent.REFLECTION,
            self.state,
            "en",
            eligible_exercises=candidates,
            llm=llm,
        )
        self.assertEqual(decision.action, RecommendationAction.OFFER_EXERCISE)


class TestSafetySignal(unittest.TestCase):
    def test_ordinary_turn_has_no_signal(self):
        assessment = SafetyAssessment(category=SafetyCategory.SAFE_CONVERSATIONAL)
        signal = build_safety_signal(
            safety_assessment=assessment,
            safety_action=ApplicationAction.NORMAL_CONVERSATION,
            gate2_triggered=False,
            gate2_violation=None,
        )
        self.assertIsNone(signal)

    def test_gate1_emits_structured_fields_without_markers(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.IMMINENT_SELF_HARM,
            detected_expression_language="en",
        )
        signal = build_safety_signal(
            safety_assessment=assessment,
            safety_action=ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE,
            gate2_triggered=False,
            gate2_violation=None,
        )
        self.assertIsNotNone(signal)
        self.assertTrue(signal["gate1_triggered"])
        self.assertEqual(signal["category"], "imminent_self_harm")
        self.assertNotIn("matched_markers", signal)

    def test_gate2_violation_category_exposed(self):
        signal = build_safety_signal(
            safety_assessment=SafetyAssessment(category=SafetyCategory.SAFE_CONVERSATIONAL),
            safety_action=ApplicationAction.NORMAL_CONVERSATION,
            gate2_triggered=True,
            gate2_violation=AssistantSafetyResult(
                is_safe=False,
                violation_category="violence_incitement",
            ),
        )
        self.assertEqual(signal["gate2_violation_category"], "violence_incitement")


class TestRecommendedExerciseIdFlow(unittest.TestCase):
    def test_unsolicited_offer_sets_top_level_id(self):
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.n_ctx = 4096
        mock_llm.max_tokens = 512
        mock_llm.count_tokens.return_value = 50
        mock_llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "I hear you.",
                "interaction_intent": "REFLECTION",
                "requires_clarification": False,
            }
        )
        mock_llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "breathing_gentle_rhythm",
        }
        from src.llm.conversation import ConversationManager

        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)
        engine._state.exercise_state.set_established_language("en")

        backend_id = "breathing_gentle_rhythm"
        candidates = [
            EligibleExerciseCandidate(id=backend_id, category=ExerciseCategory.BREATHING),
        ]
        result = engine.process_turn(
            TurnRequest(
                user_text="I feel tense and physically stressed reflecting on work.",
                language="en",
                context=TurnContext(eligible_exercises=candidates),
            )
        )
        self.assertEqual(result.recommended_exercise_id, backend_id)
        self.assertIsNotNone(result.recommended_exercise_id)


class TestApiResponseContracts(unittest.TestCase):
    def setUp(self):
        self.service = MWIZERWAApiService()
        self.mock_llm = MagicMock()
        self.service._llm = self.mock_llm
        self.service._initialized = True
        contract = json.dumps(
            {
                "response_text": "OK",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )
        self.mock_llm.generate_completion.return_value = contract
        self._patch = patch("src.api.app.get_api_service", return_value=self.service)
        self._patch.start()

        async def _noop(app_):
            yield

        app.router.lifespan_context = _noop
        self.client = TestClient(app)

    def tearDown(self):
        self._patch.stop()

    def test_chat_accepts_context_and_returns_recommended_exercise_null(self):
        r = self.client.post(
            "/chat",
            json={
                "user_text": "Hello",
                "context": {"approved_memory_summary": "Prefers mornings."},
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("recommended_exercise_id", data)
        self.assertIsNone(data["recommended_exercise_id"])

    def test_invalid_context_returns_400(self):
        r = self.client.post(
            "/chat",
            json={
                "user_text": "Hello",
                "context": {
                    "recent_turns": [{"role": "user", "content": "x"}],
                    "eligible_exercises": [{"id": "x", "category": "not_a_category"}],
                },
            },
        )
        self.assertIn(r.status_code, (400, 422))

    def test_public_metadata_includes_safety_signal_key_when_present(self):
        meta = {
            "gate1_triggered": True,
            "safety_action": "deterministic_emergency_override",
            "safety_signal": {
                "gate1_triggered": True,
                "gate2_triggered": False,
                "safety_action": "deterministic_emergency_override",
                "category": "imminent_self_harm",
            },
        }
        public = _public_metadata(meta)
        self.assertIn("safety_signal", public)
        self.assertNotIn("safety_assessment", public)


if __name__ == "__main__":
    unittest.main()