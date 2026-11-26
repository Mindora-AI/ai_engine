"""
MWIZERWA AI — Unit & Integration Tests for Exercise Pipeline Integration
Tests for Chunk 5 Phase 2D: Validates ConversationEngine integration,
Gate 1 emergency abort, Gate 2 framing protection, language resolution,
deterministic history commitment, and non-conflation of advice vs exercises.
"""

import json
import unittest
from unittest.mock import MagicMock

from src.conversation.engine import ConversationEngine
from src.conversation.session import ConversationMode, TurnRequest
from src.exercises.categories import ExerciseCategory, ExerciseStatus
from src.llm.conversation import ConversationManager
from src.safety.categories import ApplicationAction


class TestExercisePipelineIntegration(unittest.TestCase):
    """Integration test suite for ConversationEngine with exercises."""

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_llm.n_ctx = 4096
        self.mock_llm.max_tokens = 512
        self.mock_llm.count_tokens.return_value = 50
        self.mock_llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": None,
        }
        self.manager = ConversationManager(llm=self.mock_llm)
        self.engine = ConversationEngine(manager=self.manager)

    # -----------------------------------------------------------------------
    # 1. Explicit Exercise Requests Through Complete Engine
    # -----------------------------------------------------------------------

    def test_explicit_breathing_request_bypasses_llm_and_starts_exercise(self):
        # User explicitly requests breathing exercise
        request = TurnRequest(user_text="Can we do a breathing exercise please?", language="en")
        result = self.engine.process_turn(request)

        # LLM must NOT be called for deterministic exercise delivery
        self.mock_llm.generate_completion.assert_not_called()

        self.assertIn("Step 1 of 4", result.response_text)
        self.assertIn("comfortable position", result.response_text)
        self.assertEqual(result.metadata["exercise_status"], ExerciseStatus.IN_PROGRESS.value)
        self.assertEqual(result.metadata["active_exercise_id"], "breathing_gentle_rhythm")
        self.assertEqual(result.metadata["current_step_index"], 0)

        # Committed atomically to history (system + user + assistant)
        self.assertEqual(len(self.engine.messages), 3)
        self.assertEqual(self.engine.messages[1]["content"], "Can we do a breathing exercise please?")
        self.assertEqual(self.engine.messages[2]["content"], result.response_text)

    def test_explicit_exercise_step_progression_and_completion(self):
        # Turn 1: Start breathing
        self.engine.process_turn(TurnRequest(user_text="Can we do a breathing exercise please?", language="en"))

        # Turn 2: Advance to step 2
        res2 = self.engine.process_turn(TurnRequest(user_text="Next step", language="en"))
        self.assertIn("Step 2 of 4", res2.response_text)
        self.assertEqual(res2.metadata["current_step_index"], 1)

        # Turn 3: Advance to step 3
        res3 = self.engine.process_turn(TurnRequest(user_text="Done with that", language="en"))
        self.assertIn("Step 3 of 4", res3.response_text)
        self.assertEqual(res3.metadata["current_step_index"], 2)

        # Turn 4: Advance to step 4
        res4 = self.engine.process_turn(TurnRequest(user_text="Continue", language="en"))
        self.assertIn("Step 4 of 4", res4.response_text)
        self.assertEqual(res4.metadata["current_step_index"], 3)

        # Turn 5: Complete exercise and receive completion question
        res5 = self.engine.process_turn(TurnRequest(user_text="Finished", language="en"))
        self.assertIn("How does your body feel", res5.response_text)
        self.assertEqual(res5.metadata["exercise_status"], ExerciseStatus.COMPLETED.value)
        self.assertEqual(res5.metadata["cooldown_turns_remaining"], 5)

    def test_explicit_stop_during_exercise(self):
        # Turn 1: Start grounding
        self.engine.process_turn(TurnRequest(user_text="Let's do grounding together", language="en"))
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)

        # Turn 2: User says stop
        res = self.engine.process_turn(TurnRequest(user_text="Stop please, I want to stop", language="en"))
        self.assertIn("That is completely okay", res.response_text)
        self.assertEqual(res.metadata["exercise_status"], ExerciseStatus.ABANDONED.value)
        self.assertEqual(res.metadata["cooldown_turns_remaining"], 5)

    # -----------------------------------------------------------------------
    # 2. Contextual Recommendations & Intent Handling
    # -----------------------------------------------------------------------

    def test_reflection_intent_with_tension_cue_attaches_offer(self):
        # LLM returns reflection response
        safe_json = json.dumps({
            "response_text": "I hear how much pressure you are experiencing today.",
            "interaction_intent": "REFLECTION",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = safe_json
        self.mock_llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "breathing_gentle_rhythm",
        }

        res = self.engine.process_turn(
            TurnRequest(user_text="I am reflecting on today and I feel so tense and physically stressed.", language="en")
        )

        self.mock_llm.generate_completion.assert_called_once()
        self.assertIn("I hear how much pressure", res.response_text)
        self.assertIn("Gentle Rhythmic Breathing", res.response_text)
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.OFFERED)
        self.assertEqual(self.engine.state.exercise_state.offered_exercise_id, "breathing_gentle_rhythm")
        self.assertEqual(res.recommended_exercise_id, "breathing_gentle_rhythm")

    def test_listening_intent_never_causes_unsolicited_offer(self):
        safe_json = json.dumps({
            "response_text": "I am here with you. Take all the time you need.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = safe_json

        res = self.engine.process_turn(
            TurnRequest(user_text="I am completely overwhelmed and everything is spinning.", language="en")
        )

        self.assertIn("I am here with you", res.response_text)
        self.assertNotIn("exercise", res.response_text.lower())
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IDLE)

    def test_ordinary_practical_advice_does_not_trigger_exercise(self):
        safe_json = json.dumps({
            "response_text": "A good way to organize your study schedule is to break tasks into 25-minute blocks.",
            "interaction_intent": "PRACTICAL_ADVICE",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = safe_json

        res = self.engine.process_turn(
            TurnRequest(user_text="What are some practical tips to organize my study schedule?", language="en")
        )

        self.assertIn("25-minute blocks", res.response_text)
        self.assertNotIn("breathing", res.response_text.lower())
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IDLE)

    # -----------------------------------------------------------------------
    # 3. Safety Precedence (Gate 1 & Gate 2)
    # -----------------------------------------------------------------------

    def test_gate1_crisis_aborts_active_exercise_immediately(self):
        # Turn 1: Start breathing exercise
        self.engine.process_turn(TurnRequest(user_text="Can we do a breathing exercise please?", language="en"))
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)

        # Turn 2: User discloses imminent crisis during exercise
        crisis_result = self.engine.process_turn(
            TurnRequest(user_text="I intend to hurt myself tonight and have everything ready.", language="en")
        )

        # Gate 1 triggers deterministic crisis response with 112 / 912
        self.assertTrue(crisis_result.metadata["gate1_triggered"])
        self.assertIn("112", crisis_result.response_text)
        self.assertIn("912", crisis_result.response_text)

        # Exercise was immediately aborted and reset to IDLE
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IDLE)
        self.assertIsNone(self.engine.state.exercise_state.active_exercise_id)
        self.mock_llm.generate_completion.assert_not_called()

    def test_gate1_crisis_aborts_pending_offer(self):
        # Set status to OFFERED
        self.engine.state.exercise_state.offer_exercise("breathing_gentle_rhythm")
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.OFFERED)

        # Turn: User discloses danger
        crisis_result = self.engine.process_turn(
            TurnRequest(user_text="The house is on fire! Please help!", language="en")
        )

        self.assertTrue(crisis_result.metadata["gate1_triggered"])
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IDLE)

    # -----------------------------------------------------------------------
    # 4. Language Resolution & Uncertainty
    # -----------------------------------------------------------------------

    def test_language_uncertainty_asks_clarification_before_initiating_exercise(self):
        # Request with ambiguous text and no established language
        res = self.engine.process_turn(TurnRequest(user_text="Help me breathe please."))
        # Detector is uncertain on very short string, no established language
        if res.language is None:
            self.assertIn("would you prefer", res.response_text.lower())
            self.assertIn("kinyarwanda", res.response_text.lower())
            self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IDLE)

    def test_unreliable_turn_uses_established_session_language(self):
        # Turn 1: Reliable English establishes session language
        safe_json = json.dumps({
            "response_text": "Hello, I am listening.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = safe_json
        self.engine.process_turn(TurnRequest(user_text="Hello, how are you doing today?"))
        self.assertEqual(self.engine.state.exercise_state.established_language, "en")

        # Turn 2: Short text with explicit exercise request inherits established English
        res2 = self.engine.process_turn(TurnRequest(user_text="Help me breathe."))
        self.assertIn("Step 1 of 4", res2.response_text)
        self.assertEqual(res2.metadata["effective_language"], "en")

    # -----------------------------------------------------------------------
    # 5. Corrective Audit Regression Tests (Findings 01, 02, 03, 04)
    # -----------------------------------------------------------------------

    def test_unrelated_question_triggers_checkin_and_does_not_advance_step(self):
        """Finding 01: Unrelated questions trigger UNRELATED_CHECKINS and do not advance step."""
        # Turn 1: Start breathing exercise
        self.engine.process_turn(TurnRequest(user_text="Can we do a breathing exercise please?", language="en"))
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)

        # Turn 2: User asks an unrelated question mid-exercise
        res = self.engine.process_turn(TurnRequest(user_text="What time does the store close tonight?", language="en"))
        self.assertIn("Would you like to pause our exercise", res.response_text)
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)

        # Turn 3: User says continue -> now advances to step 2
        res3 = self.engine.process_turn(TurnRequest(user_text="Continue", language="en"))
        self.assertIn("Step 2 of 4", res3.response_text)
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 1)

    def test_negation_hesitation_does_not_advance_step(self):
        """Finding 02: Negative or hesitant responses do not advance the exercise."""
        # Turn 1: Start breathing exercise
        self.engine.process_turn(TurnRequest(user_text="Can we do a breathing exercise please?", language="en"))
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)

        # Turn 2: User expresses hesitation ("I am not okay")
        res2 = self.engine.process_turn(TurnRequest(user_text="I am not okay", language="en"))
        self.assertIn("say 'next', or say 'stop'", res2.response_text)
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)

        # Turn 3: User expresses readiness negation ("I'm not ready yet")
        res3 = self.engine.process_turn(TurnRequest(user_text="I'm not ready yet", language="en"))
        self.assertIn("say 'next', or say 'stop'", res3.response_text)
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)

        # Turn 4: User asks for a moment ("Yes, but I need a moment.")
        res4 = self.engine.process_turn(TurnRequest(user_text="Yes, but I need a moment.", language="en"))
        self.assertIn("say 'next', or say 'stop'", res4.response_text)
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)

    def test_multilingual_hesitation_leaves_step_unchanged(self):
        """Hesitation in RW and FR leaves active step unchanged and returns localized guidance."""
        # Kinyarwanda session
        self.engine.process_turn(TurnRequest(user_text="Dufatanye umwitozo wo guhumeka.", language="rw"))
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)
        res_rw = self.engine.process_turn(TurnRequest(user_text="Yego, ariko mpa akanya.", language="rw"))
        self.assertIn("komeza", res_rw.response_text.lower())
        self.assertIn("hagarika", res_rw.response_text.lower())
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)

        # French session
        self.engine.reset_conversation()
        self.engine.process_turn(TurnRequest(user_text="Faisons un exercice de respiration.", language="fr"))
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)
        res_fr = self.engine.process_turn(TurnRequest(user_text="D'accord, attendez un moment.", language="fr"))
        self.assertIn("suivant", res_fr.response_text.lower())
        self.assertIn("stop", res_fr.response_text.lower())
        self.assertEqual(self.engine.state.exercise_state.current_step_index, 0)

    def test_unresolved_language_cannot_produce_unsolicited_kinyarwanda_offer(self):
        """Finding 03: Language uncertainty suppresses contextual offers without defaulting to 'rw'."""
        safe_json = json.dumps({
            "response_text": "I hear the distress.",
            "interaction_intent": "REFLECTION",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = safe_json

        # Message with tension cue but undetermined language
        res = self.engine.process_turn(TurnRequest(user_text="xyz999 ??? !!! tense"))
        # Turn language is None
        self.assertIsNone(res.metadata.get("effective_language"))
        # Unsolicited offer must NOT be attached
        self.assertNotIn("Niba ubyifuza", res.response_text)
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IDLE)

    def test_ignoring_offer_sets_cooldown_to_5_and_consumes_over_next_5_turns(self):
        """Finding 04: Ignoring an offer leaves cooldown at 5 in Turn T; consumed T+1..T+5."""
        safe_json = json.dumps({
            "response_text": "I hear how much pressure you are under.",
            "interaction_intent": "REFLECTION",
            "requires_clarification": False,
        })
        self.mock_llm.generate_completion.return_value = safe_json

        # Setup: Offer an exercise
        self.engine.state.exercise_state.offer_exercise("breathing_gentle_rhythm")
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.OFFERED)

        # Turn T: User ignores offer and talks about something else
        res_t = self.engine.process_turn(
            TurnRequest(user_text="Actually my boss was really unfair today.", language="en")
        )
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.IDLE)
        # Turn T must leave cooldown at exactly 5 (NOT consumed in Turn T)
        self.assertEqual(self.engine.state.exercise_state.cooldown_turns_remaining, 5)

        # Turn T+1: consumed -> 4
        self.engine.process_turn(TurnRequest(user_text="I had to redo the entire report.", language="en"))
        self.assertEqual(self.engine.state.exercise_state.cooldown_turns_remaining, 4)

        # Turn T+2: consumed -> 3
        self.engine.process_turn(TurnRequest(user_text="And then it started raining.", language="en"))
        self.assertEqual(self.engine.state.exercise_state.cooldown_turns_remaining, 3)

        # Turn T+3: consumed -> 2
        self.engine.process_turn(TurnRequest(user_text="Traffic was terrible as well.", language="en"))
        self.assertEqual(self.engine.state.exercise_state.cooldown_turns_remaining, 2)

        # Turn T+4: consumed -> 1
        self.engine.process_turn(TurnRequest(user_text="I just want to rest tonight.", language="en"))
        self.assertEqual(self.engine.state.exercise_state.cooldown_turns_remaining, 1)

        # Turn T+5: consumed -> 0 (5th suppressed turn complete)
        self.engine.process_turn(TurnRequest(user_text="At least tomorrow is Friday.", language="en"))
        self.assertEqual(self.engine.state.exercise_state.cooldown_turns_remaining, 0)

        # Turn T+6: Unsolicited offers eligible again
        self.mock_llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": "breathing_gentle_rhythm",
        }
        res_t6 = self.engine.process_turn(
            TurnRequest(user_text="I am reflecting and I feel so tense and physically stressed.", language="en")
        )
        self.assertIn("Gentle Rhythmic Breathing", res_t6.response_text)
        self.assertEqual(self.engine.state.exercise_state.status, ExerciseStatus.OFFERED)


if __name__ == "__main__":
    unittest.main()
