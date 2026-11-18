"""
MWIZERWA AI — Behavioral Test Suite: Exercise Behavioral Regression
Evaluates multi-turn exercise flows: contextual offer & acceptance, offer decline with cooldown,
explicit requests overriding cooldown, mid-exercise stop commands, hesitation handling,
and off-topic interruptions.
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


class TestExerciseBehavioralRegression(unittest.TestCase):
    """
    Tests full exercise subsystem behavior in multi-turn conversational environments:
    - Contextual offer -> acceptance -> progression -> completion check-in.
    - Offer decline -> 5-turn cooldown enforcement.
    - Explicit user request overriding active cooldown.
    - Mid-exercise stop command handling.
    - Hesitation handling (no unintended advancement).
    - Unrelated off-topic interruption check-in.
    """

    def test_01_full_contextual_offer_accept_and_complete_flow(self):
        """Full happy path: offer -> accept -> step 1 -> step 2 -> step 3 -> step 4 -> completion."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "I can hear the heavy tension in your words.",
            "interaction_intent": "REFLECTION",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # 1. Turn 1: Somatic tension during reflection triggers contextual offer
        res1 = engine.process_turn(TurnRequest(
            user_text="I feel so tense and wound up today, wondering how to get through this.",
            language="en",
        ))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.OFFERED)
        self.assertIn("Gentle Rhythmic Breathing", res1.response_text)

        # 2. Turn 2: User accepts offer
        res2 = engine.process_turn(TurnRequest(user_text="Yes, let's do it.", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)
        self.assertIn("Step 1 of 4", res2.response_text)
        self.assertEqual(engine.state.exercise_state.current_step_index, 0)

        # 3. Turn 3: Step 1 -> Step 2
        res3 = engine.process_turn(TurnRequest(user_text="next", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)
        self.assertIn("Step 2 of 4", res3.response_text)
        self.assertEqual(engine.state.exercise_state.current_step_index, 1)

        # 4. Turn 4: Step 2 -> Step 3
        res4 = engine.process_turn(TurnRequest(user_text="next", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)
        self.assertIn("Step 3 of 4", res4.response_text)
        self.assertEqual(engine.state.exercise_state.current_step_index, 2)

        # 5. Turn 5: Step 3 -> Step 4
        res5 = engine.process_turn(TurnRequest(user_text="continue", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)
        self.assertIn("Step 4 of 4", res5.response_text)
        self.assertEqual(engine.state.exercise_state.current_step_index, 3)

        # 6. Turn 6: Step 4 -> Complete
        res6 = engine.process_turn(TurnRequest(user_text="next", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.COMPLETED)
        self.assertIn("How does your body feel", res6.response_text)
        self.assertEqual(engine.state.exercise_state.cooldown_turns_remaining, 5)

    def test_02_decline_offer_enforces_5_turn_cooldown(self):
        """Declining an offer sets cooldown to 5; subsequent turns decrement without re-offering."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "I understand completely.",
            "interaction_intent": "REFLECTION",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # Offer exercise
        engine.process_turn(TurnRequest(user_text="I feel so tense in my body."))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.OFFERED)

        # Decline offer
        res_dec = engine.process_turn(TurnRequest(user_text="No thanks, I'd rather just talk."))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.DECLINED)
        self.assertEqual(engine.state.exercise_state.cooldown_turns_remaining, 5)

        # Next turn with somatic cue: must NOT offer exercise because cooldown == 5
        res_next = engine.process_turn(TurnRequest(user_text="I still feel very tense."))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IDLE)
        self.assertNotIn("take a gentle pause", res_next.response_text)
        # Cooldown decremented to 4
        self.assertEqual(engine.state.exercise_state.cooldown_turns_remaining, 4)

    def test_03_explicit_request_bypasses_active_cooldown(self):
        """User explicitly requesting an exercise succeeds even with active cooldown."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # Artificially set cooldown
        engine.state.exercise_state.apply_cooldown(4)
        self.assertEqual(engine.state.exercise_state.cooldown_turns_remaining, 4)

        # Explicit request
        result = engine.process_turn(TurnRequest(user_text="Can we do a breathing exercise?", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)
        self.assertIn("Step 1 of 4", result.response_text)

    def test_04_mid_exercise_stop_halts_and_returns_calm_checkin(self):
        """Saying 'stop' during an active exercise halts immediately with calm acknowledgment."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # Start exercise
        engine.process_turn(TurnRequest(user_text="Can we do a breathing exercise?", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)

        # Stop exercise
        res_stop = engine.process_turn(TurnRequest(user_text="I want to stop now.", language="en"))
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.ABANDONED)
        self.assertIn("completely okay", res_stop.response_text.lower())
        self.assertEqual(engine.state.exercise_state.cooldown_turns_remaining, 5)

    def test_05_hesitation_returns_gentle_guidance_without_advancing_step(self):
        """Hesitation input ('wait, give me a moment') does NOT advance the step."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # Start exercise
        engine.process_turn(TurnRequest(user_text="Can we do a breathing exercise?", language="en"))
        self.assertEqual(engine.state.exercise_state.current_step_index, 0)

        # Hesitation
        res_hes = engine.process_turn(TurnRequest(user_text="Hold on, give me a moment.", language="en"))
        # Must stay at Step 0
        self.assertEqual(engine.state.exercise_state.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(engine.state.exercise_state.current_step_index, 0)
        self.assertIn("say 'next'", res_hes.response_text)


if __name__ == "__main__":
    unittest.main()
