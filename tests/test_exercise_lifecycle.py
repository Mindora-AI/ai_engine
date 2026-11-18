"""
MWIZERWA AI — Unit Tests for Exercise Lifecycle
Tests for Chunk 5 Phase 2C: Validates deterministic instruction delivery,
state transitions, user response classification (continue, stop, decline, accept),
step progression, completion, and cancellation.
"""

import unittest

from src.exercises.categories import ExerciseStatus, ExerciseTriggerType
from src.exercises.lifecycle import (
    ExerciseLifecycle,
    LifecycleResult,
    UserResponseType,
)
from src.exercises.models import (
    ExerciseSessionState,
    ExerciseValidationError,
)
from src.exercises.registry import get_exercise


class TestExerciseLifecycle(unittest.TestCase):
    """Exhaustive test suite for ExerciseLifecycle."""

    def setUp(self):
        self.lifecycle = ExerciseLifecycle()
        self.session = ExerciseSessionState()

    # -----------------------------------------------------------------------
    # 1. User Response Classification
    # -----------------------------------------------------------------------

    def test_classify_stop_requests(self):
        stops = [
            "Stop please",
            "I want to stop now",
            "Hagarika uyu mwitozo",
            "Reka tubireke",
            "Arrête l'exercice",
            "C'est bon j'arrête",
            "Quit",
            "Cancel",
        ]
        for s in stops:
            res = self.lifecycle.classify_user_response(s, ExerciseStatus.IN_PROGRESS)
            self.assertEqual(res, UserResponseType.STOP, f"Failed on stop: {s}")

    def test_classify_continuation_requests(self):
        continuations = [
            "Next step please",
            "Ready",
            "I'm ready",
            "Done",
            "Ok",
            "Komeza",
            "Nditeguye",
            "Birakozwe",
            "Suivant",
            "C'est fait",
            "On continue",
        ]
        for s in continuations:
            res = self.lifecycle.classify_user_response(s, ExerciseStatus.IN_PROGRESS)
            self.assertEqual(res, UserResponseType.CONTINUE, f"Failed on continue: {s}")

    def test_classify_offer_accept_and_decline(self):
        accepts = [
            "Yes please",
            "Sure, let's do it",
            "Yego ndabishaka",
            "Oui je veux bien",
            "D'accord",
        ]
        for s in accepts:
            res = self.lifecycle.classify_user_response(s, ExerciseStatus.OFFERED)
            self.assertEqual(res, UserResponseType.ACCEPT, f"Failed on accept: {s}")

        declines = [
            "No thank you",
            "Not now, I'd rather talk",
            "Oya ntabwo nshaka",
            "Non merci pas maintenant",
        ]
        for s in declines:
            res = self.lifecycle.classify_user_response(s, ExerciseStatus.OFFERED)
            self.assertEqual(res, UserResponseType.DECLINE, f"Failed on decline: {s}")

    def test_classify_reflective_answers_and_ambiguity(self):
        # A reflective answer with >= 3 words
        reflective = "I see a blue pen on the table."
        res = self.lifecycle.classify_user_response(reflective, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(res, UserResponseType.ANSWER)

        # A very short 1-word ambiguous text not in continue/stop
        ambiguous = "Maybe"
        res = self.lifecycle.classify_user_response(ambiguous, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(res, UserResponseType.AMBIGUOUS)

    def test_classify_unrelated_inputs_during_in_progress(self):
        unrelated_questions = [
            "What time does the store close?",
            "Why didn't my brother call back?",
            "Can you explain something else?",
            "Is there a good restaurant nearby?",
            "Kuki bitatunganye?",
            "Quelle heure est-il à Paris ?",
        ]
        for q in unrelated_questions:
            res = self.lifecycle.classify_user_response(q, ExerciseStatus.IN_PROGRESS)
            self.assertEqual(res, UserResponseType.UNRELATED, f"Failed on unrelated question: {q}")

        unrelated_statements = [
            "I need to go now.",
            "Someone is at the door, I have to leave.",
            "I am busy with work today.",
            "The weather is very hot outside today.",
            "Ngomba kugenda ubu.",
            "Je dois partir maintenant.",
        ]
        for s in unrelated_statements:
            res = self.lifecycle.classify_user_response(s, ExerciseStatus.IN_PROGRESS)
            self.assertEqual(res, UserResponseType.UNRELATED, f"Failed on unrelated statement: {s}")

    def test_classify_negation_and_hesitation_must_not_continue(self):
        hesitant = [
            "I am not okay",
            "I'm not ready yet",
            "Not yet",
            "I don't feel ready to continue",
            "Yes, but I need a moment.",
            "Okay, give me a moment.",
            "I want to continue, just not yet.",
            "Ntabwo niteguye",
            "Siniteguye gukomeza",
            "Yego, ariko mpa akanya.",
            "Ndashaka gukomeza, ariko si ubu.",
            "Je ne suis pas prêt",
            "Pas prêt encore",
            "Pas encore",
            "Oui, mais pas maintenant.",
            "D'accord, attendez un moment.",
        ]
        for h in hesitant:
            res = self.lifecycle.classify_user_response(h, ExerciseStatus.IN_PROGRESS)
            self.assertNotEqual(res, UserResponseType.CONTINUE, f"Should not continue on: {h}")
            self.assertEqual(res, UserResponseType.AMBIGUOUS, f"Expected ambiguous on: {h}")

        stops = [
            "I don't want to continue",
            "Sinshaka gukomeza",
            "Ntabwo nshaka gukomeza",
            "Je ne veux pas continuer",
        ]
        for s in stops:
            res = self.lifecycle.classify_user_response(s, ExerciseStatus.IN_PROGRESS)
            self.assertEqual(res, UserResponseType.STOP, f"Expected stop on: {s}")

    def test_multilingual_answers_across_all_five_categories(self):
        """Verifies genuine answers across all 5 exercise categories in EN, RW, and FR."""
        samples = [
            # 1. Breathing
            ("I took a deep breath in through my nose.", "breathing_gentle_rhythm"),
            ("Ninjije umwuka mu mazuru buhoro buhoro.", "breathing_gentle_rhythm"),
            ("J'ai inspiré doucement par le nez.", "breathing_gentle_rhythm"),
            # 2. Grounding
            ("I see a chair and a window.", "grounding_54321"),
            ("Nabonye intebe n'idirishya.", "grounding_54321"),
            ("Je vois une table et une chaise.", "grounding_54321"),
            # 3. Thought Reflection
            ("I am worried about my job interview.", "reflection_gentle_perspective"),
            ("Mfite impungenge z'uko ntatsinda ikizamini.", "reflection_gentle_perspective"),
            ("J'ai l'inquiétude de ne pas réussir mon projet.", "reflection_gentle_perspective"),
            # 4. Self-Compassion
            ("This moment is really difficult for me.", "compassion_pause"),
            ("Ibi biragoye kandi birananiye.", "compassion_pause"),
            ("C'est un moment difficile pour moi.", "compassion_pause"),
            # 5. Gratitude
            ("I am grateful for my friend who listened to me.", "gratitude_grounded_anchor"),
            ("Ndashimira inshuti yanjye yantegereje amatwi.", "gratitude_grounded_anchor"),
            ("Je suis reconnaissant envers mon ami qui m'a écouté.", "gratitude_grounded_anchor"),
        ]
        for text, ex_id in samples:
            res = self.lifecycle.classify_user_response(text, ExerciseStatus.IN_PROGRESS, active_exercise_id=ex_id)
            self.assertEqual(res, UserResponseType.ANSWER, f"Failed to classify genuine answer: '{text}' for {ex_id}")

    # -----------------------------------------------------------------------
    # 2. Offering, Acceptance, and Decline
    # -----------------------------------------------------------------------

    def test_offer_exercise_and_formatting(self):
        res = self.lifecycle.offer_exercise(self.session, "breathing_gentle_rhythm", "en")
        self.assertTrue(res.is_offered)
        self.assertEqual(res.status, ExerciseStatus.OFFERED)
        self.assertEqual(self.session.status, ExerciseStatus.OFFERED)
        self.assertEqual(self.session.offered_exercise_id, "breathing_gentle_rhythm")
        self.assertIn("Gentle Rhythmic Breathing", res.response_text)

    def test_decline_offer_lifecycle(self):
        self.lifecycle.offer_exercise(self.session, "breathing_gentle_rhythm", "en")
        res = self.lifecycle.decline_offer(self.session, "en")
        self.assertEqual(res.status, ExerciseStatus.DECLINED)
        self.assertEqual(self.session.status, ExerciseStatus.DECLINED)
        self.assertEqual(self.session.cooldown_turns_remaining, 5)
        self.assertIn("We will just keep talking", res.response_text)

    # -----------------------------------------------------------------------
    # 3. Explicit Initiation & Full Progression to Completion
    # -----------------------------------------------------------------------

    def test_full_exercise_progression_to_completion_en(self):
        # Start breathing exercise (4 steps total)
        res1 = self.lifecycle.start_exercise(
            self.session, "breathing_gentle_rhythm", ExerciseTriggerType.EXPLICIT_USER_REQUEST, "en"
        )
        self.assertEqual(res1.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(res1.step_number, 1)
        self.assertEqual(res1.total_steps, 4)
        self.assertIn("Step 1 of 4", res1.response_text)
        self.assertIn("comfortable position", res1.response_text)

        # Advance to step 2
        res2 = self.lifecycle.advance_exercise(self.session, "en")
        self.assertEqual(res2.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(res2.step_number, 2)
        self.assertIn("Step 2 of 4", res2.response_text)
        self.assertIn("breathe in through your nose", res2.response_text)

        # Advance to step 3
        res3 = self.lifecycle.advance_exercise(self.session, "en")
        self.assertEqual(res3.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(res3.step_number, 3)
        self.assertIn("Step 3 of 4", res3.response_text)
        self.assertIn("release the breath out", res3.response_text)

        # Advance to step 4 (final instruction step)
        res4 = self.lifecycle.advance_exercise(self.session, "en")
        self.assertEqual(res4.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(res4.step_number, 4)
        self.assertIn("Step 4 of 4", res4.response_text)

        # Advancing from step 4 completes the exercise and delivers completion check-in
        res5 = self.lifecycle.advance_exercise(self.session, "en")
        self.assertTrue(res5.is_completed)
        self.assertEqual(res5.status, ExerciseStatus.COMPLETED)
        self.assertIn("How does your body feel", res5.response_text)

        # Session state verification
        self.assertEqual(self.session.status, ExerciseStatus.COMPLETED)
        self.assertIn("breathing_gentle_rhythm", self.session.completed_exercise_ids)
        self.assertEqual(self.session.cooldown_turns_remaining, 5)

    def test_full_exercise_progression_rw(self):
        res1 = self.lifecycle.start_exercise(
            self.session, "grounding_54321", ExerciseTriggerType.EXPLICIT_USER_REQUEST, "rw"
        )
        self.assertEqual(res1.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(res1.step_number, 1)
        self.assertIn("Intambwe ya 1 kuri 5", res1.response_text)

    def test_full_exercise_progression_fr(self):
        res1 = self.lifecycle.start_exercise(
            self.session, "grounding_54321", ExerciseTriggerType.EXPLICIT_USER_REQUEST, "fr"
        )
        self.assertEqual(res1.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(res1.step_number, 1)
        self.assertIn("Étape 1 sur 5", res1.response_text)

    # -----------------------------------------------------------------------
    # 4. Stopping / Abandonment
    # -----------------------------------------------------------------------

    def test_stop_active_exercise(self):
        self.lifecycle.start_exercise(
            self.session, "grounding_54321", ExerciseTriggerType.EXPLICIT_USER_REQUEST, "en"
        )
        self.session.advance_step()
        self.assertEqual(self.session.current_step_index, 1)

        res = self.lifecycle.stop_exercise(self.session, "en")
        self.assertTrue(res.is_stopped)
        self.assertEqual(res.status, ExerciseStatus.ABANDONED)
        self.assertEqual(self.session.status, ExerciseStatus.ABANDONED)
        self.assertIsNone(self.session.active_exercise_id)
        self.assertEqual(self.session.cooldown_turns_remaining, 5)
        self.assertIn("That is completely okay", res.response_text)

    # -----------------------------------------------------------------------
    # 5. Invalid Transition Handling
    # -----------------------------------------------------------------------

    def test_advance_when_not_in_progress_raises_error(self):
        with self.assertRaises(ExerciseValidationError):
            self.lifecycle.advance_exercise(self.session, "en")


if __name__ == "__main__":
    unittest.main()
