"""
MWIZERWA AI — Unit Tests for Exercise Recommender
Tests for Chunk 5 Phase 2B: Validates explicit request detection across RW/EN/FR,
contextual recommendations, cooldown semantics, negation handling, and false-positive prevention.
"""

import unittest

from src.exercises.categories import ExerciseCategory
from src.exercises.models import ExerciseSessionState
from src.exercises.recommender import (
    ExerciseRecommender,
    RecommendationAction,
)
from src.llm.response_parser import InteractionIntent


class TestExerciseRecommender(unittest.TestCase):
    """Exhaustive test suite for ExerciseRecommender."""

    def setUp(self):
        self.recommender = ExerciseRecommender()
        self.session = ExerciseSessionState()

    # -----------------------------------------------------------------------
    # 1. Explicit Request Detection (RW / EN / FR)
    # -----------------------------------------------------------------------

    def test_explicit_breathing_requests(self):
        samples_en = [
            "Can we do a breathing exercise please?",
            "Help me breathe, I need to calm down.",
            "Let's practice a deep breathing technique.",
            "Guide me in breathing.",
        ]
        for s in samples_en:
            res = self.recommender.detect_explicit_request(s)
            self.assertTrue(res.is_explicit_request, f"Failed for EN: {s}")
            self.assertEqual(res.category, ExerciseCategory.BREATHING)

        samples_rw = [
            "Dufatanye umwitozo wo guhumeka.",
            "Mfasha guhumeka neza muri aka kanya.",
            "Reka dukore umwitozo wo guhumeka.",
        ]
        for s in samples_rw:
            res = self.recommender.detect_explicit_request(s)
            self.assertTrue(res.is_explicit_request, f"Failed for RW: {s}")
            self.assertEqual(res.category, ExerciseCategory.BREATHING)

        samples_fr = [
            "Faisons un exercice de respiration s'il vous plaît.",
            "Aide-moi à respirer calmement.",
            "Guide-moi pour respirer.",
        ]
        for s in samples_fr:
            res = self.recommender.detect_explicit_request(s)
            self.assertTrue(res.is_explicit_request, f"Failed for FR: {s}")
            self.assertEqual(res.category, ExerciseCategory.BREATHING)

    def test_explicit_grounding_requests(self):
        samples = [
            "Let's do grounding together.",
            "Can we do a sensory grounding exercise?",
            "Guide me through 5-4-3-2-1 technique.",
            "Mfasha gukora umwitozo wa 5-4-3-2-1.",
            "Dufatanye umwitozo wo gushyira ibitekerezo ku bikugose.",
            "Faisons un exercice d'ancrage sensoriel.",
            "Je veux faire un exercice d'ancrage.",
        ]
        for s in samples:
            res = self.recommender.detect_explicit_request(s)
            self.assertTrue(res.is_explicit_request, f"Failed: {s}")
            self.assertEqual(res.category, ExerciseCategory.GROUNDING)

    def test_explicit_thought_reflection_requests(self):
        samples = [
            "Can we do a thought reflection exercise?",
            "Help me reframe my thoughts.",
            "Mfasha gukora umwitozo wo gusuzuma ibitekerezo.",
            "Aide-moi à faire un exercice de réflexion sur mes pensées.",
        ]
        for s in samples:
            res = self.recommender.detect_explicit_request(s)
            self.assertTrue(res.is_explicit_request, f"Failed: {s}")
            self.assertEqual(res.category, ExerciseCategory.THOUGHT_REFLECTION)

    def test_explicit_self_compassion_requests(self):
        samples = [
            "Can we do a self-compassion exercise?",
            "Help me do a self-compassion pause.",
            "Dufatanye umwitozo wo kwiyitaho.",
            "Aide-moi à faire une pause d'auto-compassion.",
        ]
        for s in samples:
            res = self.recommender.detect_explicit_request(s)
            self.assertTrue(res.is_explicit_request, f"Failed: {s}")
            self.assertEqual(res.category, ExerciseCategory.SELF_COMPASSION)

    def test_explicit_gratitude_requests(self):
        samples = [
            "Can we do a gratitude reflection exercise?",
            "Let's do a grounded anchor exercise.",
            "Mfasha gukora umwitozo wo kuzirikana akantu ko kwiringira.",
            "Faisons un exercice de gratitude.",
        ]
        for s in samples:
            res = self.recommender.detect_explicit_request(s)
            self.assertTrue(res.is_explicit_request, f"Failed: {s}")
            self.assertEqual(res.category, ExerciseCategory.GRATITUDE_JOURNALING)

    def test_explicit_general_exercise_request(self):
        samples = [
            "Can we do an exercise together?",
            "Guide me through a calming exercise.",
            "Reka dukore umwitozo.",
            "Faisons un exercice.",
        ]
        for s in samples:
            res = self.recommender.detect_explicit_request(s)
            self.assertTrue(res.is_explicit_request, f"Failed: {s}")
            self.assertTrue(res.is_general_exercise_request)
            self.assertIsNone(res.category)

    # -----------------------------------------------------------------------
    # 2. Negation & Informational Inquiries (False-Positive Prevention)
    # -----------------------------------------------------------------------

    def test_negations_are_not_requests(self):
        negations = [
            "I don't want an exercise right now.",
            "Do not give me any exercise, I just want to talk.",
            "Sinshaka umwitozo na gato.",
            "Je ne veux pas d'exercice, écoute-moi.",
            "Pas d'exercice s'il te plaît.",
        ]
        for s in negations:
            res = self.recommender.detect_explicit_request(s)
            self.assertFalse(res.is_explicit_request, f"False positive on negation: {s}")

    def test_informational_inquiries_are_not_requests(self):
        inquiries = [
            "What is a breathing exercise?",
            "Can you explain how a grounding exercise works?",
            "Ni iki umwitozo wo guhumeka?",
            "Mbwira iby'imyitozo ngororangingo.",
            "Qu'est-ce que l'ancrage 5-4-3-2-1 ?",
            "C'est quoi un exercice de respiration ?",
        ]
        for s in inquiries:
            res = self.recommender.detect_explicit_request(s)
            self.assertFalse(res.is_explicit_request, f"False positive on inquiry: {s}")

    def test_ordinary_conversational_words_are_not_requests(self):
        ordinary = [
            "I need to breathe fresh air, my room is hot.",
            "I forgot to breathe when he startled me.",
            "I need to calm down before meeting my boss tomorrow.",
            "I have deep gratitude for my parents.",
            "I was thinking about my sister today.",
            "Ndumva nkeneye umwuka mwiza.",
            "J'ai besoin de respirer un peu dehors.",
        ]
        for s in ordinary:
            res = self.recommender.detect_explicit_request(s)
            self.assertFalse(res.is_explicit_request, f"False positive on ordinary sentence: {s}")

    # -----------------------------------------------------------------------
    # 3. Contextual Recommendation Evaluation & Intent Rules
    # -----------------------------------------------------------------------

    def test_listening_intent_strictly_suppresses_unsolicited_offers(self):
        # Even if user describes severe tension, LISTENING intent forbids offers
        decision = self.recommender.evaluate_recommendation(
            user_text="I feel so tense and my heart is beating fast.",
            interaction_intent=InteractionIntent.LISTENING,
            session_state=self.session,
            language="en",
        )
        self.assertEqual(decision.action, RecommendationAction.NO_ACTION)
        self.assertIn("active listening", decision.reason)

    def test_practical_advice_without_exercise_request_does_not_trigger_offer(self):
        # User asking general practical advice
        decision = self.recommender.evaluate_recommendation(
            user_text="What are some good ways to organize my study schedule for exams?",
            interaction_intent=InteractionIntent.PRACTICAL_ADVICE,
            session_state=self.session,
            language="en",
        )
        self.assertEqual(decision.action, RecommendationAction.NO_ACTION)
        self.assertIn("General practical advice", decision.reason)

    def test_unclear_intent_suppresses_unsolicited_offers(self):
        decision = self.recommender.evaluate_recommendation(
            user_text="Maybe, I don't know...",
            interaction_intent=InteractionIntent.UNCLEAR,
            session_state=self.session,
            language="en",
        )
        self.assertEqual(decision.action, RecommendationAction.NO_ACTION)
        self.assertIn("UNCLEAR", decision.reason)

    def test_reflection_intent_with_somatic_tension_offers_breathing(self):
        decision = self.recommender.evaluate_recommendation(
            user_text="I am reflecting on today and I feel so tense and physically stressed.",
            interaction_intent=InteractionIntent.REFLECTION,
            session_state=self.session,
            language="en",
        )
        self.assertEqual(decision.action, RecommendationAction.OFFER_EXERCISE)
        self.assertEqual(decision.category, ExerciseCategory.BREATHING)
        self.assertEqual(decision.exercise_id, "breathing_gentle_rhythm")

    def test_reflection_intent_with_cognitive_overwhelm_offers_grounding(self):
        decision = self.recommender.evaluate_recommendation(
            user_text="My mind is racing and my thoughts are spinning completely out of control.",
            interaction_intent=InteractionIntent.REFLECTION,
            session_state=self.session,
            language="en",
        )
        self.assertEqual(decision.action, RecommendationAction.OFFER_EXERCISE)
        self.assertEqual(decision.category, ExerciseCategory.GROUNDING)
        self.assertEqual(decision.exercise_id, "grounding_54321")

    def test_reflection_intent_with_self_blame_offers_compassion(self):
        decision = self.recommender.evaluate_recommendation(
            user_text="I am so hard on myself and I feel like a failure right now.",
            interaction_intent=InteractionIntent.REFLECTION,
            session_state=self.session,
            language="en",
        )
        self.assertEqual(decision.action, RecommendationAction.OFFER_EXERCISE)
        self.assertEqual(decision.category, ExerciseCategory.SELF_COMPASSION)
        self.assertEqual(decision.exercise_id, "compassion_pause")

    def test_reflection_without_distress_cues_takes_no_action(self):
        decision = self.recommender.evaluate_recommendation(
            user_text="I was reflecting on what happened in the meeting earlier today.",
            interaction_intent=InteractionIntent.REFLECTION,
            session_state=self.session,
            language="en",
        )
        self.assertEqual(decision.action, RecommendationAction.NO_ACTION)

    # -----------------------------------------------------------------------
    # 4. Explicit Request Override & Language Uncertainty
    # -----------------------------------------------------------------------

    def test_explicit_request_overrides_cooldown(self):
        # Put session into 5-turn cooldown
        self.session.apply_cooldown(5)
        self.assertEqual(self.session.cooldown_turns_remaining, 5)

        # Explicit request must STILL trigger INITIATE_EXPLICIT_REQUEST
        decision = self.recommender.evaluate_recommendation(
            user_text="Can we do a breathing exercise now please?",
            interaction_intent=InteractionIntent.PRACTICAL_ADVICE,
            session_state=self.session,
            language="en",
        )
        self.assertEqual(decision.action, RecommendationAction.INITIATE_EXPLICIT_REQUEST)
        self.assertEqual(decision.exercise_id, "breathing_gentle_rhythm")
        self.assertEqual(decision.category, ExerciseCategory.BREATHING)

    def test_explicit_request_language_uncertainty(self):
        # User makes explicit request but language is unsupported/undetermined and no established session language
        decision = self.recommender.evaluate_recommendation(
            user_text="Help me breathe please.",
            interaction_intent=InteractionIntent.PRACTICAL_ADVICE,
            session_state=self.session,
            language="es",  # Unsupported
        )
        self.assertEqual(decision.action, RecommendationAction.CLARIFY_LANGUAGE)

    # -----------------------------------------------------------------------
    # 5. Cooldown Semantics (Exact Sequence: T -> T+1..T+5 -> T+6)
    # -----------------------------------------------------------------------

    def test_exact_cooldown_turn_sequence(self):
        """
        Verifies exact 5-turn cooldown lifecycle:
        - Turn T: Cooldown established to 5 (not consumed in turn T).
        - Turns T+1, T+2, T+3, T+4, T+5: Unsolicited offers suppressed.
        - Turn T+6: Cooldown is 0, unsolicited offers become eligible again.
        """
        # Turn T: User declines an offer
        self.session.apply_cooldown(5)
        self.assertEqual(self.session.cooldown_turns_remaining, 5)

        tense_input = "My mind is racing and my thoughts are spinning."

        # Turn T+1 (subsequent turn 1)
        self.session.record_user_turn()
        self.assertEqual(self.session.cooldown_turns_remaining, 4)
        dec_t1 = self.recommender.evaluate_recommendation(
            tense_input, InteractionIntent.REFLECTION, self.session, language="en"
        )
        self.assertEqual(dec_t1.action, RecommendationAction.NO_ACTION)

        # Turn T+2 (subsequent turn 2)
        self.session.record_user_turn()
        self.assertEqual(self.session.cooldown_turns_remaining, 3)
        dec_t2 = self.recommender.evaluate_recommendation(
            tense_input, InteractionIntent.REFLECTION, self.session, language="en"
        )
        self.assertEqual(dec_t2.action, RecommendationAction.NO_ACTION)

        # Turn T+3 (subsequent turn 3)
        self.session.record_user_turn()
        self.assertEqual(self.session.cooldown_turns_remaining, 2)
        dec_t3 = self.recommender.evaluate_recommendation(
            tense_input, InteractionIntent.REFLECTION, self.session, language="en"
        )
        self.assertEqual(dec_t3.action, RecommendationAction.NO_ACTION)

        # Turn T+4 (subsequent turn 4)
        self.session.record_user_turn()
        self.assertEqual(self.session.cooldown_turns_remaining, 1)
        dec_t4 = self.recommender.evaluate_recommendation(
            tense_input, InteractionIntent.REFLECTION, self.session, language="en"
        )
        self.assertEqual(dec_t4.action, RecommendationAction.NO_ACTION)

        # Turn T+5 (subsequent turn 5)
        self.session.record_user_turn()
        self.assertEqual(self.session.cooldown_turns_remaining, 0)
        # Turn T+5 was the 5th suppressed turn; now cooldown reached 0

        # Turn T+6: Unsolicited recommendations are eligible again!
        dec_t6 = self.recommender.evaluate_recommendation(
            tense_input, InteractionIntent.REFLECTION, self.session, language="en"
        )
        self.assertEqual(dec_t6.action, RecommendationAction.OFFER_EXERCISE)
        self.assertEqual(dec_t6.exercise_id, "grounding_54321")


if __name__ == "__main__":
    unittest.main()
