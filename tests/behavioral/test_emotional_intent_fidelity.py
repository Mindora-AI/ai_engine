"""
MWIZERWA AI — Behavioral Test Suite: Emotional Intent Fidelity
Evaluates conversational intent classification and behavioral compliance across
LISTENING, REFLECTION, PRACTICAL_ADVICE, and UNCLEAR intents in EN, RW, and FR.
"""

import json
import unittest
from typing import Optional
from unittest.mock import MagicMock

from src.conversation.engine import ConversationEngine
from src.conversation.session import SessionState, TurnRequest, ConversationMode
from src.llm.conversation import ConversationManager
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.response_parser import InteractionIntent
from src.exercises import ExerciseStatus


class TestEmotionalIntentFidelity(unittest.TestCase):
    """
    Tests behavioral fidelity to conversational intents:
    - LISTENING: User shares a burden; system listens without unsolicited advice or exercises.
    - REFLECTION: User explores meaning; system facilitates exploration.
    - PRACTICAL_ADVICE: User asks for actionable steps; system responds without unrequested exercises.
    - UNCLEAR: Intent cannot be determined; system clarifies gently without unsolicited offers.
    - Clarification Contract: `requires_clarification` suppresses unsolicited recommendations.
    """

    def _create_engine_with_mock_llm(
        self,
        response_text: str,
        intent: InteractionIntent,
        requires_clarification: bool = False,
        *,
        auxiliary_exercise_id: Optional[str] = "breathing_gentle_rhythm",
    ):
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": response_text,
            "interaction_intent": intent.value,
            "requires_clarification": requires_clarification,
        })
        mock_llm.n_ctx = 4096
        mock_llm.max_tokens = 512
        mock_llm.count_tokens.return_value = 50
        mock_llm.generate_auxiliary_json.return_value = {
            "recommended_exercise_id": auxiliary_exercise_id,
        }
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)
        return engine, mock_llm

    def test_01_english_listening_intent_suppresses_unsolicited_exercise(self):
        """English bereavement disclosure with LISTENING intent: no exercise offered."""
        engine, _ = self._create_engine_with_mock_llm(
            response_text="I am so sorry to hear about your loss. I am right here listening.",
            intent=InteractionIntent.LISTENING,
        )
        result = engine.process_turn(TurnRequest(
            user_text="My dog passed away yesterday and the house feels completely empty."
        ))
        self.assertEqual(result.metadata.get("interaction_intent"), "LISTENING")
        self.assertEqual(result.metadata.get("exercise_status"), ExerciseStatus.IDLE.value)
        self.assertNotIn("exercise", result.response_text.lower())
        self.assertNotIn("umwitozo", result.response_text.lower())
        self.assertFalse(result.metadata.get("requires_clarification"))

    def test_02_kinyarwanda_listening_intent_suppresses_unsolicited_exercise(self):
        """Kinyarwanda work isolation with LISTENING intent: validates feelings without exercise."""
        engine, _ = self._create_engine_with_mock_llm(
            response_text="Birumvikana ko gukora wenyine bigoye cyane. Ndi hano kugira ngo ngutege amatwi.",
            intent=InteractionIntent.LISTENING,
        )
        result = engine.process_turn(TurnRequest(
            user_text="Ndi gukora akazi kenshi kandi numva ndi jyenyine, nta muntu wo kumfasha."
        ))
        self.assertEqual(result.metadata.get("interaction_intent"), "LISTENING")
        self.assertEqual(result.metadata.get("exercise_status"), ExerciseStatus.IDLE.value)
        self.assertNotIn("Niba ubyifuza", result.response_text)

    def test_03_french_listening_intent_suppresses_unsolicited_exercise(self):
        """French illness disclosure with LISTENING intent: pure listening presence."""
        engine, _ = self._create_engine_with_mock_llm(
            response_text="Je comprends à quel point cette situation avec votre mère est lourde. Je suis là avec vous.",
            intent=InteractionIntent.LISTENING,
        )
        result = engine.process_turn(TurnRequest(
            user_text="Ma mère est à l'hôpital depuis hier et j'ai une peur immense."
        ))
        self.assertEqual(result.metadata.get("interaction_intent"), "LISTENING")
        self.assertEqual(result.metadata.get("exercise_status"), ExerciseStatus.IDLE.value)
        self.assertNotIn("Si vous le souhaitez", result.response_text)

    def test_04_reflection_with_somatic_tension_offers_breathing(self):
        """REFLECTION intent with somatic cue (tense, tight): contextual breathing offer appended."""
        engine, _ = self._create_engine_with_mock_llm(
            response_text="It sounds like you have been carrying this pressure for weeks.",
            intent=InteractionIntent.REFLECTION,
        )
        result = engine.process_turn(TurnRequest(
            user_text="I feel so tense and tight in my shoulders, trying to figure out what went wrong."
        ))
        self.assertEqual(result.metadata.get("interaction_intent"), "REFLECTION")
        self.assertEqual(result.metadata.get("exercise_status"), ExerciseStatus.OFFERED.value)
        self.assertIn("Gentle Rhythmic Breathing", result.response_text)

    def test_05_reflection_without_cues_does_not_offer_exercise(self):
        """REFLECTION intent without somatic/cognitive cues: no exercise offer."""
        engine, _ = self._create_engine_with_mock_llm(
            response_text="Looking back at that meeting, what do you think triggered that reaction?",
            intent=InteractionIntent.REFLECTION,
            auxiliary_exercise_id=None,
        )
        result = engine.process_turn(TurnRequest(
            user_text="I was thinking about the project review and wondering why they reacted that way."
        ))
        self.assertEqual(result.metadata.get("interaction_intent"), "REFLECTION")
        self.assertEqual(result.metadata.get("exercise_status"), ExerciseStatus.IDLE.value)
        self.assertNotIn("exercise", result.response_text.lower())

    def test_06_practical_advice_intent_does_not_offer_unsolicited_exercise(self):
        """PRACTICAL_ADVICE intent: answers user's question without unsolicited exercise."""
        engine, _ = self._create_engine_with_mock_llm(
            response_text="You might try breaking your study sessions into 25-minute blocks with short breaks.",
            intent=InteractionIntent.PRACTICAL_ADVICE,
        )
        result = engine.process_turn(TurnRequest(
            user_text="What are some practical study tips for preparing for my exam next week?"
        ))
        self.assertEqual(result.metadata.get("interaction_intent"), "PRACTICAL_ADVICE")
        self.assertEqual(result.metadata.get("exercise_status"), ExerciseStatus.IDLE.value)
        self.assertNotIn("If you would like, we could take a gentle pause", result.response_text)

    def test_07_unclear_intent_suppresses_exercise_and_requests_clarification(self):
        """UNCLEAR intent: asks a gentle clarifying question, no exercise offered."""
        engine, _ = self._create_engine_with_mock_llm(
            response_text="I want to make sure I understand. Could you tell me a bit more about what you mean?",
            intent=InteractionIntent.UNCLEAR,
            requires_clarification=True,
        )
        result = engine.process_turn(TurnRequest(user_text="Maybe so..."))
        self.assertEqual(result.metadata.get("interaction_intent"), "UNCLEAR")
        self.assertTrue(result.metadata.get("requires_clarification"))
        self.assertEqual(result.metadata.get("exercise_status"), ExerciseStatus.IDLE.value)

    def test_08_requires_clarification_flag_suppresses_contextual_recommendations(self):
        """When requires_clarification is True, contextual recommendation is bypassed even if cues exist."""
        engine, _ = self._create_engine_with_mock_llm(
            response_text="Could you clarify what is causing the tension?",
            intent=InteractionIntent.REFLECTION,
            requires_clarification=True,
        )
        result = engine.process_turn(TurnRequest(
            user_text="I feel so tense in my chest, but I don't know why."
        ))
        # Even though "feel so tense" matches somatic cue, requires_clarification suppresses the offer
        self.assertEqual(result.metadata.get("exercise_status"), ExerciseStatus.IDLE.value)
        self.assertTrue(result.metadata.get("requires_clarification"))
        self.assertNotIn("Gentle Rhythmic Breathing", result.response_text)


if __name__ == "__main__":
    unittest.main()
