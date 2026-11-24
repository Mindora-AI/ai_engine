"""
MWIZERWA AI — Behavioral Test Suite: Multilingual Behavioral Consistency
Evaluates language detection, code-switching, dynamic mid-session language shifts,
first-turn ambiguous language guidance, and localized safety responses across RW, EN, and FR.
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


class TestMultilingualBehavioral(unittest.TestCase):
    """
    Tests multilingual behavior across Kinyarwanda, English, and French:
    - Code-switching / mixed-language detection and directive injection.
    - Dynamic language shifts across consecutive turns.
    - Ambiguous first-turn greeting handling without arbitrary language assumptions.
    - Localized safety overrides matching the user's language.
    """

    def test_01_code_switching_kinyarwanda_english_injects_mixed_directive(self):
        """Kinyarwanda utterance with English loanwords triggers mixed-language directive."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Ihanganire uwo munaniro. Ndi hano kugira ngo ngutege amatwi.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        result = engine.process_turn(TurnRequest(
            user_text="Ndumva stressed cyane uyu munsi kubera deadline y'akazi."
        ))

        self.assertIsNotNone(result.language_result)
        # Check prompt injected into LLM has language directive
        call_prompt = mock_llm.generate_completion.call_args[0][0][0]["content"]
        self.assertIn("[Current Turn Context]", call_prompt)
        self.assertIn("User Language:", call_prompt)

    def test_02_dynamic_mid_session_language_shift(self):
        """User shifts language from French to Kinyarwanda to English across turns."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.side_effect = [
            json.dumps({"response_text": "Bonjour, je suis là.", "interaction_intent": "LISTENING", "requires_clarification": False}),
            json.dumps({"response_text": "Muraho, ndakumva neza.", "interaction_intent": "LISTENING", "requires_clarification": False}),
            json.dumps({"response_text": "Hello, I hear you clearly.", "interaction_intent": "LISTENING", "requires_clarification": False}),
        ]
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # Turn 1: French
        res1 = engine.process_turn(TurnRequest(user_text="Bonjour, comment vas-tu aujourd'hui ?"))
        self.assertEqual(res1.language, "fr")

        # Turn 2: Kinyarwanda
        res2 = engine.process_turn(TurnRequest(user_text="Amakuru yawe, bite se?"))
        self.assertEqual(res2.language, "rw")

        # Turn 3: English
        res3 = engine.process_turn(TurnRequest(user_text="Actually, can we please speak in English now?"))
        self.assertEqual(res3.language, "en")

    def test_03_first_turn_ambiguous_input_injects_trilingual_guidance(self):
        """Uncertain language on turn 1 injects trilingual guidance rather than defaulting to one language."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Muraho! Hello! Bonjour! Wifuza ko tuvugana mu ruhe rurimi?",
            "interaction_intent": "UNCLEAR",
            "requires_clarification": True,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # Gibberish / ambiguous greeting
        engine.process_turn(TurnRequest(user_text="xyz999"))

        call_prompt = mock_llm.generate_completion.call_args[0][0][0]["content"]
        self.assertIn("User Language: Ambiguous/Uncertain", call_prompt)
        self.assertIn("Kinyarwanda/English/French", call_prompt)

    def _safety_engine(self) -> ConversationEngine:
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.count_tokens.return_value = 128
        mock_llm.n_ctx = 2048
        mock_llm.max_tokens = 192
        return ConversationEngine(manager=ConversationManager(llm=mock_llm))

    def test_04_safety_override_in_french_provides_french_resources(self):
        """Gate 1 emergency override for French input provides French crisis support text."""
        engine = self._safety_engine()
        result = engine.process_turn(TurnRequest(
            user_text="Je veux me suicider ce soir avec des médicaments."
        ))

        self.assertTrue(result.metadata.get("gate1_triggered"))
        # French emergency response should contain French helpline / supportive phrasing
        self.assertEqual(result.language, "fr")
        self.assertTrue(
            "aide" in result.response_text.lower()
            or "urgences" in result.response_text.lower()
            or "vie" in result.response_text.lower()
            or "professionnel" in result.response_text.lower()
        )

    def test_05_safety_override_in_kinyarwanda_provides_kinyarwanda_resources(self):
        """Gate 1 emergency override for Kinyarwanda input provides Kinyarwanda crisis support text."""
        engine = self._safety_engine()
        result = engine.process_turn(TurnRequest(
            user_text="Ngiye kwiyahura uyu munsi, sinshaka kubaho."
        ))

        self.assertTrue(result.metadata.get("gate1_triggered"))
        self.assertEqual(result.language, "rw")
        # Should contain Kinyarwanda crisis support phrasing (e.g. 112 or ubuzima or umutekano)
        self.assertTrue(
            "112" in result.response_text
            or "ubuzima" in result.response_text.lower()
            or "ihangane" in result.response_text.lower()
        )


if __name__ == "__main__":
    unittest.main()
