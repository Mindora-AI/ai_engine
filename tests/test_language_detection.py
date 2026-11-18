"""
MWIZERWA AI — Chunk 3 Language Detection & Per-Turn Tracking Tests.
Covers individual language detection, ambiguous inputs, mixed-language code-switching,
and conversation integration across Kinyarwanda (rw), English (en), and French (fr).
"""

import json
import unittest
from unittest.mock import MagicMock

from src.utils.language_detector import (
    DetectedLanguage,
    LanguageDetectionResult,
    LanguageDetector,
    detect_language,
)
from src.conversation.session import (
    ConversationMode,
    TurnRequest,
    TurnResult,
)
from src.conversation.engine import ConversationEngine
from src.llm.conversation import ConversationManager
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.prompts import SYSTEM_PROMPT


class TestLanguageDetection(unittest.TestCase):
    """Automated test suite verifying the 22 language detection and tracking requirements."""

    def setUp(self):
        self.detector = LanguageDetector()

    # -------------------------------------------------------------------------
    # Individual language detection
    # -------------------------------------------------------------------------

    def test_01_clear_kinyarwanda_message(self):
        """1. Reliably detects clear conversational Kinyarwanda messages."""
        text = "Ndumva naniwe cyane uyu munsi, ariko ndashaka kuganira nawe."
        result = self.detector.detect(text)
        self.assertEqual(result.language, "rw")
        self.assertTrue(result.is_reliable)
        self.assertFalse(result.is_mixed)
        self.assertGreaterEqual(result.confidence, 0.70)

    def test_02_clear_english_message(self):
        """2. Reliably detects clear English messages."""
        text = "I have been thinking about this all night and I feel very overwhelmed."
        result = self.detector.detect(text)
        self.assertEqual(result.language, "en")
        self.assertTrue(result.is_reliable)
        self.assertFalse(result.is_mixed)
        self.assertGreaterEqual(result.confidence, 0.70)

    def test_03_clear_french_message(self):
        """3. Reliably detects clear French messages."""
        text = "Je ne sais pas quoi faire pour surmonter cette situation difficile."
        result = self.detector.detect(text)
        self.assertEqual(result.language, "fr")
        self.assertTrue(result.is_reliable)
        self.assertFalse(result.is_mixed)
        self.assertGreaterEqual(result.confidence, 0.70)

    def test_04_capitalization_and_punctuation_variations(self):
        """4. Handles uppercase capitalization, exclamation marks, and punctuation."""
        rw_text = "MURAKOZE CYANE! BITE SE??? UMEZE GUTE UYU MUNSI?!"
        rw_res = self.detector.detect(rw_text)
        self.assertEqual(rw_res.language, "rw")
        self.assertTrue(rw_res.is_reliable)

        en_text = "HELLO THERE! HOW ARE YOU DOING TODAY??? I AM FINE."
        en_res = self.detector.detect(en_text)
        self.assertEqual(en_res.language, "en")
        self.assertTrue(en_res.is_reliable)

    def test_05_french_accents(self):
        """5. Handles French diacritics and accents (é, è, ê, à, â, ç, î, ô, etc.)."""
        text = "J'espère que vous vous sentez un peu plus apaisé aujourd'hui."
        result = self.detector.detect(text)
        self.assertEqual(result.language, "fr")
        self.assertTrue(result.is_reliable)

    def test_06_kinyarwanda_apostrophes_and_grammatical_forms(self):
        """6. Handles Kinyarwanda vowel-elision apostrophes and agglutinative forms."""
        text = "N'ibyo rwose, twaganira ku by'ubuzima n'agahinda k'umutima."
        result = self.detector.detect(text)
        self.assertEqual(result.language, "rw")
        self.assertTrue(result.is_reliable)

    # -------------------------------------------------------------------------
    # Ambiguous inputs
    # -------------------------------------------------------------------------

    def test_07_extremely_short_messages(self):
        """7. Extremely short or single-character messages avoid high confidence."""
        short_inputs = ["k", "ok", "...", "???", "123", "a", "me"]
        for s in short_inputs:
            res = self.detector.detect(s)
            self.assertFalse(
                res.is_reliable,
                f"Short/ambiguous token '{s}' should not be marked reliable."
            )

    def test_08_greetings_and_acknowledgements(self):
        """8. Distinctive greetings are identified without over-claiming native certainty."""
        rw_greeting = self.detector.detect("Muraho")
        self.assertEqual(rw_greeting.language, "rw")
        self.assertTrue(rw_greeting.is_reliable)

        fr_greeting = self.detector.detect("Bonjour")
        self.assertEqual(fr_greeting.language, "fr")
        self.assertTrue(fr_greeting.is_reliable)

        en_greeting = self.detector.detect("Hello")
        self.assertEqual(en_greeting.language, "en")
        self.assertTrue(en_greeting.is_reliable)

    def test_09_proper_nouns_and_names(self):
        """9. Proper nouns and names alone are not treated as definitive language proof."""
        text = "Kigali Rwanda Eric Marie"
        result = self.detector.detect(text)
        self.assertFalse(result.is_reliable)

    def test_10_messages_containing_insufficient_linguistic_evidence(self):
        """10. Unrecognized words or gibberish return an unreliable / unknown result."""
        text = "xyz qwerty asdfgh jklzxcv 98765"
        result = self.detector.detect(text)
        self.assertIsNone(result.language)
        self.assertFalse(result.is_reliable)
        self.assertLess(result.confidence, 0.50)

    # -------------------------------------------------------------------------
    # Mixed-language detection (Code-Switching)
    # -------------------------------------------------------------------------

    def test_11_kinyarwanda_english_code_switching(self):
        """11. Identifies code-switching between Kinyarwanda and English."""
        text = "Ndumva tired uyu munsi."
        result = self.detector.detect(text)
        self.assertTrue(result.is_mixed)
        self.assertEqual(result.language, "rw")  # Kinyarwanda is dominant

    def test_12_kinyarwanda_french_code_switching(self):
        """12. Identifies code-switching between Kinyarwanda and French."""
        text = "Ndumva naniwe ariko c'est la vie."
        result = self.detector.detect(text)
        self.assertTrue(result.is_mixed)

    def test_13_english_french_code_switching(self):
        """13. Identifies code-switching between English and French."""
        text = "I am feeling down today mais je ne sais pas pourquoi."
        result = self.detector.detect(text)
        self.assertTrue(result.is_mixed)

    def test_14_mixed_message_with_clear_dominant_language(self):
        """14. Selects the dominant language when evidence clearly supports it."""
        # 6 Kinyarwanda words, 2 English words
        text = "Ndumva mfite ikibazo cyinshi cyane mu mutima wanjye ariko I will manage."
        result = self.detector.detect(text)
        self.assertTrue(result.is_mixed)
        self.assertEqual(result.language, "rw")
        self.assertTrue(result.is_reliable)

    def test_15_mixed_message_without_clear_dominant_language(self):
        """15. Returns an uncertain result when mixed languages are evenly balanced."""
        # Balanced mix: French and Kinyarwanda
        text = "Je suis fatigué, ariko ndagerageza."
        result = self.detector.detect(text)
        self.assertTrue(result.is_mixed)
        # Without clear dominance, language is None or unreliable
        self.assertFalse(result.is_reliable)

    # -------------------------------------------------------------------------
    # Conversation integration
    # -------------------------------------------------------------------------

    def test_16_english_message_following_kinyarwanda(self):
        """16. A new English turn detects 'en' independently without inheriting prior 'rw'."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "I am here with you.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        res1 = engine.process_turn(TurnRequest(user_text="Ndumva naniwe uyu munsi."))
        self.assertEqual(res1.language, "rw")

        res2 = engine.process_turn(TurnRequest(user_text="I have been thinking about this all night."))
        self.assertEqual(res2.language, "en")
        self.assertEqual(res2.metadata["detected_language"], "en")

    def test_17_french_message_following_english(self):
        """17. A French turn detects 'fr' independently following an English turn."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Je vous écoute.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        res1 = engine.process_turn(TurnRequest(user_text="How are you doing today?"))
        self.assertEqual(res1.language, "en")

        res2 = engine.process_turn(TurnRequest(user_text="Je ne sais pas quoi faire."))
        self.assertEqual(res2.language, "fr")

    def test_18_unknown_language_result_following_reliably_detected_language(self):
        """18. Ambiguous/unknown input does not blindly inherit previous turn's language."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Yego ndakumva.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        res1 = engine.process_turn(TurnRequest(user_text="Muraho neza, amakuru ki?"))
        self.assertEqual(res1.language, "rw")

        res2 = engine.process_turn(TurnRequest(user_text="xyz999 ??? !!!"))
        self.assertIsNone(res2.language)
        self.assertFalse(res2.metadata["is_language_reliable"])

    def test_19_mixed_language_message_through_conversation_engine(self):
        """19. Mixed-language inputs pass through ConversationEngine with is_mixed exposed."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Ihanganire uwo munaniro.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        result = engine.process_turn(TurnRequest(user_text="Ndumva tired uyu munsi."))
        self.assertEqual(result.language, "rw")
        self.assertTrue(result.language_result.is_mixed)
        self.assertTrue(result.metadata["is_language_mixed"])

    def test_20_language_metadata_correctly_exposed_in_turn_result(self):
        """20. TurnResult exposes language, language_result, and corresponding metadata fields."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Ni byiza.",
            "interaction_intent": "REFLECTION",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        result = engine.process_turn(TurnRequest(user_text="Bonjour, je cherche de l'aide."))
        self.assertEqual(result.language, "fr")
        self.assertIsInstance(result.language_result, LanguageDetectionResult)
        self.assertEqual(result.metadata.get("detected_language"), "fr")
        self.assertIn("language_confidence", result.metadata)
        self.assertIn("is_language_reliable", result.metadata)
        self.assertIn("is_language_mixed", result.metadata)

    def test_21_empty_input_does_not_inherit_previous_language_metadata(self):
        """21. Empty input resets language metadata to None without inheriting previous turn."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Muraho!",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        res1 = engine.process_turn(TurnRequest(user_text="Muraho neza!"))
        self.assertEqual(res1.language, "rw")

        res2 = engine.process_turn(TurnRequest(user_text="   \n  "))
        self.assertEqual(res2.response_text, "")
        self.assertIsNone(res2.language)
        self.assertIsNone(res2.language_result)

    def test_22_failed_generation_does_not_expose_stale_turn_metadata(self):
        """22. Generation failure resets last_parsed_response and leaves history unmodified."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Ndakumva.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # Turn 1 succeeds
        res1 = engine.process_turn(TurnRequest(user_text="Muraho"))
        self.assertIsNotNone(manager.last_parsed_response)

        # Turn 2 crashes in LLM
        mock_llm.generate_completion.side_effect = RuntimeError("Inference crash")
        with self.assertRaises(RuntimeError):
            engine.process_turn(TurnRequest(user_text="Crash me"))

        self.assertIsNone(manager.last_parsed_response)

    # -------------------------------------------------------------------------
    # Regression & Integrity Tests
    # -------------------------------------------------------------------------

    def test_23_conversation_history_not_polluted_with_language_directives(self):
        """23. Language context directives are ephemeral and never enter committed history."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "I hear you clearly.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        engine.process_turn(TurnRequest(user_text="I am feeling very tired today."))

        # 1. Candidate sent to LLM included per-turn context
        call_prompt = mock_llm.generate_completion.call_args[0][0][0]["content"]
        self.assertIn("[Current Turn Context]", call_prompt)
        self.assertIn("User Language: English", call_prompt)

        # 2. Committed official history index 0 is base SYSTEM_PROMPT without turn context
        self.assertEqual(manager.messages[0]["content"], SYSTEM_PROMPT)
        self.assertNotIn("[Current Turn Context]", manager.messages[0]["content"])

        # 3. User and assistant messages in history contain only pure dialogue
        self.assertEqual(manager.messages[1]["content"], "I am feeling very tired today.")
        self.assertEqual(manager.messages[2]["content"], "I hear you clearly.")

    def test_24_unreliable_language_detection_does_not_inject_directive(self):
        """24. Ambiguous or unreliable detection avoids forcing an arbitrary language directive."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Subiza.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        # Ambiguous input
        engine.process_turn(TurnRequest(user_text="xyz999"))

        # Candidate sent to LLM should NOT contain a rigid language directive
        call_prompt = mock_llm.generate_completion.call_args[0][0][0]["content"]
        self.assertNotIn("[Current Turn Context]", call_prompt)

    # -------------------------------------------------------------------------
    # Chunk 3A Defect Regression Tests
    # -------------------------------------------------------------------------

    def test_25_english_false_positives_avoided(self):
        """25. English sentences containing words like software, bicycle, sinister do not match Kinyarwanda."""
        cases = [
            ("This fancy software has cyber security.", "en"),
            ("There is a sinister software between us.", "en"),
            ("The bicycle is near the subway.", "en"),
            ("The country has a new network policy.", "en"),
        ]
        for sentence, expected_lang in cases:
            res = self.detector.detect(sentence)
            self.assertEqual(res.language, expected_lang, f"Failed for '{sentence}'")
            self.assertTrue(res.is_reliable, f"Expected reliable for '{sentence}'")
            self.assertFalse(res.is_mixed, f"Expected non-mixed for '{sentence}'")

        # Unknown / non-word tokens do not get classified as Kinyarwanda solely due to software
        res_gorp = self.detector.detect("gorp snort software")
        self.assertNotEqual(res_gorp.language, "rw")
        self.assertFalse(res_gorp.is_reliable)

    def test_26_french_diacritic_false_positives_avoided(self):
        """26. English sentences with borrowed accented words (café, résumé, déjà vu) do not trigger French."""
        cases = [
            "Please send your résumé to our email.",
            "I had coffee at a café.",
            "The software is déjà vu for me.",
        ]
        for sentence in cases:
            res = self.detector.detect(sentence)
            self.assertEqual(res.language, "en", f"Failed for '{sentence}'")
            self.assertTrue(res.is_reliable, f"Expected reliable for '{sentence}'")
            self.assertFalse(res.is_mixed, f"Expected non-mixed for '{sentence}'")

    def test_27_repetition_confidence_inflation_prevented(self):
        """27. Repetition of identical tokens does not artificially inflate confidence."""
        # Ambiguous function word repeated
        res_the = self.detector.detect("the the the")
        self.assertFalse(res_the.is_reliable)
        self.assertLessEqual(res_the.confidence, 0.45)

        # Genuine greeting repeated has capped greeting-level confidence
        res_rw = self.detector.detect("muraho muraho muraho")
        self.assertEqual(res_rw.language, "rw")
        self.assertTrue(res_rw.is_reliable)
        self.assertLessEqual(res_rw.confidence, 0.80)

        res_fr = self.detector.detect("bonjour bonjour bonjour")
        self.assertEqual(res_fr.language, "fr")
        self.assertTrue(res_fr.is_reliable)
        self.assertLessEqual(res_fr.confidence, 0.80)

    def test_28_genuine_conversational_kinyarwanda(self):
        """28. Genuine natural conversational Kinyarwanda sentences are reliably identified."""
        sentences = [
            "Ndumva mfite agahinda kenshi mu mutima wanjye, uramfasha iki?",
            "Ndagushimira cyane kubw'inama nziza umpaye uyu munsi.",
            "Turashaka kuganira ku bibazo by'ubuzima n'umuryango wacu.",
        ]
        for s in sentences:
            res = self.detector.detect(s)
            self.assertEqual(res.language, "rw", f"Failed for '{s}'")
            self.assertTrue(res.is_reliable, f"Expected reliable for '{s}'")
            self.assertFalse(res.is_mixed, f"Expected non-mixed for '{s}'")
            self.assertGreaterEqual(res.confidence, 0.80)

    def test_29_proper_nouns_do_not_trigger_mixed_language(self):
        """29. Proper nouns resembling function words (e.g. 'Will') do not trigger mixed language."""
        res = self.detector.detect("Muraho Will.")
        self.assertEqual(res.language, "rw")
        self.assertTrue(res.is_reliable)
        self.assertFalse(res.is_mixed)

    def test_30_longer_kinyarwanda_with_english_clause(self):
        """30. Longer Kinyarwanda sentence with an English clause detects 'rw' as dominant and is_mixed=True."""
        text = "Ndumva mfite ikibazo gikomeye cyane mu mutima wanjye ariko I will manage."
        res = self.detector.detect(text)
        self.assertEqual(res.language, "rw")
        self.assertTrue(res.is_reliable)
        self.assertTrue(res.is_mixed)

    def test_31_longer_english_with_kinyarwanda_expression(self):
        """31. Longer English sentence with Kinyarwanda expression detects 'en' as dominant and is_mixed=True."""
        text = "I have been thinking about this all day and I feel like mfite ubwoba bwinshi."
        res = self.detector.detect(text)
        self.assertEqual(res.language, "en")
        self.assertTrue(res.is_reliable)
        self.assertTrue(res.is_mixed)

    def test_32_balanced_multilingual_input(self):
        """32. Balanced multilingual input with equal communicative weight preserves uncertainty."""
        text = "Je suis fatigué, ariko ndagerageza."
        res = self.detector.detect(text)
        self.assertIsNone(res.language)
        self.assertFalse(res.is_reliable)
        self.assertTrue(res.is_mixed)

    def test_33_english_with_kinyarwanda_wellness_expression(self):
        """33. English sentence containing a Kinyarwanda emotion term detects 'en' with is_mixed=True."""
        text = "I have a lot of agahinda in my heart."
        res = self.detector.detect(text)
        self.assertEqual(res.language, "en")
        self.assertTrue(res.is_reliable)
        self.assertTrue(res.is_mixed)


if __name__ == "__main__":
    unittest.main()
