"""
Unit tests for Chunk 2: Structured Response Contract, Response Parser, and Integration.
Tests all validation rules, edge cases, error modes, and multilingual character handling.
"""

import json
import unittest
from unittest.mock import MagicMock

from src.llm.response_parser import (
    InteractionIntent,
    ParsedResponse,
    ResponseParsingError,
    parse_response,
    safe_parse_response,
)
from src.llm.conversation import ConversationManager
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.conversation.engine import ConversationEngine
from src.conversation.session import TurnRequest


class TestResponseContract(unittest.TestCase):

    def test_01_valid_json_response(self):
        """1. Parses a canonical valid JSON response matching the contract."""
        payload = json.dumps({
            "response_text": "Muraho! Niteguye kugutega amatwi.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        parsed = parse_response(payload)
        self.assertIsInstance(parsed, ParsedResponse)
        self.assertEqual(parsed.response_text, "Muraho! Niteguye kugutega amatwi.")
        self.assertEqual(parsed.interaction_intent, InteractionIntent.LISTENING)
        self.assertFalse(parsed.requires_clarification)
        self.assertIsNotNone(parsed.raw_json)

    def test_02_all_supported_interaction_intents(self):
        """2. Validates all four supported interaction intents."""
        intents = [
            InteractionIntent.LISTENING,
            InteractionIntent.REFLECTION,
            InteractionIntent.PRACTICAL_ADVICE,
            InteractionIntent.UNCLEAR,
        ]
        for intent in intents:
            payload = json.dumps({
                "response_text": f"Response for {intent.value}",
                "interaction_intent": intent.value,
                "requires_clarification": (intent == InteractionIntent.UNCLEAR),
            })
            parsed = parse_response(payload)
            self.assertEqual(parsed.interaction_intent, intent)
            self.assertEqual(parsed.requires_clarification, (intent == InteractionIntent.UNCLEAR))

    def test_03_missing_response_text(self):
        """3. Rejects response missing 'response_text'."""
        payload = json.dumps({
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        with self.assertRaises(ResponseParsingError) as ctx:
            parse_response(payload)
        self.assertIn("Missing required field 'response_text'", str(ctx.exception))

    def test_04_missing_interaction_intent(self):
        """4. Rejects response missing 'interaction_intent'."""
        payload = json.dumps({
            "response_text": "Hello there.",
            "requires_clarification": False,
        })
        with self.assertRaises(ResponseParsingError) as ctx:
            parse_response(payload)
        self.assertIn("Missing required field 'interaction_intent'", str(ctx.exception))

    def test_05_missing_requires_clarification(self):
        """5. Rejects response missing 'requires_clarification'."""
        payload = json.dumps({
            "response_text": "Hello there.",
            "interaction_intent": "LISTENING",
        })
        with self.assertRaises(ResponseParsingError) as ctx:
            parse_response(payload)
        self.assertIn("Missing required field 'requires_clarification'", str(ctx.exception))

    def test_06_empty_response_text(self):
        """6. Rejects empty string in 'response_text'."""
        payload = json.dumps({
            "response_text": "",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        with self.assertRaises(ResponseParsingError) as ctx:
            parse_response(payload)
        self.assertIn("must not be empty", str(ctx.exception))

    def test_07_whitespace_only_response_text(self):
        """7. Rejects whitespace-only 'response_text'."""
        payload = json.dumps({
            "response_text": "   \n\t  ",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        with self.assertRaises(ResponseParsingError) as ctx:
            parse_response(payload)
        self.assertIn("must not be empty or whitespace-only", str(ctx.exception))

    def test_08_invalid_intent_value(self):
        """8. Rejects unsupported intent values without silently guessing."""
        invalid_intents = ["ADVICE", "CHATTING", "listening", "UNKNOWN", "EMPATHY", 123]
        for bad_intent in invalid_intents:
            payload = json.dumps({
                "response_text": "Hello.",
                "interaction_intent": bad_intent,
                "requires_clarification": False,
            })
            with self.assertRaises(ResponseParsingError) as ctx:
                parse_response(payload)
            self.assertTrue(
                "Invalid interaction_intent" in str(ctx.exception)
                or "must be a string" in str(ctx.exception)
            )

    def test_09_incorrect_field_types(self):
        """9. Rejects incorrect data types for fields."""
        # response_text is an int
        p1 = json.dumps({
            "response_text": 12345,
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        with self.assertRaises(ResponseParsingError) as ctx1:
            parse_response(p1)
        self.assertIn("'response_text' must be a string", str(ctx1.exception))

        # requires_clarification is a string instead of bool
        p2 = json.dumps({
            "response_text": "Hello",
            "interaction_intent": "LISTENING",
            "requires_clarification": "false",
        })
        with self.assertRaises(ResponseParsingError) as ctx2:
            parse_response(p2)
        self.assertIn("'requires_clarification' must be a boolean", str(ctx2.exception))

        # requires_clarification is an integer (1)
        p3 = json.dumps({
            "response_text": "Hello",
            "interaction_intent": "LISTENING",
            "requires_clarification": 1,
        })
        with self.assertRaises(ResponseParsingError) as ctx3:
            parse_response(p3)
        self.assertIn("'requires_clarification' must be a boolean", str(ctx3.exception))

    def test_10_malformed_json(self):
        """10. Handles malformed JSON syntax without crashing."""
        malformed_inputs = [
            "This is just plain conversational text, not JSON.",
            "{'response_text': 'Single quotes are invalid in JSON'}",
            '{"response_text": "Unclosed object...',
            "",
            "   ",
            None,
        ]
        for bad in malformed_inputs:
            with self.assertRaises(ResponseParsingError):
                parse_response(bad)

            # safe_parse_response returns (None, error_msg)
            parsed, err = safe_parse_response(bad)
            self.assertIsNone(parsed)
            self.assertIsNotNone(err)

    def test_11_json_wrapped_in_markdown_code_fences(self):
        """11. Safely strips Markdown code fences surrounding JSON payload."""
        fenced_1 = """```json
{
  "response_text": "Muraho neza! Amakuru ki?",
  "interaction_intent": "LISTENING",
  "requires_clarification": false
}
```"""
        parsed_1 = parse_response(fenced_1)
        self.assertEqual(parsed_1.response_text, "Muraho neza! Amakuru ki?")
        self.assertEqual(parsed_1.interaction_intent, InteractionIntent.LISTENING)

        fenced_2 = """```
{
  "response_text": "Plain fence response.",
  "interaction_intent": "REFLECTION",
  "requires_clarification": false
}
```"""
        parsed_2 = parse_response(fenced_2)
        self.assertEqual(parsed_2.response_text, "Plain fence response.")
        self.assertEqual(parsed_2.interaction_intent, InteractionIntent.REFLECTION)

    def test_12_extra_unexpected_fields(self):
        """12. Tolerates extra unexpected fields and preserves them in raw_json."""
        payload = json.dumps({
            "response_text": "Valid response with extra metadata.",
            "interaction_intent": "PRACTICAL_ADVICE",
            "requires_clarification": False,
            "extra_field": "bonus_info",
            "confidence": 0.95,
        })
        parsed = parse_response(payload)
        self.assertEqual(parsed.response_text, "Valid response with extra metadata.")
        self.assertEqual(parsed.interaction_intent, InteractionIntent.PRACTICAL_ADVICE)
        self.assertEqual(parsed.raw_json.get("extra_field"), "bonus_info")
        self.assertEqual(parsed.raw_json.get("confidence"), 0.95)

    def test_13_unicode_and_kinyarwanda_characters(self):
        """13. Accurately preserves Kinyarwanda apostrophes, elisions, and special characters."""
        kinya_text = "N'ibyo rwose, ndakumva. Twaganira ku by'ubuzima bwawe niba ubishaka."
        payload = json.dumps({
            "response_text": kinya_text,
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        }, ensure_ascii=False)
        parsed = parse_response(payload)
        self.assertEqual(parsed.response_text, kinya_text)
        self.assertIn("N'ibyo", parsed.response_text)
        self.assertIn("by'ubuzima", parsed.response_text)

    def test_14_french_accents_and_punctuation(self):
        """14. Accurately preserves French accents, guillemets, and apostrophes."""
        fr_text = "Bonjour ! J'espère que vous vous sentez un peu plus apaisé aujourd'hui. Qu'en pensez-vous ?"
        payload = json.dumps({
            "response_text": fr_text,
            "interaction_intent": "REFLECTION",
            "requires_clarification": False,
        }, ensure_ascii=False)
        parsed = parse_response(payload)
        self.assertEqual(parsed.response_text, fr_text)
        self.assertIn("apaisé", parsed.response_text)
        self.assertIn("aujourd'hui", parsed.response_text)

    def test_15_english_responses(self):
        """15. Accurately handles English conversational responses."""
        en_text = "I hear how exhausting that must have been. Take all the time you need to share."
        payload = json.dumps({
            "response_text": en_text,
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        parsed = parse_response(payload)
        self.assertEqual(parsed.response_text, en_text)
        self.assertEqual(parsed.interaction_intent, InteractionIntent.LISTENING)

    def test_16_newlines_quotes_and_punctuation(self):
        """16. Accurately handles internal quotes, commas, exclamation marks, and newlines."""
        complex_text = 'He said, "It is fine," but it was not.\n\nTake a slow breath. We can take this step by step.'
        payload = json.dumps({
            "response_text": complex_text,
            "interaction_intent": "PRACTICAL_ADVICE",
            "requires_clarification": False,
        })
        parsed = parse_response(payload)
        self.assertEqual(parsed.response_text, complex_text)
        self.assertIn('"It is fine,"', parsed.response_text)
        self.assertIn("\n\n", parsed.response_text)

    def test_17_conversation_manager_integration_with_json_response(self):
        """17. Verifies ConversationManager extracts response_text into history when receiving contract JSON."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        contract_json = json.dumps({
            "response_text": "Muraho neza! Amakuru ki?",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        mock_llm.generate_completion.return_value = contract_json

        manager = ConversationManager(llm=mock_llm)
        resp = manager.generate_response("Bite")

        # Returned text is pure response_text, not raw JSON
        self.assertEqual(resp, "Muraho neza! Amakuru ki?")

        # Committed assistant message in history contains pure response_text
        self.assertEqual(manager.messages[-1]["role"], "assistant")
        self.assertEqual(manager.messages[-1]["content"], "Muraho neza! Amakuru ki?")

        # Parsed response metadata is stored on manager
        self.assertIsNotNone(manager.last_parsed_response)
        self.assertEqual(manager.last_parsed_response.interaction_intent, InteractionIntent.LISTENING)

    def test_18_conversation_engine_captures_intent_metadata(self):
        """18. Verifies ConversationEngine exposes interaction_intent in TurnResult metadata."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        contract_json = json.dumps({
            "response_text": "Ni byiza cyane.",
            "interaction_intent": "PRACTICAL_ADVICE",
            "requires_clarification": False,
        })
        mock_llm.generate_completion.return_value = contract_json

        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        result = engine.process_turn(TurnRequest(user_text="Mfasha"))
        self.assertEqual(result.response_text, "Ni byiza cyane.")
        self.assertEqual(result.metadata.get("interaction_intent"), "PRACTICAL_ADVICE")
        self.assertFalse(result.metadata.get("requires_clarification"))

    def test_19_plain_text_output_raises_parsing_error(self):
        """19. ConversationManager raises ResponseParsingError when model returns plain text."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = "Just some conversational plain text, no JSON."
        manager = ConversationManager(llm=mock_llm)

        with self.assertRaises(ResponseParsingError):
            manager.generate_response("Muraho")

    def test_20_malformed_json_raises_parsing_error(self):
        """20. ConversationManager raises ResponseParsingError on malformed/unclosed JSON."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = '{"response_text": "Incomplete json...'
        manager = ConversationManager(llm=mock_llm)

        with self.assertRaises(ResponseParsingError):
            manager.generate_response("Muraho")

    def test_21_missing_required_fields_raises_parsing_error(self):
        """21. ConversationManager raises ResponseParsingError when required field is missing."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Muraho!",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)

        with self.assertRaises(ResponseParsingError):
            manager.generate_response("Muraho")

    def test_22_unsupported_intent_raises_parsing_error(self):
        """22. ConversationManager raises ResponseParsingError on unsupported interaction_intent."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Muraho!",
            "interaction_intent": "DIAGNOSIS_INVALID",
            "requires_clarification": False,
        })
        manager = ConversationManager(llm=mock_llm)

        with self.assertRaises(ResponseParsingError):
            manager.generate_response("Muraho")

    def test_23_invalid_output_never_added_to_history(self):
        """23. Conversation history is atomic: parsing failure leaves messages unmodified."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = "Broken plain text output."
        manager = ConversationManager(llm=mock_llm)
        initial_history_len = len(manager.messages)

        with self.assertRaises(ResponseParsingError):
            manager.generate_response("Muraho")

        self.assertEqual(len(manager.messages), initial_history_len)
        self.assertEqual(manager.messages[0]["role"], "system")

    def test_24_listening_turn_followed_by_malformed_output_clears_metadata(self):
        """24. Stale metadata isolation: malformed output resets last_parsed_response to None."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        valid_first_turn = json.dumps({
            "response_text": "Ndakumva neza.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        mock_llm.generate_completion.return_value = valid_first_turn
        manager = ConversationManager(llm=mock_llm)

        # Turn 1 succeeds
        manager.generate_response("Ndihebye")
        self.assertIsNotNone(manager.last_parsed_response)
        self.assertEqual(manager.last_parsed_response.interaction_intent, InteractionIntent.LISTENING)

        # Turn 2 fails with malformed output
        mock_llm.generate_completion.return_value = "Not a json"
        with self.assertRaises(ResponseParsingError):
            manager.generate_response("Urakomeye?")

        self.assertIsNone(manager.last_parsed_response)

    def test_25_listening_turn_followed_by_empty_output_clears_metadata(self):
        """25. Stale metadata isolation: empty output resets last_parsed_response to None."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        valid_first_turn = json.dumps({
            "response_text": "Ndakumva neza.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        mock_llm.generate_completion.return_value = valid_first_turn
        manager = ConversationManager(llm=mock_llm)

        manager.generate_response("Ndihebye")
        self.assertIsNotNone(manager.last_parsed_response)

        # Turn 2 generates empty
        mock_llm.generate_completion.return_value = ""
        resp = manager.generate_response("Urakomeye?")
        self.assertEqual(resp, "")
        self.assertIsNone(manager.last_parsed_response)

    def test_26_listening_turn_followed_by_generation_exception_clears_metadata(self):
        """26. Stale metadata isolation: generation exception resets last_parsed_response to None."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        valid_first_turn = json.dumps({
            "response_text": "Ndakumva neza.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        mock_llm.generate_completion.return_value = valid_first_turn
        manager = ConversationManager(llm=mock_llm)

        manager.generate_response("Ndihebye")
        self.assertIsNotNone(manager.last_parsed_response)

        # Turn 2 crashes in LLM
        mock_llm.generate_completion.side_effect = RuntimeError("GPU failure")
        with self.assertRaises(RuntimeError):
            manager.generate_response("Urakomeye?")

        self.assertIsNone(manager.last_parsed_response)

    def test_27_listening_turn_followed_by_practical_advice_updates_metadata(self):
        """27. Metadata correctly transitions from LISTENING to PRACTICAL_ADVICE."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        turn_1 = json.dumps({
            "response_text": "Ndakumva neza.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        turn_2 = json.dumps({
            "response_text": "Gerageza gufata umwuka buhoro buhoro.",
            "interaction_intent": "PRACTICAL_ADVICE",
            "requires_clarification": False,
        })
        mock_llm.generate_completion.side_effect = [turn_1, turn_2]
        manager = ConversationManager(llm=mock_llm)

        manager.generate_response("Mfite ubwoba")
        self.assertEqual(manager.last_parsed_response.interaction_intent, InteractionIntent.LISTENING)

        manager.generate_response("Inama wampa ni iyihe?")
        self.assertEqual(manager.last_parsed_response.interaction_intent, InteractionIntent.PRACTICAL_ADVICE)

    def test_28_empty_or_whitespace_turn_request_clears_metadata(self):
        """28. Empty or whitespace turn request clears last_parsed_response."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        valid_first_turn = json.dumps({
            "response_text": "Ndakumva.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        mock_llm.generate_completion.return_value = valid_first_turn
        manager = ConversationManager(llm=mock_llm)

        manager.generate_response("Muraho")
        self.assertIsNotNone(manager.last_parsed_response)

        # Empty string request
        resp = manager.generate_response("   ")
        self.assertEqual(resp, "")
        self.assertIsNone(manager.last_parsed_response)

    def test_29_conversation_engine_turn_result_metadata_isolation(self):
        """29. ConversationEngine does not leak metadata on empty turn requests or failed turns."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        valid_turn = json.dumps({
            "response_text": "Ni byiza.",
            "interaction_intent": "REFLECTION",
            "requires_clarification": False,
        })
        mock_llm.generate_completion.return_value = valid_turn
        manager = ConversationManager(llm=mock_llm)
        engine = ConversationEngine(manager=manager)

        res1 = engine.process_turn(TurnRequest(user_text="Amakuru"))
        self.assertEqual(res1.metadata.get("interaction_intent"), "REFLECTION")

        # Empty turn request
        res2 = engine.process_turn(TurnRequest(user_text="   "))
        self.assertEqual(res2.response_text, "")
        self.assertNotIn("interaction_intent", res2.metadata)
        self.assertIsNone(manager.last_parsed_response)


if __name__ == "__main__":
    unittest.main()
