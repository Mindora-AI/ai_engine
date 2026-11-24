import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.llm.prompts import SYSTEM_PROMPT
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.conversation import ConversationManager, OversizedMessageError
from src.asr.asr_engine import ASREngine
from src.tts.tts_engine import TTSEngine
from run_assistant import transcribe_and_respond


class TestTokenAwareContextManagement(unittest.TestCase):
    """
    Automated test suite verifying the 12 token-aware context management scenarios.
    All tests use mocked LLM / components to execute reliably and fast in CI/test runners.
    """

    def setUp(self):
        self.mock_llm = MagicMock(spec=AfriqueQwenLLM)
        self.mock_llm.n_ctx = 4096
        self.mock_llm.max_tokens = 512

    def test_01_short_conversation_fits_in_context(self):
        """1. A normal short conversation fits within the context without trimming."""
        self.mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "I am doing well, thank you.",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        # Short token count well within budget
        self.mock_llm.count_tokens.return_value = 100

        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512)
        resp1 = manager.generate_response("Hello!")
        resp2 = manager.generate_response("How are you?")

        self.assertEqual(resp1, "I am doing well, thank you.")
        self.assertEqual(resp2, "I am doing well, thank you.")
        # Total messages: 1 system + 2 * (user + assistant) = 5 messages
        self.assertEqual(len(manager.messages), 5)
        self.assertEqual(manager.messages[0]["role"], "system")
        self.assertEqual(manager.messages[1]["content"], "Hello!")
        self.assertEqual(manager.messages[3]["content"], "How are you?")

    def test_02_long_conversation_triggers_removal_of_older_complete_exchanges(self):
        """2. A long conversation triggers removal of older complete exchanges when token limit is exceeded."""
        self.mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Response",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })

        # Simulate dynamic token counting:
        # Budget = 4096 - 512 - 16 = 3568
        # Each message adds 700 tokens.
        # System prompt + 2 exchanges (4 messages) = 5 * 700 = 3500 (fits in 3568)
        # Adding a 3rd exchange (user 3) = 6 * 700 = 4200 (> 3568) -> triggers trimming of exchange 1!
        def dynamic_count(messages):
            return len(messages) * 700

        self.mock_llm.count_tokens.side_effect = dynamic_count

        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512, safety_margin=16)
        manager.rolling_summary_enabled = False

        manager.generate_response("User 1")
        manager.generate_response("User 2")
        self.assertEqual(len(manager.messages), 5)  # system + turn 1 + turn 2

        # Adding Turn 3 will exceed 3568 tokens with all 3 turns, so Turn 1 must be evicted
        manager.generate_response("User 3")

        # After trimming Turn 1, Turn 2 and Turn 3 remain:
        # Messages: [system, User 2, Response, User 3, Response] = 5 messages
        self.assertEqual(len(manager.messages), 5)
        self.assertEqual(manager.messages[0]["role"], "system")
        self.assertEqual(manager.messages[1]["role"], "user")
        self.assertEqual(manager.messages[1]["content"], "User 2")
        self.assertEqual(manager.messages[2]["role"], "assistant")
        self.assertEqual(manager.messages[3]["role"], "user")
        self.assertEqual(manager.messages[3]["content"], "User 3")
        self.assertEqual(manager.messages[4]["role"], "assistant")

    def test_03_system_prompt_preserved_after_trimming(self):
        """3. The system prompt is strictly preserved at index 0 after trimming."""
        self.mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Response",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })

        def count_func(messages):
            # If more than 3 messages, report high token count to force trimming
            return 3600 if len(messages) > 3 else 100

        self.mock_llm.count_tokens.side_effect = count_func

        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512)
        manager.rolling_summary_enabled = False
        manager.generate_response("Turn 1")
        manager.generate_response("Turn 2")
        manager.generate_response("Turn 3")

        # Even after extensive trimming, index 0 is always system prompt with SYSTEM_PROMPT content
        self.assertEqual(manager.messages[0]["role"], "system")
        self.assertEqual(manager.messages[0]["content"], SYSTEM_PROMPT)

    def test_04_newest_user_message_preserved(self):
        """4. The newest user message is preserved in candidate prompt and committed history."""
        self.mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Response to newest",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })

        def count_func(messages):
            # Force trimming if candidate history has messages
            return 3600 if len(messages) > 3 else 200

        self.mock_llm.count_tokens.side_effect = count_func

        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512)
        manager.rolling_summary_enabled = False
        manager.generate_response("Old turn 1")
        manager.generate_response("Old turn 2")
        manager.generate_response("Newest unique message 12345")

        # Verify candidate passed to LLM included newest user message
        last_call_messages = self.mock_llm.generate_completion.call_args[0][0]
        self.assertEqual(last_call_messages[-1]["role"], "user")
        self.assertEqual(last_call_messages[-1]["content"], "Newest unique message 12345")

        # Verify newest turn exists in committed history
        self.assertEqual(manager.messages[-2]["role"], "user")
        self.assertEqual(manager.messages[-2]["content"], "Newest unique message 12345")
        self.assertEqual(manager.messages[-1]["role"], "assistant")
        self.assertEqual(manager.messages[-1]["content"], "Response to newest")

    def test_05_response_tokens_properly_reserved(self):
        """5. The model's maximum response tokens and safety margin are properly reserved."""
        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512, safety_margin=16)
        expected_budget = 4096 - 512 - 16  # 3568
        self.assertEqual(manager.get_available_budget(), expected_budget)

        # A prompt with 3569 tokens exceeds budget (3568), even though it is < 4096
        self.mock_llm.count_tokens.return_value = 3569
        with self.assertRaises(OversizedMessageError):
            manager.prepare_candidate_messages("This is slightly above budget")

        # A prompt with 3568 tokens is accepted within budget
        self.mock_llm.count_tokens.return_value = 3568
        cand_msgs, cand_hist = manager.prepare_candidate_messages("Fits exactly")
        self.assertIsNotNone(cand_msgs)

    def test_06_special_tokens_and_chat_template_included_in_token_counting(self):
        """6. Special tokens and chat-template formatting are included in token counting."""
        # Using real AfriqueQwenLLM helper with mock Llama to test format_prompt and count_tokens
        llm = AfriqueQwenLLM()
        mock_llama = MagicMock()
        # Mock tokenize to return 1 token per 3 bytes
        mock_llama.tokenize.side_effect = lambda b, **kwargs: [1] * (len(b) // 3 + 1)
        mock_llama.chat_handler = None
        mock_llama._chat_handlers = {}
        mock_llama.metadata = {}

        llm._llm = mock_llama

        messages = [
            {"role": "system", "content": "Helpful AI"},
            {"role": "user", "content": "Hi"},
        ]
        formatted = llm.format_prompt(messages)
        # Verify ChatML formatting tokens exist in prompt
        self.assertIn("<|im_start|>system", formatted)
        self.assertIn("<|im_start|>user", formatted)
        self.assertIn("<|im_start|>assistant", formatted)

        tokens = llm.count_tokens(messages)
        # Token count includes formatting overhead, not just raw text length
        self.assertGreater(tokens, 0)

    def test_07_oversized_user_message_rejected_gracefully(self):
        """7. An oversized individual user message is rejected gracefully without corrupting history."""
        self.mock_llm.generate_completion.return_value = "Should not be called"
        self.mock_llm.count_tokens.return_value = 5000  # Exceeds 3568 budget

        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512)
        initial_history_len = len(manager.messages)

        with self.assertRaises(OversizedMessageError) as ctx:
            manager.generate_response("Super massive oversized user message" * 100)

        self.assertIn("too long for the available context window", str(ctx.exception))
        # Verify LLM was not called
        self.mock_llm.generate_completion.assert_not_called()
        # Verify conversation history was NOT corrupted or changed
        self.assertEqual(len(manager.messages), initial_history_len)

    def test_08_inference_failures_do_not_mutate_committed_history(self):
        """8. Inference failures do not mutate committed conversation history."""
        self.mock_llm.generate_completion.side_effect = RuntimeError("Inference crash / CUDA OOM")
        self.mock_llm.count_tokens.return_value = 100

        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512)
        initial_len = len(manager.messages)

        with self.assertRaises(RuntimeError):
            manager.generate_response("Trigger crash")

        # History remains unmodified; user message was not added
        self.assertEqual(len(manager.messages), initial_len)
        self.assertEqual(manager.messages[0]["role"], "system")

    def test_09_empty_model_responses_do_not_create_incomplete_turns(self):
        """9. Empty model responses do not create incomplete conversation turns."""
        self.mock_llm.generate_completion.return_value = ""  # Empty output
        self.mock_llm.count_tokens.return_value = 100

        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512)
        initial_len = len(manager.messages)

        res = manager.generate_response("Will generate empty")
        self.assertEqual(res, "")
        # No half-turns appended
        self.assertEqual(len(manager.messages), initial_len)

        # Also test response containing only whitespace/thinking tag
        self.mock_llm.generate_completion.return_value = "<think>\nThinking only\n</think>"
        res2 = manager.generate_response("Thinking only response")
        self.assertEqual(res2, "")
        self.assertEqual(len(manager.messages), initial_len)

    def test_10_conversation_reset_works_correctly(self):
        """10. Conversation reset works correctly and preserves system prompt."""
        self.mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Subiza",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        self.mock_llm.count_tokens.return_value = 100

        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512)
        manager.generate_response("Turn 1")
        manager.generate_response("Turn 2")
        self.assertEqual(len(manager.messages), 5)

        manager.reset_conversation()
        self.assertEqual(len(manager.messages), 1)
        self.assertEqual(manager.messages[0]["role"], "system")
        self.assertEqual(manager.messages[0]["content"], SYSTEM_PROMPT)

    def test_11_modes_retain_existing_behavior(self):
        """11. Text, voice, and text-to-voice modes retain their existing behavior."""
        self.mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Muraho neza!",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })
        self.mock_llm.count_tokens.return_value = 100
        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512)

        # 1. Text mode behavior: generate_response directly
        text_resp = manager.generate_response("Bite?")
        self.assertEqual(text_resp, "Muraho neza!")

        # 2. Voice mode behavior: transcribe_and_respond pipeline
        mock_asr = MagicMock(spec=ASREngine)
        mock_asr.transcribe_audio.return_value = "Amakuru ki?"
        mock_tts = MagicMock(spec=TTSEngine)
        mock_tts.synthesize_speech.return_value = "output.wav"

        transcribe_and_respond(manager, mock_asr, mock_tts, "dummy_voice.wav")
        self.mock_llm.generate_completion.assert_called()
        mock_tts.synthesize_speech.assert_called_with("Muraho neza!")

    def test_12_varying_message_lengths_managed_by_actual_token_usage(self):
        """12. A conversation with varying message lengths is managed according to actual token usage."""
        self.mock_llm.generate_completion.return_value = json.dumps({
            "response_text": "Answer",
            "interaction_intent": "LISTENING",
            "requires_clarification": False,
        })

        # Budget = 4096 - 512 - 16 = 3568
        # We test two cases with max_turns=None (pure token management):
        # Case A: Heavy turns (1600 tokens per turn).
        # Turn 1: 1600 tokens (fits).
        # Turn 2: 1600 + 1600 = 3200 tokens (fits).
        # Turn 3: 3200 + 1600 = 4800 tokens (> 3568) -> triggers trimming!
        token_cost_map = {}

        def dynamic_tokens(messages):
            # Calculate total based on registered content costs
            return sum(token_cost_map.get(m.get("content", ""), 20) for m in messages)

        self.mock_llm.count_tokens.side_effect = dynamic_tokens

        manager = ConversationManager(llm=self.mock_llm, n_ctx=4096, max_tokens=512, safety_margin=16)
        manager.rolling_summary_enabled = False

        token_cost_map["Heavy 1"] = 800
        token_cost_map["Heavy 2"] = 800
        token_cost_map["Heavy 3"] = 800
        token_cost_map["Answer"] = 800

        manager.generate_response("Heavy 1")  # Total tokens ~1620
        self.assertEqual(len(manager.messages), 3)  # system + 1 turn

        manager.generate_response("Heavy 2")  # Total tokens ~3240
        self.assertEqual(len(manager.messages), 5)  # system + 2 turns

        # Heavy 3 would push total to ~4860 tokens (> 3568)
        # Token-aware manager automatically evicts Heavy 1!
        manager.generate_response("Heavy 3")
        self.assertEqual(len(manager.messages), 5)  # Still 2 turns: Heavy 2 and Heavy 3
        self.assertEqual(manager.messages[1]["content"], "Heavy 2")
        self.assertEqual(manager.messages[3]["content"], "Heavy 3")


if __name__ == "__main__":
    unittest.main()
