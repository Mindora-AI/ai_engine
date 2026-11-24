import json
import unittest
from unittest.mock import MagicMock

from src.llm.conversation import ConversationManager
from src.llm.afrique_qwen import AfriqueQwenLLM


class TestRollingSummary(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=AfriqueQwenLLM)
        self.mock_llm.n_ctx = 200
        self.mock_llm.max_tokens = 40
        self.mock_llm.safety_margin = 8
        self.mock_llm.generate_completion.return_value = json.dumps(
            {
                "response_text": "Ndakumva.",
                "interaction_intent": "LISTENING",
                "requires_clarification": False,
            }
        )

        def _count(msgs):
            for m in msgs:
                c = m.get("content") or ""
                if "Conversation Summary" in c:
                    return 30
            if len(msgs) > 4:
                return 500
            return 30

        self.mock_llm.count_tokens.side_effect = _count
        self.mock_llm.generate_auxiliary_json.return_value = {
            "summary_text": "User shared stress about work."
        }

        self.manager = ConversationManager(
            llm=self.mock_llm,
            n_ctx=200,
            max_tokens=40,
            system_prompt="Short wellness system prompt.",
        )
        self.manager.verbatim_tail_turns = 1
        self.manager.rolling_summary_enabled = True

    def test_recent_turns_remain_verbatim(self):
        self.manager.rolling_summary_enabled = False
        self.manager.n_ctx = 4096
        self.mock_llm.n_ctx = 4096
        self.manager.messages.extend(
            [
                {"role": "user", "content": "Turn A user"},
                {"role": "assistant", "content": "Turn A assistant"},
                {"role": "user", "content": "Turn B user"},
                {"role": "assistant", "content": "Turn B assistant"},
            ]
        )
        self.manager.generate_response("Turn C user")
        roles = [m["role"] for m in self.manager.messages]
        contents = [m["content"] for m in self.manager.messages]
        self.assertIn("Turn B user", contents)
        self.assertIn("Turn C user", contents)

    def test_summary_injected_when_fold_occurs(self):
        self.manager.messages.extend(
            [
                {"role": "user", "content": "x" * 80},
                {"role": "assistant", "content": "y" * 80},
                {"role": "user", "content": "recent"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        self.manager.generate_response("new question")
        self.assertTrue(self.manager.rolling_summary.strip())
        self.mock_llm.generate_auxiliary_json.assert_called()

    def test_summarization_failure_does_not_corrupt_on_inference_error(self):
        self.manager.messages.extend(
            [
                {"role": "user", "content": "a" * 60},
                {"role": "assistant", "content": "b" * 60},
            ]
        )
        before = list(self.manager.messages)
        self.mock_llm.generate_completion.side_effect = RuntimeError("fail")
        self.mock_llm.generate_auxiliary_json.side_effect = RuntimeError("sum fail")
        with self.assertRaises(RuntimeError):
            self.manager.generate_response("hello")
        self.assertEqual(self.manager.messages, before)
        self.assertEqual(self.manager.rolling_summary, "")


if __name__ == "__main__":
    unittest.main()