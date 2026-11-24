import unittest
from unittest.mock import MagicMock

from src.conversation.session import (
    ConversationMode,
    SessionState,
    TurnRequest,
    TurnResult,
)
from src.conversation.engine import ConversationEngine
from src.llm.conversation import ConversationManager, OversizedMessageError
from src.conversation.compat import ensure_conversation_engine as _ensure_engine


class TestConversationEngine(unittest.TestCase):
    """
    Unit tests for Chunk 1: ConversationEngine and Session domain structures.
    Uses mocks to ensure fast, deterministic execution without loading model weights.
    """

    def setUp(self):
        self.mock_manager = MagicMock(spec=ConversationManager)
        self.mock_manager.messages = [
            {"role": "system", "content": "System prompt"}
        ]

    def test_01_successful_delegation_to_conversation_manager(self):
        """1. Verify TurnRequest is validated and delegated to ConversationManager.generate_response."""
        self.mock_manager.generate_response.return_value = "Muraho neza!"
        engine = ConversationEngine(manager=self.mock_manager)

        request = TurnRequest(user_text="Muraho")
        result = engine.process_turn(request)

        self.mock_manager.generate_response.assert_called_once()
        self.assertEqual(self.mock_manager.generate_response.call_args[0][0], "Muraho")
        self.assertIsInstance(result, TurnResult)
        self.assertEqual(result.response_text, "Muraho neza!")
        self.assertEqual(result.mode, ConversationMode.NORMAL)

    def test_02_correct_construction_of_turn_result(self):
        """2. Verify TurnResult is constructed with expected typed fields."""
        self.mock_manager.generate_response.return_value = "Inama nziza."
        state = SessionState(mode=ConversationMode.GENTLE)
        engine = ConversationEngine(manager=self.mock_manager, state=state)

        request = TurnRequest(user_text="Mfasha", mode=ConversationMode.MOTIVATIONAL)
        result = engine.process_turn(request)

        # Mode updated in state and reflected in result
        self.assertEqual(result.mode, ConversationMode.MOTIVATIONAL)
        self.assertEqual(engine.state.mode, ConversationMode.MOTIVATIONAL)
        self.assertEqual(result.response_text, "Inama nziza.")
        self.assertIn("raw_user_input", result.metadata)

    def test_03_empty_and_whitespace_input_handling(self):
        """3. Empty or whitespace-only input returns empty TurnResult without calling ConversationManager."""
        engine = ConversationEngine(manager=self.mock_manager)

        # Empty string
        res1 = engine.process_turn(TurnRequest(user_text=""))
        self.assertEqual(res1.response_text, "")
        self.mock_manager.generate_response.assert_not_called()

        # Whitespace string
        res2 = engine.process_turn(TurnRequest(user_text="   \n\t  "))
        self.assertEqual(res2.response_text, "")
        self.mock_manager.generate_response.assert_not_called()

        # None input
        res3 = engine.process_turn(TurnRequest(user_text=None))
        self.assertEqual(res3.response_text, "")
        self.mock_manager.generate_response.assert_not_called()

    def test_04_propagation_of_generation_errors(self):
        """4. Verify generation errors (e.g. OversizedMessageError, runtime exceptions) propagate cleanly."""
        self.mock_manager.generate_response.side_effect = OversizedMessageError("Prompt exceeds context budget")
        engine = ConversationEngine(manager=self.mock_manager)

        with self.assertRaises(OversizedMessageError) as ctx:
            engine.process_turn(TurnRequest(user_text="A" * 10000))
        self.assertIn("exceeds context budget", str(ctx.exception))

        # Runtime inference error
        self.mock_manager.generate_response.side_effect = RuntimeError("Inference engine error")
        with self.assertRaises(RuntimeError) as ctx2:
            engine.process_turn(TurnRequest(user_text="Test crash"))
        self.assertIn("Inference engine error", str(ctx2.exception))

    def test_05_conversation_history_remains_managed_by_manager(self):
        """5. Conversation history property points directly to manager.messages without duplication."""
        self.mock_manager.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ]
        engine = ConversationEngine(manager=self.mock_manager)

        # engine.messages is an alias to manager.messages
        self.assertEqual(engine.messages, self.mock_manager.messages)
        self.assertIs(engine.messages, self.mock_manager.messages)

    def test_06_no_duplicate_history_or_independent_state(self):
        """6. Ensure ConversationEngine does not maintain a separate message list."""
        engine = ConversationEngine(manager=self.mock_manager)
        self.assertFalse(hasattr(engine, "history"))
        self.assertFalse(hasattr(engine.state, "messages"))
        self.assertFalse(hasattr(engine.state, "history"))

    def test_07_reset_conversation_delegates_to_manager(self):
        """7. Verify reset_conversation delegates directly to manager.reset_conversation."""
        engine = ConversationEngine(manager=self.mock_manager)
        engine.reset_conversation()
        self.mock_manager.reset_conversation.assert_called_once()

    def test_08_ensure_engine_backward_compatibility(self):
        """8. Verify _ensure_engine accepts both ConversationEngine and ConversationManager."""
        # 1. Existing ConversationEngine passes through
        engine = ConversationEngine(manager=self.mock_manager)
        self.assertIs(_ensure_engine(engine), engine)

        # 2. ConversationManager is automatically wrapped in ConversationEngine
        wrapped = _ensure_engine(self.mock_manager)
        self.assertIsInstance(wrapped, ConversationEngine)
        self.assertIs(wrapped.manager, self.mock_manager)


if __name__ == "__main__":
    unittest.main()
