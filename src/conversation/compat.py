"""Small compatibility helpers for CLI/tests without pulling multimodal imports."""

from __future__ import annotations

from typing import Union

from src.conversation.engine import ConversationEngine
from src.llm.conversation import ConversationManager


def ensure_conversation_engine(
    target: Union[ConversationEngine, ConversationManager],
) -> ConversationEngine:
    """
    Accepts ConversationEngine or ConversationManager (legacy callers).
    Wraps a bare manager in ConversationEngine without duplicating state.
    """
    if isinstance(target, ConversationEngine):
        return target
    return ConversationEngine(manager=target)