"""Deterministic gates for unsolicited exercise recommendations (Gate 1 policy)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.safety.categories import ApplicationAction, SafetyCategory

MAX_RECOMMENDATION_CONTEXT_MESSAGES = 6


def unsolicited_offer_allowed(
    safety_action: ApplicationAction,
    safety_category: SafetyCategory,
) -> bool:
    """
    Explicit Gate 1 policy for unsolicited wellness exercise offers.
    Does not apply to explicit exercise requests (handled before LLM).
    """
    if safety_action != ApplicationAction.NORMAL_CONVERSATION:
        return False
    if safety_category == SafetyCategory.ORDINARY_DISTRESS:
        return False
    return True


def recent_conversation_excerpt(
    messages: Sequence[Dict[str, str]],
    rolling_summary: str,
    *,
    max_messages: int = MAX_RECOMMENDATION_CONTEXT_MESSAGES,
) -> Dict[str, object]:
    """Read-only slice of user/assistant turns for suitability decisions."""
    history = [
        {"role": m.get("role"), "content": (m.get("content") or "").strip()}
        for m in messages[1:]
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]
    tail = history[-max_messages:] if max_messages > 0 else []
    summary = (rolling_summary or "").strip()
    return {
        "recent_turns": tail,
        "rolling_summary": summary if summary else None,
    }