"""Rolling conversation summarization when context budget requires folding older turns."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from src.memory.prompts import ROLLING_SUMMARY_SYSTEM

DEFAULT_ROLLING_SUMMARY_MAX_CHARS = int(
    os.environ.get("MWIZERWA_ROLLING_SUMMARY_MAX_CHARS", "6000")
)


def rolling_summary_within_bound(text: str) -> bool:
    limit = DEFAULT_ROLLING_SUMMARY_MAX_CHARS
    return limit <= 0 or len(text.strip()) <= limit

ROLLING_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary_text": {"type": "string"}},
    "required": ["summary_text"],
}


def _format_turn_pairs(pairs: List[Dict[str, str]]) -> str:
    lines = []
    for msg in pairs:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def merge_turns_into_summary(
    llm,
    existing_summary: str,
    pairs_to_fold: List[Dict[str, str]],
) -> Optional[str]:
    """
    Returns updated summary_text or None on failure (caller should fall back to trim-only).
    """
    if not pairs_to_fold:
        return existing_summary or None

    payload = {
        "existing_summary": existing_summary.strip(),
        "new_turns": _format_turn_pairs(pairs_to_fold),
    }
    messages = [
        {"role": "system", "content": ROLLING_SUMMARY_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        if not hasattr(llm, "generate_auxiliary_json"):
            return None
        data = llm.generate_auxiliary_json(
            messages,
            schema=ROLLING_SUMMARY_SCHEMA,
            max_tokens=384,
        )
        text = (data.get("summary_text") or "").strip()
        if not text:
            return None
        if not rolling_summary_within_bound(text):
            return None
        return text
    except Exception:
        return None


def summary_context_message(summary: str) -> Dict[str, str]:
    """Ephemeral message block injected before recent verbatim turns."""
    return {
        "role": "system",
        "content": (
            "[Conversation Summary — older turns compressed; do not treat as new user input]\n"
            f"{summary.strip()}"
        ),
    }