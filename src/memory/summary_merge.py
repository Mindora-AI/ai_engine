"""Merge approved memory items into the persistent summary."""

from __future__ import annotations

import json
from typing import List, Optional

from src.memory.prompts import MEMORY_MERGE_SYSTEM

MERGE_SCHEMA = {
    "type": "object",
    "properties": {"summary_text": {"type": "string"}},
    "required": ["summary_text"],
}


def merge_approved_into_summary(
    llm,
    existing_summary: str,
    approved_items: List[str],
) -> Optional[str]:
    if not approved_items and not existing_summary.strip():
        return ""

    payload = {
        "previous_approved_summary": existing_summary.strip(),
        "newly_approved_items": [s.strip() for s in approved_items if s.strip()],
    }
    messages = [
        {"role": "system", "content": MEMORY_MERGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    if not hasattr(llm, "generate_auxiliary_json"):
        # Deterministic fallback for tests without auxiliary generation
        parts = [existing_summary.strip()] if existing_summary.strip() else []
        parts.extend(payload["newly_approved_items"])
        return " | ".join(parts) if parts else ""

    try:
        data = llm.generate_auxiliary_json(
            messages,
            schema=MERGE_SCHEMA,
            max_tokens=256,
        )
        return (data.get("summary_text") or "").strip()
    except Exception:
        return None