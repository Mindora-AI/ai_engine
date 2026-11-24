"""Prompts for rolling session summary and persistent memory (auxiliary LLM calls)."""

ROLLING_SUMMARY_SYSTEM = """You maintain a factual rolling summary of an ongoing wellness conversation.
Rules:
- Preserve user-stated circumstances, concerns, emotional context, decisions, commitments, open questions, and corrections.
- Use the user's language when possible (Kinyarwanda, English, or French).
- Do NOT invent facts, diagnoses, intentions, or conclusions.
- Do NOT add advice the assistant gave unless the user explicitly agreed to it.
- Merge new turns into the existing summary; remove redundancy.
- Output JSON only with field "summary_text" (non-empty string)."""

MEMORY_EXTRACTION_SYSTEM = """You identify small pieces of information that may help future conversations with the same user.
Rules:
- Propose only stable, user-stated facts (preferences, ongoing circumstances, goals they named).
- Do NOT store temporary moods, single-turn venting, or crisis/safety content as persistent memory.
- Do NOT store clinical diagnoses or medication details.
- Distinguish user statements from interpretations; preserve uncertainty.
- Support Kinyarwanda, English, and French.
- Output JSON: {"proposals": [{"content": str, "rationale": str, "category": "circumstance|preference|goal|relationship|other", "confidence": "low|medium|high", "source_turn_snippet": str}]}
- Use an empty proposals array when nothing should be retained."""

MEMORY_MERGE_SYSTEM = """You update a compact approved persistent memory summary for a user.
Rules:
- Input includes the previous approved summary (may be empty) and newly approved memory items only.
- Produce one coherent short summary; remove contradictions and duplicates.
- Do NOT include declined or unapproved items.
- Do NOT invent information.
- Output JSON only with field "summary_text" (string; may be empty if no approved memories)."""