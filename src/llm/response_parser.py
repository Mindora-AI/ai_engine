"""
MWIZERWA AI — Structured Response Contract & Parser
Defines the schema and validation logic for AfriqueQwen's structured responses.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple


class InteractionIntent(str, Enum):
    """
    Classifications of conversational intent:
    - LISTENING: The user primarily wants to express something and be heard.
    - REFLECTION: The user wants to understand, examine, or make sense of something.
    - PRACTICAL_ADVICE: The user wants suggestions, guidance, or concrete next steps.
    - UNCLEAR: The user's intention cannot reasonably be determined.
    """
    LISTENING = "LISTENING"
    REFLECTION = "REFLECTION"
    PRACTICAL_ADVICE = "PRACTICAL_ADVICE"
    UNCLEAR = "UNCLEAR"


class ResponseParsingError(ValueError):
    """
    Raised when a model response violates the structured response contract,
    contains malformed JSON, missing fields, invalid types, or invalid intent values.
    """
    pass


@dataclass
class ParsedResponse:
    """
    Typed, validated representation of a structured response from AfriqueQwen.
    """
    response_text: str
    interaction_intent: InteractionIntent
    requires_clarification: bool
    raw_json: Optional[Dict[str, Any]] = None


def extract_json_payload(text: str) -> str:
    """
    Extracts the JSON payload from raw text, safely stripping surrounding
    whitespace and Markdown code fences if present.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Handle Markdown code fences (e.g. ```json ... ``` or ``` ... ```)
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove opening fence line (e.g. ```json or ```)
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    return cleaned


def parse_response(text: str) -> ParsedResponse:
    """
    Parses and strictly validates a model response string against the response contract:
    {
      "response_text": str (required, non-empty),
      "interaction_intent": "LISTENING" | "REFLECTION" | "PRACTICAL_ADVICE" | "UNCLEAR" (required),
      "requires_clarification": bool (required)
    }

    Raises:
        ResponseParsingError: If JSON is malformed, required fields are missing,
                              fields have incorrect types, response_text is empty,
                              or interaction_intent is unsupported.
    """
    if text is None or not isinstance(text, str):
        raise ResponseParsingError("Response input must be a non-empty string.")

    payload = extract_json_payload(text)
    if not payload:
        raise ResponseParsingError("Response payload is empty.")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ResponseParsingError(f"Malformed JSON in response: {e}") from e

    if not isinstance(data, dict):
        raise ResponseParsingError(f"Expected a JSON object (dict), got {type(data).__name__}.")

    # 1. Validate response_text
    if "response_text" not in data:
        raise ResponseParsingError("Missing required field 'response_text'.")
    raw_response_text = data["response_text"]
    if not isinstance(raw_response_text, str):
        raise ResponseParsingError(
            f"Field 'response_text' must be a string, got {type(raw_response_text).__name__}."
        )
    cleaned_response_text = raw_response_text.strip()
    if not cleaned_response_text:
        raise ResponseParsingError("Field 'response_text' must not be empty or whitespace-only.")

    # 2. Validate interaction_intent
    if "interaction_intent" not in data:
        raise ResponseParsingError("Missing required field 'interaction_intent'.")
    raw_intent = data["interaction_intent"]
    if not isinstance(raw_intent, str):
        raise ResponseParsingError(
            f"Field 'interaction_intent' must be a string, got {type(raw_intent).__name__}."
        )

    # Strictly check against valid enum values (case-sensitive as specified in contract)
    valid_intents = {intent.value: intent for intent in InteractionIntent}
    if raw_intent not in valid_intents:
        allowed = list(valid_intents.keys())
        raise ResponseParsingError(
            f"Invalid interaction_intent '{raw_intent}'. Must be one of: {allowed}."
        )
    intent_enum = valid_intents[raw_intent]

    # 3. Validate requires_clarification
    if "requires_clarification" not in data:
        raise ResponseParsingError("Missing required field 'requires_clarification'.")
    raw_clarification = data["requires_clarification"]
    # Check strict boolean (type(True) is bool; isinstance(1, bool) is False)
    if not isinstance(raw_clarification, bool):
        raise ResponseParsingError(
            f"Field 'requires_clarification' must be a boolean, got {type(raw_clarification).__name__}."
        )

    return ParsedResponse(
        response_text=cleaned_response_text,
        interaction_intent=intent_enum,
        requires_clarification=raw_clarification,
        raw_json=data,
    )


def _unescape_partial_json_string(fragment: str) -> str:
    """Best-effort unescape for a JSON string value cut off before the closing quote."""
    try:
        return json.loads(f'"{fragment}"')
    except json.JSONDecodeError:
        cleaned = fragment.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        return cleaned.strip()


def salvage_truncated_contract(text: str) -> Optional[ParsedResponse]:
    """
    Recover user-facing text when generation stops at max_tokens mid-JSON.
    Returns None if the payload does not look like a truncated contract object.
    """
    payload = extract_json_payload(text)
    if not payload or '"response_text"' not in payload:
        return None

    try:
        json.loads(payload)
        return None
    except json.JSONDecodeError:
        pass

    match = re.search(r'"response_text"\s*:\s*"(.*)', payload, re.DOTALL)
    if not match:
        return None

    raw_value = match.group(1)
    response_text = _unescape_partial_json_string(raw_value).strip()
    if not response_text:
        return None

    return ParsedResponse(
        response_text=response_text,
        interaction_intent=InteractionIntent.LISTENING,
        requires_clarification=False,
        raw_json={"truncation_recovered": True},
    )


def safe_parse_response(text: str) -> Tuple[Optional[ParsedResponse], Optional[str]]:
    """
    Non-raising wrapper around parse_response.
    Returns (ParsedResponse, None) on success, or (None, error_message) on failure.
    Attempts truncation recovery when strict JSON parsing fails.
    """
    try:
        parsed = parse_response(text)
        return parsed, None
    except ResponseParsingError as e:
        salvaged = salvage_truncated_contract(text)
        if salvaged is not None:
            return salvaged, None
        return None, str(e)
