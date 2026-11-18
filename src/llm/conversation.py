from typing import List, Dict, Optional, Tuple, Any
from src.llm.prompts import SYSTEM_PROMPT
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.response_cleaner import clean_response
from src.llm.response_parser import ParsedResponse, ResponseParsingError, safe_parse_response
from src.utils.language_detector import LanguageDetectionResult
from src.safety.evaluator import SafetyEvaluator
from src.safety.guardrail import SafetyGuardrail
from src.safety.assistant_policy import AssistantSafetyValidator
from src.safety.categories import ApplicationAction


# Mode-specific delivery directives injected ephemerally per turn.
# Keyed by mode .value string to avoid circular import with session module.
_MODE_DIRECTIVES = {
    "gentle": (
        "Delivery Mode: GENTLE — Warm, patient, unhurried phrasing. "
        "Prioritize emotional acknowledgment; avoid pressure or forceful suggestions."
    ),
    "normal": (
        "Delivery Mode: NORMAL — Natural, balanced conversational tone. "
        "Balanced emotional acknowledgment and practical support without being overly soft or forceful."
    ),
    "motivational": (
        "Delivery Mode: MOTIVATIONAL — Encouraging, constructive, and action-oriented when appropriate. "
        "Help recognize manageable next steps without pressure, empty slogans, or dismissive positivity."
    ),
}


def _format_per_turn_system_prompt(
    base_prompt: str,
    detected_language: Optional[LanguageDetectionResult],
    mode: Optional[Any] = None,
    ambiguous_language: bool = False,
    requires_clarification: bool = False,
) -> str:
    """
    Constructs an ephemeral, per-turn system prompt incorporating:
    - Reliable language context from per-turn detection.
    - Active conversation delivery mode (GENTLE / NORMAL / MOTIVATIONAL).
    - Ambiguous-language guidance for unestablished first-turn sessions.
    - Clarification continuation when the previous turn flagged requires_clarification.

    The base prompt remains unchanged in committed conversation history.
    All injected directives are ephemeral and are not committed to self.messages[0].
    """
    parts = [base_prompt]
    context_lines = []

    # 1. Language directive
    if ambiguous_language:
        context_lines.append(
            "User Language: Ambiguous/Uncertain. If the user's intent is a greeting or unclear, "
            "respond with a concise, warm trilingual greeting (Kinyarwanda/English/French) "
            "asking which language they prefer."
        )
    elif detected_language is not None and detected_language.is_reliable:
        lang_code = detected_language.language
        lang_names = {
            "rw": "Kinyarwanda",
            "en": "English",
            "fr": "French",
        }
        lang_name = lang_names.get(lang_code)
        if lang_name:
            if detected_language.is_mixed:
                context_lines.append(
                    f"User Language: Mixed (primarily {lang_name}). "
                    f"You MUST respond in {lang_name}. Honor their meaning and preserve the primary language."
                )
            else:
                other_langs = [n for n in ("Kinyarwanda", "English", "French") if n != lang_name]
                context_lines.append(
                    f"User Language: {lang_name}. "
                    f"You MUST write your entire response_text in {lang_name}. "
                    f"Do NOT respond in {other_langs[0]} or {other_langs[1]}."
                )

    # 2. Mode directive (resolve enum or string key to directive text)
    if mode is not None:
        mode_key = mode.value if hasattr(mode, "value") else str(mode)
        mode_directive = _MODE_DIRECTIVES.get(mode_key)
        if mode_directive:
            context_lines.append(mode_directive)

    # 3. Clarification continuation
    if requires_clarification:
        context_lines.append(
            "Clarification Pending: The previous turn indicated that clarification is needed. "
            "Continue clarifying the user's intent before offering advice or exercises."
        )

    if context_lines:
        parts.append("\n\n[Current Turn Context]")
        parts.append("\n".join(context_lines))

    return "\n".join(parts)


class OversizedMessageError(ValueError):
    """
    Raised when an individual message or candidate prompt exceeds the
    available context window input budget.
    """
    pass


class ConversationManager:
    """
    Manages conversational history with AfriqueQwen using token-aware context management:
    1. System prompt is always preserved at index 0.
    2. Available context budget is computed by reserving max_tokens for response generation
       and a safety margin.
    3. Oldest complete user-assistant turn pairs are trimmed before inference when token
       budget is exceeded.
    4. Oversized user messages that exceed available budget are rejected without state mutation.
    5. Official conversation state is updated only upon successful non-empty inference.
    6. System prompt is preserved across resets.
    7. Enforces strict structured response contract validation on every non-empty response.
    """

    def __init__(
        self,
        llm: Optional[AfriqueQwenLLM] = None,
        max_turns: Optional[int] = None,
        n_ctx: Optional[int] = None,
        max_tokens: Optional[int] = None,
        safety_margin: int = 16,
        system_prompt: str = SYSTEM_PROMPT,
        safety_evaluator: Optional[SafetyEvaluator] = None,
        safety_guardrail: Optional[SafetyGuardrail] = None,
        assistant_validator: Optional[AssistantSafetyValidator] = None,
    ):
        self.llm = llm if llm is not None else AfriqueQwenLLM()
        self.max_turns = max_turns
        self.safety_margin = safety_margin
        self.system_prompt = system_prompt
        self.last_parsed_response: Optional[ParsedResponse] = None
        self.safety_evaluator = safety_evaluator if safety_evaluator is not None else SafetyEvaluator()
        self.safety_guardrail = safety_guardrail if safety_guardrail is not None else SafetyGuardrail()
        self.assistant_validator = assistant_validator if assistant_validator is not None else AssistantSafetyValidator()
        self.last_gate2_triggered: bool = False
        self.last_gate2_violation: Optional[str] = None

        # Determine n_ctx from parameter, llm instance, or default 4096
        if n_ctx is not None:
            self.n_ctx = n_ctx
        else:
            llm_ctx = getattr(self.llm, "n_ctx", None)
            if isinstance(llm_ctx, int):
                self.n_ctx = llm_ctx
            else:
                self.n_ctx = 4096

        # Determine max_tokens from parameter, llm instance, or default 512
        if max_tokens is not None:
            self.max_tokens = max_tokens
        else:
            llm_max = getattr(self.llm, "max_tokens", None)
            if isinstance(llm_max, int):
                self.max_tokens = llm_max
            else:
                self.max_tokens = 512

        self.messages: List[Dict[str, str]] = []
        self.reset_conversation()

    def reset_conversation(self) -> None:
        """Resets the conversation history, preserving the system prompt."""
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.last_parsed_response = None
        self.last_gate2_triggered = False
        self.last_gate2_violation = None

    def get_available_budget(self) -> int:
        """
        Calculates the maximum allowed tokens for candidate input prompt,
        reserving space for max_tokens response and safety margin.
        """
        return max(0, self.n_ctx - self.max_tokens - self.safety_margin)

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the total tokens of the formatted messages.
        Uses LLM's accurate tokenizer and chat template if available,
        with a fallback for unconfigured test mocks or non-LLM objects.
        """
        if hasattr(self.llm, "count_tokens"):
            try:
                tokens = self.llm.count_tokens(messages)
                if isinstance(tokens, int):
                    return tokens
            except Exception as e:
                # If LLM is not a test mock, re-raise error
                if not self.llm.__class__.__module__.startswith("unittest.mock"):
                    raise e

        # Fallback for unconfigured test mocks / test environments:
        total_chars = sum(len(m.get("content", "") or "") for m in messages)
        # Approximate 4 characters per token + 6 tokens per message formatting overhead
        return max(len(messages) * 6 + (total_chars + 3) // 4, 1)

    def _trim_history(self) -> None:
        """
        Trims older user-assistant turn pairs when history exceeds max_turns.
        Retained for backward compatibility.
        """
        if self.max_turns is None:
            return

        context = self.messages[1:]
        max_messages = self.max_turns * 2

        while len(context) > max_messages:
            context = context[2:]

        self.messages = [self.messages[0]] + context

    def prepare_candidate_messages(
        self,
        user_text: str,
        detected_language: Optional[LanguageDetectionResult] = None,
        mode: Optional[Any] = None,
        ambiguous_language: bool = False,
        requires_clarification: bool = False,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Prepares and trims candidate messages for a new user turn:
        1. Checks if the user message alone with the per-turn system prompt fits within the budget.
        2. Trims candidate history by removing the oldest complete user-assistant exchange
           until the candidate prompt fits within the budget.
        3. Returns (candidate_messages, candidate_history).
        Raises OversizedMessageError if the user message cannot fit even with zero history.
        """
        system_content = _format_per_turn_system_prompt(
            self.system_prompt, detected_language,
            mode=mode,
            ambiguous_language=ambiguous_language,
            requires_clarification=requires_clarification,
        )
        candidate_system_msg = {"role": "system", "content": system_content}
        user_msg = {"role": "user", "content": user_text}
        budget = self.get_available_budget()

        # Step 1: Check minimal candidate (system prompt + user message)
        minimal_candidate = [candidate_system_msg, user_msg]
        minimal_tokens = self.count_tokens(minimal_candidate)
        if minimal_tokens > budget:
            raise OversizedMessageError(
                f"User message is too long for the available context window. "
                f"Minimal prompt requires {minimal_tokens} tokens, but maximum available budget is {budget} tokens "
                f"(context: {self.n_ctx}, reserved response: {self.max_tokens}, safety margin: {self.safety_margin}). "
                f"Please shorten your message."
            )

        # Step 2: Build candidate history from official history (excluding system prompt at index 0)
        candidate_history = list(self.messages[1:])

        # If max_turns is configured, enforce turn cap (accounting for the new turn being added)
        if self.max_turns is not None:
            max_prior_messages = max(0, self.max_turns - 1) * 2
            while len(candidate_history) > max_prior_messages:
                candidate_history = candidate_history[2:]

        # Step 3: Trim oldest complete turn pairs until candidate fits within budget
        candidate_messages = [candidate_system_msg] + candidate_history + [user_msg]
        while (
            len(candidate_history) >= 2
            and self.count_tokens(candidate_messages) > budget
        ):
            candidate_history = candidate_history[2:]
            candidate_messages = [candidate_system_msg] + candidate_history + [user_msg]

        # Step 4: Final verification
        candidate_tokens = self.count_tokens(candidate_messages)
        if candidate_tokens > budget:
            raise OversizedMessageError(
                f"Candidate prompt of {candidate_tokens} tokens exceeds budget of {budget} tokens "
                f"even after trimming all conversation history."
            )

        return candidate_messages, candidate_history

    def generate_response(
        self,
        user_text: str,
        detected_language: Optional[LanguageDetectionResult] = None,
        mode: Optional[Any] = None,
        ambiguous_language: bool = False,
        requires_clarification: bool = False,
    ) -> str:
        """
        Generates a response for a user message, maintaining context and handling errors.
        Candidate history trimming and token budget checks occur before inference.
        Official conversation state is committed only upon successful non-empty response.
        Enforces strict compliance with the structured response contract.
        """
        # DEFECT-02 fix: Reset per-turn metadata before any early returns, validation, or generation
        self.last_parsed_response = None
        self.last_gate2_triggered = False
        self.last_gate2_violation = None

        user_text = user_text.strip() if user_text else ""
        if not user_text:
            return ""

        candidate_messages, candidate_history = self.prepare_candidate_messages(
            user_text, detected_language=detected_language,
            mode=mode,
            ambiguous_language=ambiguous_language,
            requires_clarification=requires_clarification,
        )

        try:
            # Inference call
            raw_response = self.llm.generate_completion(candidate_messages)
            cleaned = clean_response(raw_response)

            if not cleaned:
                # Empty response generated - do not mutate official state
                return ""

            # DEFECT-01 fix: Strict contract enforcement with zero unconstrained fallback
            parsed, parse_err = safe_parse_response(cleaned)
            if parsed is None:
                raise ResponseParsingError(
                    f"Model response violated structured response contract: {parse_err}"
                )

            candidate_text = parsed.response_text

            # Gate 2 Check A: Validate assistant output against assistant safety policy
            # Intercepts harmful advice, encouragement of harm, violence, poison, or concealment
            assistant_val = self.assistant_validator.validate_assistant_response(candidate_text)
            if not assistant_val.is_safe:
                self.last_gate2_triggered = True
                self.last_gate2_violation = assistant_val.violation_category
                self.last_parsed_response = None

                if detected_language and detected_language.is_reliable and detected_language.language in ("rw", "en", "fr"):
                    target_lang = detected_language.language
                else:
                    inferred = self.safety_evaluator._infer_language(user_text) or self.safety_evaluator._infer_language(candidate_text)
                    target_lang = inferred if inferred in ("rw", "en", "fr") else "rw"
                safe_fallback = self.assistant_validator.get_safe_fallback(target_lang)

                # Commit only the approved safe replacement response
                self.messages = [self.messages[0]] + candidate_history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": safe_fallback},
                ]
                return safe_fallback

            # Gate 2 Check B: Also evaluate candidate text using user-risk evaluator
            # In case assistant echoes or discloses active crisis or self-harm in first-person
            assistant_assessment = self.safety_evaluator.evaluate(
                candidate_text, conversation_context=candidate_messages
            )
            gate2_action = self.safety_guardrail.determine_action(assistant_assessment)

            if gate2_action in (
                ApplicationAction.SAFETY_FOCUSED_RESPONSE,
                ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE,
            ):
                self.last_gate2_triggered = True
                self.last_gate2_violation = assistant_assessment.category.value
                self.last_parsed_response = None

                target_lang = (
                    detected_language.language
                    if (detected_language and detected_language.is_reliable and detected_language.language in ("rw", "en", "fr"))
                    else assistant_assessment.detected_expression_language or "rw"
                )
                safe_fallback = self.safety_guardrail.get_response(
                    assistant_assessment, gate2_action, target_lang
                ) or self.assistant_validator.get_safe_fallback(target_lang)

                # Commit only the approved deterministic replacement response
                self.messages = [self.messages[0]] + candidate_history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": safe_fallback},
                ]
                return safe_fallback

            # Gate 2 passed: safe model output
            self.last_gate2_triggered = False
            self.last_gate2_violation = None
            self.last_parsed_response = parsed
            final_response_text = candidate_text

            # Update official state with trimmed history + new turn pair
            self.messages = [self.messages[0]] + candidate_history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": final_response_text},
            ]

            return final_response_text
        except Exception as e:
            # On error, ensure metadata remains cleared and state is not mutated
            self.last_parsed_response = None
            self.last_gate2_triggered = False
            self.last_gate2_violation = None
            print(f"Error during LLM inference: {e}")
            raise e

    def record_deterministic_turn(
        self,
        user_text: str,
        response_text: str,
        detected_language: Optional[LanguageDetectionResult] = None,
    ) -> None:
        """
        Atomically commits a deterministic turn pair (e.g. from Gate 1 safety override)
        into official conversation history, respecting token budgeting and history trimming.
        Clears last_parsed_response and resets per-turn safety flags.
        """
        self.last_parsed_response = None
        self.last_gate2_triggered = False
        self.last_gate2_violation = None

        user_text = user_text.strip() if user_text else ""
        if not user_text or not response_text:
            return

        candidate_messages, candidate_history = self.prepare_candidate_messages(
            user_text, detected_language=detected_language
        )
        self.messages = [self.messages[0]] + candidate_history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response_text},
        ]
