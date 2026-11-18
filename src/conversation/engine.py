"""
MWIZERWA AI — Conversation Engine
Orchestration layer coordinating conversational turn execution around ConversationManager.
"""

from typing import Optional, List, Dict, Any
from src.llm.conversation import ConversationManager
from src.conversation.session import SessionState, TurnRequest, TurnResult, ConversationMode
from src.utils.language_detector import LanguageDetector, LanguageDetectionResult
from src.safety import SafetyEvaluator, SafetyGuardrail, ApplicationAction, SafetyAssessment
from src.safety.assistant_policy import AssistantSafetyValidator


class ConversationEngine:
    """
    Orchestration layer around ConversationManager:
    - Accepts TurnRequest instances.
    - Validates incoming message usability.
    - Updates conversation session state (e.g. mode) if specified.
    - Performs per-turn language detection across Kinyarwanda, English, and French.
    - Delegates response generation to the existing token-aware ConversationManager with language context.
    - Captures structured metadata from the response contract and language detection.
    - Returns structured TurnResult exposing language information and contract metadata.
    - Delegates conversation resets.
    - Exposes underlying history without duplicating state.
    """

    def __init__(
        self,
        manager: Optional[ConversationManager] = None,
        state: Optional[SessionState] = None,
        language_detector: Optional[LanguageDetector] = None,
        safety_evaluator: Optional[SafetyEvaluator] = None,
        safety_guardrail: Optional[SafetyGuardrail] = None,
        assistant_validator: Optional[AssistantSafetyValidator] = None,
    ):
        self._safety_evaluator = safety_evaluator if safety_evaluator is not None else SafetyEvaluator()
        self._safety_guardrail = safety_guardrail if safety_guardrail is not None else SafetyGuardrail()
        self._assistant_validator = assistant_validator if assistant_validator is not None else AssistantSafetyValidator()
        self._manager = manager if manager is not None else ConversationManager(
            safety_evaluator=self._safety_evaluator,
            safety_guardrail=self._safety_guardrail,
            assistant_validator=self._assistant_validator,
        )
        self._state = state if state is not None else SessionState()
        self._detector = language_detector if language_detector is not None else LanguageDetector()

    @property
    def safety_evaluator(self) -> SafetyEvaluator:
        """Access the underlying SafetyEvaluator instance."""
        return self._safety_evaluator

    @property
    def safety_guardrail(self) -> SafetyGuardrail:
        """Access the underlying SafetyGuardrail instance."""
        return self._safety_guardrail

    @property
    def assistant_validator(self) -> AssistantSafetyValidator:
        """Access the underlying AssistantSafetyValidator instance."""
        return self._assistant_validator

    @property
    def language_detector(self) -> LanguageDetector:
        """Access the underlying LanguageDetector instance."""
        return self._detector

    @property
    def manager(self) -> ConversationManager:
        """Access the underlying ConversationManager instance."""
        return self._manager

    @property
    def state(self) -> SessionState:
        """Access the current SessionState."""
        return self._state

    @property
    def messages(self) -> List[Dict[str, str]]:
        """
        Direct view into official conversation history.
        History is managed exclusively by ConversationManager and is never duplicated.
        """
        return self._manager.messages

    def process_turn(self, request: TurnRequest) -> TurnResult:
        """
        Processes an incoming conversational turn:
        1. Validates that the user message is non-empty.
        2. Updates conversation mode if provided in the request.
        3. Delegates generation to ConversationManager.
        4. Extracts contract metadata if available.
        5. Returns a typed TurnResult.
        """
        if request.user_text is None:
            self._manager.last_parsed_response = None
            return TurnResult(
                response_text="",
                mode=self._state.mode,
                language=None,
                language_result=None,
                metadata={},
            )

        cleaned_text = request.user_text.strip()
        if not cleaned_text:
            self._manager.last_parsed_response = None
            return TurnResult(
                response_text="",
                mode=self._state.mode,
                language=None,
                language_result=None,
                metadata={},
            )

        # Update active conversation mode if specified in request
        if request.mode is not None:
            self._state.mode = request.mode

        # Per-turn language detection (independent for each turn, never blindly inherited)
        if request.language is not None:
            lang_result = LanguageDetectionResult(
                language=request.language,
                confidence=1.0,
                is_reliable=True,
                is_mixed=False,
                details={"source": "turn_request_override"},
            )
        else:
            lang_result = self._detector.detect(cleaned_text)

        # Gate 1: Independent safety assessment before LLM generation
        safety_assessment = self._safety_evaluator.evaluate(
            cleaned_text,
            conversation_context=self._manager.messages,
        )
        safety_action = self._safety_guardrail.determine_action(safety_assessment)

        if safety_action != ApplicationAction.NORMAL_CONVERSATION:
            # Gate 1 safety override: bypass LLM generation
            # Resolve target language for deterministic safety response
            if lang_result.is_reliable and lang_result.language in ("rw", "en", "fr"):
                target_lang = lang_result.language
            elif safety_assessment.detected_expression_language in ("rw", "en", "fr"):
                target_lang = safety_assessment.detected_expression_language
            else:
                target_lang = "rw"

            deterministic_response = self._safety_guardrail.get_response(
                safety_assessment,
                safety_action,
                target_lang,
            ) or ""

            # Atomically commit deterministic turn into conversation history
            self._manager.record_deterministic_turn(
                cleaned_text,
                deterministic_response,
                detected_language=lang_result,
            )

            metadata: Dict[str, Any] = {
                "raw_user_input": request.user_text,
                "detected_language": lang_result.language,
                "language_confidence": lang_result.confidence,
                "is_language_reliable": lang_result.is_reliable,
                "is_language_mixed": lang_result.is_mixed,
                "language_result": lang_result,
                "safety_assessment": safety_assessment,
                "safety_action": safety_action.value,
                "gate1_triggered": True,
                "gate2_triggered": False,
            }

            return TurnResult(
                response_text=deterministic_response,
                mode=self._state.mode,
                language=target_lang,
                language_result=lang_result,
                metadata=metadata,
            )

        # Gate 1 passed: Normal conversation -> delegate to ConversationManager (Gate 2 runs internally)
        response_text = self._manager.generate_response(
            cleaned_text,
            detected_language=lang_result,
        )

        metadata: Dict[str, Any] = {
            "raw_user_input": request.user_text,
            "detected_language": lang_result.language,
            "language_confidence": lang_result.confidence,
            "is_language_reliable": lang_result.is_reliable,
            "is_language_mixed": lang_result.is_mixed,
            "language_result": lang_result,
            "safety_assessment": safety_assessment,
            "safety_action": safety_action.value,
            "gate1_triggered": False,
            "gate2_triggered": getattr(self._manager, "last_gate2_triggered", False),
            "gate2_violation": getattr(self._manager, "last_gate2_violation", None),
        }
        last_parsed = getattr(self._manager, "last_parsed_response", None)
        if last_parsed is not None:
            metadata["interaction_intent"] = last_parsed.interaction_intent.value
            metadata["requires_clarification"] = last_parsed.requires_clarification
            metadata["parsed_response"] = last_parsed

        return TurnResult(
            response_text=response_text,
            mode=self._state.mode,
            language=lang_result.language,
            language_result=lang_result,
            metadata=metadata,
        )

    def reset_conversation(self) -> None:
        """Resets the conversation history via ConversationManager."""
        self._manager.reset_conversation()
