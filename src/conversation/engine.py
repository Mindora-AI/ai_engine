"""
MWIZERWA AI — Conversation Engine
Orchestration layer coordinating conversational turn execution around ConversationManager.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any

if TYPE_CHECKING:
    from src.memory.coordinator import MemoryCoordinator
from src.llm.conversation import ConversationManager
from src.conversation.session import SessionState, TurnRequest, TurnResult, ConversationMode
from src.utils.language_detector import LanguageDetector, LanguageDetectionResult
from src.safety import SafetyEvaluator, SafetyGuardrail, ApplicationAction, SafetyAssessment
from src.safety.assistant_policy import AssistantSafetyValidator
from src.exercises import (
    ExerciseRecommender,
    RecommendationAction,
    ExerciseLifecycle,
    LifecycleResult,
    UserResponseType,
    ExerciseCategory,
    ExerciseStatus,
    ExerciseTriggerType,
    CATEGORY_DEFAULT_EXERCISE,
    SUPPORTED_EXERCISE_LANGUAGES,
    UNRELATED_CHECKINS,
)


class ConversationEngine:
    """
    Orchestration layer around ConversationManager:
    - Accepts TurnRequest instances.
    - Validates incoming message usability.
    - Updates conversation session state (e.g. mode) if specified.
    - Performs per-turn language detection across Kinyarwanda, English, and French.
    - Enforces Gate 1 pre-inference safety check (crisis overrides bypass inference & abort active exercises).
    - Coordinates deterministic exercise lifecycle and recommendation subsystem.
    - Delegates response generation to token-aware ConversationManager with language context.
    - Enforces Gate 2 assistant-output safety validation.
    - Captures structured metadata from the response contract, safety gates, and exercise lifecycle.
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
        recommender: Optional[ExerciseRecommender] = None,
        lifecycle: Optional[ExerciseLifecycle] = None,
        memory_coordinator: Optional["MemoryCoordinator"] = None,
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
        self._recommender = recommender if recommender is not None else ExerciseRecommender()
        self._lifecycle = lifecycle if lifecycle is not None else ExerciseLifecycle()
        self._memory = memory_coordinator

    @property
    def memory_coordinator(self) -> Optional["MemoryCoordinator"]:
        return self._memory

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
    def recommender(self) -> ExerciseRecommender:
        """Access the underlying ExerciseRecommender instance."""
        return self._recommender

    @property
    def lifecycle(self) -> ExerciseLifecycle:
        """Access the underlying ExerciseLifecycle instance."""
        return self._lifecycle

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
        Processes an incoming conversational turn through the full pipeline:
        1. Input validation.
        2. Language detection & session language resolution.
        3. Gate 1 safety check (emergency overrides cancel active exercises).
        4. Active exercise control & lifecycle handling.
        5. Explicit exercise request evaluation.
        6. LLM conversation generation & Gate 2 validation.
        7. Contextual exercise recommendation evaluation.
        8. Atomic history commit & metadata packaging.
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

        # Phase 1B Language Resolution Policy:
        # 1. Explicit request override
        # 2. Per-turn reliable detection
        # 3. Established session language
        # 4. None (uncertain)
        if request.language is not None and request.language in SUPPORTED_EXERCISE_LANGUAGES:
            target_lang = request.language
            self._state.exercise_state.set_established_language(target_lang)
        elif lang_result.is_reliable and lang_result.language in SUPPORTED_EXERCISE_LANGUAGES:
            target_lang = lang_result.language
            self._state.exercise_state.set_established_language(target_lang)
        elif self._state.exercise_state.established_language in SUPPORTED_EXERCISE_LANGUAGES:
            target_lang = self._state.exercise_state.established_language
        else:
            target_lang = None

        # Gate 1: Independent safety assessment before LLM generation or exercise execution
        safety_assessment = self._safety_evaluator.evaluate(
            cleaned_text,
            conversation_context=self._manager.messages,
        )
        safety_action = self._safety_guardrail.determine_action(safety_assessment)

        if safety_action != ApplicationAction.NORMAL_CONVERSATION:
            # Gate 1 safety override: bypass LLM generation and abort active exercises immediately
            if self._state.exercise_state.status != ExerciseStatus.IDLE:
                self._state.exercise_state.reset_to_idle()

            # Resolve target language for deterministic safety response (fallback to rw)
            if target_lang in SUPPORTED_EXERCISE_LANGUAGES:
                safe_lang = target_lang
            elif safety_assessment.detected_expression_language in SUPPORTED_EXERCISE_LANGUAGES:
                safe_lang = safety_assessment.detected_expression_language
            else:
                safe_lang = "rw"

            deterministic_response = self._safety_guardrail.get_response(
                safety_assessment,
                safety_action,
                safe_lang,
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
                "exercise_status": self._state.exercise_state.status.value,
            }

            return TurnResult(
                response_text=deterministic_response,
                mode=self._state.mode,
                language=safe_lang,
                language_result=lang_result,
                metadata=metadata,
            )

        # Gate 1 passed: Evaluate Active / Offered Exercise Lifecycle
        ex_state = self._state.exercise_state
        current_status = ex_state.status
        cooldown_established_this_turn = False

        # -------------------------------------------------------------------
        # 1. Handle OFFERED state
        # -------------------------------------------------------------------
        if current_status == ExerciseStatus.OFFERED:
            resp_type = self._lifecycle.classify_user_response(
                cleaned_text,
                ExerciseStatus.OFFERED,
                active_exercise_id=ex_state.offered_exercise_id,
            )

            if resp_type == UserResponseType.ACCEPT:
                if target_lang is None:
                    # Language uncertainty: ask clarification before starting
                    clarification = (
                        "Mbere y'uko dutangira umwitozo, wifuza ko tuvugana mu Kinyarwanda, "
                        "English, cyangwa Français?"
                    )
                    self._manager.record_deterministic_turn(cleaned_text, clarification, detected_language=lang_result)
                    return self._build_turn_result(clarification, target_lang, lang_result, safety_assessment, safety_action, request.user_text)

                offered_id = ex_state.offered_exercise_id
                life_res = self._lifecycle.start_exercise(
                    ex_state, offered_id, ExerciseTriggerType.UNSOLICITED_OFFER, target_lang
                )
                self._manager.record_deterministic_turn(cleaned_text, life_res.response_text, detected_language=lang_result)
                return self._build_turn_result(life_res.response_text, target_lang, lang_result, safety_assessment, safety_action, request.user_text, life_res)

            elif resp_type in (UserResponseType.DECLINE, UserResponseType.STOP):
                eff_lang = target_lang or "rw"
                life_res = self._lifecycle.decline_offer(ex_state, eff_lang)
                self._manager.record_deterministic_turn(cleaned_text, life_res.response_text, detected_language=lang_result)
                return self._build_turn_result(life_res.response_text, eff_lang, lang_result, safety_assessment, safety_action, request.user_text, life_res)

            else:
                # User ignored offer and continued normal conversation -> decline offer and proceed to LLM
                ex_state.decline_offer()
                cooldown_established_this_turn = True

        # -------------------------------------------------------------------
        # 2. Handle IN_PROGRESS state
        # -------------------------------------------------------------------
        elif current_status == ExerciseStatus.IN_PROGRESS:
            resp_type = self._lifecycle.classify_user_response(
                cleaned_text,
                ExerciseStatus.IN_PROGRESS,
                active_exercise_id=ex_state.active_exercise_id,
                step_index=ex_state.current_step_index,
            )
            eff_lang = target_lang or "rw"

            if resp_type == UserResponseType.STOP:
                life_res = self._lifecycle.stop_exercise(ex_state, eff_lang)
                self._manager.record_deterministic_turn(cleaned_text, life_res.response_text, detected_language=lang_result)
                return self._build_turn_result(life_res.response_text, eff_lang, lang_result, safety_assessment, safety_action, request.user_text, life_res)

            elif resp_type in (UserResponseType.CONTINUE, UserResponseType.ANSWER):
                life_res = self._lifecycle.advance_exercise(ex_state, eff_lang)
                self._manager.record_deterministic_turn(cleaned_text, life_res.response_text, detected_language=lang_result)
                return self._build_turn_result(life_res.response_text, eff_lang, lang_result, safety_assessment, safety_action, request.user_text, life_res)

            elif resp_type == UserResponseType.UNRELATED:
                checkin = UNRELATED_CHECKINS.get(eff_lang, UNRELATED_CHECKINS["rw"])
                self._manager.record_deterministic_turn(cleaned_text, checkin, detected_language=lang_result)
                return self._build_turn_result(checkin, eff_lang, lang_result, safety_assessment, safety_action, request.user_text)

            else:  # AMBIGUOUS
                if eff_lang == "en":
                    ambig_text = "If you are ready to continue, say 'next', or say 'stop' if you would like to end the exercise."
                elif eff_lang == "fr":
                    ambig_text = "Si vous êtes prêt à continuer, dites 'suivant', ou dites 'stop' si vous souhaitez vous arrêter."
                else:
                    ambig_text = "Niba witeguye gukomeza, vuga 'komeza', cyangwa uvuge 'hagarika' niba wifuza kubireka."

                self._manager.record_deterministic_turn(cleaned_text, ambig_text, detected_language=lang_result)
                return self._build_turn_result(ambig_text, eff_lang, lang_result, safety_assessment, safety_action, request.user_text)

        # -------------------------------------------------------------------
        # 3. IDLE / Clean State: Evaluate Explicit Exercise Requests
        # -------------------------------------------------------------------
        if ex_state.status in (ExerciseStatus.COMPLETED, ExerciseStatus.DECLINED, ExerciseStatus.ABANDONED):
            ex_state.reset_to_idle()

        # Decrement cooldown counter for this non-empty user turn unless newly established this turn
        if not cooldown_established_this_turn:
            ex_state.record_user_turn()

        explicit_check = self._recommender.detect_explicit_request(cleaned_text)
        if explicit_check.is_explicit_request:
            if target_lang is None:
                # Language uncertainty
                clarif = (
                    "Before we begin the exercise, would you prefer to communicate in "
                    "Kinyarwanda, English, or French? / Mbere y'uko dutangira, wifuza ko "
                    "tuvugana mu Kinyarwanda, English, cyangwa Français?"
                )
                self._manager.record_deterministic_turn(cleaned_text, clarif, detected_language=lang_result)
                return self._build_turn_result(clarif, None, lang_result, safety_assessment, safety_action, request.user_text)

            target_cat = explicit_check.category or ExerciseCategory.BREATHING
            ex_id = CATEGORY_DEFAULT_EXERCISE[target_cat]

            life_res = self._lifecycle.start_exercise(
                ex_state, ex_id, ExerciseTriggerType.EXPLICIT_USER_REQUEST, target_lang
            )
            self._manager.record_deterministic_turn(cleaned_text, life_res.response_text, detected_language=lang_result)
            return self._build_turn_result(life_res.response_text, target_lang, lang_result, safety_assessment, safety_action, request.user_text, life_res)

        # -------------------------------------------------------------------
        # 4. Standard Conversational Flow (LLM + Gate 2 Validation)
        # -------------------------------------------------------------------

        # Determine ambiguous-language flag for first-turn unestablished sessions
        is_ambiguous_language = (
            request.language is None
            and not lang_result.is_reliable
            and self._state.exercise_state.established_language is None
        )

        # Carry forward requires_clarification from previous turn's parsed response
        last_parsed_prev = getattr(self._manager, "last_parsed_response", None)
        prev_requires_clarification = (
            last_parsed_prev.requires_clarification if last_parsed_prev else False
        )

        persistent_memory = ""
        if self._memory and request.user_id:
            persistent_memory = self._memory.get_approved_summary_text(request.user_id)

        response_text = self._manager.generate_response(
            cleaned_text,
            detected_language=lang_result,
            mode=self._state.mode,
            ambiguous_language=is_ambiguous_language,
            requires_clarification=prev_requires_clarification,
            persistent_memory_summary=persistent_memory,
        )

        last_parsed = getattr(self._manager, "last_parsed_response", None)
        intent = last_parsed.interaction_intent if last_parsed else None

        # Check Contextual Recommendation Eligibility
        # Suppressed when: language is unresolved, or current turn flagged requires_clarification
        current_requires_clarification = (
            last_parsed.requires_clarification if last_parsed else False
        )
        life_res = None
        if target_lang is not None and not current_requires_clarification:
            rec_decision = self._recommender.evaluate_recommendation(
                cleaned_text, intent, ex_state, target_lang
            )
            if rec_decision.action == RecommendationAction.OFFER_EXERCISE:
                life_res = self._lifecycle.offer_exercise(ex_state, rec_decision.exercise_id, target_lang)
                # Append deterministic pre-authored offer to Gate 2-approved LLM response
                combined_text = f"{response_text}\n\n{life_res.response_text}"
                if self._manager.messages and self._manager.messages[-1]["role"] == "assistant":
                    self._manager.messages[-1]["content"] = combined_text
                response_text = combined_text

        result = self._build_turn_result(
            response_text,
            target_lang,
            lang_result,
            safety_assessment,
            safety_action,
            request.user_text,
            life_res,
        )

        if self._memory and request.user_id and response_text.strip():
            proposals = self._memory.propose_from_turn(
                request.user_id,
                cleaned_text,
                response_text,
                language=target_lang,
            )
            if proposals:
                result.metadata["memory_proposals"] = [
                    p.to_public_dict() for p in proposals
                ]

        rolling = getattr(self._manager, "rolling_summary", "") or ""
        if rolling.strip():
            result.metadata["rolling_summary_active"] = True

        return result

    def _build_turn_result(
        self,
        response_text: str,
        target_lang: Optional[str],
        lang_result: Optional[LanguageDetectionResult],
        safety_assessment: Any,
        safety_action: Any,
        raw_user_input: str,
        lifecycle_result: Optional[LifecycleResult] = None,
    ) -> TurnResult:
        """Helper to construct typed TurnResult with consolidated metadata."""
        metadata: Dict[str, Any] = {
            "raw_user_input": raw_user_input,
            "detected_language": lang_result.language if lang_result else None,
            "language_confidence": lang_result.confidence if lang_result else None,
            "is_language_reliable": lang_result.is_reliable if lang_result else None,
            "is_language_mixed": lang_result.is_mixed if lang_result else None,
            "effective_language": target_lang,
            "language_result": lang_result,
            "safety_assessment": safety_assessment,
            "safety_action": safety_action.value if hasattr(safety_action, "value") else safety_action,
            "gate1_triggered": safety_action != ApplicationAction.NORMAL_CONVERSATION,
            "gate2_triggered": getattr(self._manager, "last_gate2_triggered", False),
            "gate2_violation": getattr(self._manager, "last_gate2_violation", None),
            "exercise_status": self._state.exercise_state.status.value,
            "active_exercise_id": self._state.exercise_state.active_exercise_id,
            "current_step_index": self._state.exercise_state.current_step_index,
            "cooldown_turns_remaining": self._state.exercise_state.cooldown_turns_remaining,
        }

        if lifecycle_result is not None:
            metadata["lifecycle_result"] = lifecycle_result

        last_parsed = getattr(self._manager, "last_parsed_response", None)
        if last_parsed is not None:
            metadata["interaction_intent"] = last_parsed.interaction_intent.value
            metadata["requires_clarification"] = last_parsed.requires_clarification
            metadata["parsed_response"] = last_parsed

        return TurnResult(
            response_text=response_text,
            mode=self._state.mode,
            language=lang_result.language if lang_result else None,
            language_result=lang_result,
            metadata=metadata,
        )

    def reset_conversation(self) -> None:
        """Resets the conversation history and clears active exercise state."""
        self._manager.reset_conversation()
        self._state.exercise_state.reset_to_idle()
