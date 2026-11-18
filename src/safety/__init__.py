"""
MWIZERWA AI — Safety Module
Chunk 4 Phase 1: Foundational safety domain models, emergency registry, and response templates.
"""

from src.safety.evaluator import SafetyEvaluator
from src.safety.guardrail import SafetyGuardrail
from src.safety.assistant_policy import AssistantSafetyValidator, AssistantSafetyResult
from src.safety.categories import (
    ApplicationAction,
    ImmediacyLevel,
    IntentLevel,
    MeansStatus,
    SafetyAssessment,
    SafetyCategory,
    SubjectTarget,
    TemporalStatus,
)
from src.safety.crisis_responses import (
    ResponseSituation,
    SUPPORTED_SAFETY_LANGUAGES,
    TemplateNotFoundError,
    get_available_situations,
    get_safety_response,
    get_supported_languages,
)
from src.safety.resources import (
    EmergencyContact,
    VERIFIED_EMERGENCY_RESOURCES,
    VerificationStatus,
    get_all_resources,
    get_emergency_contact_by_number,
    get_verified_resources,
)

__all__ = [
    # Evaluator & Guardrail & Assistant Output Policy
    "SafetyEvaluator",
    "SafetyGuardrail",
    "AssistantSafetyValidator",
    "AssistantSafetyResult",
    # Domain Models & Enums
    "SafetyCategory",
    "ApplicationAction",
    "SubjectTarget",
    "TemporalStatus",
    "IntentLevel",
    "MeansStatus",
    "ImmediacyLevel",
    "SafetyAssessment",
    # Resources
    "VerificationStatus",
    "EmergencyContact",
    "VERIFIED_EMERGENCY_RESOURCES",
    "get_all_resources",
    "get_verified_resources",
    "get_emergency_contact_by_number",
    # Crisis Responses
    "ResponseSituation",
    "TemplateNotFoundError",
    "SUPPORTED_SAFETY_LANGUAGES",
    "get_safety_response",
    "get_available_situations",
    "get_supported_languages",
]
