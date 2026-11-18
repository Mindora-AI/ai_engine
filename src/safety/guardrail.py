"""
MWIZERWA AI — Safety Guardrail & Application Action Routing
Decoupled safety guardrail that converts SafetyAssessment into ApplicationAction
and selects deterministic verified responses for non-normal conversation turns.
"""

from typing import Optional

from src.safety.categories import (
    ApplicationAction,
    ImmediacyLevel,
    MeansStatus,
    SafetyAssessment,
    SafetyCategory,
)
from src.safety.crisis_responses import (
    ResponseSituation,
    SUPPORTED_SAFETY_LANGUAGES,
    get_safety_response,
)


class SafetyGuardrail:
    """
    Translates independent safety assessments into concrete application actions
    and retrieves deterministic safety responses from the static registry.
    """

    def determine_action(self, assessment: SafetyAssessment) -> ApplicationAction:
        """
        Determines the application routing action based on a SafetyAssessment.

        Args:
            assessment: Immutable SafetyAssessment dataclass.

        Returns:
            ApplicationAction enum member.
        """
        category = assessment.category

        # 1. Standard / Benign Conversational dialogue
        if category in (
            SafetyCategory.SAFE_CONVERSATIONAL,
            SafetyCategory.ORDINARY_DISTRESS,
            SafetyCategory.GENERAL_OR_HYPOTHETICAL,
            SafetyCategory.ANGER_OR_VENTING,
        ):
            return ApplicationAction.NORMAL_CONVERSATION

        # 2. Supportive Clarification (Passive wishes, ambiguous distress)
        if category in (
            SafetyCategory.PASSIVE_DEATH_WISH,
            SafetyCategory.AMBIGUOUS_DISTRESS,
        ):
            return ApplicationAction.SUPPORTIVE_CLARIFICATION

        # 3. Safety-Focused Response (Explicit self-harm thoughts, non-imminent third-person crisis)
        if category == SafetyCategory.SELF_HARM_THOUGHTS:
            return ApplicationAction.SAFETY_FOCUSED_RESPONSE

        if category == SafetyCategory.THIRD_PERSON_CRISIS:
            # If an action is already underway or urgent medical trauma, escalate to emergency override
            if (
                assessment.means == MeansStatus.INGESTION_OR_INJURY_UNDERWAY
                or assessment.immediacy in (ImmediacyLevel.IMMINENT, ImmediacyLevel.IN_PROGRESS)
            ):
                return ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE
            return ApplicationAction.SAFETY_FOCUSED_RESPONSE

        # 4. Threats or intent towards others
        if category == SafetyCategory.THREAT_OR_INTENT_HARM_OTHERS:
            if assessment.immediacy in (ImmediacyLevel.IMMINENT, ImmediacyLevel.IN_PROGRESS):
                return ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE
            return ApplicationAction.SAFETY_FOCUSED_RESPONSE

        # 5. Deterministic Emergency Overrides
        if category in (
            SafetyCategory.IMMINENT_SELF_HARM,
            SafetyCategory.IMMINENT_VIOLENCE_OTHERS,
            SafetyCategory.ONGOING_PHYSICAL_DANGER,
            SafetyCategory.ACUTE_MEDICAL_EMERGENCY,
        ):
            return ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE

        # Fallback to normal conversation
        return ApplicationAction.NORMAL_CONVERSATION

    def resolve_situation(self, assessment: SafetyAssessment) -> ResponseSituation:
        """
        Resolves the appropriate ResponseSituation enum member for a non-normal assessment.
        """
        cat = assessment.category

        if cat in (SafetyCategory.PASSIVE_DEATH_WISH, SafetyCategory.AMBIGUOUS_DISTRESS):
            return ResponseSituation.PASSIVE_DEATH_WISH

        if cat == SafetyCategory.SELF_HARM_THOUGHTS:
            return ResponseSituation.SELF_HARM_THOUGHTS

        if cat == SafetyCategory.THIRD_PERSON_CRISIS:
            if assessment.means == MeansStatus.INGESTION_OR_INJURY_UNDERWAY:
                return ResponseSituation.ACUTE_MEDICAL_EMERGENCY
            return ResponseSituation.THIRD_PERSON_CRISIS

        if cat == SafetyCategory.IMMINENT_SELF_HARM:
            return ResponseSituation.IMMINENT_SELF_HARM

        if cat in (SafetyCategory.IMMINENT_VIOLENCE_OTHERS, SafetyCategory.THREAT_OR_INTENT_HARM_OTHERS):
            return ResponseSituation.IMMINENT_VIOLENCE_OTHERS

        if cat == SafetyCategory.ONGOING_PHYSICAL_DANGER:
            return ResponseSituation.ONGOING_PHYSICAL_DANGER

        if cat == SafetyCategory.ACUTE_MEDICAL_EMERGENCY:
            return ResponseSituation.ACUTE_MEDICAL_EMERGENCY

        # Default fallback situation
        return ResponseSituation.PASSIVE_DEATH_WISH

    def get_response(
        self,
        assessment: SafetyAssessment,
        action: ApplicationAction,
        language: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieves the deterministic response template for an assessment and action.

        Args:
            assessment: The safety assessment.
            action: The application action.
            language: Requested language code ('rw', 'en', 'fr').

        Returns:
            Deterministic response string, or None if action is NORMAL_CONVERSATION.
        """
        if action == ApplicationAction.NORMAL_CONVERSATION:
            return None

        situation = self.resolve_situation(assessment)

        # Resolve target language with safe fallback
        norm_lang = (language or "").strip().lower()
        if norm_lang not in SUPPORTED_SAFETY_LANGUAGES:
            if assessment.detected_expression_language in SUPPORTED_SAFETY_LANGUAGES:
                norm_lang = assessment.detected_expression_language
            else:
                norm_lang = "rw"

        return get_safety_response(situation, norm_lang)
