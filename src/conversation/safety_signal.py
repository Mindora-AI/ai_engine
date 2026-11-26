"""Public safety_signal metadata derived from existing Gate 1 / Gate 2 outcomes."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from src.safety import ApplicationAction, SafetyAssessment
from src.safety.assistant_policy import AssistantSafetyResult


def build_safety_signal(
    *,
    safety_assessment: Optional[SafetyAssessment],
    safety_action: ApplicationAction,
    gate2_triggered: bool,
    gate2_violation: Optional[Union[AssistantSafetyResult, str]],
) -> Optional[Dict[str, Any]]:
    """
    Return structured safety metadata only when a safety gate actually fired.
    Ordinary turns with NORMAL_CONVERSATION and no Gate 2 violation return None.
    """
    gate1_triggered = safety_action != ApplicationAction.NORMAL_CONVERSATION
    if not gate1_triggered and not gate2_triggered:
        return None

    signal: Dict[str, Any] = {
        "gate1_triggered": gate1_triggered,
        "gate2_triggered": gate2_triggered,
        "safety_action": safety_action.value,
    }

    if gate1_triggered and safety_assessment is not None:
        signal["category"] = safety_assessment.category.value
        signal["subject"] = safety_assessment.subject.value
        signal["temporal"] = safety_assessment.temporal.value
        signal["intent"] = safety_assessment.intent.value
        signal["immediacy"] = safety_assessment.immediacy.value
        if safety_assessment.detected_expression_language:
            signal["detected_expression_language"] = (
                safety_assessment.detected_expression_language
            )

    if gate2_triggered and gate2_violation is not None:
        if isinstance(gate2_violation, AssistantSafetyResult):
            if gate2_violation.violation_category:
                signal["gate2_violation_category"] = gate2_violation.violation_category
        elif isinstance(gate2_violation, str) and gate2_violation.strip():
            signal["gate2_violation_category"] = gate2_violation.strip()

    return signal