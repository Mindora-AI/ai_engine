"""
MWIZERWA AI — Unit Tests for Safety Guardrail
Chunk 4 Phase 2D: Validates ApplicationAction routing rules and deterministic template selection.
"""

import unittest

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
from src.safety.crisis_responses import ResponseSituation
from src.safety.guardrail import SafetyGuardrail


class TestSafetyGuardrail(unittest.TestCase):
    """Exhaustive unit test suite for SafetyGuardrail action routing and response retrieval."""

    def setUp(self):
        self.guardrail = SafetyGuardrail()

    # -------------------------------------------------------------
    # Action Routing Verification
    # -------------------------------------------------------------
    def test_routing_safe_conversational(self):
        assessment = SafetyAssessment(category=SafetyCategory.SAFE_CONVERSATIONAL)
        self.assertEqual(
            self.guardrail.determine_action(assessment),
            ApplicationAction.NORMAL_CONVERSATION,
        )
        self.assertIsNone(self.guardrail.get_response(assessment, ApplicationAction.NORMAL_CONVERSATION, "en"))

    def test_routing_ordinary_distress(self):
        assessment = SafetyAssessment(category=SafetyCategory.ORDINARY_DISTRESS)
        self.assertEqual(
            self.guardrail.determine_action(assessment),
            ApplicationAction.NORMAL_CONVERSATION,
        )

    def test_routing_anger_or_venting(self):
        assessment = SafetyAssessment(category=SafetyCategory.ANGER_OR_VENTING)
        self.assertEqual(
            self.guardrail.determine_action(assessment),
            ApplicationAction.NORMAL_CONVERSATION,
        )

    def test_routing_passive_death_wish(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.PASSIVE_DEATH_WISH,
            intent=IntentLevel.PASSIVE_WISH,
        )
        action = self.guardrail.determine_action(assessment)
        self.assertEqual(action, ApplicationAction.SUPPORTIVE_CLARIFICATION)
        # Must not escalate to emergency override
        self.assertNotEqual(action, ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE)

        # Response retrieval
        resp = self.guardrail.get_response(assessment, action, "en")
        self.assertIsNotNone(resp)
        self.assertIn("exhausted", resp.lower())

    def test_routing_self_harm_thoughts(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.SELF_HARM_THOUGHTS,
            intent=IntentLevel.EXPLICIT_INTENT,
        )
        action = self.guardrail.determine_action(assessment)
        self.assertEqual(action, ApplicationAction.SAFETY_FOCUSED_RESPONSE)

        resp = self.guardrail.get_response(assessment, action, "rw")
        self.assertIn("agaciro kanini", resp)

    def test_routing_third_person_non_imminent(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.THIRD_PERSON_CRISIS,
            subject=SubjectTarget.THIRD_PERSON_KNOWN,
            immediacy=ImmediacyLevel.NONE,
        )
        action = self.guardrail.determine_action(assessment)
        self.assertEqual(action, ApplicationAction.SAFETY_FOCUSED_RESPONSE)

        resp = self.guardrail.get_response(assessment, action, "en")
        self.assertIn("112", resp)
        self.assertIn("912", resp)

    def test_routing_third_person_imminent_action_underway(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.THIRD_PERSON_CRISIS,
            subject=SubjectTarget.THIRD_PERSON_KNOWN,
            means=MeansStatus.INGESTION_OR_INJURY_UNDERWAY,
            immediacy=ImmediacyLevel.IN_PROGRESS,
        )
        action = self.guardrail.determine_action(assessment)
        self.assertEqual(action, ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE)

        resp = self.guardrail.get_response(assessment, action, "en")
        self.assertIn("912", resp)

    def test_routing_imminent_self_harm(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.IMMINENT_SELF_HARM,
            immediacy=ImmediacyLevel.IMMINENT,
        )
        action = self.guardrail.determine_action(assessment)
        self.assertEqual(action, ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE)

        resp = self.guardrail.get_response(assessment, action, "fr")
        self.assertIn("112", resp)
        self.assertIn("912", resp)

    def test_routing_ongoing_physical_danger(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.ONGOING_PHYSICAL_DANGER,
            immediacy=ImmediacyLevel.IN_PROGRESS,
        )
        action = self.guardrail.determine_action(assessment)
        self.assertEqual(action, ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE)

        resp = self.guardrail.get_response(assessment, action, "rw")
        self.assertIn("112", resp)

    def test_routing_acute_medical_emergency(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.ACUTE_MEDICAL_EMERGENCY,
            immediacy=ImmediacyLevel.IN_PROGRESS,
        )
        action = self.guardrail.determine_action(assessment)
        self.assertEqual(action, ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE)

        resp = self.guardrail.get_response(assessment, action, "fr")
        self.assertIn("912", resp)

    def test_routing_means_access_alone_without_intent(self):
        # Having pills or weapons without intent must not trigger emergency override
        assessment = SafetyAssessment(
            category=SafetyCategory.SAFE_CONVERSATIONAL,
            means=MeansStatus.ACCESS_AVAILABLE,
            intent=IntentLevel.NONE,
        )
        action = self.guardrail.determine_action(assessment)
        self.assertEqual(action, ApplicationAction.NORMAL_CONVERSATION)

    # -------------------------------------------------------------
    # Language Fallback Verification
    # -------------------------------------------------------------
    def test_language_fallback_to_detected_expression_language(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.IMMINENT_SELF_HARM,
            detected_expression_language="fr",
        )
        action = ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE
        # Caller supplies invalid/empty language -> falls back to assessment language 'fr'
        resp = self.guardrail.get_response(assessment, action, language="unknown")
        self.assertIn("danger immédiat", resp)

    def test_language_fallback_to_default_kinyarwanda(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.IMMINENT_SELF_HARM,
            detected_expression_language=None,
        )
        action = ApplicationAction.DETERMINISTIC_EMERGENCY_OVERRIDE
        # Caller supplies invalid language and assessment has None -> falls back to 'rw'
        resp = self.guardrail.get_response(assessment, action, language="es")
        self.assertIn("Polisi 112", resp)


if __name__ == "__main__":
    unittest.main()
