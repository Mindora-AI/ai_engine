"""
MWIZERWA AI — Unit Tests for Safety Domain & Static Registries
Chunk 4 Phase 1: Validates safety domain models, immutability, resource registry, and response templates.
"""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import src.safety.crisis_responses as crisis_responses
from src.safety import (
    ApplicationAction,
    EmergencyContact,
    ImmediacyLevel,
    IntentLevel,
    MeansStatus,
    ResponseSituation,
    SUPPORTED_SAFETY_LANGUAGES,
    SafetyAssessment,
    SafetyCategory,
    SubjectTarget,
    TemplateNotFoundError,
    TemporalStatus,
    VERIFIED_EMERGENCY_RESOURCES,
    VerificationStatus,
    get_all_resources,
    get_available_situations,
    get_emergency_contact_by_number,
    get_safety_response,
    get_supported_languages,
    get_verified_resources,
)


class TestSafetyDomainEnums(unittest.TestCase):
    """Verifies that all required domain enums and their categorical values exist."""

    def test_safety_categories_completeness(self):
        expected_categories = {
            "safe_conversational",
            "ordinary_distress",
            "ambiguous_distress",
            "general_or_hypothetical",
            "passive_death_wish",
            "self_harm_thoughts",
            "imminent_self_harm",
            "anger_or_venting",
            "threat_or_intent_harm_others",
            "imminent_violence_others",
            "third_person_crisis",
            "ongoing_physical_danger",
            "acute_medical_emergency",
        }
        actual_categories = {cat.value for cat in SafetyCategory}
        self.assertEqual(expected_categories, actual_categories)

    def test_application_actions_completeness(self):
        expected_actions = {
            "normal_conversation",
            "supportive_clarification",
            "safety_focused_response",
            "deterministic_emergency_override",
        }
        actual_actions = {action.value for action in ApplicationAction}
        self.assertEqual(expected_actions, actual_actions)

    def test_contextual_enums_completeness(self):
        # SubjectTarget
        self.assertIn(SubjectTarget.SELF, SubjectTarget)
        self.assertIn(SubjectTarget.THIRD_PERSON_KNOWN, SubjectTarget)
        self.assertIn(SubjectTarget.THIRD_PERSON_GENERAL, SubjectTarget)
        self.assertIn(SubjectTarget.UNSPECIFIED, SubjectTarget)

        # TemporalStatus
        self.assertIn(TemporalStatus.HISTORICAL, TemporalStatus)
        self.assertIn(TemporalStatus.CURRENT, TemporalStatus)
        self.assertIn(TemporalStatus.HYPOTHETICAL, TemporalStatus)

        # IntentLevel
        self.assertIn(IntentLevel.NONE, IntentLevel)
        self.assertIn(IntentLevel.PASSIVE_WISH, IntentLevel)
        self.assertIn(IntentLevel.INTRUSIVE_THOUGHTS, IntentLevel)
        self.assertIn(IntentLevel.FEAR_OF_ACTING, IntentLevel)
        self.assertIn(IntentLevel.EXPLICIT_INTENT, IntentLevel)
        self.assertIn(IntentLevel.ACTION_IN_PROGRESS, IntentLevel)

        # MeansStatus
        self.assertIn(MeansStatus.NONE_MENTIONED, MeansStatus)
        self.assertIn(MeansStatus.MENTION_ONLY, MeansStatus)
        self.assertIn(MeansStatus.ACCESS_AVAILABLE, MeansStatus)
        self.assertIn(MeansStatus.PREPARATION, MeansStatus)
        self.assertIn(MeansStatus.INGESTION_OR_INJURY_UNDERWAY, MeansStatus)

        # ImmediacyLevel
        self.assertIn(ImmediacyLevel.NONE, ImmediacyLevel)
        self.assertIn(ImmediacyLevel.UNCLEAR, ImmediacyLevel)
        self.assertIn(ImmediacyLevel.IMMINENT, ImmediacyLevel)
        self.assertIn(ImmediacyLevel.IN_PROGRESS, ImmediacyLevel)

        # VerificationStatus
        self.assertEqual(VerificationStatus.VERIFIED.value, "verified")
        self.assertEqual(VerificationStatus.PROVISIONAL.value, "provisional")
        self.assertEqual(VerificationStatus.UNVERIFIED.value, "unverified")

        # ResponseSituation completeness
        expected_situations = {
            "passive_death_wish",
            "self_harm_thoughts",
            "third_person_crisis",
            "imminent_self_harm",
            "imminent_violence_others",
            "ongoing_physical_danger",
            "acute_medical_emergency",
        }
        actual_situations = {sit.value for sit in ResponseSituation}
        self.assertEqual(expected_situations, actual_situations)


class TestSafetyAssessmentModel(unittest.TestCase):
    """Validates the frozen SafetyAssessment dataclass behavior and validation rules."""

    def test_default_construction(self):
        assessment = SafetyAssessment(category=SafetyCategory.SAFE_CONVERSATIONAL)
        self.assertEqual(assessment.category, SafetyCategory.SAFE_CONVERSATIONAL)
        self.assertEqual(assessment.subject, SubjectTarget.UNSPECIFIED)
        self.assertEqual(assessment.temporal, TemporalStatus.CURRENT)
        self.assertEqual(assessment.intent, IntentLevel.NONE)
        self.assertEqual(assessment.means, MeansStatus.NONE_MENTIONED)
        self.assertEqual(assessment.immediacy, ImmediacyLevel.NONE)
        self.assertIsNone(assessment.detected_expression_language)
        self.assertEqual(assessment.matched_markers, ())

    def test_explicit_construction(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.IMMINENT_SELF_HARM,
            subject=SubjectTarget.SELF,
            temporal=TemporalStatus.CURRENT,
            intent=IntentLevel.ACTION_IN_PROGRESS,
            means=MeansStatus.INGESTION_OR_INJURY_UNDERWAY,
            immediacy=ImmediacyLevel.IN_PROGRESS,
            detected_expression_language="rw",
            matched_markers=("nanyoye uburozi",),
        )
        self.assertEqual(assessment.category, SafetyCategory.IMMINENT_SELF_HARM)
        self.assertEqual(assessment.subject, SubjectTarget.SELF)
        self.assertEqual(assessment.intent, IntentLevel.ACTION_IN_PROGRESS)
        self.assertEqual(assessment.immediacy, ImmediacyLevel.IN_PROGRESS)
        self.assertEqual(assessment.detected_expression_language, "rw")
        self.assertEqual(assessment.matched_markers, ("nanyoye uburozi",))

    def test_immutability(self):
        assessment = SafetyAssessment(category=SafetyCategory.ORDINARY_DISTRESS)
        with self.assertRaises(FrozenInstanceError):
            assessment.category = SafetyCategory.IMMINENT_SELF_HARM  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            assessment.immediacy = ImmediacyLevel.IMMINENT  # type: ignore

    def test_language_validation_accepted(self):
        for lang in ("rw", "en", "fr", None):
            assessment = SafetyAssessment(
                category=SafetyCategory.SAFE_CONVERSATIONAL,
                detected_expression_language=lang,
            )
            self.assertEqual(assessment.detected_expression_language, lang)

    def test_language_validation_rejected(self):
        invalid_languages = ["de", "es", "zh", "kin", "unknown", 123]
        for invalid_lang in invalid_languages:
            with self.assertRaises(ValueError):
                SafetyAssessment(
                    category=SafetyCategory.SAFE_CONVERSATIONAL,
                    detected_expression_language=invalid_lang,  # type: ignore
                )

    def test_matched_markers_list_coercion_to_tuple(self):
        assessment = SafetyAssessment(
            category=SafetyCategory.PASSIVE_DEATH_WISH,
            matched_markers=["kwipfira", "ndarambiwe"],  # type: ignore
        )
        self.assertIsInstance(assessment.matched_markers, tuple)
        self.assertEqual(assessment.matched_markers, ("kwipfira", "ndarambiwe"))


class TestEmergencyResourceRegistry(unittest.TestCase):
    """Validates the static verified emergency resources registry."""

    def test_resource_immutability(self):
        resources = get_all_resources()
        self.assertGreaterEqual(len(resources), 2)
        contact = resources[0]
        with self.assertRaises(FrozenInstanceError):
            contact.phone_number = "999"  # type: ignore

    def test_verified_resources_integrity(self):
        verified = get_verified_resources()
        self.assertGreaterEqual(len(verified), 2)
        numbers = {c.phone_number for c in verified}
        self.assertIn("112", numbers)
        self.assertIn("912", numbers)

        for c in verified:
            self.assertEqual(c.verification_status, VerificationStatus.VERIFIED)
            self.assertTrue(c.phone_number)
            self.assertTrue(c.service_name)
            self.assertTrue(c.intended_use)
            self.assertTrue(c.geographic_coverage)
            self.assertTrue(c.verification_source)
            self.assertTrue(c.verification_date)
            self.assertTrue(c.description_rw)
            self.assertTrue(c.description_en)
            self.assertTrue(c.description_fr)

    def test_unverified_not_in_registry(self):
        # 114 must not be in verified resources as a suicide hotline
        contact_114 = get_emergency_contact_by_number("114")
        self.assertIsNone(contact_114)

    def test_verified_resources_exact_provenance_urls(self):
        """Assert both registered emergency contacts record exact official source-page URLs."""
        contact_112 = get_emergency_contact_by_number("112")
        self.assertIsNotNone(contact_112)
        self.assertEqual(
            contact_112.verification_source,
            "https://www.police.gov.rw/home/?page_id=35",
        )
        self.assertNotIn("mental", contact_112.intended_use.lower())
        self.assertNotIn("suicide", contact_112.intended_use.lower())

        contact_912 = get_emergency_contact_by_number("912")
        self.assertIsNotNone(contact_912)
        self.assertEqual(
            contact_912.verification_source,
            "https://www.rbc.gov.rw/who-we-are/our-divisions-and-units/samu-services",
        )
        self.assertNotIn("mental", contact_912.intended_use.lower())
        self.assertNotIn("suicide", contact_912.intended_use.lower())

    def test_contact_lookup(self):
        contact_112 = get_emergency_contact_by_number("112")
        self.assertIsNotNone(contact_112)
        self.assertEqual(contact_112.phone_number, "112")
        self.assertIn("Police", contact_112.service_name)

        contact_912 = get_emergency_contact_by_number("  912 ")
        self.assertIsNotNone(contact_912)
        self.assertEqual(contact_912.phone_number, "912")
        self.assertIn("SAMU", contact_912.service_name)

        # Non-existent
        self.assertIsNone(get_emergency_contact_by_number("999"))
        self.assertIsNone(get_emergency_contact_by_number(""))

    def test_verified_resources_filtering_with_mixed_statuses(self):
        """Construct representative contacts with all 3 statuses and verify strict filtering."""
        contact_verified = EmergencyContact(
            phone_number="112",
            service_name="Verified Service",
            intended_use="Verified use",
            geographic_coverage="Rwanda",
            verification_source="https://example.gov.rw/verified",
            verification_date="2026-03-31",
            verification_status=VerificationStatus.VERIFIED,
            description_rw="Verified rw",
            description_en="Verified en",
            description_fr="Verified fr",
        )
        contact_provisional = EmergencyContact(
            phone_number="001",
            service_name="Provisional Service",
            intended_use="Provisional use",
            geographic_coverage="Rwanda",
            verification_source="https://example.gov.rw/prov",
            verification_date="2026-03-31",
            verification_status=VerificationStatus.PROVISIONAL,
            description_rw="Prov rw",
            description_en="Prov en",
            description_fr="Prov fr",
        )
        contact_unverified = EmergencyContact(
            phone_number="002",
            service_name="Unverified Service",
            intended_use="Unverified use",
            geographic_coverage="Rwanda",
            verification_source="https://example.gov.rw/unver",
            verification_date="2026-03-31",
            verification_status=VerificationStatus.UNVERIFIED,
            description_rw="Unver rw",
            description_en="Unver en",
            description_fr="Unver fr",
        )

        mixed_tuple = (contact_verified, contact_provisional, contact_unverified)
        with patch("src.safety.resources.VERIFIED_EMERGENCY_RESOURCES", mixed_tuple):
            filtered = get_verified_resources()
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0].phone_number, "112")
            self.assertEqual(filtered[0].verification_status, VerificationStatus.VERIFIED)

    def test_predictable_lookup_for_unknown_and_malformed_numbers(self):
        """Verify that looking up arbitrary, blank, or malformed numbers returns None."""
        self.assertIsNone(get_emergency_contact_by_number("unknown_number_123"))
        self.assertIsNone(get_emergency_contact_by_number("000"))
        self.assertIsNone(get_emergency_contact_by_number("   "))
        self.assertIsNone(get_emergency_contact_by_number("- - -"))
        self.assertIsNone(get_emergency_contact_by_number("114"))


class TestCrisisResponseTemplates(unittest.TestCase):
    """Validates deterministic crisis response templates across all supported languages."""

    def test_all_situations_and_languages_covered(self):
        situations = get_available_situations()
        languages = get_supported_languages()
        self.assertEqual(len(situations), 7)
        self.assertEqual(set(languages), {"rw", "en", "fr"})

        for situation in situations:
            for lang in languages:
                response = get_safety_response(situation, lang)
                self.assertIsInstance(response, str)
                self.assertGreater(len(response.strip()), 30)

    def test_emergency_numbers_present_in_imminent_templates(self):
        # 112 must appear in imminent self-harm, imminent violence, and physical danger
        for lang in ("rw", "en", "fr"):
            imminent_sh = get_safety_response(ResponseSituation.IMMINENT_SELF_HARM, lang)
            self.assertIn("112", imminent_sh)

            imminent_viol = get_safety_response(ResponseSituation.IMMINENT_VIOLENCE_OTHERS, lang)
            self.assertIn("112", imminent_viol)

            physical_danger = get_safety_response(ResponseSituation.ONGOING_PHYSICAL_DANGER, lang)
            self.assertIn("112", physical_danger)

        # 912 must appear in acute medical emergency
        for lang in ("rw", "en", "fr"):
            medical = get_safety_response(ResponseSituation.ACUTE_MEDICAL_EMERGENCY, lang)
            self.assertIn("912", medical)

    def test_unsupported_language_rejected(self):
        with self.assertRaises(ValueError):
            get_safety_response(ResponseSituation.PASSIVE_DEATH_WISH, "de")

        with self.assertRaises(ValueError):
            get_safety_response(ResponseSituation.PASSIVE_DEATH_WISH, "")

    def test_invalid_situation_rejected(self):
        with self.assertRaises(ValueError):
            get_safety_response("invalid_situation", "en")  # type: ignore

    def test_template_not_found_on_bogus_lookup(self):
        """Verify TemplateNotFoundError is raised when a situation is missing from the template mapping."""
        # Use patch.dict to temporarily clear the templates mapping in a controlled fixture
        with patch.dict("src.safety.crisis_responses._CRISIS_TEMPLATES", clear=True):
            with self.assertRaises(TemplateNotFoundError):
                get_safety_response(ResponseSituation.PASSIVE_DEATH_WISH, "en")

        # Verify shared state is completely restored after the test context exit
        restored_response = get_safety_response(ResponseSituation.PASSIVE_DEATH_WISH, "en")
        self.assertIsInstance(restored_response, str)
        self.assertGreater(len(restored_response), 0)

    def test_language_specific_copy_quality_and_refinements(self):
        """Verify French diacritics and Kinyarwanda phrasing refinements."""
        # Kinyarwanda refinements
        pdw_rw = get_safety_response(ResponseSituation.PASSIVE_DEATH_WISH, "rw")
        self.assertIn("Nturi wenyine", pdw_rw)
        self.assertNotIn("Ntukiri wenyine", pdw_rw)

        tpc_rw = get_safety_response(ResponseSituation.THIRD_PERSON_CRISIS, "rw")
        self.assertIn("ntumusige wenyine", tpc_rw)
        self.assertNotIn("ntumuherane", tpc_rw)

        # French diacritics & conditional nuance
        pdw_fr = get_safety_response(ResponseSituation.PASSIVE_DEATH_WISH, "fr")
        self.assertIn("êtes épuisé", pdw_fr)
        self.assertIn("santé", pdw_fr)

        tpc_fr = get_safety_response(ResponseSituation.THIRD_PERSON_CRISIS, "fr")
        self.assertIn("si elle est en danger", tpc_fr)
        self.assertIn("très éprouvant", tpc_fr)

        imminent_sh_fr = get_safety_response(ResponseSituation.IMMINENT_SELF_HARM, "fr")
        self.assertIn("à vos côtés", imminent_sh_fr)
        self.assertIn("énormément", imminent_sh_fr)

        danger_fr = get_safety_response(ResponseSituation.ONGOING_PHYSICAL_DANGER, "fr")
        self.assertIn("lieu sûr", danger_fr)

    def test_no_claims_of_calling_emergency_services(self):
        """Verify that no template claims the assistant will or can place emergency calls."""
        prohibited_markers = [
            "i called",
            "i have called",
            "i will call",
            "calling for you",
            "j'ai appelé",
            "je vais appeler",
            "nous avons appelé",
            "nahamagaye",
            "ngiye guhamagara",
            "twahamagaye",
        ]
        situations = get_available_situations()
        languages = get_supported_languages()
        for situation in situations:
            for lang in languages:
                text = get_safety_response(situation, lang).lower()
                for marker in prohibited_markers:
                    self.assertNotIn(
                        marker,
                        text,
                        f"Prohibited service-calling claim '{marker}' found in situation '{situation.value}', lang '{lang}'",
                    )


if __name__ == "__main__":
    unittest.main()
