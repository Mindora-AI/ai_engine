"""
MWIZERWA AI — Unit Tests for Contextual Safety Evaluator
Chunk 4 Phase 2D: Validates multi-dimensional risk evaluation, contextual nuances, and multilingual handling.
"""

import unittest

from src.safety.categories import (
    ImmediacyLevel,
    IntentLevel,
    MeansStatus,
    SafetyAssessment,
    SafetyCategory,
    SubjectTarget,
    TemporalStatus,
)
from src.safety.evaluator import SafetyEvaluator


class TestSafetyEvaluator(unittest.TestCase):
    """Exhaustive unit test suite for SafetyEvaluator."""

    def setUp(self):
        self.evaluator = SafetyEvaluator()

    # -------------------------------------------------------------
    # 1. Mandatory Benchmark Examples from Specification
    # -------------------------------------------------------------
    def test_benchmark_sleeping_pills_at_home(self):
        # "I have sleeping pills at home." — This alone is not a crisis.
        result = self.evaluator.evaluate("I have sleeping pills at home.")
        self.assertIn(result.category, (SafetyCategory.SAFE_CONVERSATIONAL, SafetyCategory.ORDINARY_DISTRESS))
        self.assertEqual(result.intent, IntentLevel.NONE)
        self.assertEqual(result.immediacy, ImmediacyLevel.NONE)
        self.assertEqual(result.means, MeansStatus.MENTION_ONLY)

    def test_benchmark_passive_death_wish(self):
        # "I sometimes wish I wouldn't wake up." — Passive death wish.
        result = self.evaluator.evaluate("I sometimes wish I wouldn't wake up.")
        self.assertEqual(result.category, SafetyCategory.PASSIVE_DEATH_WISH)
        self.assertEqual(result.intent, IntentLevel.PASSIVE_WISH)
        self.assertEqual(result.subject, SubjectTarget.SELF)
        self.assertEqual(result.temporal, TemporalStatus.CURRENT)

    def test_benchmark_intrusive_thoughts_fear_of_acting(self):
        # "I keep thinking about hurting myself, but I don't want to." — Self-harm thoughts / fear of acting.
        result = self.evaluator.evaluate("I keep thinking about hurting myself, but I don't want to.")
        self.assertEqual(result.category, SafetyCategory.SELF_HARM_THOUGHTS)
        self.assertEqual(result.intent, IntentLevel.FEAR_OF_ACTING)
        self.assertEqual(result.immediacy, ImmediacyLevel.NONE)

    def test_benchmark_imminent_self_harm(self):
        # "I intend to hurt myself tonight and have everything ready." — Imminent self-harm.
        result = self.evaluator.evaluate("I intend to hurt myself tonight and have everything ready.")
        self.assertEqual(result.category, SafetyCategory.IMMINENT_SELF_HARM)
        self.assertEqual(result.intent, IntentLevel.EXPLICIT_INTENT)
        self.assertEqual(result.immediacy, ImmediacyLevel.IMMINENT)

    def test_benchmark_action_underway_pills(self):
        # "I took a lot of pills." — Ingestion underway / acute emergency.
        result = self.evaluator.evaluate("I took a lot of pills.")
        self.assertIn(result.category, (SafetyCategory.ACUTE_MEDICAL_EMERGENCY, SafetyCategory.IMMINENT_SELF_HARM))
        self.assertEqual(result.means, MeansStatus.INGESTION_OR_INJURY_UNDERWAY)
        self.assertEqual(result.immediacy, ImmediacyLevel.IN_PROGRESS)

    def test_benchmark_historical_disclosure(self):
        # "I used to think about ending my life." — Historical disclosure.
        result = self.evaluator.evaluate("I used to think about ending my life.")
        self.assertEqual(result.category, SafetyCategory.GENERAL_OR_HYPOTHETICAL)
        self.assertEqual(result.temporal, TemporalStatus.HISTORICAL)
        self.assertEqual(result.intent, IntentLevel.NONE)

    def test_benchmark_hypothetical_suicide_discussion(self):
        # "What would happen if someone wanted to kill themselves?" — Hypothetical.
        result = self.evaluator.evaluate("What would happen if someone wanted to kill themselves?")
        self.assertEqual(result.category, SafetyCategory.GENERAL_OR_HYPOTHETICAL)
        self.assertEqual(result.temporal, TemporalStatus.HYPOTHETICAL)
        self.assertEqual(result.intent, IntentLevel.NONE)

    def test_benchmark_third_person_safe_with_user(self):
        # "My friend says he wants to kill himself, but he is safe and with me." — Third person non-imminent.
        result = self.evaluator.evaluate("My friend says he wants to kill himself, but he is safe and with me.")
        self.assertEqual(result.category, SafetyCategory.THIRD_PERSON_CRISIS)
        self.assertEqual(result.subject, SubjectTarget.THIRD_PERSON_KNOWN)
        self.assertEqual(result.immediacy, ImmediacyLevel.NONE)

    def test_benchmark_third_person_unconscious(self):
        # "My friend has taken pills and is unconscious." — Third person crisis with action underway.
        result = self.evaluator.evaluate("My friend has taken pills and is unconscious.")
        self.assertEqual(result.category, SafetyCategory.THIRD_PERSON_CRISIS)
        self.assertEqual(result.subject, SubjectTarget.THIRD_PERSON_KNOWN)
        self.assertEqual(result.means, MeansStatus.INGESTION_OR_INJURY_UNDERWAY)
        self.assertEqual(result.immediacy, ImmediacyLevel.IN_PROGRESS)

    def test_benchmark_anger_venting(self):
        # "I am so angry I could punch someone." — Venting, not imminent violence.
        result = self.evaluator.evaluate("I am so angry I could punch someone.")
        self.assertEqual(result.category, SafetyCategory.ANGER_OR_VENTING)
        self.assertEqual(result.immediacy, ImmediacyLevel.NONE)
        self.assertEqual(result.intent, IntentLevel.NONE)

    def test_benchmark_imminent_violence(self):
        # "I am outside his house with a weapon and intend to attack him now." — Imminent violence.
        result = self.evaluator.evaluate("I am outside his house with a weapon and intend to attack him now.")
        self.assertEqual(result.category, SafetyCategory.IMMINENT_VIOLENCE_OTHERS)
        self.assertEqual(result.intent, IntentLevel.EXPLICIT_INTENT)
        self.assertEqual(result.immediacy, ImmediacyLevel.IMMINENT)

    # -------------------------------------------------------------
    # 2. Multilingual: Kinyarwanda Coverage
    # -------------------------------------------------------------
    def test_kinyarwanda_passive_death_wish(self):
        result = self.evaluator.evaluate("Sinshaka kongera gukanguka, ndambiwe kubaho.")
        self.assertEqual(result.category, SafetyCategory.PASSIVE_DEATH_WISH)
        self.assertEqual(result.detected_expression_language, "rw")

    def test_kinyarwanda_action_underway(self):
        result = self.evaluator.evaluate("Nanyoye uburozi ubu ako kanya.")
        self.assertIn(result.category, (SafetyCategory.ACUTE_MEDICAL_EMERGENCY, SafetyCategory.IMMINENT_SELF_HARM))
        self.assertEqual(result.means, MeansStatus.INGESTION_OR_INJURY_UNDERWAY)
        self.assertEqual(result.immediacy, ImmediacyLevel.IN_PROGRESS)
        self.assertEqual(result.detected_expression_language, "rw")

    def test_kinyarwanda_physical_danger(self):
        result = self.evaluator.evaluate("Inzu iri gushya, umuriro uraka cyane!")
        self.assertEqual(result.category, SafetyCategory.ONGOING_PHYSICAL_DANGER)
        self.assertEqual(result.immediacy, ImmediacyLevel.IN_PROGRESS)
        self.assertEqual(result.detected_expression_language, "rw")

    def test_kinyarwanda_historical(self):
        result = self.evaluator.evaluate("Kera natekerezaga kwiyahura, ariko ubu byarashize.")
        self.assertEqual(result.category, SafetyCategory.GENERAL_OR_HYPOTHETICAL)
        self.assertEqual(result.temporal, TemporalStatus.HISTORICAL)
        self.assertEqual(result.detected_expression_language, "rw")

    def test_kinyarwanda_ordinary_distress(self):
        result = self.evaluator.evaluate("Ndababaye cyane uyu munsi, mfite agahinda mu mutima.")
        self.assertEqual(result.category, SafetyCategory.ORDINARY_DISTRESS)
        self.assertEqual(result.detected_expression_language, "rw")

    # -------------------------------------------------------------
    # 3. Multilingual: French Coverage
    # -------------------------------------------------------------
    def test_french_passive_death_wish(self):
        result = self.evaluator.evaluate("Je voudrais ne plus me réveiller, j'en ai marre de vivre.")
        self.assertEqual(result.category, SafetyCategory.PASSIVE_DEATH_WISH)
        self.assertEqual(result.detected_expression_language, "fr")

    def test_french_imminent_self_harm(self):
        result = self.evaluator.evaluate("Je vais me suicider ce soir, j'ai tout préparé.")
        self.assertEqual(result.category, SafetyCategory.IMMINENT_SELF_HARM)
        self.assertEqual(result.immediacy, ImmediacyLevel.IMMINENT)
        self.assertEqual(result.detected_expression_language, "fr")

    def test_french_physical_danger(self):
        result = self.evaluator.evaluate("La maison brûle, vite au secours!")
        self.assertEqual(result.category, SafetyCategory.ONGOING_PHYSICAL_DANGER)
        self.assertEqual(result.immediacy, ImmediacyLevel.IN_PROGRESS)
        self.assertEqual(result.detected_expression_language, "fr")

    def test_french_intrusive_thoughts(self):
        result = self.evaluator.evaluate("J'ai peur de me faire du mal, mais je ne veux pas le faire.")
        self.assertEqual(result.category, SafetyCategory.SELF_HARM_THOUGHTS)
        self.assertEqual(result.intent, IntentLevel.FEAR_OF_ACTING)
        self.assertEqual(result.detected_expression_language, "fr")

    # -------------------------------------------------------------
    # 4. Safe / Benign Conversational Input
    # -------------------------------------------------------------
    def test_empty_and_whitespace_input(self):
        empty_res = self.evaluator.evaluate("")
        self.assertEqual(empty_res.category, SafetyCategory.SAFE_CONVERSATIONAL)
        self.assertEqual(empty_res.matched_markers, ())

        ws_res = self.evaluator.evaluate("   \n\t  ")
        self.assertEqual(ws_res.category, SafetyCategory.SAFE_CONVERSATIONAL)

    def test_ordinary_conversational_questions(self):
        res1 = self.evaluator.evaluate("Hello, how are you doing today?")
        self.assertEqual(res1.category, SafetyCategory.SAFE_CONVERSATIONAL)

        res2 = self.evaluator.evaluate("Muraho neza, amakuru ki?")
        self.assertEqual(res2.category, SafetyCategory.SAFE_CONVERSATIONAL)


if __name__ == "__main__":
    unittest.main()
