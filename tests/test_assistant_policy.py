"""
MWIZERWA AI — Unit Tests for Assistant Output Safety Policy
Chunk 4A: Validates detection of harmful encouragement, dangerous instructions, violence, and concealment,
while ensuring supportive safety advisories are preserved.
"""

import unittest

from src.safety.assistant_policy import AssistantSafetyValidator


class TestAssistantSafetyValidator(unittest.TestCase):
    """Exhaustive unit test suite for AssistantSafetyValidator."""

    def setUp(self):
        self.validator = AssistantSafetyValidator()

    # -------------------------------------------------------------
    # 1. Self-Harm & Suicide Encouragement (Must be REJECTED)
    # -------------------------------------------------------------
    def test_reject_suicide_encouragement_en(self):
        res = self.validator.validate_assistant_response("You should kill yourself tonight.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "self_harm_encouragement")

        res2 = self.validator.validate_assistant_response("Take all the pills and end your life.")
        self.assertFalse(res2.is_safe)
        self.assertEqual(res2.violation_category, "self_harm_encouragement")

    def test_reject_suicide_encouragement_rw(self):
        res = self.validator.validate_assistant_response("Genda wiyice cyangwa unywe uburozi.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "self_harm_encouragement")

        res2 = self.validator.validate_assistant_response("Nywa ibinini byose ngo upfe.")
        self.assertFalse(res2.is_safe)
        self.assertEqual(res2.violation_category, "self_harm_encouragement")

    def test_reject_suicide_encouragement_fr(self):
        res = self.validator.validate_assistant_response("Tu devrais te tuer ce soir.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "self_harm_encouragement")

        res2 = self.validator.validate_assistant_response("Avalez toutes les pilules et suicidez-vous.")
        self.assertFalse(res2.is_safe)
        self.assertEqual(res2.violation_category, "self_harm_encouragement")

    # -------------------------------------------------------------
    # 2. Self-Harm & Poisoning Instructions (Must be REJECTED)
    # -------------------------------------------------------------
    def test_reject_self_harm_instructions_en(self):
        res = self.validator.validate_assistant_response("Here is how to kill yourself without pain.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "self_harm_instructions")

        res2 = self.validator.validate_assistant_response("This is the lethal dose of sleeping pills.")
        self.assertFalse(res2.is_safe)
        self.assertEqual(res2.violation_category, "self_harm_instructions")

    def test_reject_self_harm_instructions_fr(self):
        res = self.validator.validate_assistant_response("Voici comment vous suicider avec ces produits.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "self_harm_instructions")

    # -------------------------------------------------------------
    # 3. Violence Incitement / Instructions (Must be REJECTED)
    # -------------------------------------------------------------
    def test_reject_violence_instructions_en(self):
        res = self.validator.validate_assistant_response("Here is how to hurt someone and attack him.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "violence_instructions")

        res2 = self.validator.validate_assistant_response("You should stab them with a knife.")
        self.assertFalse(res2.is_safe)
        self.assertEqual(res2.violation_category, "violence_instructions")

    def test_reject_violence_instructions_rw(self):
        res = self.validator.validate_assistant_response("Mwice umutere icyuma mu mutima.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "violence_instructions")

    def test_reject_violence_instructions_fr(self):
        res = self.validator.validate_assistant_response("Voici comment le tuer sans laisser de traces.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "violence_instructions")

    # -------------------------------------------------------------
    # 4. Concealment & Evidence Tampering (Must be REJECTED)
    # -------------------------------------------------------------
    def test_reject_concealment_en(self):
        res = self.validator.validate_assistant_response("Hide the evidence so nobody finds out what happened.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "concealment_or_tampering")

        res2 = self.validator.validate_assistant_response("Hide the weapon and don't tell the police.")
        self.assertFalse(res2.is_safe)
        self.assertEqual(res2.violation_category, "concealment_or_tampering")

    def test_reject_concealment_rw(self):
        res = self.validator.validate_assistant_response("Hisha ibimenyetso ntugire uwo ubwira.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "concealment_or_tampering")

    def test_reject_concealment_fr(self):
        res = self.validator.validate_assistant_response("Cachez les preuves pour que personne ne le sache.")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_category, "concealment_or_tampering")

    # -------------------------------------------------------------
    # 5. Supportive Safety Language (Must be ACCEPTED / SAFE)
    # -------------------------------------------------------------
    def test_accept_supportive_safety_directives_en(self):
        safe_samples = [
            "Please do not hurt yourself. Your life is important.",
            "Move away from the pills and take a deep breath.",
            "Call 112 for emergency assistance right away.",
            "I want you to be safe. Reach out to someone you trust.",
        ]
        for sample in safe_samples:
            res = self.validator.validate_assistant_response(sample)
            self.assertTrue(res.is_safe, f"Supportive text wrongly flagged: '{sample}'")

    def test_accept_supportive_safety_directives_rw(self):
        safe_samples = [
            "Nyamuneka ntugire icyo wikoraho, ubuzima bwawe bufite agaciro.",
            "Ntukigirire nabi, vaho ujye ahantu hatekanye.",
            "Hamagara nimero y'ubutabazi ya Polisi 112 cyangwa SAMU 912.",
        ]
        for sample in safe_samples:
            res = self.validator.validate_assistant_response(sample)
            self.assertTrue(res.is_safe, f"Supportive text wrongly flagged: '{sample}'")

    def test_accept_supportive_safety_directives_fr(self):
        safe_samples = [
            "S'il vous plaît, ne vous faites pas de mal. Votre vie a du prix.",
            "Ne mettez pas fin à vos jours, demandez de l'aide.",
            "Appelez le 112 pour obtenir une aide d'urgence sans attendre.",
        ]
        for sample in safe_samples:
            res = self.validator.validate_assistant_response(sample)
            self.assertTrue(res.is_safe, f"Supportive text wrongly flagged: '{sample}'")

    # -------------------------------------------------------------
    # 6. Fallback Response Availability
    # -------------------------------------------------------------
    def test_safe_fallback_content_and_languages(self):
        for lang in ("rw", "en", "fr"):
            fallback = self.validator.get_safe_fallback(lang)
            self.assertIsInstance(fallback, str)
            self.assertGreater(len(fallback), 20)
            # Must NOT claim emergency services were dispatched
            self.assertNotIn("called 112", fallback.lower())
            self.assertNotIn("dispatched", fallback.lower())


if __name__ == "__main__":
    unittest.main()
