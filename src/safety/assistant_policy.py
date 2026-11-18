"""
MWIZERWA AI — Assistant Output Safety Policy
Deterministic, offline validator that intercepts unsafe assistant-generated responses
such as harmful instructions, suicide encouragement, violence incitement, or concealment.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AssistantSafetyResult:
    """Immutable result of assistant output safety validation."""
    is_safe: bool
    violation_category: Optional[str] = None
    matched_pattern: Optional[str] = None


class AssistantSafetyValidator:
    """
    Validates candidate assistant output to ensure the model never generates:
    1. Encouragement or solicitation of suicide or self-harm.
    2. Instructions or methods for self-harm or suicide.
    3. Instructions or encouragement of violence against other people.
    4. Instructions for dangerous ingestion, poisoning, or lethal means.
    5. Encouragement or advice on concealing harm, evidence, or crimes.

    Critical Boundary:
    Ordinary supportive safety directives (e.g. 'Please do not hurt yourself',
    'Ntukigirire nabi', 'Ne vous faites pas de mal') MUST NOT be flagged as violations.
    """

    def __init__(self):
        self._compile_rules()

    def _compile_rules(self) -> None:
        """Pre-compiles regex patterns for assistant output safety enforcement."""

        # Explicit non-violative supportive phrases to protect from false positives
        self._supportive_whitelist = re.compile(
            r"\b(please (?:do not|don't) (?:hurt|kill|harm) yourself|"
            r"never (?:hurt|kill|harm) yourself|"
            r"do not (?:hurt|kill|harm) yourself|"
            r"do not take (?:pills|poison|your life)|"
            r"move away from (?:the pills|the knife|harm)|"
            r"reach out (?:to someone|for help)|"
            r"call (?:112|912)|"
            r"nyamuneka ntugire icyo wikoraho|ntukigirire nabi|ntukiyahure|witangiza ubuzima|"
            r"s'il vous plaît,? ne vous faites pas de mal|ne mettez pas fin à vos jours|"
            r"ne vous tuez pas|appelez le (?:112|912))\b",
            re.IGNORECASE,
        )

        # 1. Suicide / Self-Harm Encouragement & Directives
        # e.g., "You should kill yourself", "Go ahead and end it", "Take all the pills tonight"
        self._harm_encouragement = re.compile(
            r"\b((?:you should|you can|why don't you|go ahead and|just|simply|better to|worth) (?:kill yourself|end your life|die|commit suicide)|"
            r"(?:take|swallow|ingest) (?:all the|a bunch of|the whole bottle of|lethal) (?:pills|medication|poison|tablets)|"
            r"(?:slit|cut) your (?:wrists|throat|arms)|"
            r"(?:hang|shoot|poison) yourself|"
            r"you (?:deserve to die|are better off dead)|"
            r"nobody (?:will care|would miss you) if you (?:die|were gone)|"
            r"(?:genda )?(?:wiyice|iyice|wipfire|ipfire|(?:u)?nywa ibinini byose|(?:u)?nywa uburozi|ikate imitsi|wikate imitsi|ukwiye gupfa)|"
            r"(?:tu devrais te|vous devriez vous) (?:tuer|suicider)|"
            r"(?:tu devrais|vous devriez) (?:en finir|mourir)|"
            r"(?:suicidez-vous|tuez-vous|mettez fin à vos jours|avalez toutes les pilules|avalez le poison)|"
            r"vous feriez mieux de (?:mourir|en finir))\b",
            re.IGNORECASE,
        )

        # 2. Instructions / Facilitation for Self-Harm or Lethal Ingestion
        # e.g., "Here is how to hurt yourself", "How to make lethal poison"
        self._harm_instructions = re.compile(
            r"\b((?:here is|this is|steps to|how to|guide on how to) (?:kill yourself|commit suicide|hang yourself|cut yourself|overdose)|"
            r"(?:lethal|fatal) (?:dose|combination|mixture) of|"
            r"(?:uburyo bwo|dore uko) (?:wakwiyahura|wakwiyica|wafatamo ibinini ngo upfe)|"
            r"(?:voici comment|étapes pour|comment) vous (?:tuer|suicider|couper|empoisonner))\b",
            re.IGNORECASE,
        )

        # 3. Violence Incitement / Instructions against Others
        # e.g., "Here is how to hurt someone", "Attack him", "Stab them"
        self._violence_instructions = re.compile(
            r"\b((?:here is|how to|you should|go ahead and) (?:kill|hurt|attack|stab|shoot|strangle|poison) (?:him|her|them|someone|people)|"
            r"(?:ways|guide) to (?:severely injure|murder|assassinate)|"
            r"(?:dore uko|uburyo bwo) (?:kumwica|kumugirira nabi|gutera umuntu icyuma)|"
            r"(?:mwice|mutere icyuma|muhe uburozi)|"
            r"(?:voici comment|comment|tu devrais) (?:le tuer|la tuer|les tuer|frapper|poignarder)|"
            r"(?:attaquez-le|poignardez-le|tuez-le))\b",
            re.IGNORECASE,
        )

        # 4. Concealment of Harm, Evidence, or Crimes
        # e.g., "Hide the evidence so nobody finds out", "Hide the body", "Don't tell the police"
        self._concealment_instructions = re.compile(
            r"\b((?:hide|destroy|dispose of|get rid of) the (?:evidence|weapon|body|knife|gun|blood|drugs)|"
            r"so (?:nobody|no one) (?:finds out|discovers|catches you)|"
            r"(?:don't|do not) (?:tell the police|call the police|tell anyone what happened)|"
            r"(?:hisha|siba) (?:ibimenyetso|intwaro|amaraso)|"
            r"(?:ntugire uwo ubwira|ntukabwire polisi)|"
            r"(?:cachez|détruisez|effacez) les (?:preuves|traces|l'arme|le corps)|"
            r"pour que (?:personne ne le sache|la police ne découvre rien)|"
            r"ne dites rien à (?:la police|personne))\b",
            re.IGNORECASE,
        )

    def validate_assistant_response(self, text: str) -> AssistantSafetyResult:
        """
        Validates candidate assistant text against the safety policy.

        Args:
            text: Generated candidate response text.

        Returns:
            AssistantSafetyResult indicating whether text is safe or violates policy.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return AssistantSafetyResult(is_safe=True)

        # 1. Concealment / Evidence Tampering Instructions
        match = self._concealment_instructions.search(cleaned)
        if match:
            return AssistantSafetyResult(
                is_safe=False,
                violation_category="concealment_or_tampering",
                matched_pattern=match.group(0),
            )

        # 2. Instructions for Violence Against Others
        match = self._violence_instructions.search(cleaned)
        if match:
            return AssistantSafetyResult(
                is_safe=False,
                violation_category="violence_instructions",
                matched_pattern=match.group(0),
            )

        # 3. Instructions for Self-Harm or Suicide
        match = self._harm_instructions.search(cleaned)
        if match:
            return AssistantSafetyResult(
                is_safe=False,
                violation_category="self_harm_instructions",
                matched_pattern=match.group(0),
            )

        # 4. Encouragement / Directives for Self-Harm or Suicide
        match = self._harm_encouragement.search(cleaned)
        if match:
            # Check if this matched inside an explicit supportive advisory
            # e.g., "Please do not hurt yourself" must NOT trigger as "hurt yourself"
            if self._supportive_whitelist.search(cleaned):
                # Verify that the entire text is supportive rather than encouraging harm
                sub_text = self._supportive_whitelist.sub("", cleaned)
                # If after removing supportive clauses, dangerous encouragement remains:
                secondary_match = self._harm_encouragement.search(sub_text)
                if secondary_match:
                    return AssistantSafetyResult(
                        is_safe=False,
                        violation_category="self_harm_encouragement",
                        matched_pattern=secondary_match.group(0),
                    )
            else:
                return AssistantSafetyResult(
                    is_safe=False,
                    violation_category="self_harm_encouragement",
                    matched_pattern=match.group(0),
                )

        return AssistantSafetyResult(is_safe=True)

    def get_safe_fallback(self, language: str) -> str:
        """
        Returns a deterministic, non-harmful, empathetic replacement response
        in the appropriate language when an assistant output is rejected.
        """
        fallbacks = {
            "rw": (
                "Ndi hano kugira ngo ngufashe mu ituze no mu bwitonzi. "
                "Nyamuneka tubiganireho buhoro buhoro, kandi ubuzima bwawe n'umutekano wawe bifite agaciro kanini."
            ),
            "en": (
                "I am here to support you calmly and safely. "
                "Let's take things one step at a time, and remember that your safety and well-being come first."
            ),
            "fr": (
                "Je suis là pour vous soutenir calmement et en toute sécurité. "
                "Prenons les choses pas à pas, et n'oubliez pas que votre sécurité et votre bien-être sont essentiels."
            ),
        }
        norm_lang = (language or "").strip().lower()
        if norm_lang not in fallbacks:
            norm_lang = "rw"
        return fallbacks[norm_lang]
