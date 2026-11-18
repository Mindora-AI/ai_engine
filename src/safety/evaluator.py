"""
MWIZERWA AI — Contextual Safety Evaluator
Offline, deterministic, multilingual contextual safety assessment for MWIZERWA AI.
Distinguishes ordinary distress and conversational venting from acute crisis and self-harm.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from src.safety.categories import (
    ImmediacyLevel,
    IntentLevel,
    MeansStatus,
    SafetyAssessment,
    SafetyCategory,
    SubjectTarget,
    TemporalStatus,
)


class SafetyEvaluator:
    """
    Offline contextual safety evaluator.
    Independent of external networks, APIs, and the statistical language detector.
    Evaluates risk dimensions across Subject, Temporal Status, Intent, Means, and Immediacy.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compiles regex patterns for rapid deterministic matching."""

        # -------------------------------------------------------------
        # Language marker expressions
        # -------------------------------------------------------------
        self._lang_rw = re.compile(
            r"\b(ndifuza|sinshaka|ndakwitayeho|nanyoye|ibinini|uburozi|umuntu|inshuti|yange|yanjye|"
            r"mugenzi|kera|cyera|ubu|ngiye|kwiyahura|kwipfira|kwiyica|gupfa|ndambiwe|ndashaka|"
            r"umunaniro|ubwoba|umuhangayiko|akaga|umuriro|intwaro|amaraso|gutakaza|ntabwo|ariko|"
            r"ndababaye|agahinda|cyane|mutima|mfite)\b",
            re.IGNORECASE,
        )
        self._lang_fr = re.compile(
            r"\b(suicide|suicider|mourir|tuer|faire du mal|médicaments|pilules|poison|ami|amie|frère|"
            r"soeur|maintenant|ce soir|passé|autrefois|feu|brûle|arme|inconscient|saignement|aide|"
            r"urgences|peur|marre|pourquoi|jamais)\b",
            re.IGNORECASE,
        )
        self._lang_en = re.compile(
            r"\b(kill|suicide|die|hurt|pills|poison|friend|brother|sister|tonight|right now|"
            r"yesterday|used to|past|fire|weapon|gun|knife|unconscious|bleeding|emergency|"
            r"angry|punch|hate|wish|afraid|scared)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Subject / Target Patterns
        # -------------------------------------------------------------
        self._third_person_known = re.compile(
            r"\b(my friend|my brother|my sister|my mother|my father|my mom|my dad|my cousin|"
            r"my partner|my spouse|my son|my daughter|my classmate|my coworker|"
            r"inshuti yanjye|mugenzi wanjye|umuvandimwe wanjye|mubyara wanjye|umwana wanjye|"
            r"umugabo wanjye|umugore wanjye|inshuti|mugenzi|"
            r"mon ami|mon amie|mon frère|ma soeur|mon père|ma mère|mon fils|ma fille|mon copain|ma copine)\b",
            re.IGNORECASE,
        )
        self._third_person_general = re.compile(
            r"\b(someone|somebody|a person|people|he wants to|she wants to|they want to|"
            r"umuntu|abantu|umuntu ashaka|ashaka|"
            r"quelqu'un|une personne|les gens|il veut|elle veut|ils veulent)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Temporal Patterns
        # -------------------------------------------------------------
        self._temporal_historical = re.compile(
            r"\b(used to|in the past|years ago|months ago|long ago|when i was young|back then|previously|i once|"
            r"kera|cyera|mu gihe cyashize|mu myaka yashize|mbere|byarashize|"
            r"dans le passé|autrefois|il y a des années|il y a longtemps|j'y pensais avant|à l'époque)\b",
            re.IGNORECASE,
        )
        self._temporal_hypothetical = re.compile(
            r"\b(what would happen if|what if someone|hypothetically|curious if someone|why do people|"
            r"ese byagenda bite|ese biba byifashe bite|bigenze bite|iyo umuntu|kuki abantu|"
            r"que se passerait-il si|et si quelqu'un|hypothétiquement|pourquoi les gens)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Negation & Fear of Acting / Intrusive Thoughts
        # -------------------------------------------------------------
        self._fear_or_intrusive = re.compile(
            r"(keep thinking about (?:hurting|killing) myself,? but i don't want to|"
            r"thoughts of (?:death|dying|hurting myself),? but i don't want|"
            r"scared (?:that|of) (?:hurting|killing) myself|"
            r"afraid (?:that|i might) (?:hurt|kill) myself|"
            r"i would never actually do it|"
            r"ariko sinshaka kubikora|ndabitekereza ariko sinshaka|mfite ubwoba bwo kwigirira nabi|"
            r"mais je ne veux pas le faire|j'ai peur de me faire du mal|pensées intrusives)",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Underway / Active Ingestion or Harm Patterns
        # -------------------------------------------------------------
        self._action_underway = re.compile(
            r"\b((?:has |have |had )?taken (?:a lot of |all the |some |the |)?pills|"
            r"took (?:a lot of |all the |some |the |)?pills|swallowed (?:all the|a bunch of|the|some)? ?pills|"
            r"swallowed poison|drinking bleach|drank (?:poison|bleach)|bleeding out|already cut my|"
            r"nanyoye (?:ibinini|uburozi)|naciye imitsi|namaze kunywa|"
            r"j'ai avalé (?:des|tous les|les)? ?(?:médicaments|pilules)|j'ai pris beaucoup de pilules|j'ai bu du poison|"
            r"je saigne abondamment)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Imminent Self-Harm (Explicit intent + urgency/preparation)
        # -------------------------------------------------------------
        self._imminent_self_harm = re.compile(
            r"\b(intend to hurt myself (?:tonight|now|today)|"
            r"(?:going to|will) (?:kill|end) my(?:self| life) (?:tonight|now|today)|"
            r"have everything ready to (?:die|kill myself|end it)|"
            r"goodbye everyone,? this is (?:my end|it)|"
            r"ngiye kwiyica (?:ubu|uyu munsi|kuri uyu mugoroba)|"
            r"nateguye byose ngo nipfire|"
            r"je vais me (?:tuer|suicider) (?:ce soir|maintenant)|"
            r"j'ai tout préparé pour (?:en finir|mourir))\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Active Self-Harm Thoughts (without confirmed execution underway)
        # -------------------------------------------------------------
        self._active_self_harm = re.compile(
            r"\b(want(?:s)? to kill (?:myself|himself|herself|themselves|oneself)|"
            r"thinking of killing (?:myself|himself|herself|themselves|oneself)|"
            r"want(?:s)? to die|thinking about suicide|commit(?:ting)? suicide|"
            r"end(?:ing)? (?:my|his|her|their) life|end it all|"
            r"hurt(?:ing)? (?:myself|himself|herself|themselves)|"
            r"cut(?:ting)? (?:myself|himself|herself|themselves)|"
            r"take (?:my|his|her|their) own life|"
            r"kill(?:ing)? (?:myself|himself|herself|themselves|oneself)|"
            r"kwiyahura|kwiyica|kwigirira nabi|ndashaka kwipfira|ndifuza kwiyica|ndatekereza kwiyahura|"
            r"me tuer|me suicider|se tuer|se suicider|mettre fin à mes jours|mettre fin a mes jours|me faire du mal|"
            r"se faire du mal|envie de mourir|pense au suicide)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Passive Death Wish Patterns
        # -------------------------------------------------------------
        self._passive_death_wish = re.compile(
            r"\b(wish i wouldn't wake up|wish i didn't wake up|wish i were dead|wish i was dead|"
            r"better off dead|no reason to live|tired of living|don't want to exist|"
            r"disappear forever|tired of being here|tired of this life|"
            r"sinshaka gukanguka|sinshaka kongera gukanguka|ndambiwe kubaho|napfa nkaruhuka|"
            r"nta mpamvu yo kubaho|sinshaka kubaho|"
            r"ne plus me réveiller|ne plus me reveiller|voudrais être mort|marre de vivre|"
            r"plus envie de vivre|ne plus exister)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Violence Towards Others Patterns
        # -------------------------------------------------------------
        self._imminent_violence = re.compile(
            r"\b(outside (?:his|her|their) house with a weapon|"
            r"outside with a (?:gun|knife|weapon) and (?:intend|going) to|"
            r"about to (?:attack|kill|shoot) (?:him|her|them) now|"
            r"ndi hanze (?:ye|y'inzu) n'intwaro|"
            r"devant chez (?:lui|elle) avec une arme)\b",
            re.IGNORECASE,
        )
        self._threat_violence = re.compile(
            r"\b(going to kill (?:him|her|them)|want to kill (?:him|her|them)|"
            r"intend to attack (?:him|her|them)|will stab (?:him|her|them)|"
            r"nzamwica|nzabica|ngiye kumutera|ngiye kubagirira nabi|"
            r"je vais le tuer|je vais la tuer|je vais les massacrer)\b",
            re.IGNORECASE,
        )
        self._anger_venting = re.compile(
            r"\b(so angry i could punch (?:someone|a wall)|"
            r"so mad|pissed off|furious|i hate everyone|want to punch a wall|"
            r"ndarakaye cyane|ndumva nakubita|ndarakaye|"
            r"tellement en colère|envie de frapper dans le mur|marre de tout le monde)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Ongoing Physical Danger Patterns
        # -------------------------------------------------------------
        self._ongoing_danger = re.compile(
            r"\b(someone is breaking in|intruder in (?:my|the) house|"
            r"house is on fire|building on fire|fire in my room|"
            r"being attacked right now|someone is following me with a knife|"
            r"inzu iri gushya|umuriro uraka|abajura binjiye|ndi guterwa ubu|"
            r"la maison brûle|le feu a pris|on s'introduit chez moi|on m'attaque maintenant)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Acute Medical Emergency Patterns
        # -------------------------------------------------------------
        self._acute_medical = re.compile(
            r"\b(unconscious|not breathing|drank pesticide|drank bleach|severe bleeding won't stop|"
            r"collapsed and not moving|heart attack symptoms|"
            r"yataye ubwenge|ntari guhumeka|yanyoye uburozi|amaraso menshi adahagarara|"
            r"inconscient|ne respire plus|saignement incontrôlable|arrêt cardiaque|bu de l'eau de javel)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Means mentions (benign or contextual)
        # -------------------------------------------------------------
        self._means_mention = re.compile(
            r"\b(sleeping pills|pills|medication|knife|knives|rope|poison|gun|weapon|"
            r"ibinini by'ibitotsi|ibinini|uburozi|icyuma|umugozi|imbunda|intwaro|"
            r"somnifères|médicaments|pilules|couteau|corde|poison|arme)\b",
            re.IGNORECASE,
        )

        # -------------------------------------------------------------
        # Ordinary Distress Patterns
        # -------------------------------------------------------------
        self._ordinary_distress = re.compile(
            r"\b(feeling sad|feeling down|so stressed|exhausted|depressed|overwhelmed|heartbroken|"
            r"lonely|grieving|failed my exam|hard day|crying|"
            r"ndababaye|mfite agahinda|ndaniwe cyane|umunaniro|ndumva nguye isari|ndigunga|"
            r"triste|angoissé|stressé|épuisé|déprimé|démoralisé|journée difficile)\b",
            re.IGNORECASE,
        )

    def _infer_language(self, text: str) -> Optional[str]:
        """Infers the language marker profile strictly from current-turn safety markers."""
        score_rw = len(self._lang_rw.findall(text))
        score_fr = len(self._lang_fr.findall(text))
        score_en = len(self._lang_en.findall(text))

        max_score = max(score_rw, score_fr, score_en)
        if max_score == 0:
            return None

        if score_rw == max_score and score_rw > score_en and score_rw > score_fr:
            return "rw"
        if score_fr == max_score and score_fr > score_en and score_fr > score_rw:
            return "fr"
        if score_en == max_score and score_en > score_rw and score_en > score_fr:
            return "en"

        return None

    def evaluate(
        self,
        text: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> SafetyAssessment:
        """
        Performs an offline, multi-dimensional contextual safety evaluation on user input.

        Args:
            text: Current turn message text.
            conversation_context: Optional prior message history list.

        Returns:
            SafetyAssessment frozen dataclass instance.
        """
        raw_text = (text or "").strip()
        if not raw_text:
            return SafetyAssessment(
                category=SafetyCategory.SAFE_CONVERSATIONAL,
                subject=SubjectTarget.UNSPECIFIED,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.NONE,
                means=MeansStatus.NONE_MENTIONED,
                immediacy=ImmediacyLevel.NONE,
                detected_expression_language=None,
                matched_markers=(),
            )

        expr_lang = self._infer_language(raw_text)
        matched_markers: List[str] = []

        # --- Dimension 1: Subject Target ---
        subject = SubjectTarget.SELF
        if self._third_person_known.search(raw_text):
            subject = SubjectTarget.THIRD_PERSON_KNOWN
            matched = self._third_person_known.findall(raw_text)
            matched_markers.extend(matched[:2])
        elif self._third_person_general.search(raw_text):
            subject = SubjectTarget.THIRD_PERSON_GENERAL
            matched = self._third_person_general.findall(raw_text)
            matched_markers.extend(matched[:2])

        # --- Dimension 2: Temporal Status ---
        temporal = TemporalStatus.CURRENT
        if self._temporal_hypothetical.search(raw_text):
            temporal = TemporalStatus.HYPOTHETICAL
            matched = self._temporal_hypothetical.findall(raw_text)
            matched_markers.extend(matched[:2])
        elif self._temporal_historical.search(raw_text):
            temporal = TemporalStatus.HISTORICAL
            matched = self._temporal_historical.findall(raw_text)
            matched_markers.extend(matched[:2])

        # --- Dimension 3: Means Status ---
        means = MeansStatus.NONE_MENTIONED
        if self._means_mention.search(raw_text):
            means = MeansStatus.MENTION_ONLY
            matched = self._means_mention.findall(raw_text)
            matched_markers.extend(matched[:2])

        # --- Dimension 4: Acute Medical / Ingestion Underway ---
        if self._action_underway.search(raw_text):
            matched = self._action_underway.findall(raw_text)
            matched_markers.extend(matched[:2])
            means = MeansStatus.INGESTION_OR_INJURY_UNDERWAY
            immediacy = ImmediacyLevel.IN_PROGRESS
            intent = IntentLevel.ACTION_IN_PROGRESS

            if subject in (SubjectTarget.THIRD_PERSON_KNOWN, SubjectTarget.THIRD_PERSON_GENERAL):
                category = SafetyCategory.THIRD_PERSON_CRISIS
            else:
                category = SafetyCategory.ACUTE_MEDICAL_EMERGENCY

            return SafetyAssessment(
                category=category,
                subject=subject,
                temporal=TemporalStatus.CURRENT,
                intent=intent,
                means=means,
                immediacy=immediacy,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 5: Ongoing Physical Danger ---
        if self._ongoing_danger.search(raw_text):
            matched = self._ongoing_danger.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.ONGOING_PHYSICAL_DANGER,
                subject=subject,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.NONE,
                means=means,
                immediacy=ImmediacyLevel.IN_PROGRESS,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 6: Acute Medical Emergency (Trauma / Unconscious) ---
        if self._acute_medical.search(raw_text):
            matched = self._acute_medical.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.ACUTE_MEDICAL_EMERGENCY,
                subject=subject,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.NONE,
                means=means,
                immediacy=ImmediacyLevel.IN_PROGRESS,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 7: Imminent Violence Towards Others ---
        if self._imminent_violence.search(raw_text):
            matched = self._imminent_violence.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.IMMINENT_VIOLENCE_OTHERS,
                subject=SubjectTarget.SELF,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.EXPLICIT_INTENT,
                means=MeansStatus.ACCESS_AVAILABLE,
                immediacy=ImmediacyLevel.IMMINENT,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 8: Non-Imminent Threat Towards Others ---
        if self._threat_violence.search(raw_text):
            matched = self._threat_violence.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.THREAT_OR_INTENT_HARM_OTHERS,
                subject=SubjectTarget.SELF,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.EXPLICIT_INTENT,
                means=means,
                immediacy=ImmediacyLevel.UNCLEAR,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 9: Anger / Venting ---
        if self._anger_venting.search(raw_text):
            matched = self._anger_venting.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.ANGER_OR_VENTING,
                subject=SubjectTarget.SELF,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.NONE,
                means=MeansStatus.NONE_MENTIONED,
                immediacy=ImmediacyLevel.NONE,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 10: Imminent Self-Harm ---
        if self._imminent_self_harm.search(raw_text) and temporal == TemporalStatus.CURRENT:
            matched = self._imminent_self_harm.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.IMMINENT_SELF_HARM,
                subject=subject,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.EXPLICIT_INTENT,
                means=MeansStatus.PREPARATION if means != MeansStatus.NONE_MENTIONED else MeansStatus.NONE_MENTIONED,
                immediacy=ImmediacyLevel.IMMINENT,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 11: Hypothetical or Historical Discussion of Suicide ---
        if temporal in (TemporalStatus.HISTORICAL, TemporalStatus.HYPOTHETICAL):
            if self._active_self_harm.search(raw_text) or self._passive_death_wish.search(raw_text):
                matched = self._active_self_harm.findall(raw_text) or self._passive_death_wish.findall(raw_text)
                matched_markers.extend(matched[:2])
                return SafetyAssessment(
                    category=SafetyCategory.GENERAL_OR_HYPOTHETICAL,
                    subject=subject,
                    temporal=temporal,
                    intent=IntentLevel.NONE,
                    means=means,
                    immediacy=ImmediacyLevel.NONE,
                    detected_expression_language=expr_lang,
                    matched_markers=tuple(matched_markers[:4]),
                )

        # --- Dimension 12: Fear of Acting / Intrusive Thoughts ---
        if self._fear_or_intrusive.search(raw_text):
            matched = self._fear_or_intrusive.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.SELF_HARM_THOUGHTS,
                subject=SubjectTarget.SELF,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.FEAR_OF_ACTING,
                means=means,
                immediacy=ImmediacyLevel.NONE,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 13: Third-Person Crisis (Non-imminent / disclosure) ---
        if subject in (SubjectTarget.THIRD_PERSON_KNOWN, SubjectTarget.THIRD_PERSON_GENERAL):
            if self._active_self_harm.search(raw_text) or self._passive_death_wish.search(raw_text):
                matched = self._active_self_harm.findall(raw_text) or self._passive_death_wish.findall(raw_text)
                matched_markers.extend(matched[:2])
                return SafetyAssessment(
                    category=SafetyCategory.THIRD_PERSON_CRISIS,
                    subject=subject,
                    temporal=temporal,
                    intent=IntentLevel.PASSIVE_WISH if self._passive_death_wish.search(raw_text) else IntentLevel.EXPLICIT_INTENT,
                    means=means,
                    immediacy=ImmediacyLevel.NONE if "safe" in raw_text.lower() or "with me" in raw_text.lower() else ImmediacyLevel.UNCLEAR,
                    detected_expression_language=expr_lang,
                    matched_markers=tuple(matched_markers[:4]),
                )

        # --- Dimension 14: Active Self-Harm Thoughts (First Person) ---
        if self._active_self_harm.search(raw_text):
            matched = self._active_self_harm.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.SELF_HARM_THOUGHTS,
                subject=SubjectTarget.SELF,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.EXPLICIT_INTENT,
                means=means,
                immediacy=ImmediacyLevel.NONE,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 15: Passive Death Wish ---
        if self._passive_death_wish.search(raw_text):
            matched = self._passive_death_wish.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.PASSIVE_DEATH_WISH,
                subject=SubjectTarget.SELF,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.PASSIVE_WISH,
                means=means,
                immediacy=ImmediacyLevel.NONE,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Dimension 16: Ordinary Distress ---
        if self._ordinary_distress.search(raw_text):
            matched = self._ordinary_distress.findall(raw_text)
            matched_markers.extend(matched[:2])
            return SafetyAssessment(
                category=SafetyCategory.ORDINARY_DISTRESS,
                subject=SubjectTarget.SELF,
                temporal=TemporalStatus.CURRENT,
                intent=IntentLevel.NONE,
                means=means,
                immediacy=ImmediacyLevel.NONE,
                detected_expression_language=expr_lang,
                matched_markers=tuple(matched_markers[:4]),
            )

        # --- Default: Safe Conversational ---
        return SafetyAssessment(
            category=SafetyCategory.SAFE_CONVERSATIONAL,
            subject=SubjectTarget.UNSPECIFIED,
            temporal=TemporalStatus.CURRENT,
            intent=IntentLevel.NONE,
            means=means,
            immediacy=ImmediacyLevel.NONE,
            detected_expression_language=expr_lang,
            matched_markers=tuple(matched_markers[:4]),
        )
