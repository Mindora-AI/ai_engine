"""
MWIZERWA AI — Multilingual Language Detector
Lightweight, offline, CPU-friendly language detection for Kinyarwanda (rw), English (en), and French (fr).
Supports per-turn language tracking, code-switching/mixed-language detection, and uncertainty handling.

Linguistic Design Notes:
- Confidence is a bounded heuristic reliability score (0.0 to 1.0) based on accumulated
  lexical and grammatical evidence diversity, NOT a statistically calibrated probability.
- Diminishing returns are enforced on repeated identical tokens to prevent artificial confidence inflation.
- Morphological heuristics require valid word boundaries, Kinyarwanda phonotactics (open vowel endings),
  and verified verbal prefix patterns, preventing false matches on English/French vocabulary.
- French diacritics are evaluated in token context; common borrowed words in English (e.g. café, résumé)
  do not trigger false French or mixed-language classifications.
- Mid-sentence title-cased words matching ambiguous function words (e.g. 'Will', 'May') are treated as
  proper nouns/names rather than foreign language markers.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Tuple


class DetectedLanguage(str, Enum):
    """Supported language codes."""
    KINYARWANDA = "rw"
    ENGLISH = "en"
    FRENCH = "fr"


@dataclass
class LanguageDetectionResult:
    """
    Structured language detection result.
    - language: 'rw', 'en', 'fr', or None if uncertain / unsupported.
    - confidence: normalized heuristic reliability value between 0.0 and 1.0.
    - is_reliable: whether detection meets the reliability threshold without ambiguity.
    - is_mixed: whether meaningful evidence of multiple supported languages was detected.
    - details: supplementary scoring and token breakdown for debugging and inspection.
    """
    language: Optional[str]
    confidence: float
    is_reliable: bool
    is_mixed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def capitalize(self) -> str:
        """Helper for display/legacy string formatting."""
        code_to_name = {"en": "English", "fr": "French", "rw": "Kinyarwanda"}
        return code_to_name.get(self.language, "Unknown")

    def __str__(self) -> str:
        code_to_name = {"en": "english", "fr": "french", "rw": "kinyarwanda"}
        return code_to_name.get(self.language, "unknown")

    def __eq__(self, other):
        """
        Backwards compatibility: supports comparison against legacy string names
        ('english', 'french', 'kinyarwanda', 'unknown') or codes ('en', 'fr', 'rw').
        """
        if isinstance(other, str):
            code_to_name = {
                "en": "english",
                "fr": "french",
                "rw": "kinyarwanda",
                None: "unknown",
            }
            return (
                self.language == other
                or code_to_name.get(self.language) == other
                or (self.language is None and other in ["unknown", "None"])
            )
        if isinstance(other, LanguageDetectionResult):
            return (
                self.language == other.language
                and abs(self.confidence - other.confidence) < 1e-4
                and self.is_reliable == other.is_reliable
                and self.is_mixed == other.is_mixed
            )
        return False


# -------------------------------------------------------------------------
# Lexical and Morphological Dictionaries
# -------------------------------------------------------------------------

# Kinyarwanda Grammatical & Conversational Vocabulary
RW_STRONG_MARKERS: Set[str] = {
    # Pronouns & Demonstratives
    "ubu", "iki", "ibi", "icyo", "ibyo", "uyu", "aba", "uwo", "bariya",
    "hano", "aho", "hariya", "njye", "wowe", "twe", "mwe", "bo",
    # Possessives
    "wanjye", "wawe", "wacu", "wanyu", "wabo",
    "yanjye", "yawe", "yacu", "yanyu", "yabo",
    "cyanjye", "cyawe", "cyacu", "cyanyu", "cyabo",
    "byanjye", "byawe", "byacu", "byanyu", "byabo",
    "bwanjye", "bwawe", "bwacu", "bwanyu", "bwabo",
    "kwanjye", "kwawe", "kwacu", "kwanyu", "kwabo",
    # Conjunctions & Prepositions
    "ariko", "kandi", "cyangwa", "kuko", "nuko", "nka", "nko", "muri", "kuri",
    "ntabwo", "ntago", "ntacyo", "ntibisanzwe", "ntibishoboka",
    # Common Stems, Inflected Verbs & Copulas
    "ndumva", "urakoze", "ndabizi", "sinzi", "nshaka", "urashaka",
    "ndashobora", "turashobora", "ndabyumva", "ndagukunda", "tubane",
    "twaganira", "ndimo", "gukora", "kugenda", "kuvuga", "kumva",
    "gutega", "kureba", "gutekereza", "kwiga", "kwibaza", "kugira",
    "kumenya", "guhitamo", "bivuga", "bifite", "ndagerageza", "ndaniwe",
    "naniwe", "twibuke", "mfite", "ufite", "afite", "dufite", "bafite",
    "ndabona", "urabona", "arabona", "turabona", "barabona",
    "ndagenda", "uragenda", "aragenda", "turagenda", "baragenda",
    "ndakora", "urakora", "arakora", "turakora", "barakora",
    "ndavuga", "uravuga", "aravuga", "turavuga", "baravuga",
    "ndakomeye", "urakomeye", "arakomeye", "turakomeye", "komera", "ihangane",
    # Conversational, Emotional & Wellness Terms
    "ubuzima", "umunsi", "ijambo", "umuntu", "abantu", "ikibazo",
    "ibibazo", "umusanzu", "ubwoba", "umunezero", "agahinda", "umunaniro",
    "kwihangana", "umutima", "imitima", "umuryango", "inshuti", "amahoro",
    "akazi", "ishuri", "amafaranga", "uburwayi", "ubuvuzi", "gufasha",
    "kwishima", "gushavura", "kuremererwa", "kunanirwa", "birakomeye",
    "biroroshye", "gikomeye", "bwinshi", "cyinshi", "cyiza", "byiza",
    "neza", "nabi", "cyane", "rwose", "amatsiko", "ngaho",
    # Greetings & Common Expressions
    "muraho", "mwiriwe", "mwaramutse", "uraho", "amakuru", "yego",
}

RW_FUNCTION_WORDS: Set[str] = {
    "ni", "si", "ko", "mu", "ku", "na", "nta", "ngo", "bwa", "bya",
    "wa", "ba", "za", "ya", "rya", "kwa", "gusa", "ejo", "hose",
    "byose", "bose", "cyose", "nde", "he", "ryari", "kuki", "gute",
    "nanjye", "nawe", "natwe", "namwe", "nabo", "nuko",
}

# French Grammatical & Conversational Vocabulary
FR_STRONG_MARKERS: Set[str] = {
    # Pronouns & Articles
    "cette", "cette-ci", "celles", "celui", "ceux", "notre", "votre", "leurs",
    # Conjunctions & Prepositions
    "avec", "pour", "dans", "sans", "chez", "sous", "vers", "pendant",
    "parce", "lorsque", "puisque", "mais", "donc",
    # Verbs
    "suis", "sommes", "etes", "sont", "avons", "avez", "etais", "etait",
    "faire", "dire", "sais", "sait", "veux", "veut", "peux", "peut",
    "pense", "sens", "comprends", "fatigue", "fatiguee", "partir",
    "aide", "aider", "cherche", "chercher", "surmonter", "situation", "difficile",
    # Conversational / Adverbs
    "bonjour", "bonsoir", "salut", "merci", "toujours", "jamais", "aujourd'hui",
    "maintenant", "pourquoi", "comment", "beaucoup", "vraiment", "aussi",
    "apaisé", "apaise", "espère", "espere",
}

FR_FUNCTION_WORDS: Set[str] = {
    "le", "la", "les", "un", "une", "des", "du", "de", "ce", "cet", "ces",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "leur", "y", "en", "est", "ai", "as", "a",
    "ont", "va", "sur", "par", "ou", "et", "ni", "car", "si", "qui",
    "que", "quoi", "oui", "non", "bien", "mal", "tres", "trop", "très",
}

# English Grammatical & Conversational Vocabulary
EN_STRONG_MARKERS: Set[str] = {
    # Pronouns & Determiners
    "these", "those", "their", "theirs", "myself", "yourself", "themselves",
    # Conjunctions & Prepositions
    "with", "about", "into", "through", "after", "before", "between",
    "under", "without", "because", "although", "while", "since", "unless",
    # Verbs & Participles
    "have", "having", "been", "being", "does", "doing", "could", "would",
    "should", "might", "feel", "feeling", "felt", "think", "thinking",
    "thought", "know", "knowing", "want", "wanted", "need", "needed",
    "tired", "exhausted", "helping", "please", "manage", "managed",
    "talk", "talking", "work", "working", "send", "sent", "email",
    # Domain / Common Nouns
    "software", "network", "policy", "accuracy", "security", "bicycle",
    "subway", "country", "story", "theory", "economy", "coffee", "heart",
    "director", "overwhelmed",
    # Conversational / Adverbs
    "hello", "thanks", "thank", "today", "tonight", "tomorrow", "yesterday",
    "always", "never", "sometimes", "really", "quite", "sorry", "lot",
}

EN_FUNCTION_WORDS: Set[str] = {
    "the", "this", "that", "my", "your", "his", "her", "its", "our",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them",
    "am", "is", "are", "was", "were", "be", "has", "had", "do", "did",
    "can", "will", "shall", "may", "must", "for", "from", "and", "but",
    "or", "nor", "so", "as", "at", "by", "in", "on", "to", "up", "out",
    "if", "what", "which", "who", "whom", "why", "how", "yes", "no",
    "not", "just", "very", "good", "bad", "fine", "of", "a", "an", "all",
}

# Common French words borrowed into English that frequently retain accents in English writing
BORROWED_ACCENTED_WORDS: Set[str] = {
    "cafe", "café", "resume", "résumé", "resumé", "deja", "déjà",
    "cliche", "cliché", "fiance", "fiancé", "fiancee", "fiancée",
    "facade", "façade", "naive", "naïve", "elite", "élite",
}

# Common Proper Nouns / Location Names to avoid as sole definitive evidence
COMMON_PROPER_NOUNS: Set[str] = {
    "kigali", "rwanda", "africa", "afrique", "france", "paris", "london",
    "america", "usa", "mwizerwa", "qwen", "afriqueqwen", "google", "meta",
    "john", "eric", "marie", "paul", "jean", "david", "will", "peter", "barack",
}


# -------------------------------------------------------------------------
# Feature Extraction Helpers
# -------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    """Removes combining diacritical marks."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _extract_tokens_with_casing(text: str) -> List[Tuple[str, str, bool]]:
    """
    Extracts tokens with casing context:
    Returns list of (token_lower, raw_token, is_mid_sentence_title).
    Preserves apostrophe contractions like n'ibyo, c'est, don't, j'ai.
    """
    normalized = re.sub(r"[’‘'`]", "'", text)
    # Find word tokens with optional internal apostrophe
    matches = list(re.finditer(r"\b[a-zA-Zà-ÿÀ-Ÿ]+(?:'[a-zA-Zà-ÿÀ-Ÿ]+)?\b", normalized))
    results = []
    for idx, match in enumerate(matches):
        raw = match.group(0)
        token_lower = raw.lower()
        # Mid-sentence title-cased words (e.g. names like 'Will', 'May')
        is_mid_title = (idx > 0) and raw.istitle()
        results.append((token_lower, raw, is_mid_title))
    return results


def _has_kinyarwanda_apostrophe(token: str) -> bool:
    """
    Checks for Kinyarwanda vowel-elision apostrophe patterns.
    E.g. n'ibyo, y'ubuzima, rw'ikinyarwanda, kw'ishuri, by'ukuri, bw'ejo, tw'abantu.
    """
    return bool(re.match(r"^(n|y|rw|by|kw|bw|tw|ny|ry)'[a-z]+$", token))


def _has_kinyarwanda_morphology(token: str) -> bool:
    """
    Linguistically defensible check for genuine Kinyarwanda verbal and nominal morphology.
    Requirements:
    1. Must contain only letters in the standard Kinyarwanda alphabet (no l, q, x).
    2. Must end in an open vowel (a, e, i, o, u) — native Kinyarwanda words never end in consonants.
    3. Minimum length of 5 letters.
    4. Must exhibit recognized Kinyarwanda verbal prefix patterns followed by valid consonant onset:
       - Affirmative prefixes: nda-, tura-, mura-, bara-
       - Negative prefixes: sin-, ntidu-, ntiba-
       - Infinitives: ku-, gu-, kwi-, gute-
    """
    # 1. Native Kinyarwanda words do not contain l, q, or x
    if re.search(r"[lqx]", token):
        return False

    # 2. Native Kinyarwanda words strictly end in a vowel
    if not re.search(r"[aeiou]$", token):
        return False

    # 3. Minimum length for an inflected verbal form
    if len(token) < 5:
        return False

    # Exclude common borrowed proper nouns or foreign loanwords that resemble prefixes
    if token in {"banana", "panama", "havana", "alabama", "nevada", "barack"}:
        return False

    # 4. Genuine verbal prefix structures followed by a valid consonant onset
    # E.g. ndagenda, turakora, barakora
    if re.match(r"^(nda|tura|mura|bara)[bcdfghjkmnprstvwz][a-z]+[aeiou]$", token):
        return True

    # Negative verbs: sinshaka, ntidukora, ntibakora
    if re.match(r"^(sin|ntidu|ntiba)[bcdfghjkmnprstvwz][a-z]+[aeiou]$", token):
        return True

    # Infinitives: gukora, kugenda, kureba, gutekereza
    if re.match(r"^(ku|gu|kwi|gute)[bcdfghjkmnprstvwz][a-z]+[aeiou]$", token):
        return True

    return False


# -------------------------------------------------------------------------
# Core Language Detector
# -------------------------------------------------------------------------

class LanguageDetector:
    """
    Lightweight rule- and feature-based language detector for:
    - Kinyarwanda ('rw')
    - English ('en')
    - French ('fr')
    """

    def __init__(self, reliability_threshold: float = 0.55):
        self.reliability_threshold = reliability_threshold

    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detects the primary language of incoming text with evidence-diversity scoring,
        diminishing returns for repeated tokens, and contextual mixed-language evaluation.
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                language=None,
                confidence=0.0,
                is_reliable=False,
                is_mixed=False,
                details={"reason": "empty_input"},
            )

        cleaned_text = text.strip()
        tokens_info = _extract_tokens_with_casing(cleaned_text)

        if not tokens_info:
            return LanguageDetectionResult(
                language=None,
                confidence=0.0,
                is_reliable=False,
                is_mixed=False,
                details={"reason": "no_alphabetic_tokens"},
            )

        tokens = [t[0] for t in tokens_info]

        # Extremely short or purely non-linguistic inputs (e.g. "?", "ok", "k", "123")
        total_letters = sum(len(t) for t in tokens)
        if len(tokens) == 1 and total_letters <= 2 and tokens[0] not in {"rw", "en", "fr"}:
            return LanguageDetectionResult(
                language=None,
                confidence=0.2,
                is_reliable=False,
                is_mixed=False,
                details={"reason": "too_short_ambiguous", "tokens": tokens},
            )

        # Track token occurrences and unique tokens per language
        token_counts: Dict[str, Dict[str, int]] = {"rw": {}, "en": {}, "fr": {}}
        scores: Dict[str, float] = {"rw": 0.0, "en": 0.0, "fr": 0.0}
        has_french_grammatical_marker = False

        # First pass: check for French grammatical context
        for token, raw, is_mid_title in tokens_info:
            stripped = _strip_accents(token)
            if stripped in FR_FUNCTION_WORDS or stripped in FR_STRONG_MARKERS:
                has_french_grammatical_marker = True
                break

        # Helper to add language score with diminishing returns for repeated markers
        def add_language_evidence(lang: str, token_key: str, base_weight: float):
            count = token_counts[lang].get(token_key, 0)
            token_counts[lang][token_key] = count + 1
            if count == 0:
                scores[lang] += base_weight
            elif count == 1:
                # Diminishing return on duplicate token
                scores[lang] += base_weight * 0.25
            # 3rd and subsequent occurrences yield zero additional evidence

        # 1. Token-level analysis
        for token, raw, is_mid_title in tokens_info:
            # Skip isolated proper nouns/names
            if token in COMMON_PROPER_NOUNS:
                continue

            # If a mid-sentence title-cased word matches an ambiguous function word (e.g. 'Will', 'May'),
            # treat it as a proper noun and do not award English points
            if is_mid_title and token in {"will", "may", "in", "on", "to", "me", "no", "so", "or"}:
                continue

            stripped_token = _strip_accents(token)
            token_has_accents = bool(re.search(r"[éèêàâçîïôùû]", token, re.IGNORECASE))

            # --- English Contractions ---
            if re.match(r"^(i'm|don't|can't|won't|it's|that's|i've|you're|we're|they're|doesn't|didn't)$", token):
                add_language_evidence("en", token, 2.5)
                continue

            # --- French Contractions ---
            if re.match(r"^(c'est|j'ai|j'espere|j'aime|d'un|d'une|qu'un|qu'il|qu'elle|n'est|n'a|s'il|s'est)$", stripped_token):
                add_language_evidence("fr", stripped_token, 2.5)
                continue

            # --- Kinyarwanda Apostrophe Elisions ---
            if _has_kinyarwanda_apostrophe(token):
                add_language_evidence("rw", token, 2.5)
                continue

            # --- Kinyarwanda Characteristic Greetings ---
            if token in {"bite", "bitese"}:
                add_language_evidence("rw", "bite", 2.0)
                continue

            # --- Handle French Diacritics in Token Context ---
            if token_has_accents:
                # Common loanword in English (e.g. café, résumé, déjà)
                if token in BORROWED_ACCENTED_WORDS or stripped_token in BORROWED_ACCENTED_WORDS:
                    # In an English or Kinyarwanda context, do not credit French points
                    pass
                elif stripped_token in FR_STRONG_MARKERS:
                    add_language_evidence("fr", stripped_token, 2.5)
                    continue
                elif stripped_token in FR_FUNCTION_WORDS:
                    add_language_evidence("fr", stripped_token, 1.5)
                    continue
                elif has_french_grammatical_marker:
                    # Genuine French word in French context
                    add_language_evidence("fr", stripped_token, 1.5)
                    continue

            # --- Lexical Dictionary Matching ---
            token_matched = False

            # Kinyarwanda lexical check
            if token in RW_STRONG_MARKERS:
                add_language_evidence("rw", token, 2.0)
                token_matched = True
            elif token in RW_FUNCTION_WORDS:
                add_language_evidence("rw", token, 1.0)
                token_matched = True

            # French lexical check
            if stripped_token in FR_STRONG_MARKERS:
                add_language_evidence("fr", stripped_token, 2.0)
                token_matched = True
            elif stripped_token in FR_FUNCTION_WORDS:
                add_language_evidence("fr", stripped_token, 1.0)
                token_matched = True

            # English lexical check
            if token in EN_STRONG_MARKERS:
                add_language_evidence("en", token, 2.0)
                token_matched = True
            elif token in EN_FUNCTION_WORDS:
                add_language_evidence("en", token, 1.0)
                token_matched = True

            # If not matched by dictionary, inspect genuine Kinyarwanda verbal morphology
            if not token_matched and len(token) >= 5:
                if _has_kinyarwanda_morphology(token):
                    # Weak supporting evidence for inflected unfamiliar verb
                    add_language_evidence("rw", token, 1.0)

        # 2. Evidence Diversity & Ranking
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_lang, top_score = sorted_scores[0]
        second_lang, second_score = sorted_scores[1]
        total_evidence = sum(scores.values())

        top_unique_count = len(token_counts[top_lang])
        second_unique_count = len(token_counts[second_lang])

        # 3. Mixed-Language Evaluation
        # A secondary language must represent genuine, non-borrowed evidence:
        # either a strong marker (score >= 1.8) or at least 2 distinct words
        is_mixed = False
        if second_score >= 1.8 or (second_unique_count >= 2 and second_score >= 1.5):
            is_mixed = True

        # 4. Confidence & Reliability Calculation
        if total_evidence == 0:
            return LanguageDetectionResult(
                language=None,
                confidence=0.1,
                is_reliable=False,
                is_mixed=False,
                details={"reason": "no_language_evidence_found", "tokens": tokens},
            )

        # Baseline ratio of evidence
        evidence_ratio = top_score / (total_evidence + 0.5)

        # Distinctive evidence diversity scaling
        if top_unique_count == 1:
            first_marker = next(iter(token_counts[top_lang].keys()))
            is_strong_greeting = first_marker in {
                "muraho", "bonjour", "hello", "urakoze", "merci", "thanks", "bite"
            }
            if is_strong_greeting:
                # Distinctive single-word greeting
                raw_confidence = min(0.75, evidence_ratio * 0.92)
            else:
                # Repeated or ambiguous single function word (e.g. "the the the")
                raw_confidence = min(0.40, evidence_ratio * 0.45)
        elif top_unique_count == 2:
            raw_confidence = min(0.85, evidence_ratio * 0.90)
        else:
            # 3 or more distinct linguistic markers
            raw_confidence = min(0.98, evidence_ratio * 0.96 + 0.02)

        normalized_confidence = round(max(0.0, min(1.0, raw_confidence)), 3)

        # Balanced multilingual input without a dominant language
        # When both languages have substantial evidence and close score margins
        if is_mixed and second_score > 0 and (top_score < 1.75 * second_score):
            return LanguageDetectionResult(
                language=None,
                confidence=round(normalized_confidence * 0.65, 3),
                is_reliable=False,
                is_mixed=True,
                details={
                    "scores": scores,
                    "reason": "balanced_mixed_without_dominant_language",
                    "top_languages": [top_lang, second_lang],
                },
            )

        # Reliability threshold
        is_reliable = (normalized_confidence >= self.reliability_threshold) and (top_score > second_score)

        # If confidence is below threshold, language is None
        final_lang = top_lang if is_reliable else None

        return LanguageDetectionResult(
            language=final_lang,
            confidence=normalized_confidence,
            is_reliable=is_reliable,
            is_mixed=is_mixed,
            details={
                "scores": scores,
                "token_count": len(tokens),
                "unique_matched_tokens": {k: list(v.keys()) for k, v in token_counts.items()},
                "top_score": top_score,
                "second_score": second_score,
            },
        )


# Default module-level instance
_DEFAULT_DETECTOR = LanguageDetector()


def detect_language(text: str, threshold: float = 0.55) -> LanguageDetectionResult:
    """
    Detects language using the default or configured LanguageDetector instance.
    Returns a LanguageDetectionResult.
    """
    if abs(threshold - 0.55) > 1e-4:
        return LanguageDetector(reliability_threshold=threshold).detect(text)
    return _DEFAULT_DETECTOR.detect(text)
