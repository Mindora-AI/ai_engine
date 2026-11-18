# MWIZERWA AI — Developer Handover and Engineering Guide

This document provides a comprehensive handover for future software engineers and researchers maintaining and extending MWIZERWA AI without requiring external agent assistance.

---

## 1. Executive Summary & Project Status

MWIZERWA AI is a fully functional, offline-capable, multilingual supportive conversational companion equipped with multimodal voice capabilities (ASR, LLM, TTS), two-tier deterministic safety guardrails, and a deterministic wellness exercise subsystem.

### Completed Milestones
- **Chunk 1 — Conversational Domain & Orchestration:** `ConversationEngine`, lightweight typed session state, and TurnRequest/TurnResult contracts.
- **Chunk 2 — Token-Aware Context Management & Response Contract:** Sliding-window context budgeting (`n_ctx=4096`), strict JSON schema parser, and zero unconstrained fallback.
- **Chunk 3 — Tri-Lingual Language Detection:** Per-turn language detection (Kinyarwanda, English, French) with code-switching and false-positive protection.
- **Chunk 4 — Two-Tier Safety Architecture:** Gate 1 pre-inference emergency overrides (suicide/crisis prevention) and Gate 2 post-inference assistant safety validation.
- **Chunk 5 — Exercise Recommendation & Lifecycle Subsystem:** 5 static exercise categories (Breathing, Grounding, Reflection, Compassion, Gratitude), deterministic step sequencing, 5-turn unsolicited cooldown, and explicit request overrides.
- **Chunk 6 — Behavioral Evaluation & Quality Assurance:** Comprehensive multi-turn behavioral evaluation suite in `tests/behavioral/` bringing test coverage to **298 passing tests**.

### Git State
- **Active Branch:** `chunk-4-safety-implementation`
- **Baseline Commit:** `b1efd56bb8c833166b224302c56ddb6769a89e66`
- **Main Branch:** Untouched and preserved.

---

## 2. Core Architectural Invariants

When extending the codebase, strictly uphold these non-negotiable architectural rules:

1. **Exclusive History Ownership:**  
   `ConversationManager.messages` is the **sole source of truth** for conversation history. `ConversationEngine` and other modules must never maintain secondary history lists or parallel message states.
2. **Gate 1 Must Precede Inference and Abort Active Exercises:**  
   The pre-inference safety check (`SafetyEvaluator.evaluate()`) must always run before any LLM generation or exercise progression. If Gate 1 triggers an emergency override, active exercises must immediately reset to `IDLE` and LLM inference must be completely bypassed.
3. **Gate 2 Must Validate Assistant Outputs Before Commit:**  
   The model's candidate `response_text` must be inspected by `AssistantSafetyValidator` before it is committed to history. If unsafe content is detected, the turn must be replaced with the approved safe fallback.
4. **Structured Response Contract Enforcement:**  
   Every non-empty response from `AfriqueQwen` must strictly validate against the JSON schema (`response_text`, `interaction_intent`, `requires_clarification`). Responses that violate the contract must raise `ResponseParsingError` without corrupting state.
5. **Deterministic Pre-Authored Exercises:**  
   Exercise instructions are static, verified assets authored in Kinyarwanda, English, and French in `src/exercises/registry.py`. The LLM does not generate exercise steps dynamically.

---

## 3. Important Files and Directory Guide

| File / Directory | Subsystem | Responsibility |
| :--- | :--- | :--- |
| `src/conversation/engine.py` | Orchestration | `ConversationEngine`: coordinates turn flow, safety gates, exercises, and metadata packaging. |
| `src/conversation/session.py` | Domain Models | Typed models: `TurnRequest`, `TurnResult`, `SessionState`, `ConversationMode`. |
| `src/llm/conversation.py` | LLM & Context | `ConversationManager`: context token budgeting, history trimming, ephemeral prompt formatting, inference. |
| `src/llm/prompts.py` | Prompt Engineering | Base system prompt, identity boundary, conversational principles, and response contract schema. |
| `src/llm/response_parser.py` | Parsing & Contract | Strict JSON extraction and validation for `ParsedResponse` and `InteractionIntent`. |
| `src/safety/evaluator.py` | Gate 1 Safety | Regex-based evaluation of user risk categories across 3 languages. |
| `src/safety/guardrail.py` | Gate 1 Policy | Maps safety assessments to application actions (`DETERMINISTIC_EMERGENCY_OVERRIDE`, etc.). |
| `src/safety/assistant_policy.py`| Gate 2 Safety | Validates assistant generation against medicalization, toxicity, and harm encouragement. |
| `src/exercises/models.py` | Exercise Domain | Immutable `ExerciseStep`, `Exercise`, and mutable `ExerciseSessionState`. |
| `src/exercises/registry.py` | Exercise Catalog | Static registry containing all 5 trilingual exercises. |
| `src/exercises/recommender.py` | Recommendation | Explicit request detector and contextual recommendation decision engine. |
| `src/exercises/lifecycle.py` | Lifecycle Engine | State machine for step sequencing, user response classification, and pause/stop controls. |
| `src/asr/asr_engine.py` | Speech Recognition | Hugging Face pipeline wrapper for Kinyarwanda Health ASR. |
| `src/tts/tts_engine.py` | Speech Synthesis | Neural TTS synthesis wrapper for `FlexKinyaTTS` (MB-iSTFT-VITS2). |
| `tests/behavioral/` | Behavioral QA | 28 multi-turn behavioral evaluation tests verifying intent fidelity, continuity, and safety. |

---

## 4. How to Extend MWIZERWA Safely

### 4.1 Adding a New Exercise
1. Open `src/exercises/registry.py`.
2. Define consecutive, 1-indexed `ExerciseStep` objects with verified instructions in `instruction_rw`, `instruction_en`, and `instruction_fr`.
3. Construct the `Exercise` instance with trilingual names, descriptions, and duration.
4. Register the exercise in `_EXERCISE_CATALOG` and update `CATEGORY_DEFAULT_EXERCISE` in `recommender.py` if adding a new default.
5. Add unit and behavioral tests in `tests/test_exercise_domain.py` and `tests/test_exercise_lifecycle.py`.

### 4.2 Adding a New Intent or Response Field
1. Update `src/llm/prompts.py` with the updated JSON contract schema.
2. Update `src/llm/response_parser.py`:
   - Add new enum values to `InteractionIntent` if applicable.
   - Update `ParsedResponse` dataclass.
   - Update `parse_response()` validation checks.
3. Run `python -m unittest tests/test_response_contract.py` to verify schema validation.

### 4.3 Adding New Safety Patterns
1. In `src/safety/evaluator.py`, locate the appropriate category regex (e.g. `_imminent_self_harm`, `_passive_death_wish`).
2. Add regex patterns using word boundaries (`\b... \b`) with appropriate dialectal and colloquial variations in Kinyarwanda, English, and French.
3. In `src/safety/guardrail.py`, ensure localized crisis response text covers any new languages or scenarios.
4. Run `python -m unittest discover tests` to verify zero regression across safety gates.

---

## 5. Development and Testing Workflow

### 5.1 Verification Commands
Before submitting any pull request or deploying changes, run:
```powershell
# 1. Full regression test suite (must report 298 passed, OK)
python -m unittest discover tests

# 2. Behavioral test suite specifically
python -m unittest discover tests/behavioral

# 3. Model assets check
python scripts/download_models.py --check
```

### 5.2 Interactive Smoke Testing
Test your changes in text mode:
```powershell
python run_assistant.py --mode text
```
Test with speech synthesis:
```powershell
python run_assistant.py --mode text-voice --speaker 0
```

---

## 6. Known Limitations and Future Roadmap

### Current Limitations
1. **TTS Language Restriction:** `KinyaFlex TTS` is trained exclusively for Kinyarwanda. English and French responses are rendered as text; audio generation is safely skipped.
2. **CPU Latency:** On quad-core CPUs, local LLM generation takes ~2-5 seconds per turn. GPU acceleration (CUDA) is recommended for production deployment.
3. **ASR Dialectal Variation:** `Kinyarwanda-Health-ASR` performs best on standard central Rwandan speech; heavy regional accents may show reduced accuracy.

### Recommended Next Steps
1. **Multilingual Speech Synthesis:** Train or integrate English and French neural TTS voices (e.g. Piper or MMS-TTS) to provide full trilingual voice output.
2. **Streaming Audio Pipeline:** Implement streaming token generation and streaming TTS synthesis to reduce perceived latency on voice turns.
3. **Web / Mobile Interface:** Build a lightweight FastAPI or WebRTC interface around `ConversationEngine` for browser and mobile deployment.
