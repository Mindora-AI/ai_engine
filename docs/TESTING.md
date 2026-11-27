# MWIZERWA AI — Testing and Quality Assurance Guide

MWIZERWA AI includes an automated test suite (run `pytest tests/`) across unit, integration, safety, API contracts, exercise lifecycle, and behavioral evaluation suites.

All test suites in `tests/` are designed for fast, deterministic, offline execution using mocked inference harnesses. No model weights or external network access are required to run the automated test suite.

---

## 1. Test Suite Organization

```text
tests/
├── behavioral/                                # Multi-Turn Behavioral Evaluation Suite (28 tests)
│   ├── test_emotional_intent_fidelity.py      # Intent fidelity across LISTENING, REFLECTION, ADVICE (8 tests)
│   ├── test_conversational_continuity.py      # Context tracking, topic shifts, sliding window (5 tests)
│   ├── test_multilingual_behavioral.py        # Code-switching, mid-session language shifts (5 tests)
│   ├── test_safety_boundary_behavior.py       # Gate 1 & Gate 2, therapist boundary, crisis abortion (5 tests)
│   └── test_exercise_behavioral_regression.py # Contextual offer, progression, cooldowns, stops (5 tests)
├── test_assistant.py                          # Full multimodal pipeline integration & mock audio (7 tests)
├── test_assistant_policy.py                   # Gate 2 assistant safety policy validation (16 tests)
├── test_conversation_engine.py                # ConversationEngine orchestration & session state (8 tests)
├── test_exercise_domain.py                    # Exercise & ExerciseStep invariants, bounds, types (58 tests)
├── test_exercise_lifecycle.py                 # Exercise lifecycle transitions & control vocabulary (35 tests)
├── test_exercise_pipeline_integration.py      # Pipeline integration: offers, acceptance, cooldowns (15 tests)
├── test_exercise_recommender.py               # Explicit request detection & contextual rules (44 tests)
├── test_language_detection.py                 # Tri-lingual language detector & false positive guards (33 tests)
├── test_response_contract.py                  # Structured response JSON schema & contract validation (22 tests)
├── test_safety_domain.py                      # Safety assessment dataclasses & severity models (11 tests)
├── test_safety_evaluator.py                   # Gate 1 crisis pattern recognition (10 tests)
├── test_safety_gates.py                       # Gate 1 & Gate 2 pipeline interception (6 tests)
├── test_safety_guardrail.py                   # Guardrail policy & localized response selection (10 tests)
└── test_token_context_management.py           # Sliding-window context budgeting & trimming (18 tests)
```

**Total Automated Test Count:** **298 Tests**

---

## 2. Running Tests

### 2.1 Running the Complete Test Suite
To discover and run the full suite:
```powershell
python -m unittest discover tests
```
*Expected Output:*
```text
pytest tests/ -q  (typically ~400+ tests in under a minute with mocks)
OK
```

*(Note: During test execution, mock error-injection tests will print simulated log lines like `Error during LLM inference: Inference boom!` and `[ASR Warning]`. These are normal, verified negative test executions).*

### 2.2 Running Behavioral Tests Only
To run the 28 multi-turn behavioral evaluation tests:
```powershell
python -m unittest discover tests/behavioral
```
*Expected Output:*
```text
Ran 28 tests in ~14s
OK
```

### 2.3 Running Specific Test Modules
To run individual test files:
- **Conversation Engine:**
  ```powershell
  python -m unittest tests/test_conversation_engine.py
  ```
- **Exercise Lifecycle:**
  ```powershell
  python -m unittest tests/test_exercise_lifecycle.py
  ```
- **Safety Gates:**
  ```powershell
  python -m unittest tests/test_safety_gates.py
  ```
- **Language Detection:**
  ```powershell
  python -m unittest tests/test_language_detection.py
  ```
- **Structured Response Contract:**
  ```powershell
  python -m unittest tests/test_response_contract.py
  ```

### 2.4 Running a Single Test Case or Method
```powershell
python -m unittest tests.behavioral.test_emotional_intent_fidelity.TestEmotionalIntentFidelity.test_01_english_listening_intent_suppresses_unsolicited_exercise
```

---

## 3. Mocked vs. Live Model Tests

| Test Suite | Execution Target | Model Weights Required? | Network Required? | Typical Duration |
| :--- | :--- | :---: | :---: | :---: |
| `tests/*.py` | Unit / API / integration mocks | No | No | ~1 minute |
| `tests/behavioral/*.py` | Multi-turn behavioral mocks | No | No | included above |
| `scripts/benchmark_whisper_asr.py` | Live Whisper (EN/FR) | Yes (`faster-whisper`) | First run may download HF weights | variable |
| `scripts/benchmark_coreidency_llm_whisper.py` | Live LLM + Whisper (sequential) | Yes (GGUF + Whisper) | No | variable |
| `scripts/validate_token_context_real_model.py` | Live GGUF Model | Yes (`AfriqueQwen`) | No | ~2-5 minutes |
| `scripts/benchmark_cpu_threads.py` | Live GGUF Model | Yes (`AfriqueQwen`) | No | ~3-10 minutes |

### Running Live Model Benchmarks (Optional)
To test token budgeting against the real 3 GB `AfriqueQwen` GGUF:
```powershell
python scripts/validate_token_context_real_model.py
```
To benchmark CPU thread performance (evaluating 2, 4, 8 threads):
```powershell
python scripts/benchmark_cpu_threads.py
```

---

## 4. Interpreting Test Results and Troubleshooting Failures

- **`429 passed`** (typical CI run with `pytest tests/`): Regression boundaries, API contracts, safety gates, and behavioral tests are preserved.
- **`AssertionError: <ExerciseStatus.IN_PROGRESS> != <ExerciseStatus.COMPLETED>`**: Check step count for the exercise. `breathing_gentle_rhythm` has 4 steps (indices 0..3), requiring 4 continuation turns to complete.
- **`ResponseParsingError: Missing required field 'interaction_intent'`**: The model output parser strictly rejects responses lacking required contract fields (`response_text`, `interaction_intent`, `requires_clarification`).
- **`OversizedMessageError`**: The input user turn exceeds the token budget (`n_ctx - max_tokens - safety_margin`).
