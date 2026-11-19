# MWIZERWA AI — Multilingual Supportive Conversational Companion

**Repository:** [Mindora-AI/ai_engine](https://github.com/Mindora-AI/ai_engine)

MWIZERWA AI is an offline-capable, multimodal, and multilingual conversational companion designed to provide attentive emotional presence, supportive listening, guided reflection, and deterministic wellness exercises across Kinyarwanda, English, and French.

> [!IMPORTANT]
> **Critical Identity Boundary:**  
> MWIZERWA AI is a conversational companion. It is **NOT** a therapist, psychologist, psychiatrist, doctor, or medical provider. It does not provide clinical diagnoses, psychiatric assessments, medical advice, or crisis intervention services. Users in acute crisis are immediately directed to emergency support services via deterministic safety guardrails.

---

## 1. Core Purpose and Capabilities

MWIZERWA AI provides:
- **Supportive Conversational Presence:** Attentive listening without premature judgment, unsolicited problem-solving, or manufactured optimism.
- **Intent Recognition via Structured Contract:** Strict structured response generation enforcing explicit conversational intentions (`LISTENING`, `REFLECTION`, `PRACTICAL_ADVICE`, `UNCLEAR`) and clarification tracking.
- **Multilingual Support:** Native interaction in Kinyarwanda, English, and French, with dynamic per-turn language detection and code-switching handling.
- **Two-Tier Safety Architecture:**
  - **Gate 1 (Pre-Inference):** Deterministic emergency detection (suicide, self-harm, severe crisis) that immediately bypasses LLM inference and returns localized crisis resources.
  - **Gate 2 (Post-Inference):** Independent safety validation of model outputs intercepting toxic, dangerous, or medicalized responses before user presentation.
- **Deterministic Wellness Exercises:** Offline, structured step-by-step guidance for Breathing, 5-4-3-2-1 Sensory Grounding, Thought Reflection, Self-Compassion, and Gratitude Anchoring.
- **Full Voice Pipeline:** End-to-end speech recognition (Kinyarwanda Health ASR), local LLM inference (AfriqueQwen3.5-4B GGUF), and **trilingual speech synthesis** via KinyaFlex (Kinyarwanda) and Piper (English / French).

---

## 2. Supported Languages

| Language | Text Interaction | Speech Recognition (ASR) | Speech Synthesis (TTS) | Wellness Exercises | Safety Overrides |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kinyarwanda (`rw`)** | Yes | Yes (Native) | Yes — KinyaFlex (3 speakers) | Yes (All 5 Categories) | Yes (112, Police, Support) |
| **English (`en`)** | Yes | English Audio | Yes — Piper (`en_US-ljspeech-medium`) | Yes (All 5 Categories) | Yes (911/112/Hotlines) |
| **French (`fr`)** | Yes | French Audio | Yes — Piper (`fr_FR-siwis-medium`) | Yes (All 5 Categories) | Yes (112/SAMU 912) |

**TTS routing:** The assistant detects the language of the **final response text** and routes synthesis automatically — KinyaFlex for `rw`, Piper for `en` and `fr`. The same router is used for text-to-voice and voice-to-voice modes.

---

## 3. Technology Stack

- **Large Language Model (LLM):** `AfriqueQwen3.5-4B-Instruct-v1` quantized at `Q4_K_M` (2.86 GB GGUF) running locally via `llama-cpp-python` on CPU or GPU.
- **Automatic Speech Recognition (ASR):** `ElvisTata2024/Kinyarwanda-Health-ASR` via HuggingFace `transformers` pipeline.
- **Text-to-Speech (TTS):** Dual-engine routing — `KinyaFlex TTS` (`FlexKinyaTTS`, Kinyarwanda, 3 speakers) and [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) (English / French ONNX voices).
- **Context Management:** Token-aware sliding window with deterministic system prompt preservation (`n_ctx=4096`, `max_tokens=512`).
- **Safety Architecture:** Deterministic regex-based evaluator + Guardrail policy + Assistant output validator.
- **Exercise Engine:** Pre-authored trilingual static content catalog with deterministic lifecycle state machine.

---

## 4. Repository Structure

```text
ai_engine/
├── ac-ai-models/              # Submodule: DeepKIN-AgAI TTS architecture
├── docs/                      # Comprehensive technical documentation
│   ├── ARCHITECTURE.md        # Complete architectural specification & pipeline flow
│   ├── DEVELOPER_HANDOVER.md  # Handover guide for future engineering teams
│   ├── INSTALLATION.md        # Step-by-step setup for Windows & Linux
│   ├── MODELS_AND_AUDIO.md    # Model weights, TTS details, and audio catalog
│   ├── RUNNING.md             # Execution guide for all operating modes
│   ├── TESTING.md             # Complete testing guide & test inventory
│   └── TROUBLESHOOTING.md     # Common issues and verified resolutions
├── models/                    # Model weights directory (gitignored)
│   ├── afriqueqwen/           # AfriqueQwen GGUF model file
│   ├── kinya-flex-tts/        # KinyaFlex TTS PyTorch weights & metadata
│   └── piper/                 # Piper ONNX voices (en_US-ljspeech-medium, fr_FR-siwis-medium)
├── outputs/                   # Generated evaluation outputs and manifests
│   ├── prosody_reference_set/ # Reference dataset manifests and inspection reports
│   └── tts_baseline_evaluation/ # 54-sample baseline TTS benchmark manifest & audio
├── scripts/                   # Operational and benchmarking scripts
│   ├── benchmark_cpu_threads.py
│   ├── download_models.py     # Verified model downloader with SHA-256 checks
│   ├── run_tts_baseline_evaluation.py
│   ├── test_asr.py
│   └── test_kinya_flex.py
├── src/                       # Primary application source code
│   ├── asr/                   # Speech recognition engine wrapper
│   ├── conversation/          # ConversationEngine orchestration & session models
│   ├── exercises/             # Exercise catalog, models, recommender, and lifecycle
│   ├── llm/                   # ConversationManager, prompts, cleaner, parser
│   ├── safety/                # Gate 1 evaluator, Gate 2 assistant policy, guardrails
│   ├── tts/                   # TTS router, KinyaFlex engine, Piper engine
│   └── utils/                 # Tri-lingual language detector and helpers
├── tests/                     # 298 Automated regression and behavioral tests
│   ├── behavioral/            # Multi-turn behavioral fidelity test suite (28 tests)
│   └── test_*.py              # Core unit, safety, contract, and exercise tests (270 tests)
├── launch_menu.py             # Interactive console launcher for all 4 modes
├── run_assistant.py           # Primary application CLI entry point
├── setup_env.bat              # Automated Windows setup script
├── setup_env.sh               # Automated Linux/macOS setup script
└── requirements.txt           # Verified Python package dependencies
```

---

## 5. Quick Start

### 5.1 Installation

1. **Clone the repository (Windows PowerShell):**
   ```powershell
   git clone https://github.com/Mindora-AI/ai_engine.git
   cd ai_engine
   ```

2. **Run the automated environment setup:**
   - **Windows (PowerShell / Command Prompt):**
     ```cmd
     setup_env.bat
     ```
   - **Linux / macOS:**
     ```bash
     chmod +x setup_env.sh
     ./setup_env.sh
     ```

3. **Download required model assets** (LLM, KinyaFlex, and Piper voices):
   ```powershell
   .\.venv\Scripts\activate
   python scripts/download_models.py
   ```
   Piper-only download:
   ```powershell
   python scripts/download_models.py --model piper
   ```
   Verify without downloading:
   ```powershell
   python scripts/download_models.py --check
   ```

### 5.2 Running the Application

Launch the interactive mode menu:
```powershell
python launch_menu.py
```

Or run directly via `run_assistant.py`:
- **Text Chat Mode (no audio required):**
  ```powershell
  python run_assistant.py --mode text
  ```
- **Text-to-Voice Mode (type text, receive voice response):**
  ```powershell
  python run_assistant.py --mode text-voice --speaker 0
  ```
- **Voice-to-Voice Mode (WAV audio input -> ASR -> LLM -> TTS):**
  ```powershell
  python run_assistant.py --mode voice --audio-file test.wav --speaker 0
  ```

Voice replies are written to `outputs/tts_responses/latest_response.wav` (KinyaFlex or Piper, depending on response language).

### 5.3 Running the Test Suite

```powershell
python -m unittest discover tests
python -m unittest tests.test_tts_routing
```

Or with pytest:
```powershell
pytest tests/test_tts_routing.py -q
```

---

## 6. Complete Documentation Index

For in-depth guides, consult the dedicated documentation files:
- [Installation and Setup Guide](docs/INSTALLATION.md)
- [Running MWIZERWA](docs/RUNNING.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Testing Guide](docs/TESTING.md)
- [Model and TTS Guide](docs/MODELS_AND_AUDIO.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [Developer Handover](docs/DEVELOPER_HANDOVER.md)
