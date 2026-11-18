# MWIZERWA AI — Multilingual Supportive Conversational Companion

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
- **Full Voice Pipeline:** End-to-end speech recognition (Kinyarwanda Health ASR), local LLM inference (AfriqueQwen3.5-4B GGUF), and multi-speaker neural speech synthesis (KinyaFlex TTS).

---

## 2. Supported Languages

| Language | Text Interaction | Speech Recognition (ASR) | Speech Synthesis (TTS) | Wellness Exercises | Safety Overrides |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kinyarwanda (`rw`)** | Yes | Yes (Native) | Yes (3 Speakers) | Yes (All 5 Categories) | Yes (112, Police, Support) |
| **English (`en`)** | Yes | English Audio | Non-Kinya Skipped* | Yes (All 5 Categories) | Yes (911/112/Hotlines) |
| **French (`fr`)** | Yes | French Audio | Non-Kinya Skipped* | Yes (All 5 Categories) | Yes (112/SAMU 912) |

*\*Note on TTS:* Current neural TTS (`KinyaFlex TTS`) is specialized exclusively for Kinyarwanda phonology. English and French responses are displayed in text mode; TTS synthesis is safely bypassed when non-Kinyarwanda responses are generated.

---

## 3. Technology Stack

- **Large Language Model (LLM):** `AfriqueQwen3.5-4B-Instruct-v1` quantized at `Q4_K_M` (2.86 GB GGUF) running locally via `llama-cpp-python` on CPU or GPU.
- **Automatic Speech Recognition (ASR):** `ElvisTata2024/Kinyarwanda-Health-ASR` via HuggingFace `transformers` pipeline.
- **Text-to-Speech (TTS):** `KinyaFlex TTS` (`FlexKinyaTTS` MB-iSTFT-VITS2 architecture) with 3 local speaker identities.
- **Context Management:** Token-aware sliding window with deterministic system prompt preservation (`n_ctx=4096`, `max_tokens=512`).
- **Safety Architecture:** Deterministic regex-based evaluator + Guardrail policy + Assistant output validator.
- **Exercise Engine:** Pre-authored trilingual static content catalog with deterministic lifecycle state machine.

---

## 4. Repository Structure

```text
kinyarwanda-tts-evaluation/
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
│   └── kinya-flex-tts/        # KinyaFlex TTS PyTorch weights & metadata
├── notebooks/                 # Exploratory and fine-tuning Colab notebooks
├── outputs/                   # Generated evaluation outputs and manifests
│   ├── prosody_reference_set/ # Reference dataset manifests and inspection reports
│   ├── tts_baseline_evaluation/ # 54-sample baseline TTS benchmark manifest & audio
│   └── tts_phase2_experiments/  # Phase 2 speed and elision experiment outputs
├── scratch/                   # Research scripts, dataset analysis, and intermediate tools
├── scripts/                   # Operational and benchmarking scripts
│   ├── benchmark_cpu_threads.py
│   ├── download_models.py     # Verified model downloader with SHA-256 checks
│   ├── inspect_prosody_reference_datasets.py
│   ├── run_phase2_experiments.py
│   ├── run_tts_baseline_evaluation.py
│   ├── test_asr.py
│   ├── test_kinya_flex.py
│   └── train_colab_kinya_flex.py
├── src/                       # Primary application source code
│   ├── asr/                   # Speech recognition engine wrapper
│   ├── conversation/          # ConversationEngine orchestration & session models
│   ├── exercises/             # Exercise catalog, models, recommender, and lifecycle
│   ├── llm/                   # ConversationManager, prompts, cleaner, parser
│   ├── safety/                # Gate 1 evaluator, Gate 2 assistant policy, guardrails
│   ├── tts/                   # KinyaFlex TTS synthesis engine wrapper
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

1. **Clone the repository:**
   ```bash
   git clone https://github.com/blessed-winner/MWIZERWA.git
   cd MWIZERWA
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

3. **Download required model assets:**
   ```powershell
   .\.venv\Scripts\activate
   python scripts/download_models.py
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

### 5.3 Running the Test Suite

Execute the complete regression and behavioral test suite (298 tests):
```powershell
python -m unittest discover tests
```

Execute behavioral tests specifically:
```powershell
python -m unittest discover tests/behavioral
```

---

## 6. Complete Documentation Index

For in-depth guides, consult the dedicated documentation files:
- [Installation and Setup Guide](file:///docs/INSTALLATION.md)
- [Running MWIZERWA](file:///docs/RUNNING.md)
- [Architecture Guide](file:///docs/ARCHITECTURE.md)
- [Testing Guide](file:///docs/TESTING.md)
- [Model and TTS Guide](file:///docs/MODELS_AND_AUDIO.md)
- [Troubleshooting Guide](file:///docs/TROUBLESHOOTING.md)
- [Developer Handover](file:///docs/DEVELOPER_HANDOVER.md)
