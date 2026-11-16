# MINDORA — Kinyarwanda Local Voice & Mental Health Assistant

An integrated, fully local multimodal voice assistant for Kinyarwanda mental-health support and conversational guidance:

1. **ASR (Speech Recognition):** `ElvisTata2024/Kinyarwanda-Health-ASR` (HuggingFace Transformers pipeline).
2. **LLM (Language Model):** `AfriqueQwen3.5-4B-Instruct-v1` GGUF loaded locally via `llama-cpp-python` with token-aware context management.
3. **TTS (Text-to-Speech):** `KinyaFlex TTS` (`FlexKinyaTTS` MB-iSTFT-VITS2 PyTorch model via DeepKIN-AgAI).

---

## 1. System Architecture

```text
[ Audio WAV File / Microphone ]
               │
               ▼
┌───────────────────────────────┐
│          ASR Engine           │ (Kinyarwanda Health ASR)
└──────────────┬────────────────┘
               │ Transcribed Text
               ▼
┌───────────────────────────────┐
│     Conversation Manager      │ (Token-Aware Context Budgeting & Eviction)
└──────────────┬────────────────┘
               │ Formatted Prompt (Jinja2 Chat Template, n_ctx=4096)
               ▼
┌───────────────────────────────┐
│       AfriqueQwen LLM         │ (AfriqueQwen3.5-4B-Instruct-v1 Q4_K_M GGUF)
└──────────────┬────────────────┘
               │ Raw LLM Output
               ▼
┌───────────────────────────────┐
│       Response Cleaner        │ (Strips <think> tags & control tokens)
└──────────────┬────────────────┘
               │ Cleaned Text
               ▼
┌───────────────────────────────┐
│       Language Detector       │ (Ensures Kinyarwanda before TTS)
└──────────────┬────────────────┘
               │ Kinyarwanda Text
               ▼
┌───────────────────────────────┐
│        KinyaFlex TTS          │ (Synthesizes 24kHz 16-bit Mono PCM WAV)
└───────────────────────────────┘
```

---

## 2. Requirements & Prerequisites

- **Operating System:** Windows 10/11 or Linux (Ubuntu 20.04/22.04 LTS tested).
- **Python Version:** Python **3.10**, **3.11**, or **3.12** (tested with Python 3.12.10).
- **Hardware Profile:** Baseline runs on **CPU-only** (e.g. Intel Core i5/i7/i9 or AMD Ryzen).
- **RAM:** Minimum 16 GB physical RAM recommended.
- **Disk Space:**
  - Git repository & dependencies: ~2 GB
  - Model weights:
    - AfriqueQwen GGUF (`Q4_K_M`): **2.86 GB**
    - KinyaFlex TTS Checkpoint: **1.03 GB**
    - ASR Cache (downloaded on first run): **~0.95 GB**
  - Total recommended free disk space: **$\ge$ 8 GB**.

---

## 3. Quick Start Installation (Cold Clone)

### Option A: Automated Setup (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/blessed-winner/MWIZERWA.git
   cd MWIZERWA
   ```

2. **Run the environment setup script:**
   - **On Windows:**
     ```cmd
     setup_env.bat
     ```
   - **On Linux / macOS:**
     ```bash
     chmod +x setup_env.sh
     ./setup_env.sh
     ```

   *The setup script creates `.venv`, installs dependencies from `requirements.txt`, clones `ac-ai-models` (pinned to verified commit `c28db6b`), and installs `DeepKIN-AgAI` in editable mode.*

3. **Download required model checkpoints:**
   ```bash
   # Activate environment if not already activated
   # Windows: .\.venv\Scripts\activate
   # Linux:   source .venv/bin/activate

   python scripts/download_models.py
   ```
   *To inspect model presence and verify SHA-256 hashes without re-downloading, run:*
   ```bash
   python scripts/download_models.py --check
   ```

---

### Option B: Manual Step-by-Step Installation

If you prefer manual setup or are operating in a customized environment:

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install core Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > **Note for Windows Users without Visual Studio C++ Build Tools:**
   > If `pip install llama-cpp-python` fails to compile, install a prebuilt CPU wheel:
   > ```bash
   > pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   > ```

3. **Clone and install DeepKIN-AgAI:**
   ```bash
   git clone https://github.com/c4ir-rw/ac-ai-models.git ac-ai-models
   git -C ac-ai-models checkout c28db6bfb6b80666583717c87e8edaa2a8f384d5
   pip install -e ac-ai-models/DeepKIN-AgAI
   ```

4. **Download model checkpoints:**
   ```bash
   python scripts/download_models.py
   ```

---

## 4. How to Run

### Interactive Unified Launcher
The easiest way to explore all features:
```bash
python launch_menu.py
```
Provides an interactive terminal menu for Text Mode, Voice Mode, ASR testing, TTS testing, and Baseline Evaluation.

---

### Text Mode (Interactive Terminal Chat)
Runs conversation with AfriqueQwen using the token-aware context manager without loading audio models:
```bash
python run_assistant.py --mode text
```
- `/clear`: Resets conversation history while preserving the `SYSTEM_PROMPT`.
- `exit` or `quit`: Exits the session.

---

### Voice Mode (End-to-End Multimodal Pipeline)
Runs the full ASR $\to$ LLM $\to$ TTS pipeline:
```bash
python run_assistant.py --mode voice
```

Or pass a specific WAV audio file directly for non-interactive / headless execution:
```bash
python run_assistant.py --mode voice --audio-file path/to/input.wav
```

Select a specific voice for KinyaFlex TTS with `--speaker`:
- `0`: female_1 (default)
- `1`: female_2
- `2`: male

Example:
```bash
python run_assistant.py --mode voice --audio-file test.wav --speaker 1
```

*Note: Voice mode without `--audio-file` requires working microphone input via system audio capture.*

---

## 5. Testing & Evaluation

### Run Automated Unit Tests
Runs the full 24-test suite (token budgeting, Turn eviction, safety margin, error handling, pipeline mocks):
```bash
python -m unittest discover tests
```

### Run Baseline TTS Benchmark Evaluation
Synthesizes the 18 standardized benchmark sentences across all 3 speaker voices (54 WAV files total) and outputs audio metrics and real-time factors to `outputs/tts_baseline_evaluation/manifest.json`:
```bash
python scripts/run_tts_baseline_evaluation.py
```
*Generated audio is saved locally to `outputs/tts_baseline_evaluation/*.wav` (ignored by Git); the benchmark metadata manifest is tracked.*

### Test Individual Components Independently
- **Test ASR:**
  ```bash
  python scripts/test_asr.py
  ```
- **Test TTS:**
  ```bash
  python scripts/test_kinya_flex.py
  ```
- **Validate LLM Context Management on Live Model:**
  ```bash
  python scripts/validate_token_context_real_model.py
  ```
- **Benchmark CPU Threads ($n=4, 6, 8$):**
  ```bash
  python scripts/benchmark_cpu_threads.py
  ```

---

## 6. Supported Languages & Assistant Behavior

- **ASR Input:** Supports spoken Kinyarwanda speech recognition.
- **LLM Output:** Fully multilingual in **Kinyarwanda**, **English**, and **French**.
- **TTS Output:** **Kinyarwanda only.** If the LLM generates a response in English or French, the response text is printed to the console, and speech synthesis is skipped with an informative notice.
- **Mental Health Focus:** The system prompt instructs the assistant to listen actively, clarify gently, and validate emotional experiences before providing measured guidance.

---

## 7. Troubleshooting

1. **`FileNotFoundError: AfriqueQwen model GGUF file not found` or `KinyaFlex TTS model file not found`:**
   - Run `python scripts/download_models.py` to download missing weights from Hugging Face.
   - Run `python scripts/download_models.py --check` to verify local file integrity.
2. **`ImportError: deepkin package is required for KinyaFlex TTS`:**
   - Ensure `ac-ai-models` has been cloned and installed:
     ```bash
     git clone https://github.com/c4ir-rw/ac-ai-models.git ac-ai-models
     pip install -e ac-ai-models/DeepKIN-AgAI
     ```
3. **`llama-cpp-python` Compilation Errors:**
   - On Windows, if MSYS2/MinGW or conflicting headers are in your system `PATH`, remove them before pip install, or install pre-built wheels:
     ```bash
     pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
     ```
4. **Microphone Audio Not Capturing:**
   - Use `--audio-file <path.wav>` to pass recorded 16-bit PCM WAV audio directly to the assistant.
