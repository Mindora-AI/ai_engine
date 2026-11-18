# MWIZERWA AI — Troubleshooting and Maintenance Guide

This document lists common issues, error messages, root causes, and verified resolutions.

---

## 1. Environment and Installation Issues

### 1.1 `ModuleNotFoundError: No module named 'deepkin'`
- **Symptom:** Running `run_assistant.py` or TTS scripts fails with `No module named 'deepkin'`.
- **Cause:** `DeepKIN-AgAI` was not installed in editable mode inside the active virtual environment.
- **Resolution:**
  ```powershell
  pip install -e ac-ai-models/DeepKIN-AgAI --no-deps
  ```

### 1.2 `llama-cpp-python` Compilation Errors on Windows
- **Symptom:** `pip install llama-cpp-python` fails requiring Visual Studio C++ compilers.
- **Cause:** No pre-built binary wheel matched your Python/OS version.
- **Resolution:**
  1. Ensure you are using a supported Python version (Python **3.10**, **3.11**, or **3.12** 64-bit).
  2. Install using the pre-compiled wheel repository:
     ```powershell
     pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
     ```

### 1.3 Python Architecture Mismatch
- **Symptom:** `ImportError: DLL load failed` or architecture mismatch.
- **Cause:** 32-bit Python was installed instead of 64-bit.
- **Resolution:** Verify your Python architecture:
  ```powershell
  python -c "import struct; print(struct.calcsize('P') * 8)"
  ```
  Must print `64`. If it prints `32`, install 64-bit Python.

---

## 2. Model Loading Issues

### 2.1 `FileNotFoundError: models/afriqueqwen/...`
- **Symptom:** Startup fails because the LLM GGUF model file cannot be found.
- **Resolution:**
  Run the automated model downloader:
  ```powershell
  python scripts/download_models.py
  ```
  Or verify existing model placement:
  ```powershell
  python scripts/download_models.py --check
  ```

### 2.2 Memory Exhaustion / System Crash During LLM Loading
- **Symptom:** System freezes or process terminates abruptly with code 137 / Out of Memory when loading `AfriqueQwen3.5-4B`.
- **Cause:** Less than 8 GB of available physical RAM.
- **Resolution:**
  1. Close memory-intensive background applications.
  2. On Windows, ensure a paging file (virtual memory) of at least 8-16 GB is enabled on an SSD.
  3. Verify that `n_ctx` in `ConversationManager` is not set above 4096.

---

## 3. Audio and Multimodal Pipeline Issues

### 3.1 `[TTS Info] Speech output skipped`
- **Symptom:** The assistant prints a response in English or French, but no audio is produced.
- **Cause:** **Expected Behavior.** `KinyaFlex TTS` is trained exclusively for Kinyarwanda phonology. Non-Kinyarwanda text is intentionally prevented from passing to the Kinyarwanda synthesizer to avoid garbled speech.
- **Resolution:** None needed. If Kinyarwanda audio is desired, interact with the assistant in Kinyarwanda.

### 3.2 ASR First-Run Download Hangs
- **Symptom:** `run_assistant.py --mode voice` pauses at `[1/4] Transcribing audio with ASR...` for several minutes.
- **Cause:** First-time download of `ElvisTata2024/Kinyarwanda-Health-ASR` (~0.95 GB) from Hugging Face Hub.
- **Resolution:** Allow the initial download to complete. Subsequent runs load directly from the local Hugging Face cache (`~/.cache/huggingface/`).

### 3.3 ASR Returns Empty or Silent Transcription
- **Symptom:** `[ASR Warning] Transcription empty or silent. Skipping LLM generation.`
- **Cause:** Audio file is silent, below volume threshold, or in an unsupported format (e.g. non-16kHz, stereo, corrupted header).
- **Resolution:**
  - Verify that the input WAV is 16 kHz 16-bit mono.
  - Test using the verified reference sample `test.wav`:
    ```powershell
    python scripts/test_asr.py
    ```

---

## 4. Test Suite and Discovery Issues

### 4.1 Tests Fail to Discover or Import Modules
- **Symptom:** Running `python -m unittest discover tests` yields `ModuleNotFoundError: No module named 'src'`.
- **Cause:** Current working directory is not the repository root, or `.venv` is not activated.
- **Resolution:**
  ```powershell
  Set-Location <path-to-MWIZERWA>
  .\.venv\Scripts\activate
  python -m unittest discover tests
  ```

### 4.2 Error Messages Printed During Test Runs
- **Symptom:** Output displays lines like:
  ```text
  Error during LLM inference: Inference boom!
  Error during LLM inference: GPU failure
  [ASR Warning] Transcription empty or silent.
  ```
- **Cause:** **Expected Behavior.** These log lines originate from negative test cases in `test_token_context_management.py` and `test_conversation_engine.py` that intentionally verify error handling and state preservation during simulated failures.
- **Resolution:** Look at the final summary line. If it concludes with `OK` (e.g. `Ran 298 tests in ~14s OK`), all tests have passed.
