# MWIZERWA AI — Installation and Setup Guide

This guide provides verified, step-by-step instructions for installing, configuring, and initializing MWIZERWA AI on a fresh development machine.

---

## 1. Prerequisites and System Requirements

### 1.1 Operating System
- **Windows:** Windows 10 or Windows 11 (64-bit). Primary development platform.
- **Linux:** Ubuntu 20.04 or 22.04 LTS (x86_64). Fully supported.

### 1.2 Python Runtime
- **Supported Versions:** Python **3.10**, **3.11**, or **3.12**.
- **Verified Environment:** Python **3.12.10** (64-bit).
- Ensure Python and `pip` are added to your system `PATH`.

### 1.3 System Tools
- **Git:** Required for cloning and submodule management.
- **C/C++ Build Tools (Optional but Recommended):**
  - *Windows:* Visual Studio Build Tools (C++ Desktop development workload) if building `llama-cpp-python` with custom flags. Pre-built wheels are typically used.
  - *Linux:* `build-essential`, `cmake`, and `ffmpeg`.

### 1.4 Hardware Requirements
- **Processor:** Modern multi-core CPU (Intel Core i5/i7/i9 or AMD Ryzen 5/7/9, 4+ cores).
- **RAM:** Minimum 16 GB physical memory.
- **Disk Space:**
  - Git repository & dependencies: ~2.5 GB
  - Model weights:
    - `AfriqueQwen3.5-4B-Instruct-v1` GGUF: **2.86 GB**
    - `KinyaFlex TTS` checkpoint: **1.03 GB**
    - Piper voices (English + French): **~120 MB**
    - `ElvisTata2024/Kinyarwanda-Health-ASR` cache: **~0.95 GB**
  - Total minimum free disk space: **$\ge$ 10 GB**.
- **GPU (Optional):** NVIDIA GPU with CUDA 12.x for accelerated inference. Default setup runs completely on CPU.

---

## 2. Automated Installation

### 2.1 Windows (PowerShell / Command Prompt)

1. Open PowerShell and navigate to your projects directory:
   ```powershell
   git clone https://github.com/blessed-winner/MWIZERWA.git
   cd MWIZERWA
   ```

2. Run the automated bootstrap script:
   ```cmd
   setup_env.bat
   ```
   *What this script does:*
   - Creates a dedicated virtual environment at `.venv` using your system Python.
   - Upgrades `pip`, `setuptools`, and `wheel`.
   - Installs all dependencies from `requirements.txt`.
   - Clones the external `DeepKIN-AgAI` repository into `ac-ai-models/` at pinned commit `c28db6b`.
   - Installs `DeepKIN-AgAI` in editable mode (`pip install -e ac-ai-models/DeepKIN-AgAI --no-deps`).

3. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\activate
   ```

### 2.2 Linux / macOS

1. Clone repository:
   ```bash
   git clone https://github.com/blessed-winner/MWIZERWA.git
   cd MWIZERWA
   ```

2. Run the shell setup script:
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

3. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```

---

## 3. Manual Installation (Alternative)

If you prefer installing dependencies step by step:

1. **Create and activate virtual environment:**
   - *Windows:*
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - *Linux:*
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. **Upgrade packaging tools:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

3. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Clone and install DeepKIN-AgAI:**
   ```bash
   git clone https://github.com/AntoineM7/DeepKIN-AgAI.git ac-ai-models/DeepKIN-AgAI
   cd ac-ai-models/DeepKIN-AgAI
   git checkout c28db6b
   cd ../..
   pip install -e ac-ai-models/DeepKIN-AgAI --no-deps
   ```

---

## 4. Model Setup and Verification

MWIZERWA requires these local assets for full voice output:
1. `models/afriqueqwen/AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf`
2. `models/kinya-flex-tts/kinya_flex_tts_base_trained.pt`
3. Piper ONNX bundles under `models/piper/` (English + French)

### 4.1 Automated Model Download (Windows PowerShell)

```powershell
.\.venv\Scripts\activate
python scripts/download_models.py
```

This downloads:
- AfriqueQwen GGUF (`mradermacher/AfriqueQwen3.5-4B-Instruct-v1-i1-GGUF`)
- KinyaFlex checkpoint (`C4IR-RW/kinya-flex-tts`)
- Piper voices `en_US-ljspeech-medium` and `fr_FR-siwis-medium` from `rhasspy/piper-voices`

Piper only:
```powershell
python scripts/download_models.py --model piper
```

Install Piper dependencies (included in `requirements.txt`):
```powershell
pip install piper-tts onnxruntime
```

### 4.2 Verifying Model Integrity

To check whether model files exist and verify their SHA-256 integrity without downloading:
```powershell
python scripts/download_models.py --check
```

Expected output:
```text
Checking AfriqueQwen GGUF... OK (Found: models/afriqueqwen/AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf)
Checking KinyaFlex TTS Checkpoint... OK (Found: models/kinya-flex-tts/kinya_flex_tts_base_trained.pt)
All models verified.
```

---

## 5. GPU Acceleration (Optional)

By default, `llama-cpp-python` runs on CPU using AVX2 instructions. To enable NVIDIA CUDA GPU offloading:

### Windows:
```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### Linux:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

When CUDA is enabled, specify GPU layer offload in code or initialize with `n_gpu_layers > 0`.

---

## 6. Verifying the Installation

Run the complete test suite to confirm all subsystems are operational:
```powershell
python -m unittest discover tests
```
A successful installation will output:
```text
pytest tests/ -q  (mocked suite; expect hundreds of passes in under ~1 minute)
OK
```
