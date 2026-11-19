# MWIZERWA — Colab deployment audit

**Repository:** `https://github.com/Mindora-AI/ai_engine`  
**Status:** Implementation present in workspace (push to `main` required for fresh Colab clones).

## Root cause of prior Colab failure

The notebook imported `scripts.colab_runtime`, but `colab_runtime.py` was not on the remote `ai_engine` branch. Local workspace now ships:

| File | Role |
|------|------|
| `scripts/colab_runtime.py` | Hardware, Drive env, deps, llama-cpp, DeepKIN |
| `scripts/colab_validate.py` | Real inference + pipeline metadata + Gradio build |
| `scripts/colab_gradio.py` | Production Gradio UI |
| `scripts/verify_notebook_imports.py` | Static import parity check |
| `notebooks/MWIZERWA_Colab.ipynb` | Ordered runtime cells |
| `requirements-colab.txt` | Colab-safe deps (no torch / llama-cpp) |
| `src/runtime/model_paths.py` | `MWIZERWA_*` path resolution |

## Reused project components (not duplicated)

- `scripts/download_models.py` + `config/model_registry.json`
- `src/conversation/engine.py` — full safety + exercise pipeline
- `src/llm/afrique_qwen.py` — reads `MWIZERWA_LLM_N_GPU_LAYERS`
- `src/tts/tts_router.py` — KinyaFlex / Piper routing
- `src/asr/asr_engine.py`
- `run_assistant.py` semantics mirrored in `colab_gradio.py`

## Colab-only validation still required

Not executed in this development environment:

- Google Drive `drive.mount` authentication
- `install_llama_cpp_for_colab()` against live Colab CUDA wheels
- Multi-GB model download to Drive
- Full GPU LLM + KinyaFlex + Piper latency / OOM behavior

Local/static validation performed:

- `pytest tests/test_colab_runtime.py`
- `python scripts/verify_notebook_imports.py`
- `py_compile` on Colab scripts

## Known limitations

- Free Colab GPU RAM may require `MWIZERWA_LLM_N_GPU_LAYERS=0`.
- `colab_validate.py` loads models twice (validation + `build_ui` smoke); acceptable for deploy check.
- ASR depends on HF cache under `HF_HOME` on first run.