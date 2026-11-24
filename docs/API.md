# MWIZERWA Inference HTTP API

Serve with:

```bash
pip install -r requirements-api.txt
python scripts/serve_api.py
```

Environment: `MWIZERWA_API_HOST`, `MWIZERWA_API_PORT`, `MWIZERWA_MODELS_DIR`, `MWIZERWA_LLM_N_GPU_LAYERS`, `MWIZERWA_MAX_UPLOAD_BYTES` (default 25 MiB).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | LLM ready status |
| POST | `/chat` | JSON text chat |
| POST | `/tts` | Form text → WAV |
| POST | `/transcribe` | Multipart audio → text |
| POST | `/voice` | Multipart audio → JSON + base64 WAV reply |

### Languages

- **Chat / voice:** detected per turn (`rw`, `en`, `fr`).
- **TTS:** `rw` → KinyaFlex; `en` / `fr` → Piper. Omit `language` on `/tts` to auto-detect from text.

### Audio uploads

- **Formats:** `.wav`, `.webm`, `.mp3`, `.ogg`, `.flac`, `.m4a`, … (`audio/*` or `application/octet-stream`).
- **Max size:** `MWIZERWA_MAX_UPLOAD_BYTES` (default 25 MB).
- Uploads are stored in temp files only and deleted after the request.

---

## `GET /health`

**Response**

```json
{"status": "ready", "engine_initialized": true}
```

---

## `POST /chat` (Stage 1 — text conversation)

Primary integration surface for the main Mwizerwa application.

### Supported languages

| Code | Language |
|------|----------|
| `rw` | Kinyarwanda |
| `en` | English |
| `fr` | French |

Optional `language` must be one of these codes (case-insensitive). Unsupported values return **422**.

### Session behavior

- **`session_id` omitted:** server assigns a new UUID; history starts empty (system prompt only).
- **Same `session_id` on later requests:** prior user/assistant turns for that id are kept in memory and passed into the model with token-aware trimming.
- **Different `session_id` values:** isolated histories (in-memory map in the API process).
- **Durability:** sessions are **not** persisted. Process restart or deploy clears all context. There is no cross-device or permanent memory.
- **`session_id` constraints:** non-empty after trim, max 128 characters.

### Request body

```json
{
  "user_text": "Muraho",
  "session_id": "optional-client-id",
  "mode": "normal",
  "language": null
}
```

| Field | Required | Notes |
|-------|----------|--------|
| `user_text` | yes | Non-empty after trim |
| `session_id` | no | Client-provided id for continuity |
| `mode` | no | `gentle` \| `normal` \| `motivational` |
| `language` | no | Hint: `rw` \| `en` \| `fr` |

### Response

```json
{
  "session_id": "...",
  "response_text": "...",
  "mode": "normal",
  "language": "rw",
  "metadata": {
    "interaction_intent": "LISTENING",
    "requires_clarification": false,
    "detected_language": "rw",
    "gate1_triggered": false,
    "gate2_triggered": false
  }
}
```

`metadata` only includes keys from the public allowlist (no raw safety objects or internal paths).

### Errors

| Status | When |
|--------|------|
| **422** | Empty `user_text`, invalid `language`, invalid `session_id`, or schema violation |
| **503** | LLM not loaded (`/health` → `unavailable`) |
| **500** | Unexpected inference failure (generic message; no stack trace) |

### Model readiness

`POST /chat` requires the shared AfriqueQwen GGUF to be loaded at API startup (`engine_initialized: true` on `/health`). No external network calls are made per chat request once the model is loaded.

### Optional `user_id` (persistent memory)

Supply only from a **trusted application backend** after authentication. When set, the engine may return `metadata.memory_proposals` for consent UI. Approved memory is injected into future turns; see [MEMORY.md](MEMORY.md).

---

## `POST /tts`

**multipart/form-data**

| Field | Type | Required |
|-------|------|----------|
| `text` | string | yes |
| `language` | `rw` \| `en` \| `fr` | no |
| `speaker_id` | int (0–2) | no (KinyaFlex) |

**Response:** `audio/wav` bytes.

**Example (curl)**

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -F "text=Hello from MWIZERWA" \
  -F "language=en" \
  -o speech.wav
```

---

## `POST /transcribe`

**multipart/form-data**

| Field | Type | Required |
|-------|------|----------|
| `audio` | file | yes |

**Response**

```json
{"transcription": "..."}
```

---

## `POST /voice`

**multipart/form-data**

| Field | Type | Required |
|-------|------|----------|
| `audio` | file | yes |
| `session_id` | string | no (reuse `/chat` session) |
| `mode` | `gentle` \| `normal` \| `motivational` | no |
| `speaker_id` | int | no |

**Response (JSON)**

```json
{
  "session_id": "...",
  "transcription": "user speech text",
  "response_text": "assistant reply",
  "mode": "normal",
  "language": "rw",
  "metadata": {},
  "audio_base64": "<base64 WAV>",
  "audio_media_type": "audio/wav"
}
```

Decode `audio_base64` to play the assistant voice.

**Example (curl)**

```bash
curl -X POST http://127.0.0.1:8000/voice \
  -F "audio=@user.wav" \
  -F "session_id=my-session-1"
```

---

## Colab / ngrok

1. Install API deps and ensure models on Drive (`MWIZERWA_MODELS_DIR`).
2. `python scripts/serve_api.py` (bind `0.0.0.0` via `MWIZERWA_API_HOST=0.0.0.0`).
3. Expose with ngrok: `ngrok http 8000`.

First `/transcribe` or `/voice` call loads ASR (HF). First `/tts` or `/voice` loads KinyaFlex/Piper lazily.