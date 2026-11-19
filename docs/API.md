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

## `POST /chat`

**JSON body**

```json
{
  "user_text": "Muraho",
  "session_id": "optional-uuid",
  "mode": "normal",
  "language": null
}
```

**Response**

```json
{
  "session_id": "...",
  "response_text": "...",
  "mode": "normal",
  "language": "rw",
  "metadata": {}
}
```

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