# Implementation Guidelines — Python AI Workers

This document outlines the system design and implementation rules for the Python AI Worker subsystem of our graduation project. It lives in its own server and repository, separate from the Node.js Core Backend. All team members working on this subsystem must read and understand this architecture before writing or committing code.

---

## 1. System Decomposition

Our system is decomposed into these main components:

- **FastAPI Gateway (`gateway.py`):** The primary entry point and protection proxy. It handles asynchronous HTTP requests, validates stream ticket credentials against Redis, and routes valid payloads to the correct worker. It exposes **`POST /ingest`** (single HTTP endpoint for media batches) and **`GET /health`** (readiness for the Node.js backend).
- **Redis clients:** **`infrastructure/redis_async_client.py`** (`redis.asyncio`) is used by the **gateway** for **ticket validation** reads only (§6). **`infrastructure/redis_client.py`** is a **sync** helper for **worker** Pub/Sub publish paths — one process-wide instance per worker process (same “no per-request churn” discipline).
- **Audio Worker (`workers/audio/`):** Processes incoming audio chunks. Outputs **VAD**, **speaker diarization**, **talk-time metrics for the current ingest window only** (not cumulative per-speaker totals for the whole meeting unless added later upstream of Python), **transcript (ASR)** for that timeline window, and publishes a single structured payload **per window** to `meeting:{id}:audio`. Implementation spans multiple modules (ASR sliding context buffer lives here — see §1.3).
- **Video Worker (`workers/video/`):** Consumes video chunks for attention/focus tracking; publishes to `meeting:{id}:vision`. Multiple modules allowed under `workers/video/`.
- **Text Worker (`workers/text/`):** Runs **slower-cycle** NLP (e.g. agenda adherence / staying in context). It does **not** receive a multipart ingest part. It is fed **internally** from transcripts produced by the audio pipeline (in-process queue, IPC, or equivalent — see §3.2). It maintains a **per-meeting sliding transcript ring buffer** and a **periodic timer**; it publishes analysis to `meeting:{id}:text`. Multiple modules allowed under `workers/text/`.

> **Scope note:** Emotion detection is strictly out of scope for the current phase. The Video Worker handles focus/attention tracking only.

### 1.1. End-to-end contract (concise)

Let **`D`** = configured media chunk duration in ms (`MEDIA_CHUNK_DURATION_MS`, **default `2000`**). Values below use **`D`**; defaults are **`D=2000`**, ASR context **10s**, text interval **30s**.

| Concern | Agreement |
|---------|-----------|
| **Ingest** | One HTTP request per timeline window of length **`D`**: `POST /ingest`, `multipart/form-data`. |
| **Multipart parts** | `meetingId`, `streamTicket`, `offsetMs`, `audioChunk`, `videoChunk`. |
| **Timeline** | `offsetMs` = ms from **`startedAt`** (meeting `IN_PROGRESS`); **source of truth: client**. First window **`offsetMs = 0`**. Subsequent windows: **`offsetMs` increases by `D`** (`0`, `D`, `2D`, …) when the client sends contiguous windows. |
| **Implicit window** | Audio/video bytes for one request represent **`[offsetMs, offsetMs + D)`** relative to meeting start. End bound is not sent separately. |
| **Alignment** | Same request’s `audioChunk` and `videoChunk` share the **same** `offsetMs`; **audio** and **vision** Redis messages for that ingest **MUST** use the **same** `offsetMs`. |
| **Transcript on Redis** | **Canonical per-window transcript** is published on **`meeting:{meetingId}:audio`** together with other audio-analysis fields (**not** owned solely by the text worker). |
| **Text Redis** | **`meeting:{meetingId}:text`**: periodic **semantic** output (agenda/context), **`D`-independent cadence**; see §5. |

### 1.2. Timing — tunable defaults (2 / 10 / 30)

All three durations are **configuration-driven** (environment variables below). **Do not hardcode** magic numbers in business logic — read from config at startup.

| Role | Env variable | Default | Meaning |
|------|----------------|---------|---------|
| **Media / ingest / fusion stride** | `MEDIA_CHUNK_DURATION_MS` | `2000` | Nominal duration **`D`** (ms) of each `audioChunk`/`videoChunk` and step between successive `offsetMs` values **when uploads are contiguous**. Frontend and gateway validation MUST stay consistent with whatever value is deployed. |
| **ASR sliding context** | `ASR_CONTEXT_WINDOW_MS` | `10000` | **Internal audio pipeline only:** approximate length of the **rolling acoustic/context window** used to stabilize transcript decoding for **each new chunk** (~10 s default). Does **not** change HTTP frequency or **`offsetMs`**. |
| **Text analysis period** | `TEXT_ANALYSIS_INTERVAL_MS` | `30000` | How often (ms) the text worker runs agenda/context inference **approximately** (~30 s default). **Not** triggered by “ring full”; timing is **timer-driven**. |

**Ring buffer sizing:** Maintain a **per-meeting sliding ring** of transcript **segments** (as produced by the audio path for ingestion steps). Capacity SHOULD default to **`ceil(TEXT_ANALYSIS_INTERVAL_MS / MEDIA_CHUNK_DURATION_MS)`** segments so the buffer nominally spans **about one analysis interval** of transcript history. Optional override:

| Env variable | Default | Meaning |
|----------------|---------|---------|
| `TEXT_TRANSCRIPT_RING_BUFFER_SLOTS` | *(derived)* | If unset, derive from **`ceil(TEXT_ANALYSIS_INTERVAL_MS / MEDIA_CHUNK_DURATION_MS)`**. Set explicitly only when you intentionally decouple slot count from the two vars above. |

### 1.3. Stateless vs meeting-scoped state

- **Gateway and routing** remain stateless.
- **Video** processing is treated as **stateless** per chunk (aside from any model-internal buffers).
- **Audio** MAY keep a **bounded sliding buffer** of recent audio samples/feature state for **`ASR_CONTEXT_WINDOW_MS`** only; scope by `meetingId` and drop on meeting end or process policy.
- **Text** MAY keep **per-meeting ring buffer + timer state** as above. Document cleanup when a meeting ends or the process resets.

---

## 2. Project Folder Structure

The repository follows the **Separation of Concerns** principle, mirroring the philosophy of the Core Backend. Worker logic is grouped under **`workers/`** in three modality packages **`audio/`**, **`video/`**, and **`text/`**. Each package may contain several `snake_case.py` modules (processors, adapters, subprocess entrypoints); there is **no requirement** that a modality be implemented in a single file.

```
grad-project-ai-workers/
├── gateway.py                        # FastAPI entry — routing and auth only (`GET /health`, `POST /ingest`)
├── gateway_supervisor.py             # Spawns audio/video Processes + Pipe IPC (§6)
├── gateway_schemas.py                # Standard HTTP envelope helpers (§10)
├── workers/
│   ├── audio/                        # Audio + live pipeline + offline batch_runner
│   │   ├── pipeline.py               # ``processLiveChunk`` → Redis :audio (+ text handoff)
│   │   ├── worker_main.py            # ``audio_worker_process_target`` / ``audioWorkerLoop`` (Pipe from supervisor)
│   │   ├── batch_runner.py           # ``runBatchRunner`` — offline WAV pipeline (legacy ``AudioWorker/app.py`` logic)
│   │   ├── context_buffer.py          # rolling ASR context (§1.2)
│   │   ├── io_audio.py               # WAV bytes → mono float PCM @ ``TARGET_SAMPLE_RATE``
│   │   ├── asr.py                     # Whisper singleton + window-aligned transcript clip
│   │   ├── vad_window.py             # lightweight **window** VAD (live)
│   │   ├── vad_batch.py              # Pyannote **file** VAD (offline)
│   │   ├── diarization.py            # Pyannote **file** diarization + streaming stub (live)
│   │   ├── transcript_batch.py        # Whisper file + alignment (offline)
│   │   ├── transcript_refiner.py      # Llama refinement — **offline only**
│   │   └── schemas.py                # Redis + Pipe Pydantic models
│   ├── video/                        # Vision worker → ``meeting:{id}:vision``
│   │   ├── pipeline.py               # ``processLiveVisionChunk`` stub → Redis
│   │   ├── worker_main.py            # ``video_worker_process_target`` / ``videoWorkerLoop``
│   │   └── schemas.py                # Redis vision payload models
│   └── text/                         # Agenda / context NLP → :text Redis
├── core/
│   └── fusion_publisher.py           # Redis Pub/Sub publish helpers
├── infrastructure/
│   ├── redis_client.py               # Sync Redis helper for worker publish paths
│   └── redis_async_client.py         # ``redis.asyncio`` singleton for gateway ticket reads (§6)
├── config.py                         # Optional: central place to load durations from env (recommended)
├── models/
│   ├── audio_model.py
│   ├── video_model.py
│   └── text_model.py                 # Text / agenda model when used
├── utils/
│   └── errors.py
├── .env
└── requirements.txt
```

*(If you prefer zero new file, load the same variables in `gateway.py` / worker mains — keep a single module that defines defaults `2000` / `10000` / `30000`.)*

**Naming rules**

- **Module files:** `snake_case.py` (Python import path).
- **`workers/` packages (audio / video / text):** **Functions and public instance methods** use **lowerCamelCase** (`processLiveChunk`, `runTranscription`, `buildConcatenated`, `warmupWhisper`). **Private helpers** use a leading underscore and camelCase (`_stubPayload`, `_resampleLinear`).
- **Exceptions (unchanged spelling):** Python special methods (`__init__`, `__getattr__`, …) and **`config.py` / `gateway.py` / `infrastructure/`** may keep conventional `snake_case` where they mirror backend/PEP8 style.
- **Classes:** **`PascalCase`** (e.g. `AudioChunkPayload`, `RollingAudioContextBuffer`).
- **Infrastructure clients:** module-level singleton pattern unchanged; one primary public class per infra module where applicable.

---

## 3. Incoming Request Interface

### 3.1. HTTP Ingest — `POST /ingest` (Media chunks from frontend)

Each ingest covers one timeline window of **`D`** milliseconds (`MEDIA_CHUNK_DURATION_MS`). That window arrives as a single **`multipart/form-data`** POST to **`/ingest`**.

| Field | Type | Description |
|-------|------|-------------|
| `meetingId` | `str` | Active meeting UUID |
| `streamTicket` | `str` | Redis stream ticket from Node (`POST /api/v1/meetings/:id/start`) |
| `offsetMs` | `str` *(→ `int`)* | Start of this window relative to **`startedAt`**, in ms (**client authoritative**). First window **`0`**. Contiguous uploads: **`0`, `D`, `2D`, …**. Gateway MAY validate multiples of **`D`** and return **`400`** + standard envelope if not. |
| `videoChunk` | `UploadFile` | Video bytes for **`[offsetMs, offsetMs + D)`** |
| `audioChunk` | `UploadFile` | Audio bytes for the **same** interval |

**Gateway** validates the ticket, parses `offsetMs`, routes `{ meetingId, videoChunk, offsetMs }` → **video** worker and `{ meetingId, audioChunk, offsetMs }` → **audio** worker. The **`streamTicket`** **must never** reach workers.

**Node/Data Fusion:** The backend sliding window SHOULD be consistent with **`MEDIA_CHUNK_DURATION_MS`** (defaults to 2 s; if you change `D`, coordinate Node fusion config).

### 3.2. Text worker — **no multipart field**

There is **no** `textChunk` on `POST /ingest`.

**Data path:** Audio pipeline emits **transcript segments** (and optional metadata) **from `workers/audio/`** into an **internal handoff** consumed by **`workers/text/`**.

**Canonical IPC (implemented):** a **one-way `multiprocessing.Pipe`** from **audio → text**. The supervisor / gateway parent process constructs the Pipe and passes:

- **Writer end** (`conn.send`): into the audio worker (e.g. as `textPipe` in each `chunk` IPC message alongside `audioWavBytes`), so `workers/audio/pipeline.py` forwards a JSON-serializable **`TextHandoffMessage`** after each ingest.
- **Reader end** (`conn.recv`): into the **text worker** loop (`workers/text/` — to be wired).

*`multiprocessing.Connection` objects generally cannot cross machine boundaries; nesting them inside messages only works reliably when supervisors use `spawn`/fork-compatible patterns (`workers/audio/worker_main.py`). If this breaks on Windows, fall back to a process-safe `multiprocessing.Queue` created once at supervisor startup.*

The text worker:

1. Appends arriving segments into a **per-meeting sliding ring** (capacity §1.2).
2. On each **`TEXT_ANALYSIS_INTERVAL_MS`** tick (per meeting with active traffic), runs agenda/context analysis over the **current ring contents**.
3. Publishes results to **`meeting:{meetingId}:text`** (see §5). **Optional:** include a short **excerpt** or **covered `offsetMs` range** for traceability; **canonical** per-window transcript remains on **`:audio`**.

---

## 4. Access Control & Authentication

*(Unchanged in behavior.)*

- Validate `GET meeting:<meetingId>:ticket` matches `streamTicket`; else **`401`**. No Redis key delete on validate; Node owns TTL + refresh.

---

## 5. Redis Pub/Sub — Publishing Results

### 5.1. Channels

| Channel | Published by | Consumed by |
|---------|--------------|-------------|
| `meeting:{meetingId}:vision` | Video worker | Data Fusion Engine (Node) |
| `meeting:{meetingId}:audio` | Audio worker | Data Fusion Engine (Node) |
| `meeting:{meetingId}:text` | Text worker | Data Fusion Engine (Node) |

### 5.2. Publish cadence (two time scales)

| Channel | Cadence | Notes |
|---------|---------|--------|
| **`…:audio`**, **`…:vision`** | **Once per ingest window** (~every **`D`** ms) | Each message **MUST** include **`meetingId`**, **`offsetMs`** matching that ingest. **Audio** payload **MUST** include **transcript** for that window (or `null` on graceful degradation) plus VAD / diarization / **window-local** talk-time fields. |
| **`…:text`** | **Periodic** (~every **`TEXT_ANALYSIS_INTERVAL_MS`**) | Not tied to each `D` step. Payload **SHOULD** include analysis fields and **MAY** include `analyzedOffsetRange` / excerpt; **must not** be the only source of per-window transcript. |

### 5.3. Payload schema

Exact JSON fields are finalized in Phase 3. Minimum contract:

- **Audio:** `meetingId`, `offsetMs`, transcript text (or `null`), VAD/diarization/speaking metrics **for `[offsetMs, offsetMs + D)` only**, optional raw payload bag.
- **Vision:** `meetingId`, `offsetMs`, focus-related fields per SDD.
- **Text:** `meetingId`, timestamp or **`offsetMs`/range** describing what was analyzed, agenda/context result fields; transcript duplication optional.

### 5.4. Graceful degradation

Malformed/empty chunks: publish **stub** payload with **`offsetMs`** preserved and analysis fields **`null`** where appropriate — no process crash.

---

## 6. Software Control & Concurrency

Target end-to-end latency remains aligned with SDD (≤ **`D`** under nominal conditions when **`MEDIA_CHUNK_DURATION_MS`** defaults to 2000). Use **`asyncio`** for I/O-bound paths; **`redis.asyncio`** for ticket reads.

Use **separate OS processes** for audio, video, and text workers (**not** threads for compute-heavy modality work — SDD §3.5.3). **Gateway:** no inference, no Redis publish.

---

## 7. Data Management & Global Resources

- **`offsetMs`:** Authoritative timeline placement for multimodal fusion and correlation with **`startedAt`**.
- **Transcript source of truth on Redis:** **`:audio`** messages per **`offsetMs`**. **`:text`** carries **higher-level** interpretations on a slower schedule.
- No direct PostgreSQL access; no per-request Redis client instances.

---

## 8. Model Warm-Up & Startup Readiness

Load audio + video models before ingest; load text models when NLP path enabled. Each required module MUST become ready within **15 seconds** of process start or it is treated as failed. **`GET /health`**: **`200`** `{ "success": true, "data": {}, "message": "AI workers ready" }` when all required modules passed warm-up; on failure **`503`** with `{ "success": false, "error": { "code": 503, "message": "..." } }` naming failing modules (**`audio`**, **`video`**, **`text`**). Frontend blocks ticket until **`200`** (SDD §3.6.1).

---

## 9. Boundary Conditions

Startup: Redis + models mandatory. Runtime Redis failure: **`503`** on ingest, never bypass auth. Worker crash: structured gateway error. Meeting-scoped buffers (ASR context, transcript ring) **SHOULD** be bounded and cleaned up when a meeting completes or exceeds policy.

Watchdog / shutdown refinement: Phase 4 (SDD references).

---

## 10. HTTP Response Format Contract

Standard envelope: **`{ success, data, message }`** / **`{ success, error: { code, message } }`**. **`401`** on ticket failure → frontend stops capture.

---

## 11. Python Code Standards

Type hints, typed config for durations, **Pydantic** for wire shapes, no bare `except`, **no business logic in `gateway.py`**, secrets in **`.env`**.

**Worker modules (`workers/audio`, `workers/video`, `workers/text`):** define **functions and methods in lowerCamelCase** (e.g. `runTranscription`, `pcmMonoF32FromWavBytes`, `payloadToRedisDict`). This matches the project’s JS/TS naming on the wire while keeping **`.py` filenames** in `snake_case`.

---

## 12. Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | Yes | — | Same Redis as Node |
| `PORT` | No | `8000` | FastAPI listen port |
| `MEDIA_CHUNK_DURATION_MS` | No | `2000` | Ingest/fusion stride **`D`** (ms); client `offsetMs` step must match |
| `ASR_CONTEXT_WINDOW_MS` | No | `10000` | Internal ASR sliding context (ms) |
| `TEXT_ANALYSIS_INTERVAL_MS` | No | `30000` | Text worker timer period (ms) |
| `TEXT_TRANSCRIPT_RING_BUFFER_SLOTS` | No | *(derive)* | Override ring capacity; default `ceil(TEXT_ANALYSIS_INTERVAL_MS / MEDIA_CHUNK_DURATION_MS)` |
| `HF_TOKEN` | No | *(empty)* | Hugging Face token for gated **pyannote** models (**batch**/offline paths in `vad_batch.py` / `diarization.py`; also respects `HUGGINGFACE_HUB_TOKEN`) |
| `WHISPER_MODEL_SIZE` | No | `small` | Live ASR singleton + batch runner model id (`tiny`/`base`/`small`/…) |
| `WHISPER_LANGUAGE` | No | `tr` | Whisper language code |
| `TARGET_SAMPLE_RATE` | No | `16000` | Decode/resample target for ingest WAV bytes (**must match Whisper expectations**) |
| `RUN_LIVE_VAD_ENERGY` | No | `1` | RMS frame **window-local** metrics on live ingest (`0` disables) |
| `RUN_LIVE_ASR` | No | `1` | Whisper on rolling context (`0` stubs transcript null) |
| `RUN_LIVE_DIARIZATION_STUB` | No | `1` | Placeholder **`UNKNOWN`** speaker list per window (`0` skips field) |
| `VAD_ENERGY_RMS_QUANTILE` | No | `0.35` | RMS percentile threshold divider for speech vs silence heuristic |

**Offline batch CLI** *(non-ingest regression / lab tool)*:

| Env | Purpose |
|-----|---------|
| `BATCH_AUDIO_FILE` | Input WAV (`audio/meeting_clean.wav` default relative CWD) |
| `BATCH_RUN_VAD` / `…_DIARIZATION` / `…_TRANSCRIPTION` / `…_REFINEMENT` | Enable pipeline stages (`1`/`0`) |
| `BATCH_OUTPUT_DIR` | Write text outputs (`outputs` default). Run from repo root: ``python -m workers.audio.batch_runner`` (entry: ``runBatchRunner()``). |

Frontend: **`VITE_PYTHON_INGEST_BASE_URL`** and **the same effective `D`** (or client-read config) must match deployment.

**Legacy folder:** [`AudioWorker/`](AudioWorker/) (original script layout) remains for historical reference — new code belongs under **`workers/audio/`**.

---

## 13. Phase Context & Implementation Order

| Step | Deliverable | Phase |
|------|-------------|-------|
| 1 | Config module reading **`MEDIA_CHUNK_DURATION_MS`**, **`ASR_CONTEXT_WINDOW_MS`**, **`TEXT_ANALYSIS_INTERVAL_MS`** (+ optional ring slots) | **3** |
| 2 | `infrastructure/redis_client.py` | **3** |
| 3 | `GET /health` | **3** |
| 4 | `POST /ingest` — ticket, **`offsetMs`** validation vs **`D`**, route A/V | **3** |
| 5 | Audio stub: publish `:audio` with **`offsetMs`** + placeholder transcript + metrics | **3** |
| 6 | Video stub: publish `:vision` | **3** |
| 7 | Internal audio→text handoff + ring + timer; text stub: publish `:text` on interval | **3** |
| 8 | Model loaders + warm-up | **3** |
| 9 | Real VAD/diarization/ASR in **`workers/audio/`** (ASR uses **`ASR_CONTEXT_WINDOW_MS`**) | **3+** |
| 10 | Real vision in **`workers/video/`** | **3+** |
| 11 | Real agenda/context model in **`workers/text/`** | **3+** |
| 12 | Graceful shutdown / Watchdog polish | **4** |

**Gateway run:** From the directory that contains `gateway.py`, with Redis reachable and `.env` aligned with Node:

```bash
uvicorn gateway:app --host 0.0.0.0 --port ${PORT:-8000}
```

