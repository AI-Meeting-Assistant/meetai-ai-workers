# meetai-ai-workers — quick setup (team)

## 1. Prerequisites

Download and install **uv** (fast Python package manager):
- **macOS/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

## 2. Environment Setup

From the project root, create the virtual environment and install all dependencies:

```powershell
# This automatically handles Python version, venv creation, and dependencies
uv sync --index-strategy unsafe-best-match
```

### Current platform support
The lockfile selects wheels by operating system, not by detecting GPU hardware:
- **Windows/Linux x86-64 with NVIDIA:** `uv sync` selects the CUDA 12.4 PyTorch wheels. CUDA runs when a compatible NVIDIA GPU and driver are available.
- **Apple Silicon macOS:** `uv sync` selects the PyPI PyTorch wheel. An offline install dry run passes for macOS 13 ARM64. PyTorch audio models select MPS when available. Whisper word timestamps move their alignment calculation to CPU because MPS cannot use float64. The MediaPipe Python video path still runs on CPU; full Metal processing is not yet supported.

The Docker worker image is Linux-based and uses the CUDA wheels. It requires an NVIDIA GPU for GPU acceleration.
Build it independently from this repository with `docker build -t meetai-ai-workers .`.

## 3. Verification

Check if the environment is correctly set up:

```powershell
# Verify the installed wheel and available accelerator
uv run python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available(), 'mps=', torch.backends.mps.is_available())"
```

`cuda=True` requires a working NVIDIA GPU and driver. `mps=True` means the PyTorch audio models will select Metal on this Mac.

## 4. Configuration

```powershell
copy .env.example .env
```

Edit `REDIS_URL`, `HF_TOKEN` (for batch diarization), and `WHISPER_MODEL_SIZE`.

## 5. Run Gateway

```powershell
uv run gateway.py
```

Health check: `GET http://localhost:8000/health` should show all workers ready.

---

## 6. Local LLM — Qwen 2.5 3B via Ollama (text worker)

The text worker calls a locally-running LLM to analyse meeting transcript adherence. Ollama is the recommended runtime — it handles model download, quantisation, and serving automatically.

### macOS

```bash
# Install Ollama
brew install ollama

# Start the Ollama service (runs on http://localhost:11434)
ollama serve

# In a new terminal — pull and run Qwen 2.5 3B
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

Ollama starts automatically on login after `brew install`. To run it as a background service:

```bash
brew services start ollama
```

### Windows

1. Download the Ollama installer from **https://ollama.com/download/windows** and run it.
2. Ollama installs as a system service and starts automatically.
3. Open a terminal and pull the model:

```powershell
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

### Verify Ollama is running

```bash
curl http://localhost:11434/api/tags
```

Should return a JSON list that includes `qwen2.5:3b`.

### `.env` settings

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
TEXT_TRANSCRIPT_RING_BUFFER_SLOTS=4
```

`TEXT_TRANSCRIPT_RING_BUFFER_SLOTS` controls how many transcript chunks are buffered before the LLM is called. At 6-second chunks, `4` means analysis runs every ~24 seconds.

### Alternative models

Any model available on Ollama works — just change `OLLAMA_MODEL` in `.env`:

| Model | Size | Notes |
|-------|------|-------|
| `qwen2.5:3b` | ~2 GB | Recommended — fast, good quality |
| `qwen2.5:7b` | ~4.5 GB | Higher accuracy, slower on CPU |
| `llama3.2:3b` | ~2 GB | Good alternative |
| `phi3:mini` | ~2.3 GB | Microsoft, very fast |

## 7. Recorded Meeting Upload (`POST /ingest-recorded`)

Used when the frontend uploads a full audio/video file (not live chunks).

1. Node creates a `RECORDED` meeting (`IN_PROGRESS`) and returns `streamTicket`.
2. Frontend multipart POST to `http://localhost:8000/ingest-recorded` with `meetingId`, `streamTicket`, `file`, `title`, `agenda`.
3. Gateway stores file under `UPLOAD_DIR/{meetingId}/`, converts to WAV, runs batch VAD + diarization + Whisper, then Ollama adherence + summary.
4. Result is published to Redis `meeting:{id}:recorded-complete`; Node persists `TIMELINE_DATA` at `offset_ms=0` and sets meeting `COMPLETED`.

Env (see `.env.example`):

```env
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE_MB=500
RECORDED_KEEP_FILES=0
```

Requires `HF_TOKEN` for pyannote diarization in batch mode.

**PyTorch 2.6+:** batch pyannote checkpoints use `torch.load` with Lightning objects; the repo imports [`workers/audio/_torch_compat.py`](workers/audio/_torch_compat.py) before pyannote so `weights_only` defaults to `False` for trusted HF weights (avoids `UnpicklingError` / `ModelCheckpoint` allowlist issues).
