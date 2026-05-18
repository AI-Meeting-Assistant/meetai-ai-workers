# meetai-ai-workers — quick setup (team)

## 1. Python env

Python 3.10+ recommended. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. PyTorch with CUDA (recommended)

NVIDIA driver installed (`nvidia-smi` works). **Do not** rely on plain `pip install torch` on Windows — it often installs **CPU** wheels.

```powershell
pip install "torch>=2.6.0" "torchaudio>=2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip install "numpy>=1.24,<2"
pip install -r requirements.txt
```

Check:

```powershell
python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
```

Expect `2.6.x+cu124` and `cuda= True`.

## 3. Environment

```powershell
copy .env.example .env
```

Edit `REDIS_URL`, `HF_TOKEN` (if using batch pyannote), and keep `WHISPER_MODEL_SIZE=medium` when GPU is available.

## 4. Run gateway

```powershell
python gateway.py
```

Health: `GET http://localhost:8000/health` → workers ready.

## CPU-only fallback

Use `WHISPER_MODEL_SIZE=small` in `.env` and accept slower / queued live ingest.

---

## 5. Local LLM — Qwen 2.5 3B via Ollama (text worker)

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
