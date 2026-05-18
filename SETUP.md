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
