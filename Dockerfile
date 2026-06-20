# LABAD — insider-threat detection pipeline
# CPU image (no CUDA): inference + scoring run fine on CPU in a container;
# GPU/MPS is a host-only concern for training.
FROM python:3.11-slim

WORKDIR /app

# build-essential covers any source builds pulled in by faiss-cpu / torch deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch FIRST from PyTorch's CPU index. The default PyPI
# wheel bundles the multi-GB NVIDIA CUDA stack (nvidia-cu*), which is dead
# weight in a CPU container — it bloats the image by ~3-4GB and is what
# exhausted the build disk. Pinning the CPU build keeps the image lean.
RUN pip install --no-cache-dir torch==2.12.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining deps: torch is already satisfied, so pip won't re-pull CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source only — data/, checkpoints/, .venv are excluded via .dockerignore
# and bind-mounted at runtime by docker-compose.
COPY . .

# Ollama lives in a sibling container; reach it by service name, not localhost.
ENV OLLAMA_URL=http://ollama:11434/api/generate

# Default: generate threat reports (Week 3). Assumes a trained checkpoint and
# Week 2 scores already exist in the mounted ./data and ./checkpoints.
# Override at run time, e.g.:  docker compose run labad python train.py
CMD ["python", "eval/run_week3.py"]
