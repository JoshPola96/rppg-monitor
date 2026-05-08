# rPPG_bpm/Dockerfile

# ─────────────────────────────────────────────────
# Wise AI rPPG Vitals Monitor — Dockerfile
# Targets Render's free/starter tier (Linux, x86-64)
# ─────────────────────────────────────────────────

FROM python:3.11-slim

# System libs required by OpenCV and PyAV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application files
COPY app.py .
COPY static/ static/

# Render injects PORT at runtime; default to 8000
ENV PORT=8000

# open-rppg downloads model weights on first use → pre-warm at build time
# This prevents cold-start model download in production
RUN python -c "import rppg; rppg.Model('FacePhys.rlap')" || true

# ─────────────────────────────────────────────────
# RAM & Thread Constraints for ML Frameworks
# ─────────────────────────────────────────────────
# Force CPU math libraries to use exactly 1 thread
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Stop JAX/TensorFlow from pre-allocating all available memory
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=3

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1"]