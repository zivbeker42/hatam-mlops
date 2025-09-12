# syntax=docker/dockerfile:1

# CUDA-enabled PyTorch runtime (includes Python & torch)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Basics
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (audio + utils)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Workdir maps to your git project when mounted
WORKDIR /workspace

# Copy the project in (so "pip install -r requirements.txt" can work if present)
# When you run the container, you'll typically mount your live project over this.
COPY . /workspace

# Python deps:
# - JupyterLab for notebooks
# - torchaudio/librosa are common with audio ML
# - If requirements.txt exists, install it (preferred)
RUN pip install --upgrade pip && \
    pip install jupyterlab ipykernel && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    # Ensure CUDA-matched wheels for torchaudio; librosa for feature extraction
    pip install --index-url https://download.pytorch.org/whl/cu121 torchaudio && \
    pip install librosa

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab (no token/password; bind on 0.0.0.0)
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'"]
