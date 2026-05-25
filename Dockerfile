# syntax=docker/dockerfile:1.7
#
# voice-dubbing — DeepFilterNet speech enhancement HTTP API for the
# Vocence /studio/ops fleet manager.
#
# Build:
#   docker build -t vocence/voice-dubbing:latest .
# Run:
#   docker run -d --gpus all --restart=unless-stopped -p 8116:8116 \
#     -e DUBBING_API_KEY=<key> \
#     vocence/voice-dubbing:latest

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8116

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        build-essential \
        git curl ca-certificates \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# Install torch + torchaudio from the cu126 index FIRST with exact
# pins. DeepFilterNet needs torchaudio.backend.common which was
# removed in torchaudio 2.6+, so we pin to 2.5.1. Then install
# the rest from PyPI with a constraints file so pip doesn't
# re-resolve torch from a different index.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu126 \
         torch==2.5.1 torchaudio==2.5.1 \
    && pip freeze | grep -iE '^(torch|torchaudio)' > /tmp/torch-pin.txt \
    && pip install -r requirements.txt -c /tmp/torch-pin.txt

COPY main.py ./

# Build-time import smoke test — catches dep mismatches before
# the image reaches Docker Hub.
RUN python3 -c "import torch; print('torch', torch.__version__)" \
    && python3 -c "import df; print('deepfilternet OK')" \
    && python3 -c "import main; print('main import OK')"

EXPOSE 8116

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8116}/health || exit 1

CMD ["python3", "main.py"]
