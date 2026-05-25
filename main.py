"""
Voice Dubbing — DeepFilterNet speech enhancement HTTP API.

Upload noisy audio → get clean, noise-reduced audio back.
Uses DeepFilterNet3 (48 kHz full-band noise suppression).

Vocence /studio/ops integration: /healthz + /metrics with bearer auth,
inflight cap middleware, same patterns as voice_clone / text-to-music.
"""

from __future__ import annotations

import asyncio
import collections
import io
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("voice-dubbing")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("PORT") or "8116")
HOST = os.environ.get("HOST") or "0.0.0.0"
API_KEY = (os.environ.get("DUBBING_API_KEY") or "").strip()
CAP = int(os.environ.get("DUBBING_CAP") or "4")
MODEL_NAME = os.environ.get("DUBBING_MODEL") or "DeepFilterNet3"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Voice Dubbing (DeepFilterNet)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model (lazy-loaded on first request)
# ---------------------------------------------------------------------------

_model = None
_df_state = None
_model_lock = threading.Lock()


def _ensure_model():
    global _model, _df_state
    if _model is not None:
        return
    with _model_lock:
        if _model is not None:
            return
        _log.info("Loading %s model...", MODEL_NAME)
        t0 = time.perf_counter()
        from df.enhance import enhance, init_df
        _model, _df_state, _ = init_df()
        _log.info("Model loaded in %.1fs", time.perf_counter() - t0)


def _enhance_audio(audio_bytes: bytes, filename: str = "input.wav") -> bytes:
    """Run DeepFilterNet enhancement on raw audio bytes. Returns WAV bytes."""
    _ensure_model()
    from df.enhance import enhance

    # Read input audio
    buf = io.BytesIO(audio_bytes)
    try:
        audio, sr = sf.read(buf, dtype="float32")
    except Exception:
        buf.seek(0)
        import librosa
        audio, sr = librosa.load(buf, sr=None, mono=False)

    # DeepFilterNet expects 48kHz
    if sr != 48000:
        import librosa
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        else:
            audio = np.stack([
                librosa.resample(audio[ch], orig_sr=sr, target_sr=48000)
                for ch in range(audio.shape[0])
            ])
        sr = 48000

    # Convert to torch tensor: (channels, samples)
    if audio.ndim == 1:
        tensor = torch.from_numpy(audio).unsqueeze(0)
    else:
        if audio.shape[0] > audio.shape[1]:
            audio = audio.T
        tensor = torch.from_numpy(audio)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

    # Enhance
    enhanced = enhance(_model, _df_state, tensor)

    # Convert back to numpy and write WAV
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.numpy()
    if enhanced.ndim == 2:
        enhanced = enhanced.T  # (samples, channels) for soundfile

    out_buf = io.BytesIO()
    sf.write(out_buf, enhanced, sr, format="WAV", subtype="PCM_16")
    return out_buf.getvalue()


# ---------------------------------------------------------------------------
# Ops integration: metrics + inflight + bearer auth
# ---------------------------------------------------------------------------

class _Metrics:
    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.start_ts = time.time()
        self.requests_total = 0
        self.requests_ok = 0
        self.requests_err: dict[str, int] = {}
        self.duration_ms_sum = 0.0
        self.duration_ms_count = 0
        self.recent: collections.deque[float] = collections.deque(maxlen=window)
        self.bytes_sent_total = 0

    def record_success(self, ms: float, bytes_sent: int = 0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_ok += 1
            self.duration_ms_sum += ms
            self.duration_ms_count += 1
            self.recent.append(ms)
            self.bytes_sent_total += bytes_sent

    def record_error(self, code: str, ms: float = 0.0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_err[code] = self.requests_err.get(code, 0) + 1
            if ms > 0:
                self.duration_ms_sum += ms
                self.duration_ms_count += 1
                self.recent.append(ms)

    def snapshot(self) -> dict:
        with self._lock:
            d = sorted(self.recent)
            n = len(d)
            def pct(p: float) -> float:
                return d[min(n - 1, int(p * n))] if n else 0.0
            return {
                "uptime_seconds": int(time.time() - self.start_ts),
                "requests_total": self.requests_total,
                "requests_ok": self.requests_ok,
                "requests_err": dict(self.requests_err),
                "duration_ms_sum": self.duration_ms_sum,
                "duration_ms_count": self.duration_ms_count,
                "duration_ms_avg": (self.duration_ms_sum / self.duration_ms_count) if self.duration_ms_count else 0.0,
                "duration_ms_p50": pct(0.50),
                "duration_ms_p95": pct(0.95),
                "duration_ms_p99": pct(0.99),
                "bytes_sent_total": self.bytes_sent_total,
            }


_metrics = _Metrics()


class _InflightTracker:
    def __init__(self, cap: int) -> None:
        self._cap = max(1, cap)
        self._count = 0
        self._lock = asyncio.Lock()

    @property
    def cap(self) -> int: return self._cap

    @property
    def inflight(self) -> int: return self._count

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._count >= self._cap:
                return False
            self._count += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._count = max(0, self._count - 1)


_inflight = _InflightTracker(cap=CAP)

_OPS_PATHS = {"/healthz", "/metrics", "/health", "/docs", "/redoc", "/openapi.json"}


def _check_bearer(request: Request) -> Optional[JSONResponse]:
    if not API_KEY:
        return None
    if request.headers.get("authorization", "") == f"Bearer {API_KEY}":
        return None
    return JSONResponse(
        {"type": "error", "code": "auth", "message": "missing or invalid bearer token"},
        status_code=401,
    )


@app.middleware("http")
async def _ops_middleware(request: Request, call_next):
    path = request.url.path
    if path in _OPS_PATHS:
        return await call_next(request)
    if not await _inflight.try_acquire():
        _metrics.record_error("server_busy", 0)
        return JSONResponse(
            {"type": "error", "code": "server_busy", "message": f"inflight cap ({_inflight.cap}) exhausted"},
            status_code=503,
        )
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if response.status_code < 400:
            _metrics.record_success(elapsed)
        else:
            _metrics.record_error(f"http_{response.status_code}", elapsed)
        return response
    except Exception as e:
        _metrics.record_error(f"exception_{type(e).__name__}", (time.perf_counter() - t0) * 1000.0)
        raise
    finally:
        await _inflight.release()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/healthz")
async def healthz(request: Request):
    err = _check_bearer(request)
    if err is not None:
        return err
    return JSONResponse({
        "status": "ok",
        "service": "dubbing",
        "model_id": MODEL_NAME,
        "sample_rate": 48000,
        "inflight": _inflight.inflight,
        "cap": _inflight.cap,
        "loaded": _model is not None,
        "dev_stub": False,
    })


@app.get("/metrics")
async def metrics_endpoint(request: Request):
    err = _check_bearer(request)
    if err is not None:
        return err
    snap = _metrics.snapshot()
    snap["service"] = "dubbing"
    snap["inflight"] = _inflight.inflight
    snap["cap"] = _inflight.cap
    return JSONResponse(snap)


@app.post("/enhance")
async def enhance_audio(
    audio: UploadFile = File(..., description="Noisy audio file (WAV, MP3, M4A, etc.)"),
    output_format: str = Form("wav"),
):
    """Upload noisy audio, get back enhanced/denoised audio."""
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(raw) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100 MB)")

    try:
        enhanced_wav = await asyncio.get_event_loop().run_in_executor(
            None, _enhance_audio, raw, audio.filename or "input.wav"
        )
    except Exception as e:
        _log.exception("Enhancement failed")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {e}")

    return Response(
        content=enhanced_wav,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="enhanced.wav"',
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
