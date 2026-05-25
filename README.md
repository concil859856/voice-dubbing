# voice-dubbing

DeepFilterNet speech enhancement HTTP API for the Vocence platform.

Upload noisy audio → get clean, noise-reduced audio back. Uses DeepFilterNet3
(48 kHz full-band noise suppression).

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/enhance` | none | Upload audio file → enhanced WAV |
| GET | `/health` | none | Basic health check |
| GET | `/healthz` | bearer | Ops dashboard health probe |
| GET | `/metrics` | bearer | Ops dashboard metrics scrape |

## Quick start

```bash
docker build -t vocence/voice-dubbing:latest .
docker run -d --gpus all -p 8116:8116 -e DUBBING_API_KEY=mykey vocence/voice-dubbing:latest
curl -F audio=@noisy.wav http://localhost:8116/enhance -o clean.wav
```

## Environment

| Var | Default | Description |
|-----|---------|-------------|
| `PORT` | `8116` | HTTP port |
| `HOST` | `0.0.0.0` | Bind address |
| `DUBBING_API_KEY` | _(none)_ | Bearer token for /healthz + /metrics |
| `DUBBING_CAP` | `4` | Max concurrent enhance requests |
| `DUBBING_MODEL` | `DeepFilterNet3` | Model variant |
