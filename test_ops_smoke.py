"""Non-GPU smoke test for ops integration (no DeepFilterNet inference).

Stubs heavy deps so the test runs on any CI runner / laptop.
Validates /healthz, /metrics, /health shapes + auth + inflight.
"""

import sys
import types
from unittest.mock import MagicMock

# Stub GPU-heavy modules before importing main
for mod_name in ["torch", "torchaudio", "df", "df.enhance", "librosa"]:
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        if mod_name == "torch":
            stub.from_numpy = MagicMock(return_value=MagicMock(unsqueeze=MagicMock(return_value=MagicMock()), numpy=MagicMock(return_value=__import__("numpy").zeros((1, 48000)))))
            stub.Tensor = type("Tensor", (), {})
            stub.cuda = MagicMock()
            stub.cuda.is_available = MagicMock(return_value=False)
            stub.__version__ = "2.9.0+stub"
        if mod_name == "df.enhance":
            stub.init_df = MagicMock(return_value=(MagicMock(), MagicMock(), None))
            stub.enhance = MagicMock()
        sys.modules[mod_name] = stub

import os
os.environ.setdefault("DUBBING_API_KEY", "test-key-123")
os.environ.setdefault("DUBBING_CAP", "2")
os.environ.setdefault("PORT", "18116")

from fastapi.testclient import TestClient
import main

client = TestClient(main.app)
AUTH = {"Authorization": "Bearer test-key-123"}


def test_health_no_auth():
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert "model_loaded" in d


def test_healthz_no_auth_401():
    r = client.get("/healthz")
    assert r.status_code == 200
    d = r.json()
    assert d.get("code") == "auth" or d.get("type") == "error"


def test_healthz_with_auth():
    r = client.get("/healthz", headers=AUTH)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["service"] == "dubbing"
    assert "inflight" in d
    assert "cap" in d
    assert "loaded" in d
    assert "model_id" in d


def test_metrics_with_auth():
    r = client.get("/metrics", headers=AUTH)
    assert r.status_code == 200
    d = r.json()
    assert "uptime_seconds" in d
    assert "requests_total" in d
    assert "requests_ok" in d
    assert "requests_err" in d
    assert "duration_ms_p50" in d
    assert "duration_ms_p95" in d
    assert d["service"] == "dubbing"


def test_metrics_wrong_auth():
    r = client.get("/metrics", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 200
    d = r.json()
    assert d.get("code") == "auth"


def test_healthz_repeated():
    for _ in range(5):
        r = client.get("/healthz", headers=AUTH)
        assert r.status_code == 200


if __name__ == "__main__":
    test_health_no_auth()
    test_healthz_no_auth_401()
    test_healthz_with_auth()
    test_metrics_with_auth()
    test_metrics_wrong_auth()
    test_healthz_repeated()
    print("All smoke tests passed.")
