"""API-only configuration and spending guards; all HTTP traffic is simulated."""
import json
from pathlib import Path

import httpx
import pytest

from ai_player_lab.diagnostics import machine_report
from ai_player_lab.providers import MultimodalReviewer
from ai_player_lab.runtime import run_session


@pytest.fixture(autouse=True)
def clear_api_environment(monkeypatch):
    for name in ("VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def job():
    return {"event_id": "event-api", "event": "death_marked", "states": [],
            "frames": [{"frame_hash": "frame-a", "captured_at": 1.0,
                        "jpeg": b"synthetic-contract-bytes"}]}


def response():
    return {"usage": {"total_tokens": 23}, "choices": [{"message": {"content": json.dumps({
        "hypothesis": "A causa da morte permanece incerta.", "confidence": 0.2,
        "evidence_frame_ids": ["frame-a"], "suggested_action": "wait"
    })}}]}


def test_no_local_or_remote_fallback_without_configuration():
    with pytest.raises(ValueError, match="VLM_BASE_URL"):
        MultimodalReviewer()


def test_environment_selects_endpoint_model_and_authorization(monkeypatch):
    monkeypatch.setenv("VLM_BASE_URL", "https://vision.test/v1/")
    monkeypatch.setenv("VLM_MODEL", "selected-model")
    monkeypatch.setenv("VLM_API_KEY", "secret-test-value")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response())

    client = MultimodalReviewer(transport=httpx.MockTransport(handle))
    try:
        result = client(job())
        assert str(requests[0].url) == "https://vision.test/v1/chat/completions"
        assert requests[0].headers["Authorization"] == "Bearer secret-test-value"
        body = json.loads(requests[0].content)
        assert body["model"] == "selected-model"
        assert "secret-test-value" not in json.dumps(result)
        assert client.calls == 1 and client.last_usage == {"total_tokens": 23}
    finally:
        client.close()


@pytest.mark.parametrize("url", [
    "http://vision.test/v1", "http://127.0.0.1:8080/v1",
    "https://name:secret@vision.test/v1", "https://vision.test/v1?key=secret",
    "https://vision.test/v1#fragment", "https:///v1"
])
def test_invalid_or_unsafe_api_url_rejected(url):
    with pytest.raises(ValueError, match="HTTPS"):
        MultimodalReviewer(url, "model", api_key="test-key")


def test_explicit_model_required():
    with pytest.raises(ValueError, match="VLM_MODEL"):
        MultimodalReviewer("https://vision.test/v1", api_key="test-key")


def test_remote_authentication_required():
    with pytest.raises(ValueError, match="VLM_API_KEY"):
        MultimodalReviewer("https://vision.test/v1", "model")


def test_call_budget_prevents_additional_http():
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response())
    client = MultimodalReviewer("https://vision.test/v1", "model", api_key="key",
                                max_calls=1, transport=httpx.MockTransport(handle))
    try:
        client(job())
        with pytest.raises(RuntimeError, match="Limite"):
            client(job())
        assert len(requests) == 1 and client.calls == 1
    finally:
        client.close()


def test_http_failure_counts_and_does_not_retry():
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(429, json={"error": "rate limited"})
    client = MultimodalReviewer("https://vision.test/v1", "model", api_key="key",
                                max_calls=1, transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client(job())
        with pytest.raises(RuntimeError):
            client(job())
        assert len(requests) == 1 and client.calls == 1
    finally:
        client.close()


def test_timeout_counts_against_budget_without_retry():
    calls = []
    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("simulated", request=request)
    client = MultimodalReviewer("https://vision.test/v1", "model", api_key="key",
                                max_calls=1, transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(httpx.ReadTimeout):
            client(job())
        assert client.calls == 1 and len(calls) == 1
    finally:
        client.close()


def test_redirect_does_not_forward_credentials():
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(307, headers={"location": "https://other.test/collect"})
    client = MultimodalReviewer("https://vision.test/v1", "model", api_key="key",
                                transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client(job())
        assert len(calls) == 1
    finally:
        client.close()


def test_config_can_defer_endpoint_model_to_environment(monkeypatch):
    monkeypatch.setenv("VLM_BASE_URL", "https://vision.test/v1")
    monkeypatch.setenv("VLM_MODEL", "remote-model")
    monkeypatch.setenv("VLM_API_KEY", "key")
    client = MultimodalReviewer.from_config({"vlm_base_url": "", "vlm_model": "", "max_vlm_calls": 2})
    try:
        assert client.model == "remote-model" and client.max_calls == 2
    finally:
        client.close()


def test_missing_api_configuration_fails_before_workers_or_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"capture_region": [0, 0, 320, 180]}))
    with pytest.raises(ValueError, match="VLM_BASE_URL"):
        run_session(cfg, seconds=1, vlm=True, headless=True)
    assert not (tmp_path / "runs").exists()


def test_diagnostics_never_record_secret_values(monkeypatch):
    monkeypatch.setenv("VLM_API_KEY", "private-test-key")
    report = machine_report()
    assert report["model_execution"] == "remote_api"
    assert report["vlm_key_present"] is True
    assert "private-test-key" not in json.dumps(report)
    assert "llama_server_found" not in report


def test_default_profile_has_no_local_inference_dependency():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/mario-observe.json").read_text())
    assert cfg["vlm_base_url"] == cfg["vlm_model"] == ""
    assert cfg["max_vlm_calls"] == 12
    assert not (root / "scripts/start-vlm.ps1").exists()
