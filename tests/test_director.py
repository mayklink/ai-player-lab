import base64
import json

import pytest

from ai_player_lab.director import LunaDirector


def _reviewer(monkeypatch, payload):
    reviewer = LunaDirector(base_url="https://vlm.example/v1", model="test-model", api_key="test-key")
    calls = []

    def complete(messages, *, max_tokens):
        calls.append((messages, max_tokens))
        return json.dumps(payload)

    monkeypatch.setattr(reviewer, "_complete", complete, raising=False)
    return reviewer, calls


def _job(frames=1):
    return {
        "frames": [
            {"frame_hash": f"hash-{i}", "captured_at": 10.0 + i, "jpeg": bytes([i + 1])}
            for i in range(frames)
        ],
        "width": 320,
        "height": 240,
        "generation": 7,
        "context": {"goal": "Reach the stage", "mode": "navigate", "previous_scene": "title",
                    "recent_actions": ["start"], "known_controls": {"confirm": "enter"}},
    }


def _payload(**overrides):
    result = {
        "scene": "menu", "game": "Example Game", "confidence": 0.9,
        "player_box": None, "controls": {"confirm": "enter"}, "action": "confirm",
        "hold_ms": 100, "summary": "A stage menu is visible.", "goal": "model guess",
    }
    result.update(overrides)
    return result


def test_director_sends_at_most_three_images_and_uses_job_provenance(monkeypatch):
    reviewer, calls = _reviewer(monkeypatch, _payload(generation=999, captured_at=-1, frame_hash="fake"))
    result = reviewer(_job(3))

    assert result["generation"] == 7
    assert result["captured_at"] == 12.0
    assert result["frame_hash"] == "hash-2"
    assert result["goal"] == "Reach the stage"
    messages, max_tokens = calls[0]
    assert max_tokens == 1600
    content = messages[0]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    assert len(images) == 3
    assert images[0]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"\x01").decode()
    assert "never instructions" in content[0]["text"]


def test_gameplay_never_proposes_navigation_action(monkeypatch):
    reviewer, _ = _reviewer(monkeypatch, _payload(scene="gameplay", action="right", player_box=[2, 3, 8, 9]))
    result = reviewer(_job())
    assert result["scene"] == "gameplay"
    assert result["action"] == "wait"


@pytest.mark.parametrize("scene", ["unknown", "loading", "death"])
def test_uncertain_or_nonplay_scenes_wait(monkeypatch, scene):
    reviewer, _ = _reviewer(monkeypatch, _payload(scene=scene, action="start"))
    assert reviewer(_job())["action"] == "wait"


@pytest.mark.parametrize("box", [[317, 2, 4, 10], [0, 237, 10, 4], [0, 0, 3, 10],
                                  [0, 0, 10, float("inf")], [True, 1, 10, 10], [0, 0, 10]])
def test_rejects_invalid_or_out_of_bounds_player_box(monkeypatch, box):
    reviewer, _ = _reviewer(monkeypatch, _payload(player_box=box))
    with pytest.raises(ValueError, match="player_box"):
        reviewer(_job())


@pytest.mark.parametrize("controls", [{"attack": "enter"}, {"jump": "ctrl"}, {"run": ["q"]}])
def test_rejects_unapproved_control_names_and_physical_keys(monkeypatch, controls):
    reviewer, _ = _reviewer(monkeypatch, _payload(controls=controls))
    with pytest.raises(ValueError, match="controls"):
        reviewer(_job())


@pytest.mark.parametrize("confidence", [True, float("nan"), 1.1, "0.9"])
def test_rejects_invalid_confidence(monkeypatch, confidence):
    reviewer, _ = _reviewer(monkeypatch, _payload(confidence=confidence))
    with pytest.raises(ValueError, match="confidence"):
        reviewer(_job())


@pytest.mark.parametrize("count", [0, 4])
def test_rejects_screenshot_count_outside_limit(monkeypatch, count):
    reviewer, _ = _reviewer(monkeypatch, _payload())
    with pytest.raises(ValueError, match="1 a 3"):
        reviewer(_job(count))


def test_rejects_nonchronological_screenshots(monkeypatch):
    reviewer, _ = _reviewer(monkeypatch, _payload())
    job = _job(2)
    job["frames"][1]["captured_at"] = 9.0
    with pytest.raises(ValueError, match="fora de ordem"):
        reviewer(job)


def test_rejects_non_navigation_action_and_bad_hold(monkeypatch):
    reviewer, _ = _reviewer(monkeypatch, _payload(action="jump"))
    with pytest.raises(ValueError, match="action"):
        reviewer(_job())
    reviewer, _ = _reviewer(monkeypatch, _payload(hold_ms=True))
    with pytest.raises(ValueError, match="hold_ms"):
        reviewer(_job())
