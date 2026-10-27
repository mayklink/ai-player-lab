from __future__ import annotations

import cv2
import numpy as np

from ai_player_lab.event_evidence import enlarge_hud, evidence_summary, select_event_frames


def frame(i: int, *, scene: str = "gameplay", hazards=None, frame_hash=None):
    return {
        "frame_hash": frame_hash or f"hash-{i}",
        "captured_at": i / 8,
        "jpeg": bytes([i % 255]),
        "state": {
            "captured_at": i / 8,
            "scene": scene,
            "player": {"id": 1} if scene == "gameplay" else None,
            "hazards": hazards or [],
        },
    }


def test_collision_anchor_is_not_lost_in_long_sampled_history():
    history = [frame(i) for i in range(100)]
    history[92]["state"]["hazards"] = [{"kind": "visual_overlap", "ttc_ms": 0}]
    current = frame(100, scene="death")

    selected = select_event_frames(history, current)
    hashes = {item["frame_hash"] for item in selected}

    assert len(selected) <= 12
    assert current["frame_hash"] in hashes
    assert history[92]["frame_hash"] in hashes
    # Collision evidence retains a dense neighborhood around the anchor.
    around = [item for item in selected if abs(item["captured_at"] - history[92]["captured_at"]) <= 0.26]
    assert len(around) >= 4
    summary = evidence_summary(selected)
    assert summary["anchor_frame_hash"] == history[92]["frame_hash"]
    assert summary["max_gap_ms"] == max(
        (b["captured_at"] - a["captured_at"]) * 1000
        for a, b in zip(selected, selected[1:])
    )


def test_current_frame_is_mandatory_even_between_sampler_ticks():
    history = [frame(i) for i in range(0, 80, 8)]
    current = {
        "frame_hash": "actual-current",
        "captured_at": 9.91,
        "jpeg": b"current-map",
        "state": {"scene": "death", "hazards": []},
    }
    selected = select_event_frames(history, current)
    assert selected[-1]["frame_hash"] == "actual-current"
    assert selected[-1]["jpeg"] == b"current-map"


def test_hash_duplicates_collapse_and_current_copy_wins():
    history = [frame(0, frame_hash="same"), frame(8, frame_hash="same"), frame(16)]
    current = frame(17, scene="death", frame_hash="same")

    selected = select_event_frames(history, current)
    same = [item for item in selected if item["frame_hash"] == "same"]
    assert len(same) == 1
    assert same[0]["captured_at"] == current["captured_at"]
    assert [item["captured_at"] for item in selected] == sorted(item["captured_at"] for item in selected)


def test_first_gameplay_context_and_event_transition_are_preserved():
    history = [frame(0, scene="title"), frame(8), frame(16), frame(24, scene="death")]
    current = frame(25, scene="death")
    selected = select_event_frames(history, current)

    assert history[1]["frame_hash"] in {item["frame_hash"] for item in selected}
    assert history[3]["frame_hash"] in {item["frame_hash"] for item in selected}
    assert len(selected) <= 12


def test_hud_crop_is_top_40_rows_scaled_nearest_neighbor():
    image = np.zeros((60, 20, 3), dtype=np.uint8)
    image[:40] = (20, 100, 220)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    result = enlarge_hud(encoded.tobytes())
    assert result is not None
    decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (120, 60)
    assert enlarge_hud(b"not a jpeg") is None
