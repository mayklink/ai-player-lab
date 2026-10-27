from __future__ import annotations

import pytest

from ai_player_lab.action_replay import ActionReplay


def test_survival_only_prefix_is_not_authorized_for_replay():
    replay=ActionReplay()
    experience={'event_id':'prefix-1','outcome':'safe_prefix','status':'observed_prefix','match_kind':'scene_only',
        'successful_sequence':[{'action':'right','hold_ms':100,'offset_s':.1}],
        'success_evidence':{'kind':'observed_forward_segment','duration_s':4.,'forward_progress_px':48.,'survived_s':1.,'no_contact':True}}
    assert not replay.start(experience,100.,[100,160,20,24],4)
    assert not replay.active


def _experience(**overrides):
    result = {
        "event_id": "event-123",
        "outcome": "success",
        "status": "observed_success",
        "match_kind": "scene_and_objects",
        "motion_verified": True,
        "action_authorized": True,
        "success_evidence": {"kind":"tracked_object_passed","track_id":7,"no_contact":True,
            "survived_s":1.2,"forward_progress_px":48.,"separation_px":30.},
        "alternative_failed": False,
        "observed_at": 100.0,
        "match_evidence": {"nearby_count": 2},
        "sequence": [
            {"action": "right", "hold_ms": 100, "offset_s": 0.4},
            {"action": "jump", "hold_ms": 75, "offset_s": 0.8},
        ],
    }
    result.update(overrides)
    return result


def _prefix(**overrides):
    result = {
        "event_id": "prefix-456",
        "outcome": "safe_prefix",
        "status": "observed_prefix",
        "match_kind": "scene_only",
        "successful_sequence": True,
        "success_evidence": {"kind": "observed_forward_segment"},
        "duration_s": 6.0,
        "forward_progress_px": 48.0,
        "survived_s": 1.2,
        "no_contact": True,
        "observed_at": 100.0,
        "sequence": [{"action": "right", "hold_ms": 100, "offset_s": 0.4}],
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize("overrides", [
    {"outcome": "death"},
    {"outcome": "survived"},
    {"status": "unverified_cause"},
    {"status": "death"},
    {"match_kind": "scene_only"},
    {"motion_verified": False},
    {"alternative_failed": True},
    {"action_authorized": False},
    {"success_evidence": None},
    {"success_evidence": {"kind":"survived_without_progress"}},
])
def test_start_rejects_experiences_without_verified_observed_success(overrides):
    replay = ActionReplay()
    assert replay.start(_experience(**overrides), 100.0, [10, 20, 12, 12], episode=4) is False
    assert replay.active is False


@pytest.mark.parametrize("sequence", [
    [],
    [{"action": "right", "hold_ms": 251, "offset_s": 0}],
    [{"action": "shoot", "hold_ms": 100, "offset_s": 0}],
    [{"action": "right", "hold_ms": 100, "offset_s": -0.01}],
    [{"action": "right", "hold_ms": 100, "offset_s": 0.5},
     {"action": "left", "hold_ms": 100, "offset_s": 0.4}],
    [{"action": "right", "hold_ms": 100, "offset_s": 8.01}],
    [{"action": "right", "hold_ms": 250, "offset_s": 7.9}],
])
def test_start_rejects_invalid_or_overlong_sequences(sequence):
    replay = ActionReplay()
    assert replay.start(_experience(sequence=sequence), 100.0, [10, 20, 12, 12], 4) is False


def test_start_rejects_stale_experience():
    replay = ActionReplay()
    assert replay.start(_experience(observed_at=39.9), 100.0, [10, 20, 12, 12], 4) is False
    assert replay.active is False


@pytest.mark.parametrize("overrides", [
    {"status": "observed_success"},
    {"match_kind": "scene_and_objects"},
    {"successful_sequence": False},
    {"success_evidence": {"kind": "survived_without_progress"}},
    {"duration_s": 8.1},
    {"forward_progress_px": 23.9},
    {"survived_s": 0.99},
    {"no_contact": False},
])
def test_start_rejects_unproven_safe_prefix(overrides):
    replay = ActionReplay()
    assert replay.start(_prefix(**overrides), 100.0, [10, 20, 12, 12], 4) is False


def test_forward_survival_prefix_is_rejected_even_when_scene_is_clear():
    replay = ActionReplay()
    assert not replay.start(_prefix(), 100.0, [10, 20, 12, 12], 4)
    assert replay.next(100.4, 4, [10, 20, 12, 12], "gameplay", True, [], objects=[]) is None


@pytest.mark.parametrize("hazards,objects", [
    ([{"kind": "approaching_object", "ttc_ms": 500}], []),
    ([{"kind": "visual_overlap", "ttc_ms": 500}], []),
    ([{"kind": "gap"}], []),
    ([{"kind": "gap_candidate"}], []),
    ([], [{"bbox": [35, 24, 10, 12]}]),
])
def test_survival_prefix_stays_rejected_for_forward_hazards_or_objects(hazards, objects):
    replay = ActionReplay()
    assert not replay.start(_prefix(), 100.0, [10, 20, 12, 12], 4)
    assert replay.next(100.4, 4, [10, 20, 12, 12], "gameplay", True,
                      hazards, objects=objects) is None
    assert replay.active is False


def test_steps_are_deterministic_due_once_and_preserve_offsets():
    replay = ActionReplay()
    assert replay.start(_experience(), 100.0, [10, 20, 12, 12], 4)

    assert replay.next(100.399, 4, [10, 20, 12, 12], "gameplay", True, []) is None
    first = replay.next(100.4, 4, [10, 20, 12, 12], "gameplay", True, [])
    assert first == {"action": "right", "hold_ms": 100, "event_id": "event-123"}
    assert replay.active is True
    assert replay.next(100.4, 4, [10, 20, 12, 12], "gameplay", True, []) is None

    second = replay.next(100.8, 4, [10, 20, 12, 12], "gameplay", True, [])
    assert second == {"action": "jump", "hold_ms": 75, "event_id": "event-123"}
    assert replay.active is False
    assert replay.next(100.8, 4, [10, 20, 12, 12], "gameplay", True, []) is None


@pytest.mark.parametrize("kwargs", [
    {"scene": "menu", "stable": True, "playerbbox": [10, 20, 12, 12], "episode": 4, "hazards": []},
    {"scene": "gameplay", "stable": False, "playerbbox": [10, 20, 12, 12], "episode": 4, "hazards": []},
    {"scene": "gameplay", "stable": True, "playerbbox": None, "episode": 4, "hazards": []},
    {"scene": "gameplay", "stable": True, "playerbbox": [10, 20, 12, 12], "episode": 5, "hazards": []},
    {"scene": "gameplay", "stable": True, "playerbbox": [100, 20, 12, 12], "episode": 4, "hazards": []},
    {"scene": "gameplay", "stable": True, "playerbbox": [10, 20, 12, 12], "episode": 4,
     "hazards": [{"kind": "gap"}]},
    {"scene": "gameplay", "stable": True, "playerbbox": [10, 20, 12, 12], "episode": 4,
     "hazards": [{"kind": "approaching_object", "ttc_ms": 119}]},
    {"scene": "gameplay", "stable": True, "playerbbox": [10, 20, 12, 12], "episode": 4,
     "hazards": [{"kind": "visual_overlap", "ttc_ms": 300}]},
])
def test_safety_changes_cancel_replay(kwargs):
    replay = ActionReplay()
    assert replay.start(_experience(), 100.0, [10, 20, 12, 12], 4)
    assert replay.next(100.4, kwargs["episode"], kwargs["playerbbox"], kwargs["scene"],
                       kwargs["stable"], kwargs["hazards"]) is None
    assert replay.active is False


def test_late_action_cancels_without_replaying_backlog():
    replay = ActionReplay()
    assert replay.start(_experience(), 100.0, [10, 20, 12, 12], 4)
    # 200 ms late exceeds the allowed 150 ms window, so no old step is emitted.
    assert replay.next(100.6, 4, [10, 20, 12, 12], "gameplay", True, []) is None
    assert replay.active is False


def test_observed_success_can_replay_with_the_same_matched_hazard():
    replay = ActionReplay()
    experience = _experience(match_evidence={"nearby_count": 1, "relative_object_ids": [17]})
    assert replay.start(experience, 100.0, [10, 20, 12, 12], 4)
    result = replay.next(100.4, 4, [10, 20, 12, 12], "gameplay", True,
                        [{"kind": "approaching_object", "object_id": 17, "ttc_ms": 100}])
    assert result == {"action": "right", "hold_ms": 100, "event_id": "event-123"}


def test_replay_cancels_after_eight_seconds():
    replay = ActionReplay()
    assert replay.start(_experience(sequence=[{"action": "right", "hold_ms": 100, "offset_s": 7.9}]),
                        100.0, [10, 20, 12, 12], 4)
    assert replay.next(108.01, 4, [10, 20, 12, 12], "gameplay", True, []) is None
    assert replay.active is False
