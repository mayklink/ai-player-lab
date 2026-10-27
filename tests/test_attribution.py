from ai_player_lab.attribution import attribute_review


def _review(**changes):
    value = {
        "event_id": "death-1",
        "outcome": "death",
        "confidence": 0.9,
        "trigger_frame_id": "trigger-frame",
        "hypothesis": "The old memory says to jump, so jumping caused the death.",
    }
    value.update(changes)
    return value


def _candidate(**changes):
    value = {
        "frame_id": "trigger-frame",
        "scene": "gameplay",
        "captured_at": 10.0,
        "context": {"player": {"bbox": [100, 100, 20, 20]},
            'objects':[{'id':7,'bbox':[150,100,20,20],'age':5}]},
    }
    value.update(changes)
    return value


def _state(event_id="memory-1", action="jump", *, executed_at=10.1,
           captured_at=10.0, bbox=(102, 100, 20, 20), hypotheses=None):
    episode = {
        "source": "episodic_experience",
        "event_id": event_id,
        "suggested_action": "jump",
        "match_kind": "scene_and_objects",
        "match_evidence": {"motion_verified": True},
    }
    return {
        "executed_at": executed_at,
        "action": action,
        "hold_ms": 100,
        "state": {
            "captured_at": captured_at,
            "player": {"bbox": list(bbox)},
            'objects':[{'id':7,'bbox':[150,100,20,20],'age':5}],
            "hypotheses": hypotheses if hypotheses is not None else [episode],
        },
    }


def test_close_action_matches_but_labels_non_causal_association():
    result = attribute_review(_review(), [_candidate()], [_state()])

    assert result == [{
        "event_id": "memory-1",
        "action": "jump",
        "executed_at": 10.1,
        "trigger_time": 10.0,
        "attribution": "local_action_match_not_causal_proof",
    }]


def test_old_retrieved_memory_alone_does_not_match_later_action():
    old = _state(executed_at=8.0, captured_at=8.0)
    assert attribute_review(_review(), [_candidate()], [old]) == []


def test_mismatched_executed_action_does_not_match_suggestion():
    assert attribute_review(_review(), [_candidate()], [_state(action="right")]) == []


def test_nearby_different_encounter_is_not_blame_for_trigger():
    state=_state();state['state']['objects'][0]['id']=99
    assert attribute_review(_review(),[_candidate()],[state])==[]


def test_missing_trigger_and_unknown_scene_fail_closed():
    state = [_state()]
    assert attribute_review(_review(), [], state) == []
    assert attribute_review(_review(), [_candidate(scene="unknown")], state) == []


def test_review_text_is_not_enough_without_trigger_context():
    assert attribute_review(_review(), [_candidate(context={})], [_state()]) == []


def test_requires_motion_verified_episode_and_deduplicates_event_ids():
    invalid = {
        "source": "episodic_experience", "event_id": "memory-2",
        "suggested_action": "jump", "match_kind": "scene_only",
        "match_evidence": {"motion_verified": False},
    }
    first = _state(hypotheses=[invalid])
    second = _state(event_id="memory-1")
    assert [row["event_id"] for row in attribute_review(
        _review(), [_candidate()], [first, second, second]
    )] == ["memory-1"]
