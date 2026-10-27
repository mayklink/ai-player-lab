import json

import pytest

from ai_player_lab.goal import GameGoal, parse_goal


@pytest.mark.parametrize(
    ("text", "mode", "target"),
    [
        ("chegar na fase 3", "reach", 3),
        ("chegue à fase 3", "reach", 3),
        ("concluir fase 3", "complete", 3),
        ("zerar", "win", None),
        ("vença", "win", None),
        ("vencer o jogo", "win", None),
    ],
)
def test_parse_supported_portuguese_goals(text, mode, target):
    goal = parse_goal(text)
    assert goal.description == text
    assert goal.mode == mode
    assert goal.target_stage == target
    assert goal.status == "pending"
    assert goal.evidence["verification"] == "unverified"


def test_unsupported_generic_goal_is_custom_without_invented_target():
    goal = GameGoal.from_text("explore o mundo e me surpreenda")
    assert goal.mode == "custom"
    assert goal.target_stage is None
    assert goal.status == "pending"
    assert goal.evidence == {"verification": "unverified", "reason": "unsupported_goal"}
    assert not goal.observe("gameplay", 3, {"source": "local_ocr"}, 0.5)
    assert goal.current_stage == 3
    assert goal.status == "pending"
    assert goal.evidence["verification"] == "unverified"


def test_reach_requires_two_distinct_confirmed_gameplay_observations():
    goal = parse_goal("chegar na fase 3")
    assert not goal.observe("gameplay", 3, {"source": "local_ocr"}, 1.0)
    assert goal.status == "running"
    # A repeated frame timestamp does not count as independent confirmation.
    assert not goal.observe("gameplay", 3, {"source": "local_ocr"}, 1.0)
    assert goal.status == "running"
    assert goal.observe("gameplay", 3, {"source": "labelled_template"}, 1.1)
    assert goal.status == "completed"
    assert goal.current_stage == 3


def test_map_header_alone_and_non_gameplay_scenes_do_not_complete_reach_goal():
    goal = parse_goal("chegue à fase 3")
    assert not goal.observe("map", 3, {"source": "local_ocr"}, 1.0)
    assert not goal.observe("map", 3, {"source": "local_ocr"}, 1.1)
    assert goal.status == "running"
    assert goal.current_stage == 3
    assert not goal.observe("loading", 3, {"source": "local_ocr"}, 1.2)
    assert goal.status == "running"


def test_far_stage_requires_same_visual_stage_to_be_confirmed_twice():
    goal = parse_goal("chegar na fase 3")
    assert not goal.observe("gameplay", 2, {"source": "local_ocr"}, 1.0)
    assert not goal.observe("gameplay", 4, {"source": "local_ocr"}, 1.1)
    assert goal.status == "running"
    assert not goal.observe("gameplay", 5, {"source": "local_ocr"}, 1.2)
    assert goal.status == "running"
    assert goal.observe("gameplay", 5, {"source": "local_ocr"}, 1.3)
    assert goal.status == "completed"


@pytest.mark.parametrize("source", ["luna", "model", "inference", "unknown"])
def test_model_or_untrusted_stage_label_never_advances_goal(source):
    goal = parse_goal("chegar na fase 3")
    for timestamp in (1.0, 1.1, 1.2):
        assert not goal.observe("gameplay", 3, {"source": source, "confidence": 1.0}, timestamp)
    assert goal.status == "pending"
    assert goal.current_stage is None
    assert goal.evidence["verification"] == "unverified"


def test_complete_goal_requires_entering_target_then_local_stage_complete():
    goal = parse_goal("concluir fase 3")
    assert not goal.observe("stage_complete", 3, {"source": "local_ocr"}, 1.0)
    assert goal.status == "running"
    assert not goal.observe("gameplay", 2, {"source": "local_ocr"}, 1.1)
    assert not goal.observe("stage_complete", 3, {"source": "labelled_template"}, 1.2)
    assert goal.status == "running"
    assert not goal.observe("gameplay", 3, {"source": "local_ocr"}, 1.3)
    assert goal.observe("stage_complete", 3, {"source": "local_ocr"}, 1.4)
    assert goal.status == "completed"


def test_win_requires_two_locally_verified_victory_observations():
    goal = parse_goal("vença")
    assert not goal.observe("victory", None, {"source": "model"}, 1.0)
    assert not goal.observe("victory", None, {"source": "labelled_template"}, 1.1)
    assert goal.status == "pending"
    assert goal.observe("victory", None, {"source": "local_ocr"}, 1.2)
    assert goal.status == "completed"


def test_out_of_order_timestamps_are_ignored_and_dict_is_json_friendly():
    goal = parse_goal("zerar")
    assert not goal.observe("victory", None, {"source": "local_ocr"}, 2.0)
    assert not goal.observe("victory", None, {"source": "local_ocr"}, 1.9)
    serialized = json.dumps(goal.to_dict(), ensure_ascii=False)
    assert '"mode": "win"' in serialized
    assert goal.status == "pending"
