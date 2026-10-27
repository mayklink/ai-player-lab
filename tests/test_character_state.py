from types import SimpleNamespace

import cv2
import numpy as np

from ai_player_lab.character_state import CharacterStateEstimator


def _frame(y=100, floor=True):
    image = np.full((224, 256, 3), (190, 135, 80), np.uint8)
    if floor:
        # Support line across the inset foot width, three pixels under bbox.
        cv2.line(image, (103, y + 25), (120, y + 25), (15, 20, 25), 1)
    return image


def _obs(y=100, time=0.0, vx=0, vy=0, objects=None):
    return SimpleNamespace(
        captured_at=time,
        player=SimpleNamespace(bbox=(100, y, 24, 24), vx=vx, vy=vy),
        objects=objects or [],
    )


def test_empty_blue_pixels_never_confirm_grounded_without_support():
    estimator = CharacterStateEstimator()
    results = [estimator.update(_frame(floor=False), _obs(time=i / 30), set()) for i in range(8)]
    assert all(result["grounded"] is None for result in results)
    assert all(result["ground_support"] is False for result in results)
    assert results[-1]["can_start_jump"] is None


def test_uncommanded_vertical_bbox_changes_do_not_invent_a_jump():
    estimator=CharacterStateEstimator()
    states=[estimator.update(_frame(y,floor=False),_obs(y,i/30),set())
        for i,y in enumerate([171,181,179,172,168,127,110,173])]
    assert all(s['jump_phase']=='unknown' and s['can_start_jump'] is None for s in states)
    assert all(s['rise_samples']==0 for s in states)


def test_visual_floor_requires_stability_then_reports_rising_falling_and_release():
    estimator = CharacterStateEstimator({"jump_rise_ms": 180})
    stable = [estimator.update(_frame(), _obs(time=i / 30), set()) for i in range(4)]
    assert stable[-1]["grounded"] is True
    assert stable[-1]["grounded_evidence"] == "visual_estimate"
    assert stable[-1]["can_start_jump"] is True

    estimator.update(_frame(100),_obs(100,4/30),{'x'})
    first=estimator.update(_frame(94, floor=False), _obs(94, 5 / 30), {"x"})
    assert first['jump_phase']=='unknown'
    rising = estimator.update(_frame(86, floor=False), _obs(86, 6 / 30), {"x"})
    assert rising["jump_phase"] == "rising"
    assert rising["grounded"] is None
    assert rising["can_start_jump"] is False
    released = estimator.update(_frame(78, floor=False), _obs(78, 7 / 30), set())
    assert released["jump_key_released"] is True
    falling = estimator.update(_frame(84, floor=False), _obs(84, 8 / 30), set())
    assert falling["jump_phase"] == "falling"


def test_approach_prediction_and_deadline_include_configured_delay_and_rise():
    enemy = SimpleNamespace(id=7, bbox=(100, 100, 10, 12), vx=-100, vy=10)
    obs = SimpleNamespace(captured_at=0.0, player=SimpleNamespace(bbox=(0, 100, 10, 15), vx=0), objects=[enemy])
    estimator = CharacterStateEstimator({"jump_rise_ms": 180})
    # Prime a confirmed visual-grounded state so the decision flag can be true.
    for i in range(4):
        obs.captured_at = i / 30
        state = estimator.update(_frame(100, floor=True), obs, set(), actuation_delay_ms=300)
    approach = state["approaching_object"]
    assert approach["ttc_ms"] == 900.0
    assert approach["jump_deadline_ms"] == 420.0
    assert approach["predicted_object_box"] == [70.0, 103.0, 10.0, 12.0]
    assert approach["predicted_player_box"] == [0.0, 100.0, 10.0, 15.0]

    obs.captured_at += 1 / 30
    slower_delay = estimator.update(_frame(100), obs, set(), actuation_delay_ms=100)["approaching_object"]
    assert slower_delay["jump_deadline_ms"] > approach["jump_deadline_ms"]


def test_loss_reset_clears_motion_but_preserves_observed_rise_calibration():
    estimator = CharacterStateEstimator()
    # Actual key edge followed by observed >16px upward movement in 100 ms.
    estimator.update(_frame(100, floor=False), _obs(100, 1.0), {"x"})
    estimator.update(_frame(94, floor=False), _obs(94, 1.04), {'x'})
    learned = estimator.update(_frame(82, floor=False), _obs(82, 1.1), {"x"})
    assert learned["rise_samples"] == 1
    assert learned["jump_rise_time_ms"] == 100.0
    estimator.reset()
    assert estimator.update(_frame(), SimpleNamespace(player=None), set())["jump_phase"] == "unknown"
    after_reset = estimator.update(_frame(100, floor=False), _obs(100, 2.0), set())
    assert after_reset["jump_rise_time_ms"] == 100.0
    assert after_reset["jump_rise_time_source"] == "observed"


def test_stationary_held_jump_does_not_fabricate_rise_sample():
    estimator = CharacterStateEstimator({"jump_rise_ms": None})
    states = [estimator.update(_frame(100, floor=False), _obs(100, 1 + i / 30), {"x"}) for i in range(8)]
    assert states[-1]["rise_samples"] == 0
    assert states[-1]["jump_rise_time_ms"] is None


def test_invalid_rise_parameter_keeps_deadline_unknown():
    enemy = SimpleNamespace(id=7, bbox=(100, 100, 10, 12), vx=-100, vy=0)
    obs = SimpleNamespace(captured_at=1.0, player=SimpleNamespace(bbox=(0, 100, 10, 15), vx=0), objects=[enemy])
    approach = CharacterStateEstimator({"jump_rise_ms": "invalid"}).update(
        _frame(100), obs, set(), actuation_delay_ms=100
    )["approaching_object"]
    assert approach["jump_deadline_ms"] is None
    assert approach["deadline_known"] is False
    assert approach["too_late"] is None


def test_jump_prompt_threshold_and_too_late_flag():
    estimator = CharacterStateEstimator({"jump_rise_ms": 180})
    enemy = SimpleNamespace(id=3, bbox=(160, 100, 10, 12), vx=-100, vy=0)
    obs = SimpleNamespace(captured_at=0.0, player=SimpleNamespace(bbox=(100, 100, 24, 24), vx=0), objects=[enemy])
    for index in range(4):
        obs.captured_at = index / 30
        state = estimator.update(_frame(), obs, set(), actuation_delay_ms=100)
    approach = state["approaching_object"]
    assert approach["jump_deadline_ms"] == 80.0
    assert approach["jump_must_start_this_decision"] is True
    assert approach["too_late"] is False

    obs.objects = [SimpleNamespace(id=4, bbox=(130, 100, 10, 12), vx=-100, vy=0)]
    obs.captured_at += 1 / 30
    late = estimator.update(_frame(), obs, set(), actuation_delay_ms=100)["approaching_object"]
    assert late["jump_deadline_ms"] < 0
    assert late["too_late"] is True


def test_one_pixel_box_jitter_on_visible_support_keeps_ground_available():
    estimator=CharacterStateEstimator()
    states=[estimator.update(_frame(y),_obs(y,i/60),set())
            for i,y in enumerate([100,101,100,101,100,101,100,101])]
    assert states[-1]['grounded'] is True
    assert states[-1]['can_start_jump'] is True
