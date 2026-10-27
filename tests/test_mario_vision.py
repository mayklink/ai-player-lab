
import pytest as _asset_pytest
from pathlib import Path as _AssetPath
if not (_AssetPath(__file__).parent / "fixtures" / "small-mario-stopped.png").is_file():
    _asset_pytest.skip("Optional game imagery is not distributed; see docs/ASSETS.md", allow_module_level=True)

from pathlib import Path

import cv2
import numpy as np
import pytest

from ai_player_lab.mario_vision import LocalMarioVision, detect_player
from ai_player_lab import mario_vision

FIXTURES = Path(__file__).parent / "fixtures"


def _frame(color=(95, 165, 80)):
    return np.full((224, 256, 3), color, dtype=np.uint8)


def _draw_sprite(frame, x=105, y=145):
    # Discrete pixel-art palette patches: red cap, skin face, blue overalls.
    cv2.rectangle(frame, (x + 2, y), (x + 10, y + 3), (0, 0, 255), -1)
    cv2.rectangle(frame, (x + 5, y + 4), (x + 11, y + 8), (120, 180, 240), -1)
    cv2.rectangle(frame, (x + 3, y + 9), (x + 10, y + 15), (255, 0, 0), -1)
    return frame


def _shift_sprite_patch(frame, box, dx=4, dy=2, cover_original=True):
    x, y, w, h = box
    shifted = frame.copy()
    patch = frame[y:y+h, x:x+w].copy()
    if cover_original:
        background = np.median(frame[max(0, y-4):y, x:x+w], axis=(0, 1)).astype(np.uint8)
        shifted[y:y+h, x:x+w] = background
    shifted[y+dy:y+dy+h, x+dx:x+dx+w] = patch
    return shifted


def _vision_with_gameplay_hud(tmp_path, frame):
    mask = np.zeros((224, 256), np.uint8)
    mask[:35, :] = 255
    setup = LocalMarioVision(tmp_path)
    setup.register_template("gameplay", frame, "hud-only", mask=mask)
    return LocalMarioVision(tmp_path)


def test_unknown_blue_scene_is_never_promoted_to_gameplay():
    vision = LocalMarioVision()
    blue_hub = _frame((210, 130, 70))
    first = vision.observe(blue_hub, 0.0)
    second = vision.observe(blue_hub, 0.1)
    assert first["scene"] == second["scene"] == "unknown"
    assert first["evidence"] == second["evidence"] == {}
    assert first["stable"] is False and second["stable"] is False


@pytest.mark.parametrize("scene", ["hub", "intro"])
def test_human_labelled_hub_and_intro_templates_are_not_gameplay(tmp_path: Path, scene):
    sample = _frame((220, 150, 95))
    if scene == "hub":
        # Tree canopy and several orange/red fruit are a common hub false positive.
        cv2.ellipse(sample, (128, 105), (105, 45), 0, 0, 360, (0, 220, 0), -1)
        for x, y in [(30, 105), (75, 130), (155, 110), (205, 135)]:
            cv2.circle(sample, (x, y), 7, (0, 90, 245), -1)
    else:
        cv2.putText(sample, "INTRO", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    vision = LocalMarioVision(tmp_path)
    # Variation only in the HUD/animation strip masked by the human profile.
    mask = np.zeros((224, 256), np.uint8)
    mask[20:, :] = 255
    vision.register_template(scene, sample, "known", mask=mask)
    vision = LocalMarioVision(tmp_path)
    changed = sample.copy()
    changed[:20] = (0, 0, 0)
    result = vision.observe(changed, 1.0)
    assert result["scene"] == scene
    assert result["scene"] != "gameplay"
    assert result["evidence"]["source"] == "labelled_template"
    assert result["stable"] is False


def test_registered_templates_need_two_matching_frames_for_stability(tmp_path: Path):
    vision = LocalMarioVision(tmp_path)
    example = _frame((18, 30, 210))
    vision.register_template("gameplay", example, "level-a")
    first = vision.observe(example, 2.0)
    second = vision.observe(example, 2.1)
    assert first["scene"] == second["scene"] == "gameplay"
    assert first["stable"] is False
    assert second["stable"] is True
    assert first["evidence"]["template"] == "gameplay__level-a"


def test_conflicting_near_identical_scene_templates_stay_unknown(tmp_path: Path):
    vision = LocalMarioVision(tmp_path)
    base = _frame((25, 70, 145))
    almost_same = base.copy()
    almost_same[120, 120] = (26, 70, 145)
    vision.register_template("hub", base, "one")
    vision.register_template("intro", almost_same, "two")
    result = vision.observe(base, 2.2)
    assert result["scene"] == "unknown"
    assert result["evidence"] == {}


def test_black_loading_rule_and_recovery_from_unknown():
    vision = LocalMarioVision()
    loading = vision.observe(np.zeros((448, 512, 3), np.uint8), 3.0)
    unknown = vision.observe(_frame((100, 125, 80)), 3.1)
    assert loading["scene"] == "loading"
    assert loading["evidence"]["rule"] == "near_black_frame"
    assert loading["stable"] is False
    assert unknown["scene"] == "unknown" and unknown["stable"] is False


def test_player_palette_detector_rejects_large_red_fruit():
    frame = _frame()
    # Red/orange objects have no skin and blue palette combination.
    for x, y in [(20, 80), (60, 100), (180, 90)]:
        cv2.ellipse(frame, (x, y), (11, 7), 0, 0, 360, (0, 90, 245), -1)
    _draw_sprite(frame)
    box = detect_player(frame)
    assert box is not None
    assert 105 <= box[0] <= 112
    assert 140 <= box[1] <= 149
    assert box[2] <= 23 and box[3] <= 32


def test_player_detector_resizes_canvas_and_returns_none_without_palette_candidate():
    no_player = _frame((30, 80, 120))
    assert detect_player(cv2.resize(no_player, (512, 448))) is None
    assert detect_player(_draw_sprite(_frame())) is not None


def test_small_mario_in_stopped_browser_frame_is_detected():
    # Raw canvas pixels from the run where the palette detector lost small Mario.
    frame = cv2.imread(str(FIXTURES / "small-mario-stopped.png"), cv2.IMREAD_COLOR)
    assert frame is not None
    box = detect_player(frame)
    assert box is not None
    assert 148 <= box[0] <= 153
    assert 169 <= box[1] <= 173
    assert box[2:] == [20, 24]


def test_player_detector_prefers_mario_to_intro_fruit():
    # This raw intro frame has two conspicuous fruits near the shrubs. Keep the
    # Mario candidate at x~133 rather than selecting either fruit as the player.
    frame = cv2.imread(str(FIXTURES / "intro-mario-fruit.png"), cv2.IMREAD_COLOR)
    assert frame is not None
    box = detect_player(frame)
    assert box is not None
    assert 125 <= box[0] <= 142


def test_player_detector_rejects_a_dark_outlined_red_fruit():
    frame = _frame()
    cv2.circle(frame, (105, 155), 6, (15, 25, 35), -1)
    cv2.circle(frame, (105, 155), 4, (0, 0, 255), -1)
    assert detect_player(frame) is None


def test_player_detector_does_not_reuse_previous_box_when_current_frame_is_empty():
    frame = cv2.imread(str(FIXTURES / "transition-no-player.jpg"), cv2.IMREAD_COLOR)
    assert frame is not None
    assert detect_player(frame, previous_box=[150, 171, 20, 24]) is None


def test_user_reported_rex_is_not_a_player():
    frame=cv2.imread(str(FIXTURES/'rex-misidentified.png'))
    assert frame is not None
    assert detect_player(frame) is None


def test_player_identity_cannot_teleport_to_another_sprite():
    frame=_draw_sprite(_frame(),x=160,y=145)
    assert detect_player(frame) is not None
    assert detect_player(frame,previous_box=[20,145,20,24]) is None


def test_masked_tracker_never_confirms_a_cloud_even_if_appearance_matches():
    profile=Path(__file__).resolve().parents[1]/'src/ai_player_lab/profiles/mario'
    frame=cv2.imread(str(FIXTURES/'mario/phase-cloud-identity.png'))
    vision=LocalMarioVision(profile)
    cloud=[88,60,20,24]
    assert vision.identity_tracker.bank.add(frame,cloud,source='regression_bad_reference')
    assert vision.identity_tracker.bank.score(frame,cloud)['accepted']
    result=vision.identity_tracker.update(frame,1.,cloud)
    assert result['box'] is None


def test_masked_tracker_never_accepts_rex_legs_as_a_learned_player_pose():
    profile=Path(__file__).resolve().parents[1]/'src/ai_player_lab/profiles/mario'
    frame=cv2.imread(str(FIXTURES/'mario/rex-identity-adversarial.png'))
    vision=LocalMarioVision(profile)
    enemy=[124,175,20,24]
    assert vision.identity_tracker.bank.add(frame,enemy,source='regression_bad_reference')
    assert vision.identity_tracker.bank.score(frame,enemy)['accepted']
    assert vision.identity_tracker.update(frame,1.,enemy)['box'] is None


def test_one_palette_miss_bridges_only_to_a_current_matching_airborne_sprite(tmp_path: Path, monkeypatch):
    frame = cv2.imread(str(FIXTURES / "airborne-mario-rex.png"), cv2.IMREAD_COLOR)
    assert frame is not None
    vision = _vision_with_gameplay_hud(tmp_path, frame)
    first = vision.observe(frame, 10.0)
    assert first["scene"] == "gameplay"
    assert first["player_box"] == [67, 144, 20, 24]

    current = _shift_sprite_patch(frame, first["player_box"], dx=4, dy=2)
    monkeypatch.setattr(mario_vision, "detect_player", lambda *args, **kwargs: None)
    bridged = vision.observe(current, 10.08)
    assert bridged["scene"] == "gameplay"
    assert bridged["player_box"] == [71, 146, 20, 24]


def test_bounded_reacquisition_runs_inside_prediction_window_and_reports_reference(tmp_path: Path, monkeypatch):
    frame = cv2.imread(str(FIXTURES / "airborne-mario-rex.png"), cv2.IMREAD_COLOR)
    box = [67, 144, 20, 24]
    moved = _shift_sprite_patch(frame, box, dx=12, dy=3)
    _vision_with_gameplay_hud(tmp_path, frame)
    vision = LocalMarioVision(tmp_path, appearance_root=tmp_path / "identity")
    assert vision.identity_tracker.bank.add(frame, box, source="airborne-seed")
    vision.identity_tracker.validator = lambda image, candidate: True
    monkeypatch.setattr(vision, "_valid_player_crop", lambda image, candidate: True)

    calls = {"direct": 0}
    def proposals(image, previous_box=None, *, all_candidates=False):
        if all_candidates:
            return [[67, 144, 20, 24]] if calls["direct"] == 1 else [[79, 147, 20, 24]]
        calls["direct"] += 1
        return box if calls["direct"] == 1 else None

    monkeypatch.setattr(mario_vision, "detect_player", proposals)
    monkeypatch.setattr(vision.identity_tracker, "_search", lambda *args, **kwargs: None)
    first = vision.observe(frame, 10.0)
    assert first["player_box"] == box
    reacquired = vision.observe(moved, 10.05)
    assert reacquired["player_box"] == [79, 147, 20, 24]
    assert reacquired["identity"]["source"] == "bounded_predicted_reacquisition"
    assert reacquired["identity"]["status"] == "confirmed"
    assert reacquired["identity"]["reference"]["source"] == "airborne-seed"


def test_recent_player_fallback_requires_current_red_evidence(tmp_path: Path, monkeypatch):
    frame = cv2.imread(str(FIXTURES / "airborne-mario-rex.png"), cv2.IMREAD_COLOR)
    assert frame is not None
    vision = _vision_with_gameplay_hud(tmp_path, frame)
    first = vision.observe(frame, 20.0)
    assert first["player_box"] is not None

    empty = _shift_sprite_patch(frame, first["player_box"], dx=4, dy=2)
    x, y, w, h = first["player_box"]
    empty[y+2:y+2+h, x+4:x+4+w] = (160, 175, 190)
    monkeypatch.setattr(mario_vision, "detect_player", lambda *args, **kwargs: None)
    result = vision.observe(empty, 20.05)
    assert result["scene"] == "gameplay"
    assert result["player_box"] is None


def test_recent_player_fallback_expires_from_last_direct_sighting(tmp_path: Path, monkeypatch):
    frame = cv2.imread(str(FIXTURES / "airborne-mario-rex.png"), cv2.IMREAD_COLOR)
    assert frame is not None
    vision = _vision_with_gameplay_hud(tmp_path, frame)
    first = vision.observe(frame, 30.0)
    current = _shift_sprite_patch(frame, first["player_box"], dx=4, dy=2)
    monkeypatch.setattr(mario_vision, "detect_player", lambda *args, **kwargs: None)

    assert vision.observe(current, 30.19)["player_box"] is not None
    assert vision.observe(current, 30.201)["player_box"] is None


def test_loading_and_map_transitions_clear_recent_player_fallback(tmp_path: Path, monkeypatch):
    frame = cv2.imread(str(FIXTURES / "airborne-mario-rex.png"), cv2.IMREAD_COLOR)
    map_frame = cv2.imread(str(FIXTURES / "map-stage1.png"), cv2.IMREAD_COLOR)
    assert frame is not None and map_frame is not None
    vision = _vision_with_gameplay_hud(tmp_path, frame)
    vision.register_template("map", map_frame, "map-transition")
    vision = LocalMarioVision(tmp_path)
    assert vision.observe(frame, 40.0)["player_box"] is not None
    monkeypatch.setattr(mario_vision, "detect_player", lambda *args, **kwargs: None)

    assert vision.observe(np.zeros_like(frame), 40.05)["scene"] == "loading"
    assert vision.observe(frame, 40.08)["player_box"] is None

    # A separately learned map label must also terminate the gameplay track.
    assert vision.observe(frame, 41.0)["player_box"] is None
    map_result = vision.observe(map_frame, 41.05)
    assert map_result["scene"] == "map"
    assert map_result["player_box"] is None


@pytest.mark.parametrize("scene,fixture", [
    ("title", "title-mario.png"),
    ("intro", "intro-mario-fruit.png"),
    ("map", "map-stage1.png"),
])
def test_actual_title_intro_and_map_frames_keep_their_label(tmp_path: Path, scene, fixture):
    frame = cv2.imread(str(FIXTURES / fixture), cv2.IMREAD_COLOR)
    assert frame is not None
    vision = LocalMarioVision(tmp_path)
    vision.register_template(scene, frame, "actual-frame")
    result = LocalMarioVision(tmp_path).observe(frame, 4.0)
    assert result["scene"] == scene
    assert result["scene"] != "gameplay"


def test_rejects_unexpected_dimensions_and_unlabelled_scene():
    vision = LocalMarioVision()
    with pytest.raises(ValueError):
        vision.observe(np.zeros((100, 100, 3), np.uint8), 0.0)
    with pytest.raises(ValueError):
        vision.register_template("unknown", _frame(), "bad")


def test_reacquires_actual_mario_partially_hidden_by_bush(tmp_path):
    frame=cv2.imread(str(FIXTURES/'mario-partially-hidden-bush.png'))
    profile=Path(mario_vision.__file__).parent/'profiles'/'mario'
    vision=LocalMarioVision(profile,appearance_root=tmp_path,online_learning=False)
    result=vision.observe(frame,1.0)
    box=result['player_box']
    assert box is not None
    assert 98<=box[0]<=110 and 168<=box[1]<=178
    assert result['identity']['score']['accepted']
    assert not result['identity']['score']['partial_match']
    assert not result['identity']['reference']['learned']
