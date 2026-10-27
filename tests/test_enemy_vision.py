
import pytest as _asset_pytest
from pathlib import Path as _AssetPath
if not (_AssetPath(__file__).parent / "fixtures" / "small-mario-stopped.png").is_file():
    _asset_pytest.skip("Optional game imagery is not distributed; see docs/ASSETS.md", allow_module_level=True)

from pathlib import Path

import cv2
import numpy as np

from ai_player_lab.enemy_vision import RexDetector


ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "src/ai_player_lab/profiles/mario/enemy-seeds"
FIXTURES = ROOT / "tests/fixtures"


def _frame(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def test_detects_upright_rex_in_full_verified_fixture():
    detector = RexDetector()
    found = detector.detect(_frame("rex-misidentified.png"))

    assert found
    result = found[0]
    assert result["label"] == "enemy_rex"
    assert result["confidence"] >= 0.80
    x, y, width, height = result["bbox"]
    assert abs(x - 91) <= 1 and abs(y - 157) <= 1
    assert width == 16 and height == 28


def test_detects_both_flattened_rex_poses_in_the_opening_encounter():
    detector = RexDetector()
    for name in ("rex-flattened-opening.png", "rex-flattened-initial.png"):
        found = detector.detect(_frame(f"mario/{name}"))
        assert found, name
        assert found[0]["label"] == "enemy_rex"
        assert found[0]["bbox"] == (70, 175, 20, 17)

    # The later frame shows Mario partly covering Rex. Keep only independently
    # visible Rex pixels when applying the player mask.
    partially_hidden = detector.detect(_frame("mario/rex-flattened-initial.png"),
                                       player_box=(62, 170, 20, 24))
    assert partially_hidden
    assert partially_hidden[0]["bbox"] == (70, 175, 20, 17)


def test_static_and_camera_scrolled_rex_do_not_need_motion_bootstrap():
    detector = RexDetector()
    reference = cv2.imread(str(SEEDS / "rex-upright.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(SEEDS / "rex-upright.mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    assert reference is not None and mask.any()

    scene = np.full((224, 256, 3), (55, 95, 135), dtype=np.uint8)
    x, y = 84, 150
    roi = scene[y:y + reference.shape[0], x:x + reference.shape[1]]
    roi[mask] = reference[mask]

    # Repeated unchanged pixels still produce a current detection.
    first = detector.detect(scene)
    second = detector.detect(scene.copy())
    assert first == second
    assert first and first[0]["bbox"] == (x, y, 16, 28)

    shifted = cv2.warpAffine(scene, np.float32([[1, 0, -11], [0, 1, 0]]),
                             (256, 224), borderMode=cv2.BORDER_REPLICATE)
    scrolled = detector.detect(shifted)
    assert scrolled and scrolled[0]["bbox"] == (x - 11, y, 16, 28)


def test_rejects_map_mario_hud_and_missing_actor_frames():
    detector = RexDetector()
    for name in ("map-stage1.png", "small-mario-stopped.png", "title-mario.png",
                 "transition-no-player.jpg"):
        assert detector.detect(_frame(name)) == [], name

    # An otherwise valid candidate inside the supplied player ROI is masked
    # out of the required Rex-color counts, so a fully hidden sprite is omitted.
    rex = _frame("rex-misidentified.png")
    assert detector.detect(rex, player_box=(91, 157, 16, 28)) == []


def test_partial_player_overlap_keeps_rex_only_when_both_colors_remain_visible():
    detector = RexDetector()
    frame = _frame("rex-misidentified.png")
    mario = cv2.imread(str(SEEDS.parent / "identity-seeds" / "small-mario-stopped.png"),
                       cv2.IMREAD_COLOR)
    mario_mask = cv2.imread(str(SEEDS.parent / "identity-seeds" /
                                "small-mario-stopped.mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    assert mario is not None and mario_mask.any()

    # Place the verified Mario crop over one edge of Rex, leaving visible
    # purple/blue and orange Rex pixels outside the player ROI.
    x, y = 100, 160
    roi = frame[y:y + mario.shape[0], x:x + mario.shape[1]]
    roi[mario_mask] = mario[mario_mask]
    found = detector.detect(frame, player_box=(x, y, 20, 24))
    assert found and found[0]["label"] == "enemy_rex"
    assert found[0]["bbox"] == (91, 157, 16, 28)
