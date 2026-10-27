
import pytest as _asset_pytest
from pathlib import Path as _AssetPath
if not (_AssetPath(__file__).parent / "fixtures" / "small-mario-stopped.png").is_file():
    _asset_pytest.skip("Optional game imagery is not distributed; see docs/ASSETS.md", allow_module_level=True)

from pathlib import Path

import cv2
import numpy as np

from ai_player_lab.perception import VisualPipeline


ROOT = Path(__file__).resolve().parents[1]
MARIO_FRAME = cv2.imread(str(ROOT / "tests/fixtures/small-mario-stopped.png"))
REX = cv2.imread(str(ROOT / "src/ai_player_lab/profiles/mario/identity-seeds/rex-blue-background.png"))
REX_MASK = cv2.imread(str(ROOT / "src/ai_player_lab/profiles/mario/identity-seeds/rex-blue-background.mask.png"),
                      cv2.IMREAD_GRAYSCALE) > 0
MARIO_BOX = (150, 171, 20, 24)


def _scene(mario=(40, 168), rex=(160, 168), *, scroll=0, include_rex=True, dark=False):
    frame = np.zeros((224, 256, 3), np.uint8) if dark else np.full((224, 256, 3), (40, 80, 120), np.uint8)
    if not dark:
        # Stable scenery texture gives camera estimation actual background features.
        for x in range(0, 256, 32):
            cv2.rectangle(frame, (x, 35), (x + 1, 223), (50, 100, 140), 1)
        for y in range(55, 220, 28):
            cv2.rectangle(frame, (4, y), (251, y + 1), (55, 90, 135), 1)
        mx, my = mario
        if mx >= 0 and my >= 0 and mx + 20 <= 256 and my + 24 <= 224:
            frame[my:my+24, mx:mx+20] = MARIO_FRAME[MARIO_BOX[1]:MARIO_BOX[1]+24,
                                                       MARIO_BOX[0]:MARIO_BOX[0]+20]
        if include_rex:
            rx, ry = rex
            if rx >= 0 and ry >= 0 and rx + 20 <= 256 and ry + 24 <= 224:
                roi = frame[ry:ry+24, rx:rx+20]
                roi[REX_MASK] = REX[REX_MASK]
    if scroll:
        frame = cv2.warpAffine(frame, np.float32([[1, 0, scroll], [0, 1, 0]]),
                               (256, 224), borderMode=cv2.BORDER_REPLICATE)
    return frame


def test_real_rex_pixels_remain_current_when_stopped_near_mario_contact_and_scroll():
    # The Rex pixels come from the repository's captured Rex identity fixture;
    # the surrounding level, Mario motion, and camera shifts are controlled.
    first = _scene(rex=(165, 168))
    pipeline = VisualPipeline(first, (40, 168, 20, 24), hud_px=35)

    moving = pipeline.process(_scene(rex=(160, 168)), 10.03, (40, 168, 20, 24))
    assert len(moving.objects) == 1
    rex_id = moving.objects[0].id
    assert rex_id in pipeline._object_appearance

    stopped = pipeline.process(_scene(rex=(160, 168)), 10.06, (40, 168, 20, 24))
    assert len(stopped.objects) == 1
    assert stopped.objects[0].id == rex_id
    assert stopped.objects[0].missed == 0
    assert stopped.objects[0].label == "appearance_confirmed_motion"

    # Mario moves into visible contact; the enemy anchor compares only pixels
    # outside Mario's current box and does not inherit Mario's identity.
    contact = pipeline.process(_scene(mario=(143, 168), rex=(160, 168)),
                               10.09, (143, 168, 20, 24))
    assert len(contact.objects) == 1
    assert contact.objects[0].id == rex_id
    assert contact.objects[0].label == "appearance_confirmed_motion"

    # When Mario covers the Rex's entire distinctive visible region, retain
    # only a short prediction instead of calling hidden pixels confirmed.
    covered = pipeline.process(_scene(mario=(150, 168), rex=(160, 168)),
                               10.105, (150, 168, 20, 24))
    assert len(covered.objects) == 1
    assert covered.objects[0].id == rex_id
    assert covered.objects[0].label == "predicted_motion"
    assert covered.objects[0].missed > 0

    contact = pipeline.process(_scene(mario=(143, 168), rex=(160, 168)),
                               10.12, (143, 168, 20, 24))
    assert contact.objects[0].id == rex_id
    assert contact.objects[0].label == "appearance_confirmed_motion"

    scrolled_frame = _scene(mario=(143, 168), rex=(160, 168), scroll=-8)
    scrolled = pipeline.process(scrolled_frame, 10.15, (135, 168, 20, 24))
    assert len(scrolled.objects) == 1
    assert scrolled.objects[0].id == rex_id
    stopped_scroll = pipeline.process(scrolled_frame.copy(), 10.18, (135, 168, 20, 24))
    assert len(stopped_scroll.objects) == 1
    assert stopped_scroll.objects[0].id == rex_id


def test_real_rex_track_adapts_to_flattened_current_pose_without_new_identity():
    first = _scene(rex=(165, 168))
    pipeline = VisualPipeline(first, (40, 168, 20, 24), hud_px=35)
    moving = pipeline.process(_scene(rex=(160, 168)), 40.03, (40, 168, 20, 24))
    rex_id = moving.objects[0].id
    squash = cv2.resize(REX, (20, 12), interpolation=cv2.INTER_NEAREST)
    squash_mask = cv2.resize(REX_MASK.astype(np.uint8), (20, 12),
                             interpolation=cv2.INTER_NEAREST) > 0
    flattened = _scene(rex=(160, 180), include_rex=False)
    flattened[180:192, 160:180][squash_mask] = squash[squash_mask]
    result = pipeline.process(flattened, 40.06, (40, 168, 20, 24))
    assert len(result.objects) == 1
    assert result.objects[0].id == rex_id
    assert result.objects[0].missed == 0
    assert result.objects[0].label == "appearance_confirmed_motion"
    still_flat = pipeline.process(flattened.copy(), 40.09, (40, 168, 20, 24))
    assert len(still_flat.objects) == 1
    assert still_flat.objects[0].id == rex_id
    assert still_flat.objects[0].missed == 0


def test_swept_mario_roi_does_not_erase_known_rex_or_create_replacement_track():
    first = _scene(mario=(40, 168), rex=(112, 150))
    pipeline = VisualPipeline(first, (40, 168, 20, 24), hud_px=35)
    moving = pipeline.process(_scene(mario=(40, 168), rex=(108, 150)),
                              20.03, (40, 168, 20, 24))
    assert len(moving.objects) == 1
    rex_id = moving.objects[0].id

    # This jump places Rex inside the rectangle spanning Mario's old and new
    # boxes, but outside both endpoint ROIs. Only endpoint masking preserves it.
    jumping = pipeline.process(_scene(mario=(120, 118), rex=(108, 150)),
                               20.06, (120, 118, 20, 24), allow_new_objects=False)
    assert len(jumping.objects) == 1
    assert jumping.objects[0].id == rex_id
    assert jumping.objects[0].label == "appearance_confirmed_motion"


def test_dropout_exports_only_short_prediction_then_expires_and_transition_clears():
    first = _scene(rex=(165, 168))
    pipeline = VisualPipeline(first, (40, 168, 20, 24), hud_px=35)
    seen = pipeline.process(_scene(rex=(160, 168)), 30.03, (40, 168, 20, 24))
    rex_id = seen.objects[0].id

    # No current Rex pixels: a residual motion contour cannot refresh the
    # known track, and unknown-player mode cannot invent another actor.
    missing = _scene(rex=(160, 168), include_rex=False)
    predicted = pipeline.process(missing, 30.08, (40, 168, 20, 24), allow_new_objects=False)
    assert len(predicted.objects) == 1
    assert predicted.objects[0].id == rex_id
    assert predicted.objects[0].missed > 0
    assert predicted.objects[0].label == "predicted_motion"
    assert predicted.hazards == [] or all(
        h.get("predicted_geometry") is True for h in predicted.hazards)

    expired = pipeline.process(missing, 30.24, (40, 168, 20, 24), allow_new_objects=False)
    assert all(obj.id != rex_id for obj in expired.objects)

    transition = pipeline.process(_scene(dark=True), 30.27, None, allow_new_objects=False)
    assert transition.camera.get("scene_transition") is True
    assert transition.objects == []
    assert transition.hazards == []
    assert not pipeline.tracker.tracks
    assert not pipeline._object_appearance
