from __future__ import annotations

import pytest as _asset_pytest
from pathlib import Path as _AssetPath
if not (_AssetPath(__file__).parent / "fixtures" / "small-mario-stopped.png").is_file():
    _asset_pytest.skip("Optional game imagery is not distributed; see docs/ASSETS.md", allow_module_level=True)


from pathlib import Path

import cv2
import numpy as np
import pytest

from ai_player_lab.life_evidence import LIFE_DIGIT_BOXES, LifeCounter, life_loss_evidence


def glyph(digit: int) -> np.ndarray:
    image = np.full((14, 12, 3), 190, dtype=np.uint8)
    cv2.putText(image, str(digit), (1, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (12, 12, 12), 1, cv2.LINE_8)
    return image


def samples() -> dict[str, list[tuple[int, np.ndarray]]]:
    return {
        scene: [(digit, glyph(digit)) for digit in range(3, 7)]
        for scene in ("gameplay", "map")
    }


def scene_frame(scene: str, digit: int | None) -> np.ndarray:
    image = np.full((224, 256, 3), 165, dtype=np.uint8)
    if digit is not None:
        x, y, w, h = LIFE_DIGIT_BOXES[scene]
        image[y:y + h, x:x + w] = cv2.resize(glyph(digit), (w, h), interpolation=cv2.INTER_NEAREST)
    return image


def records(before: int, after: int, *, noise: bool = False):
    ids = ["g0", "g1", "m0", "m1"]
    digits = [before, before, after, after]
    return [
        {"frame_id": frame_id, "scene": "gameplay" if i < 2 else "map",
         "frame": scene_frame("gameplay" if i < 2 else "map", None if noise and i == 2 else digit)}
        for i, (frame_id, digit) in enumerate(zip(ids, digits))
    ]


def test_same_count_does_not_confirm_death():
    result = life_loss_evidence(records(5, 5), LifeCounter(samples()))
    assert result["confirmed"] is False
    assert result["status"] == "no_life_loss"
    assert result["decrement"] == 0


def test_increased_count_does_not_confirm_death():
    result = life_loss_evidence(records(4, 5), LifeCounter(samples()))
    assert result["confirmed"] is False
    assert result["decrement"] == -1


def test_noisy_or_blank_reading_stays_unknown():
    result = life_loss_evidence(records(5, 4, noise=True), LifeCounter(samples()))
    assert result["confirmed"] is False
    assert result["status"] == "unknown"
    assert result["reason"] == "insufficient_stable_readings"


def test_map_alone_or_wrong_scene_never_confirms_a_loss():
    counter = LifeCounter(samples())
    map_only = [
        {"frame_id": "m0", "scene": "map", "frame": scene_frame("map", 4)},
        {"frame_id": "m1", "scene": "map", "frame": scene_frame("map", 4)},
    ]
    assert life_loss_evidence(map_only, counter)["status"] == "unknown"
    assert counter.read(scene_frame("gameplay", 5), "title")["value"] is None


def test_reader_accepts_the_supported_512_by_448_size():
    counter = LifeCounter(samples())
    frame = cv2.resize(scene_frame("gameplay", 5), (512, 448), interpolation=cv2.INTER_NEAREST)
    assert counter.read(frame, "gameplay")["value"] == 5


def test_packaged_references_distinguish_reviewed_two_and_three_without_clipping_digit():
    fixtures=Path(__file__).parent/'fixtures/mario';counter=LifeCounter()
    for name,scene,value in [('map-life-two.png','map',2),('map-life-three.png','map',3),('gameplay-life-three.png','gameplay',3)]:
        frame=cv2.imread(str(fixtures/name))
        assert counter.read(frame,scene)['value']==value


def test_manually_verified_retest_map_counts_four_through_one(tmp_path):
    fixtures=Path(__file__).parent/'fixtures/mario';counter=LifeCounter()
    results=[]
    for digit in (4,3,2,1):
        frame=cv2.imread(str(fixtures/f'map-life-{digit}-retest.png'))
        assert frame is not None
        result=counter.read(frame,'map')
        results.append(result['value'])
        assert result['score']>=.88
        assert result['margin']>=.10
    assert results==[4,3,2,1]


def test_map_search_recovers_a_two_pixel_hud_shift_without_relaxing_thresholds():
    fixture=Path(__file__).parent/'fixtures/mario/map-life-1-retest.png'
    frame=cv2.imread(str(fixture));matrix=np.float32([[1,0,2],[0,1,-1]])
    shifted=cv2.warpAffine(frame,matrix,(frame.shape[1],frame.shape[0]),flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REPLICATE)
    result=LifeCounter().read(shifted,'map')
    assert result['value']==1
    assert result['score']>=.88 and result['margin']>=.10
    assert result['offset']==(2,-1)


def test_map_digit_sequence_without_gameplay_baseline_is_still_not_a_death():
    fixtures=Path(__file__).parent/'fixtures/mario';counter=LifeCounter()
    frames=[{'frame_id':str(digit),'scene':'map',
             'frame':cv2.imread(str(fixtures/f'map-life-{digit}-retest.png'))}
            for digit in (4,3,2,1)]
    result=life_loss_evidence(frames,counter)
    assert result['confirmed'] is False
    assert result['status']=='unknown'
    assert [counter.read(item['frame'],'map')['value'] for item in frames]==[4,3,2,1]


def test_actual_run_frames_confirm_stable_five_to_four_transition():
    episode = Path(__file__).resolve().parents[1] / "runs/browser/20261024-232700/episodes/20261024-232700-1"
    if not episode.is_dir():
        pytest.skip("Optional live-run fixture is not present in this checkout")
    frames = []
    for index in range(12):
        image = cv2.imread(str(episode / f"{index:02}.png"), cv2.IMREAD_COLOR)
        assert image is not None
        scene = "gameplay" if index <= 9 else "map"
        frames.append({"frame_id": f"{index:02}.png", "scene": scene, "frame": image})
    # Labels are explicit verified exemplars: the x5 gameplay crop and x4 map crop.
    gx, gy, gw, gh = LIFE_DIGIT_BOXES["gameplay"]
    mx, my, mw, mh = LIFE_DIGIT_BOXES["map"]
    gameplay_sample = frames[4]["frame"][gy:gy + gh, gx:gx + gw]
    map_sample = frames[10]["frame"][my:my + mh, mx:mx + mw]
    counter = LifeCounter({"gameplay": [(5, gameplay_sample)], "map": [(4, map_sample)]})
    result = life_loss_evidence(frames, counter)
    assert result["confirmed"] is True
    assert result["before_value"] == 5
    assert result["after_value"] == 4
    assert result["decrement"] == 1
    assert len(result['evidence_frame_ids'])==4
    assert result['evidence_frame_ids'][-2:]==['10.png','11.png']
