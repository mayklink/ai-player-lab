
import pytest as _asset_pytest
from pathlib import Path as _AssetPath
if not (_AssetPath(__file__).parent / "fixtures" / "small-mario-stopped.png").is_file():
    _asset_pytest.skip("Optional game imagery is not distributed; see docs/ASSETS.md", allow_module_level=True)

from pathlib import Path
import time

import cv2
import numpy as np

from ai_player_lab.appearance import AppearanceBank, IdentityTracker, _foreground_mask
from ai_player_lab.mario_vision import detect_player


FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name):
    image = cv2.imread(str(FIXTURES / name), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _shift(frame, box, dx=4, dy=2):
    x, y, w, h = box
    out = frame.copy()
    patch = frame[y:y+h, x:x+w].copy()
    out[y:y+h, x:x+w] = np.median(frame[max(0, y-3):y, x:x+w], axis=(0, 1)).astype(np.uint8)
    out[y+dy:y+dy+h, x+dx:x+dx+w] = patch
    return out


def test_bank_scores_masked_positive_and_negative_examples():
    bank = AppearanceBank()
    positive = _fixture("airborne-mario-rex.png")
    negative = _fixture("rex-misidentified.png")
    assert bank.add(positive, [67, 144, 20, 24], source="verified_fixture")
    assert bank.add(negative, [97, 173, 20, 24], label="rex", source="verified_fixture")

    pos = bank.score(positive, [67, 144, 20, 24])
    neg = bank.score(negative, [97, 173, 20, 24])
    assert pos["accepted"] and pos["positive_score"] > 0.95
    assert not neg["accepted"]
    assert neg["negative_score"] > neg["positive_score"]


def test_partial_comparison_tolerates_a_bounded_occluder_but_not_known_enemy():
    frame=_fixture('airborne-mario-rex.png');box=[67,144,20,24]
    bank=AppearanceBank();bank.add(frame,box)
    enemy=_fixture('rex-misidentified.png');bank.add(enemy,[97,173,20,24],label='enemy')
    changed=frame.copy();x,y,w,h=box
    mask=_foreground_mask(frame[y:y+h,x:x+w]);ys,xs=np.where(mask>0)
    count=max(1,len(xs)//4)
    changed[y+ys[-count:],x+xs[-count:]]=(0,255,0)
    full=bank.score(changed,box);partial=bank.score(changed,box,partial=True)
    assert partial['positive_score']>full['positive_score']
    assert partial['partial_match'] and partial['accepted']
    assert not bank.score(enemy,[97,173,20,24],partial=True)['accepted']


def test_masked_match_ignores_changed_pixels_outside_sprite_mask():
    frame = _fixture("small-mario-stopped.png")
    box = [150, 171, 20, 24]
    bank = AppearanceBank()
    crop = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    mask = _foreground_mask(crop)
    assert int(mask.sum()) >= 8
    assert bank.add(frame, box, mask=mask)
    changed = frame.copy()
    changed[:140] = (22, 180, 230)
    changed_crop = changed[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    changed_crop[mask == 0] = (40, 220, 60)
    changed[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = changed_crop
    assert bank.score(changed, box)["accepted"]


def test_tracker_rejects_identity_jump_and_never_confirms_prediction():
    frame = _fixture("airborne-mario-rex.png")
    box = [67, 144, 20, 24]
    bank = AppearanceBank()
    assert bank.add(frame, box)
    tracker = IdentityTracker(bank)
    first = tracker.update(frame, 1.0, box)
    assert first["confirmed_box"] == box and first["status"] == "confirmed"

    empty = np.full_like(frame, 100)
    missing = tracker.update(empty, 1.05, None)
    assert missing["box"] is None and missing["confirmed_box"] is None
    assert missing["predicted_box"] is not None
    assert missing["status"] == "missing"

    moved = _shift(frame, box, dx=120, dy=36)
    jump = tracker.update(moved, 1.06, [187, 180, 20, 24])
    assert jump["box"] is None
    assert jump["status"] == "identity_jump_rejected"


def test_tracker_searches_near_prior_only_with_masked_positive_match():
    frame = _fixture("airborne-mario-rex.png")
    box = [67, 144, 20, 24]
    bank = AppearanceBank()
    assert bank.add(frame, box)
    tracker = IdentityTracker(bank)
    assert tracker.update(frame, 2.0, box)["confirmed_box"] == box
    shifted = _shift(frame, box)
    result = tracker.update(shifted, 2.05, None)
    assert result["source"] == "masked_local_search"
    assert result["confirmed_box"] == [71, 146, 20, 24]


def test_tracker_local_search_keeps_seed_after_many_learned_references():
    frame = _fixture("airborne-mario-rex.png")
    box = [67, 144, 20, 24]
    bank = AppearanceBank()
    assert bank.add(frame, box, source="explicit-airborne-seed")
    from ai_player_lab.appearance import _Reference
    rng = np.random.default_rng(17)
    for index in range(12):
        image = rng.integers(0, 256, (24, 20, 3), dtype=np.uint8)
        mask = np.zeros((24, 20), np.uint8)
        mask[:12, :12] = 1
        bank._references.append(_Reference(image, mask, "player", "confirmed_tracker",
                                           True, f"{index + 1:032x}", True))
    tracker = IdentityTracker(bank)
    assert tracker.update(frame, 2.0, box)["status"] == "confirmed"
    result = tracker.update(_shift(frame, box), 2.05, None)
    assert result["confirmed_box"] == [71, 146, 20, 24]
    assert result["reference"]["source"] == "explicit-airborne-seed"


def test_tracker_recovers_player_when_detector_candidate_is_negative():
    frame = _fixture("airborne-mario-rex.png")
    rex = _fixture("rex-misidentified.png")
    box = [67, 144, 20, 24]
    bank = AppearanceBank()
    assert bank.add(frame, box)
    assert bank.add(rex, [97, 173, 20, 24], label="rex")
    tracker = IdentityTracker(bank)
    assert tracker.update(frame, 2.5, box)["status"] == "confirmed"

    shifted = _shift(frame, box)
    # A palette detector can nominate the enemy box; the bank rejects it and
    # performs a bounded local search for the known positive appearance.
    result = tracker.update(shifted, 2.55, [97, 173, 20, 24])
    assert result["source"] == "masked_local_search"
    assert result["confirmed_box"] == [71, 146, 20, 24]


def test_tracker_does_not_confirm_a_seeded_negative_scene():
    positive = _fixture("airborne-mario-rex.png")
    rex = _fixture("rex-misidentified.png")
    box = [67, 144, 20, 24]
    bank = AppearanceBank()
    assert bank.add(positive, box)
    assert bank.add(rex, [97, 173, 20, 24], label="rex")
    tracker = IdentityTracker(bank)
    assert tracker.update(positive, 3.0, box)["status"] == "confirmed"
    result = tracker.update(rex, 3.05, [97, 173, 20, 24])
    assert result["confirmed_box"] is None
    assert result["score"]["negative_score"] > result["score"]["positive_score"]


def test_camera_offset_is_compensated_without_resetting_identity():
    frame = _fixture("airborne-mario-rex.png")
    box = [67, 144, 20, 24]
    bank = AppearanceBank()
    assert bank.add(frame, box)
    tracker = IdentityTracker(bank)
    assert tracker.update(frame, 2.0, box, camera={"dx": 0, "dy": 0})["status"] == "confirmed"
    moved = _shift(frame, box, dx=20, dy=0)
    result = tracker.update(moved, 2.05, [87, 144, 20, 24], camera={"dx": 20, "dy": 0})
    assert result["status"] == "confirmed"
    assert result["confirmed_box"] == [87, 144, 20, 24]


def test_learned_capacity_retains_explicit_seeds():
    frame = _fixture("small-mario-stopped.png")
    other = _fixture("airborne-mario-rex.png")
    box, other_box = [150, 171, 20, 24], [67, 144, 20, 24]
    bank = AppearanceBank()
    assert bank.add(frame, box, source="seed-one")
    assert bank.add(other, other_box, source="seed-two")
    crop = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    mask = _foreground_mask(crop)
    # Emulate a bank at its learned-reference cap while keeping two real seeds.
    seed = bank._references[0]
    for _ in range(16):
        import uuid
        from ai_player_lab.appearance import _Reference
        bank._references.append(_Reference(seed.image.copy(), seed.mask.copy(), "player",
                                            "confirmed_tracker", True, uuid.uuid4().hex, True))
    variant = frame.copy()
    variant_crop = variant[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    ys, xs = np.where(mask > 0)
    variant_crop[ys[:max(1, len(ys)//12)], xs[:max(1, len(xs)//12)]] = (250, 20, 220)
    variant[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = variant_crop
    assert bank._add_learned(variant, box, mask)
    assert bank.reference_count == 18
    assert [r.source for r in bank._references if not r.learned] == ["seed-one", "seed-two"]
    assert sum(r.learned for r in bank._references) == 16


def test_learned_full_crop_masks_are_rejected_and_existing_data_is_unchanged(tmp_path):
    frame = _fixture("small-mario-stopped.png")
    box = [150, 171, 20, 24]
    bank = AppearanceBank(tmp_path)
    assert bank.add(frame, box, source="seed")
    assert not bank._add_learned(frame, box, np.full((24, 20), 255, np.uint8))

    # Simulate an old persisted learned reference with a contaminated full
    # mask. It remains on disk and in the manifest, but is quarantined at use.
    from ai_player_lab.appearance import _Reference
    key = "a" * 32
    crop = frame[171:195, 150:170].copy()
    bank._references.append(_Reference(crop, np.ones((24, 20), np.uint8), "player",
                                       "confirmed_tracker", True, key, True))
    bank._save()
    manifest_before = (bank.root / "references.json").read_bytes()
    image_before = (bank.root / f"{key}.png").read_bytes()
    loaded = AppearanceBank(tmp_path)
    assert loaded._quarantined(next(ref for ref in loaded._references if ref.key == key))
    dense_mask = np.zeros((24, 20), np.uint8)
    dense_mask.flat[:int(dense_mask.size * 0.80)] = 1
    dense = _Reference(crop, dense_mask, "player", "confirmed_tracker", True,
                       "b" * 32, True)
    assert not loaded._quarantined(dense)  # 80% alone is not evidence of contamination.
    result = loaded.score(frame, box)
    assert result["positive_ref"]["source"] == "seed"
    loaded._save()
    assert (bank.root / "references.json").read_bytes() == manifest_before
    assert (bank.root / f"{key}.png").read_bytes() == image_before


def test_frozen_bank_adds_seed_in_memory_without_persisting(tmp_path):
    frame = _fixture("small-mario-stopped.png")
    box = [150, 171, 20, 24]
    writable = AppearanceBank(tmp_path)
    assert writable.add(frame, box, source="persisted-seed")
    manifest_before = (writable.root / "references.json").read_bytes()
    frozen = AppearanceBank(tmp_path, online_learning=False)
    assert frozen.add(frame, box, source="session-seed")
    assert frozen.reference_count == 2
    assert (writable.root / "references.json").read_bytes() == manifest_before
    assert len(list(writable.root.glob("*.png"))) == 2


def test_frozen_bank_does_not_learn_from_tracker_updates():
    frame = _fixture("airborne-mario-rex.png")
    box = [67, 144, 20, 24]
    bank = AppearanceBank(online_learning=False)
    assert bank.add(frame, box)
    tracker = IdentityTracker(bank)
    for index in range(4):
        result = tracker.update(frame, 5.0 + index * 0.1, box)
        assert result["status"] == "confirmed"
    assert bank.reference_count == 1


def test_masked_search_cost_is_measured_for_one_miss():
    frame = _fixture("airborne-mario-rex.png")
    box = [67, 144, 20, 24]
    bank = AppearanceBank()
    for _ in range(8):
        assert bank.add(frame, box)
    tracker = IdentityTracker(bank)
    assert tracker.update(frame, 4.0, box)["status"] == "confirmed"
    shifted = _shift(frame, box)
    started = time.perf_counter()
    result = tracker.update(shifted, 4.05, None)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert result["confirmed_box"] == [71, 146, 20, 24]
    print(f"masked local search with 8 references: {elapsed_ms:.2f} ms")


def test_reset_and_persistent_banks_are_isolated_by_root(tmp_path):
    frame = _fixture("small-mario-stopped.png")
    box = [150, 171, 20, 24]
    first = AppearanceBank(tmp_path / "one")
    assert first.add(frame, box)
    loaded = AppearanceBank(tmp_path / "one")
    other = AppearanceBank(tmp_path / "two")
    assert loaded.score(frame, box)["accepted"]
    assert other.reference_count == 0
    assert len(list((tmp_path / "one" / "appearance_bank").glob("*.png"))) == 2
    tracker = IdentityTracker(loaded)
    tracker.update(frame, 3.0, box)
    tracker.reset()
    assert tracker.update(frame, 3.01, None)["confirmed_box"] is None


def test_bank_is_bounded_and_flat_crops_are_not_references():
    bank = AppearanceBank()
    frame = _fixture("small-mario-stopped.png")
    assert not bank.add(np.full_like(frame, 70), [0, 0, 20, 24])
    for _ in range(26):
        assert bank.add(frame, [150, 171, 20, 24])
    assert bank.reference_count == 24


def test_intro_fixture_uses_explicit_detected_box_as_positive_seed():
    frame = _fixture("intro-mario-fruit.png")
    box = detect_player(frame)
    assert box is not None
    bank = AppearanceBank()
    assert bank.add(frame, box, source="verified_fixture")
    assert bank.score(frame, box)["accepted"]
