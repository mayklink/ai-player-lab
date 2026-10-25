"""Local-only regressions: no game, screen capture, keyboard or model API."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from ai_player_lab.diagnostics import fixture_frame
from ai_player_lab.perception import VisualPipeline, SeededPlayer, floor_gaps
from ai_player_lab.providers import choose_action, local_policy
from ai_player_lab.tracking import MultiTracker
from ai_player_lab.types import Decision, Observation, Track


CAMERA = {'dx': 0., 'dy': 0., 'valid': True}


def test_calibration_appearance_not_recropped_from_later_background():
    from ai_player_lab.sources import encode_player_template, decode_player_template
    first, box = fixture_frame(0)
    later, actual = fixture_frame(50)
    cfg = {'player_box': list(box), 'player_template': encode_player_template(first, box)}
    anchor = decode_player_template(cfg, later)
    assert np.array_equal(anchor, cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)[122:144, 40:54])
    obs = VisualPipeline(later, box, player_template=anchor).process(later, 1.)
    assert obs.player is None or abs(obs.player.bbox[0] - actual[0]) < 2


def test_calibration_rejects_legacy_bbox_and_resolution_change():
    from ai_player_lab.sources import encode_player_template, decode_player_template
    frame, box = fixture_frame(0)
    with pytest.raises(ValueError, match='calibr'):
        decode_player_template({'player_box': list(box)}, frame)
    cfg = {'player_box': list(box), 'player_template': encode_player_template(frame, box)}
    with pytest.raises(ValueError, match='[Tt]amanho|[Rr]esolu'):
        decode_player_template(cfg, frame[:150])
    cfg['player_template']['png_base64'] = 'not base64!'
    with pytest.raises(ValueError):
        decode_player_template(cfg, frame)


def test_horizontal_pose_change_preserves_player_without_learning_distractor():
    frame, box = fixture_frame(0); x, y, w, h = box
    seed = SeededPlayer(frame, box)
    changed = frame.copy()
    changed[y:y+h, x:x+w] = changed[y:y+h, x:x+w][:, ::-1]
    for _ in range(12):
        track = seed.update(cv2.cvtColor(changed, cv2.COLOR_BGR2GRAY), 1/30, CAMERA)
        assert track is not None and track.bbox == box


def test_two_plausible_player_copies_are_rejected_as_ambiguous():
    frame, box = fixture_frame(0); x, y, w, h = box
    seed = SeededPlayer(frame, box)
    changed = frame.copy()
    changed[y:y+h, x:x+w] = changed[y:y+h, x:x+w][:, ::-1]
    changed[y:y+h, x+20:x+20+w] = frame[y:y+h, x:x+w]
    assert seed.update(cv2.cvtColor(changed, cv2.COLOR_BGR2GRAY), 1/30, CAMERA) is None


def test_player_reacquisition_requires_confirmation_but_is_not_permanently_disabled():
    frame, box = fixture_frame(0); seed = SeededPlayer(frame, box)
    for _ in range(12):
        assert seed.update(np.zeros(frame.shape[:2], np.uint8), 1/30, CAMERA) is None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert seed.update(gray, 1/30, CAMERA) is None
    recovered = seed.update(gray, 1/30, CAMERA)
    assert recovered is not None and recovered.bbox == box


def test_long_player_loss_requires_human_reseed():
    frame, box = fixture_frame(0); seed = SeededPlayer(frame, box)
    for _ in range(20):
        seed.update(np.zeros(frame.shape[:2], np.uint8), .1, CAMERA)
    for _ in range(3):
        assert seed.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), .1, CAMERA) is None
    assert seed.requires_reseed


def test_hud_is_excluded_before_camera_estimation():
    rng = np.random.default_rng(75)
    first = rng.integers(25, 65, (180, 320), dtype=np.uint8)
    first[:60] = rng.integers(0, 255, (60, 320), dtype=np.uint8)
    current = cv2.warpAffine(first, np.float32([[1,0,-3],[0,1,0]]), (320,180), borderMode=cv2.BORDER_REPLICATE)
    current[:60] = first[:60]
    pipe = VisualPipeline(cv2.cvtColor(first, cv2.COLOR_GRAY2BGR), hud_px=60)
    obs = pipe.process(cv2.cvtColor(current, cv2.COLOR_GRAY2BGR), 1.)
    assert obs.camera['valid'] and abs(obs.camera['dx'] + 3) < .2
    assert not obs.camera.get('scene_transition', False)


def test_old_player_mask_outside_viewport_does_not_erase_other_objects(monkeypatch):
    from ai_player_lab import perception
    rng = np.random.default_rng(4)
    background = rng.integers(20,65,(180,320,3),dtype=np.uint8)
    sprite = fixture_frame(0)[0][122:144,40:54].copy()
    first = background.copy(); first[122:144,2:16] = sprite
    first[127:144,180:194] = (170,200,170)
    current = cv2.warpAffine(background,np.float32([[1,0,-20],[0,1,0]]),(320,180),borderMode=cv2.BORDER_REPLICATE)
    current[122:144,5:19] = sprite
    current[127:144,150:164] = (170,200,170)
    monkeypatch.setattr(perception, 'estimate_camera', lambda *args: {'dx':-20.,'dy':0.,'valid':True,'inliers':100})
    pipe = VisualPipeline(first,(2,122,14,22)); pipe.last_time = 0.
    obs = pipe.process(current,.1)
    assert obs.player is not None
    assert any(140 < t.bbox[0] < 160 for t in obs.objects)


def test_approaching_object_does_not_disappear_when_close_to_player():
    frame, box = fixture_frame(0); pipe = VisualPipeline(frame, box)
    for i in range(71):
        obs = pipe.process(fixture_frame(i)[0], 1+i/30)
        if 66 <= i <= 70:
            assert obs.player is not None
            assert obs.objects, f'lost object at frame {i}'
            assert any(h['kind'] in ('approaching_object', 'visual_overlap') for h in obs.hazards)


def test_stationary_object_persists_briefly_as_uncertain_then_expires():
    frame, box = fixture_frame(0); pipe = VisualPipeline(frame, box)
    for i in range(10):
        obs = pipe.process(fixture_frame(i)[0], 1+i/30)
    object_id = obs.objects[0].id
    frozen = fixture_frame(9)[0]
    for j in range(1, 4):
        obs = pipe.process(frozen, 1+(9+j)/30)
        assert any(t.id == object_id and t.missed > 0 and t.confidence < .9 for t in obs.objects)
    obs = pipe.process(frozen, 2.)
    assert not obs.objects


def test_gap_open_to_right_boundary_is_not_silently_dropped():
    frame, box = fixture_frame(0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray[140:,80:] = 30
    gaps = floor_gaps(gray, Track(0, box), 145)
    assert gaps and gaps[0]['right_boundary_visible'] is False
    assert gaps[0]['semantic_danger'] == 'unverified'
    assert gaps[0]['width_is_lower_bound'] is True


def test_continuous_floor_does_not_invent_gap():
    frame, box = fixture_frame(0)
    assert floor_gaps(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),Track(0,box),145) == []


def test_track_velocity_accumulates_actual_elapsed_time_through_occlusion():
    tracker = MultiTracker()
    tracker.update([(0,10,10,10)], .1)
    tracker.update([], 1.)
    track = tracker.update([(20,10,10,10)], .1)[0]
    assert track.vx == pytest.approx(.6*20/1.1)


def test_seed_velocity_accumulates_time_including_reacquisition_confirmation():
    frame, box = fixture_frame(0); seed = SeededPlayer(frame, box)
    seed.update(np.zeros(frame.shape[:2], np.uint8), 1., CAMERA)
    shifted = cv2.warpAffine(frame, np.float32([[1,0,20],[0,1,0]]), (320,180), borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    assert seed.update(gray, .1, {'dx':20., 'dy':0., 'valid':True}) is None
    track = seed.update(gray, .1, CAMERA)
    assert track is not None and track.vx == pytest.approx(.6*20/1.2)


@pytest.mark.parametrize('kind,ttc', [('approaching_object', 150.), ('visual_overlap', 0.)])
def test_local_collision_veto_without_api(kind, ttc):
    player = Track(0, (10,10,10,20), confidence=.9)
    obj = Track(1, (25,10,10,20), confidence=.8, age=4)
    obs = Observation(1, 10., 'hash', '0', 320, 180, player, [obj], {},
                      hazards=[{'kind':kind, 'object_id':1, 'ttc_ms':ttc, 'confidence':.8}])
    assert local_policy(obs).action == 'wait'
    chosen, reason = choose_action(obs, Decision('right', 10., 1), 10.01)
    assert chosen.action == 'wait' and reason == 'local_collision_veto'
    # An approaching object on the right is not a reason to block retreat left.
    assert choose_action(obs, Decision('left', 10., 1), 10.01)[0].action == 'left'
    assert choose_action(obs, Decision('left_jump', 10., 1), 10.01)[0].action == 'left_jump'


def test_replay_media_time_does_not_make_decisions_stale():
    first, box = fixture_frame(0); pipe = VisualPipeline(first, box)
    for i in range(10):
        obs = pipe.process(fixture_frame(i)[0], i/30, captured_at=100+i*.1)
        assert obs.captured_at == pytest.approx(100+i*.1)
        assert obs.media_time == pytest.approx(i/30)
        assert choose_action(obs, local_policy(obs), 100+i*.1)[1] != 'stale_decision'
    assert obs.player.vx == pytest.approx(30., abs=.02)


def test_runtime_replay_uses_media_clock_even_when_decode_is_slow(tmp_path, monkeypatch):
    from ai_player_lab import runtime
    from ai_player_lab.sources import encode_player_template
    first, box = fixture_frame(0)
    class Video:
        def __init__(self, _): self.index = 0
        def isOpened(self): return True
        def get(self, _): return 30.
        def read(self):
            if self.index >= 10: return False, None
            time.sleep(.05)
            frame = fixture_frame(self.index)[0]; self.index += 1
            return True, frame
        def release(self): pass
    monkeypatch.setattr(runtime.cv2, 'VideoCapture', Video)
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path/'config.json'
    cfg.write_text(json.dumps({'capture_region':[0,0,320,180], 'player_box':box,
                               'player_template':encode_player_template(first, box)}))
    summary = runtime.run_session(cfg, video_path=Path('fake.avi'), headless=True, seconds=3)
    rows = [json.loads(x) for x in (Path(summary['output'])/'state.jsonl').read_text().splitlines()]
    assert len(rows) == 10 and rows[-1]['observation']['player']['vx'] == pytest.approx(30., abs=.02)
    assert summary['jev_calls'] == summary['vlm_calls'] == 0
    assert summary['real_input_enabled'] is False
