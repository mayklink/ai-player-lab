import cv2
import numpy as np
import pytest
from ai_player_lab.types import Track, Decision, iou
from ai_player_lab.features import exact_hash, phash, hamming, geometric_match
from ai_player_lab.perception import VisualPipeline, estimate_camera, SeededPlayer, floor_gaps
from ai_player_lab.tracking import MultiTracker, collision_estimate
from ai_player_lab.diagnostics import fixture_frame


def test_exact_hash_distinguishes_one_changed_pixel():
    a=np.zeros((20,20,3),np.uint8); b=a.copy(); b[0,0,0]=1
    assert exact_hash(a)!=exact_hash(b)
    assert exact_hash(a)==exact_hash(a.copy())


def test_exact_hash_includes_dimensions():
    assert exact_hash(np.zeros((20,20,3),np.uint8))!=exact_hash(np.zeros((10,40,3),np.uint8))


def test_perceptual_hash_tolerates_small_brightness_change():
    a,_=fixture_frame(0); b=cv2.convertScaleAbs(a,alpha=1,beta=4)
    assert hamming(phash(a),phash(b))<=4


def test_geometry_does_not_invent_features_on_blank_scene():
    a=np.zeros((180,320,3),np.uint8)
    assert geometric_match(a,a)['verified'] is False


def test_geometry_accepts_translation_of_textured_scene():
    a,_=fixture_frame(0)
    b=cv2.warpAffine(a,np.float32([[1,0,-8],[0,1,0]]),(320,180))
    match=geometric_match(a,b)
    assert match['verified']
    assert abs(match['transform'][0][2]+8)<2


def test_camera_translation():
    rng=np.random.default_rng(7)
    a=rng.integers(0,255,(180,320),dtype=np.uint8)
    b=cv2.warpAffine(a,np.float32([[1,0,-3],[0,1,1]]),(320,180),borderMode=cv2.BORDER_REPLICATE)
    camera=estimate_camera(a,b,[])
    assert camera['valid']
    assert abs(camera['dx']+3)<0.5
    assert abs(camera['dy']-1)<0.5


def test_fixed_synthetic_player_is_tracked_without_training():
    first,box=fixture_frame(0); pipe=VisualPipeline(first,box)
    errors=[]
    for i in range(35):
        frame,truth=fixture_frame(i)
        obs=pipe.process(frame,10+i/30)
        assert obs.player is not None
        errors.append(abs(obs.player.bbox[0]-truth[0]))
    assert max(errors)<2


def test_unknown_motion_is_not_labeled_enemy():
    first,box=fixture_frame(0); pipe=VisualPipeline(first,box)
    all_tracks=[]
    for i in range(10):
        obs=pipe.process(fixture_frame(i)[0],10+i/30)
        all_tracks+=obs.objects
    assert all_tracks
    assert all(t.label=='unknown_motion' for t in all_tracks)


def test_static_scene_does_not_generate_moving_objects():
    first,box=fixture_frame(0); pipe=VisualPipeline(first,box)
    for i in range(5):
        obs=pipe.process(first,1+i/30)
        assert obs.objects==[]


def test_invalid_player_seed_is_rejected():
    frame=np.zeros((100,100,3),np.uint8)
    with pytest.raises(ValueError): SeededPlayer(frame,(0,0,20,20))
    with pytest.raises(ValueError): SeededPlayer(frame,(90,90,20,20))


def test_non_monotonic_timestamp_rejected():
    frame,box=fixture_frame(0); pipe=VisualPipeline(frame,box); pipe.process(frame,2)
    with pytest.raises(ValueError): pipe.process(frame,2)


def test_frame_size_change_rejected():
    frame,box=fixture_frame(0); pipe=VisualPipeline(frame,box)
    with pytest.raises(ValueError): pipe.process(frame[:50],1)


def test_tracking_preserves_id_and_estimates_velocity():
    tracker=MultiTracker()
    ids=[]
    for i in range(8):
        item=tracker.update([(10+2*i,30,10,10)],0.1)[0]; ids.append(item.id)
    assert len(set(ids))==1
    assert abs(item.vx-20)<1


def test_lost_track_expires():
    tracker=MultiTracker(max_missed=2); tracker.update([(10,10,10,10)],.1)
    for _ in range(3): tracker.update([],.1)
    assert not tracker.tracks


def test_contact_uses_relative_velocity():
    p=Track(0,(50,100,10,20),vx=10,confidence=.9,age=5)
    e=Track(1,(100,100,10,20),vx=-30,confidence=.9,age=5)
    h=collision_estimate(p,e)
    assert h['ttc_ms']==1000
    # Common camera translation cancels in relative velocities.
    p.vx-=100; e.vx-=100
    assert collision_estimate(p,e)['ttc_ms']==1000


def test_diverging_or_different_height_has_no_contact_prediction():
    p=Track(0,(0,100,10,20),vx=0,age=5)
    assert collision_estimate(p,Track(1,(100,100,10,20),vx=30,age=5)) is None
    assert collision_estimate(p,Track(1,(100,0,10,20),vx=-30,age=5)) is None


def test_floor_unknown_without_calibration():
    frame,box=fixture_frame(0)
    assert floor_gaps(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),Track(0,box),None)==[]


def test_gap_detection_is_explicitly_a_hypothesis():
    frame,box=fixture_frame(0); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray[140:,80:110]=30
    gaps=floor_gaps(gray,Track(0,box),145)
    assert gaps
    assert gaps[0]['semantic_danger']=='unverified'


@pytest.mark.parametrize('action',['enter','alt_f4','type_password',''])
def test_unknown_action_rejected(action):
    with pytest.raises(ValueError): Decision(action,1,1)
