import numpy as np
from ai_player_lab.detections import DetectionOverlay
from ai_player_lab.diagnostics import fixture_frame
from ai_player_lab.types import Observation,Track


def test_overlay_never_changes_model_input():
    frame,box=fixture_frame(0);original=frame.copy()
    annotated,objects=DetectionOverlay().update(frame,1,'gameplay',list(box))
    assert np.array_equal(frame,original)
    assert not np.array_equal(annotated,original)
    assert objects[0]['kind']=='player' and objects[0]['label']=='Mario'


def test_still_background_is_not_marked_as_enemy():
    frame,_=fixture_frame(0);overlay=DetectionOverlay()
    for i in range(5):
        _,objects=overlay.update(frame,1+i*.05,'gameplay')
        assert objects==[]


def test_motion_is_labelled_as_motion_not_invented_enemy_identity():
    overlay=DetectionOverlay();found=[]
    for i in range(12):
        frame,box=fixture_frame(i)
        obs=Observation(i,1+i*.05,'f','s',256,224,Track(0,box),
            [Track(7,(190-i*2,100,12,15),confidence=.8,age=4,label='independent_motion')],{})
        _,objects=overlay.update(frame,1+i*.05,'gameplay',list(box),observation=obs,reuse=True)
        found.extend(o for o in objects if o['kind']=='motion')
    assert found
    assert all(o['label'].startswith('Movimento') for o in found)


def test_overlay_reuses_exact_control_observation_without_running_perception(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Overlay must not recompute vision')
    monkeypatch.setattr('ai_player_lab.detections.VisualPipeline',forbidden)
    frame,box=fixture_frame(0)
    observation=Observation(1,1.,'f','s',256,224,Track(0,box),
        [Track(7,(190,100,12,15),confidence=.9,age=4,label='independent_motion')],{})
    _,drawn=DetectionOverlay().update(frame,1.,'gameplay',box,observation=observation,reuse=True)
    assert next(o for o in drawn if o['kind']=='motion')['bbox']==[190,100,12,15]
