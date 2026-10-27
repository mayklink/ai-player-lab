
import pytest as _asset_pytest
from pathlib import Path as _AssetPath
if not (_AssetPath(__file__).parent / "fixtures" / "small-mario-stopped.png").is_file():
    _asset_pytest.skip("Optional game imagery is not distributed; see docs/ASSETS.md", allow_module_level=True)

from pathlib import Path

import cv2
import numpy as np

from ai_player_lab.game_object_vision import MarioObjectDetector
from ai_player_lab.perception import VisualPipeline
from ai_player_lab.tracking import collision_estimate
from ai_player_lab.types import Track, Observation, actionable_actor
from ai_player_lab.detections import DetectionOverlay


FIXTURES=Path(__file__).parent/'fixtures'


def test_old_high_confidence_patch_and_prediction_cannot_authorize_collision():
    player=Track(0,(100,170,20,24),vx=100)
    for label,missed in [('appearance_confirmed_motion',0),('unknown_motion',0),('enemy_rex_predicted',1)]:
        obj=Track(7,(119,170,16,24),confidence=.99,age=90,missed=missed,label=label)
        assert not actionable_actor(obj)
        assert collision_estimate(player,obj,require_current_evidence=True) is None


def test_scrolling_real_scenery_does_not_become_independent_motion():
    first=cv2.imread(str(FIXTURES/'small-mario-stopped.png'))
    pipe=VisualPipeline(first,hud_px=35,classify_motion=True)
    for index in range(1,10):
        frame=cv2.warpAffine(first,np.float32([[1,0,-index*2],[0,1,0]]),(256,224),
                             borderMode=cv2.BORDER_REPLICATE)
        obs=pipe.process(frame,1+index*.04,player_bbox=(150-index*2,171,20,24))
        assert not any(actionable_actor(o) for o in obs.objects)


def test_large_projectile_detected_in_two_real_frames_without_chopping_its_box():
    detector=MarioObjectDetector()
    for name,x in [('large-projectile-approach.jpg',128),('large-projectile-close.jpg',78)]:
        frame=cv2.imread(str(FIXTURES/name))
        found=[o for o in detector.detect(frame) if o['label']=='enemy_projectile_large']
        assert len(found)==1 and abs(found[0]['bbox'][0]-x)<=1
        box=found[0]['bbox'];assert box[2]>55 and box[3]>55
        obj=Track(100001,box,confidence=found[0]['confidence'],age=4,label='enemy_projectile_large')
        obs=Observation(1,1.,'','',256,224,None,[obj],{})
        _,drawn=DetectionOverlay().update(frame,1.,'gameplay',observation=obs,reuse=True)
        assert drawn[0]['bbox']==list(box) and drawn[0]['label'].startswith('Projetil')


def test_scenery_and_unverified_candidates_do_not_draw_enemy_or_reaction_boxes():
    detector=MarioObjectDetector()
    for name in ['map-stage1.png','small-mario-stopped.png','mario-partially-hidden-bush.png']:
        assert detector.detect(cv2.imread(str(FIXTURES/name)))==[]
    frame=np.zeros((224,256,3),np.uint8)
    obs=Observation(1,1.,'','',256,224,None,
        [Track(5,(100,100,15,15),confidence=.99,age=40,label='unknown_motion')],{})
    _,drawn=DetectionOverlay().update(frame,1.,'gameplay',observation=obs,reuse=True)
    assert drawn==[]
