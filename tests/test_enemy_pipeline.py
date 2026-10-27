import numpy as np

from ai_player_lab.perception import VisualPipeline
from ai_player_lab.detections import DetectionOverlay


def test_verified_enemy_is_tracked_and_drawn_even_without_motion():
    class Detector:
        def detect(self,frame,player_box=None):
            return [{'bbox':[160,160,18,32],'label':'enemy_rex','confidence':.96}]
    frame=np.full((224,256,3),80,np.uint8)
    pipeline=VisualPipeline(frame,hud_px=34,object_detector=Detector())
    first=pipeline.process(frame,1.0)
    second=pipeline.process(frame,1.03)
    assert len(second.objects)==1
    enemy=second.objects[0]
    assert enemy.id==first.objects[0].id
    assert enemy.label=='enemy_rex' and enemy.missed==0
    _,drawn=DetectionOverlay().update(frame,1.03,'gameplay',observation=second,reuse=True)
    assert drawn[0]['kind']=='enemy'
    assert drawn[0]['label'].startswith('Rex')
    assert drawn[0]['color']==[80,230,100]


def test_disappeared_enemy_is_predicted_then_expires():
    class Detector:
        present=True
        def detect(self,frame,player_box=None):
            return [{'bbox':[160,160,18,32],'label':'enemy_rex','confidence':.96}] if self.present else []
    frame=np.full((224,256,3),80,np.uint8);detector=Detector()
    pipeline=VisualPipeline(frame,hud_px=34,object_detector=detector)
    pipeline.process(frame,1.0);detector.present=False
    missing=pipeline.process(frame,1.04)
    assert missing.objects[0].label=='enemy_rex_predicted'
    assert missing.objects[0].missed>0
    assert pipeline.process(frame,1.25).objects==[]
