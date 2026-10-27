import cv2
import numpy as np
from ai_player_lab.perception import VisualPipeline


def test_known_player_swept_motion_is_not_another_actor():
    rng=np.random.default_rng(17)
    background=rng.integers(40,110,(224,256,3),dtype=np.uint8)
    def frame(box):
        image=background.copy();x,y,w,h=box
        cv2.rectangle(image,(x,y),(x+w-1,y+h-1),(20,20,230),-1)
        cv2.rectangle(image,(x+3,y+5),(x+9,y+12),(220,20,20),-1)
        return image
    box=(40,160,16,24);pipeline=VisualPipeline(frame(box),box,hud_px=35)
    for index in range(1,15):
        box=(40+index*2,160-min(index,14-index)*5,16,24)
        obs=pipeline.process(frame(box),index*.05,player_bbox=box)
        assert obs.objects==[]
