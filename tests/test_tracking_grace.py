from ai_player_lab.tracking_grace import recent_visual_lease
from ai_player_lab.types import Observation,Track
from ai_player_lab.local_controller import LocalController

def test_brief_occlusion_uses_only_fresh_real_observation_and_bounded_lease():
    obs=Observation(1,1.,'f','s',256,224,Track(0,(20,170,20,24)),[],{},episode=3)
    args=dict(episode=3,scene='gameplay',stable=True,paused=False,identity={'predicted_box':[22,170,20,24]})
    assert 0<recent_visual_lease(obs,1.08,**args)<=40
    assert recent_visual_lease(obs,1.121,**args)==0
    assert recent_visual_lease(obs,1.04,**{**args,'episode':4})==0
    assert recent_visual_lease(obs,1.04,**{**args,'scene':'map'})==0
    assert recent_visual_lease(obs,1.04,**{**args,'identity':{}})==0

def test_jump_reply_can_execute_during_short_occlusion_without_new_geometry():
    obs=Observation(1,1.,'f','s',256,224,Track(0,(20,170,20,24)),[],{},episode=3)
    lease=recent_visual_lease(obs,1.08,episode=3,scene='gameplay',stable=True,paused=False,
        identity={'predicted_box':[22,170,20,24]})
    control=LocalController();control.accept('right_jump',1.08,3)
    command=control.update(obs,1.08)
    assert command['action']=='right_jump' and min(lease,command['hold_ms'])<=40
    assert obs.player.bbox==(20,170,20,24)
