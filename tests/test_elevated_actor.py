from ai_player_lab.local_controller import LocalController
from ai_player_lab.types import Observation,Track


def state(now,player=(105,170,20,24),rex=(195,109,16,28),*,grounded=True,missed=0,vy=0,extra=None):
    enemy=Track(7,rex,vx=-35,vy=45,confidence=.9,age=8,missed=missed,
                label='enemy_rex_predicted' if missed else 'enemy_rex')
    return Observation(1,now,'','',256,224,Track(0,player,vy=vy,confidence=.9,age=8),
        [enemy]+list(extra or []),{'valid':True,'dx':0,'dy':0},episode=1,
        character_state={'grounded':grounded,'can_start_jump':True if grounded else False})


def test_pipe_encounter_waits_then_ascends_before_advancing_and_does_not_reverse_midair():
    controller=LocalController();controller.accept('right',0.,1)
    first=controller.update(state(0.),0.)
    assert first['action']=='wait' and first['reason']=='elevated_actor_wait'
    still_waiting=controller.update(state(.15,rex=(175,114,16,28),grounded=None),.15)
    assert still_waiting['action']=='wait'
    controller.accept('right',.4,1)
    launch=controller.update(state(.4,rex=(165,125,16,28)),.4)
    assert launch['action']=='jump' and launch['reason']=='elevated_actor_ascent'
    rising=controller.update(state(.55,player=(105,150,20,24),rex=(149,134,16,28),grounded=False,vy=-120),.55)
    assert rising['action']=='jump'
    cleared=controller.update(state(.72,player=(105,106,20,24),rex=(140,145,16,28),grounded=False,vy=-10),.72)
    assert cleared['action']=='right_jump' and cleared['reason']=='elevated_actor_clearance'
    controller.accept('left',.74,1)
    assert controller.update(state(.78,player=(110,108,20,24),rex=(136,145,16,28),grounded=False,vy=15),.78)['action']=='right_jump'
    landed=controller.update(state(.99,player=(139,135,20,24),rex=(120,157,16,28)),.99)
    assert landed['action']=='left'


def test_elevated_reflex_requires_current_rex_and_current_ground():
    for grounded,missed in [(None,0),(False,0),(True,1)]:
        controller=LocalController();controller.accept('right',0.,1)
        result=controller.update(state(0.,rex=(165,125,16,28),grounded=grounded,missed=missed),0.)
        assert result['action']=='right'


def test_current_overhead_projectile_prevents_new_elevated_jump():
    controller=LocalController();controller.accept('right',0.,1)
    projectile=Track(9,(80,90,66,67),confidence=.9,age=5,label='enemy_projectile_large')
    result=controller.update(state(0.,rex=(165,125,16,28),extra=[projectile]),0.)
    assert result['action']=='wait'
    assert result['reason']=='elevated_actor_ground_or_projectile_pending'


def test_lost_target_before_clearance_never_starts_blind_forward_motion():
    controller=LocalController();controller.accept('right',0.,1)
    controller.update(state(0.,rex=(165,125,16,28)),0.)
    result=controller.update(state(.12,player=(105,152,20,24),rex=(155,130,16,28),grounded=False,missed=1,vy=-100),.12)
    assert result['action']=='jump'


def test_live_pipe_geometry_launches_before_rex_reaches_side_of_mario():
    controller=LocalController();controller.accept('right',0.,1)
    launch=controller.update(state(0.,player=(105,171,20,24),rex=(161,114,16,28)),0.)
    assert launch['action']=='jump'
    assert launch['reason']=='elevated_actor_ascent'
