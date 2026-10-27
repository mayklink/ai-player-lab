from ai_player_lab.local_controller import LocalController
from ai_player_lab.types import Observation, Track


def _obs(now, *, episode=1, player=True, hazards=None, objects=None):
    return Observation(
        seq=1, captured_at=now, frame_hash="f", scene_hash="s", width=320,
        height=240,
        player=Track(0, (40, 100, 12, 20), confidence=0.9, age=10) if player else None,
        objects=list(objects or []), camera={}, hazards=list(hazards or []), episode=episode,
        character_state={'can_start_jump':True,'grounded':True},
    )


def _ground_hazard(ttc=300, *, bbox=(66, 104, 10, 10), object_id=7):
    obj = Track(object_id, bbox, confidence=0.8, age=4, label="enemy_rex")
    hazard = {"kind": "approaching_object", "object_id": object_id, "ttc_ms": ttc,
              "confidence": 0.8, "semantic_danger": "unknown"}
    return obj, hazard


def test_bounded_learned_correction_does_not_renew_as_eight_hundred_ms_intent():
    controller=LocalController();controller.accept('right',1.,1,100,bounded=True)
    assert controller.update(_obs(1.01),1.01)['action']=='right'
    assert controller.update(_obs(1.12),1.12)['action']=='wait'


def test_unknown_ground_and_predicted_overlap_do_not_trigger_reflexes():
    controller=LocalController();controller.accept('right',1.,1)
    obj,hazard=_ground_hazard()
    obs=_obs(1.,objects=[obj],hazards=[hazard]);obs.character_state={'can_start_jump':None}
    assert controller.update(obs,1.)['action']=='right'
    controller=LocalController();controller.accept('right',2.,1)
    obs=_obs(2.,hazards=[{'kind':'visual_overlap','predicted_geometry':True}])
    assert controller.update(obs,2.)['action']=='right'


def test_unclassified_patch_cannot_authorize_ground_jump():
    controller=LocalController();controller.accept('right',1.,1)
    obj,hazard=_ground_hazard();obj.label='appearance_confirmed_motion'
    assert controller.update(_obs(1.,objects=[obj],hazards=[hazard]),1.)['action']=='right'


def test_airborne_temporal_state_prevents_another_ground_reflex():
    controller=LocalController();controller.accept('right',1.,1)
    obj,hazard=_ground_hazard()
    obs=_obs(1.,objects=[obj],hazards=[hazard])
    obs.character_state={'jump_phase':'falling','can_start_jump':False}
    assert controller.update(obs,1.)['action']=='right'


def test_intent_is_continuous_while_waiting_for_remote_reply_and_renews_bounded_lease():
    controller = LocalController()
    assert controller.accept("right", 1.0, episode=1)

    first = controller.update(_obs(1.0), 1.0)
    assert first == {"action": "right", "hold_ms": 100, "reason": "intent"}
    assert controller.update(_obs(1.04), 1.04) is None
    renewed = controller.update(_obs(1.06), 1.06)
    assert renewed == {"action": "right", "hold_ms": 100, "reason": "intent"}
    assert controller.update(_obs(1.81), 1.81) == {
        "action": "wait", "hold_ms": 100, "reason": "intent_expired"
    }


def test_late_observation_paused_and_lost_player_cancel_motion_immediately():
    controller = LocalController()
    controller.accept("right", 2.0, 1, 800)
    assert controller.update(_obs(2.0), 2.0)["action"] == "right"
    assert controller.update(_obs(2.0), 2.151)["reason"] == "observation_stale"

    controller.accept("left", 3.0, 1, 800)
    assert controller.update(_obs(3.0), 3.0, paused=True)["reason"] == "paused"
    controller.accept("right", 4.0, 1, 800)
    assert controller.update(_obs(4.0, player=False), 4.0)["reason"] == "player_missing"


def test_gap_and_visual_overlap_veto_motion():
    controller = LocalController()
    controller.accept("right", 0.0, 1, 800)
    gap = {"kind": "gap_candidate", "distance_px": 18, "confidence": 0.45}
    assert controller.update(_obs(0.0, hazards=[gap]), 0.0)["reason"] == "gap_veto"

    controller.accept("right", 0.1, 1, 800)
    overlap = {"kind": "visual_overlap", "object_id": 5, "ttc_ms": 0}
    assert controller.update(_obs(0.1, hazards=[overlap]), 0.1)["reason"] == "immediate_overlap"


def test_ground_threat_triggers_early_jump_and_keeps_jump_through_track_change():
    controller = LocalController()
    obj, hazard = _ground_hazard(300)
    controller.accept("right", 0.0, 1, 800)
    result = controller.update(_obs(0.0, hazards=[hazard], objects=[obj]), 0.0)
    assert result == {"action": "right_jump", "hold_ms": 100, "reason": "ground_threat_jump"}

    # Tracking may lose the ground classification after the player leaves the floor.
    continued = controller.update(_obs(0.1), 0.1)
    assert continued is not None
    assert continued["action"] == "right_jump"


def test_left_intent_jumps_left_but_tall_or_airborne_threats_do_not_trigger_jump():
    controller = LocalController()
    tall, hazard = _ground_hazard(300, bbox=(66, 40, 10, 55))
    controller.accept("left", 0.0, 1, 800)
    result = controller.update(_obs(0.0, hazards=[hazard], objects=[tall]), 0.0)
    assert result["action"] == "left"

    airborne, hazard = _ground_hazard(300, bbox=(66, 40, 10, 10))
    controller.accept("left", 0.1, 1, 800)
    result = controller.update(_obs(0.1, hazards=[hazard], objects=[airborne]), 0.1)
    assert result["action"] == "left"

    ground, hazard = _ground_hazard(300, bbox=(15, 104, 10, 10), object_id=8)
    controller.accept("left", 0.2, 1, 800)
    result = controller.update(_obs(0.2, hazards=[hazard], objects=[ground]), 0.2)
    assert result["action"] == "left_jump"


def test_jump_cooldown_prevents_key_spam_and_unknown_immediate_threat_stops():
    controller = LocalController()
    obj, hazard = _ground_hazard(300)
    controller.accept("right", 0.0, 1, 800)
    assert controller.update(_obs(0.0, hazards=[hazard], objects=[obj]), 0.0)["action"] == "right_jump"

    # Beyond the 200ms jump, the same track cannot retrigger until cooldown passes.
    controller.update(_obs(0.25, hazards=[hazard], objects=[obj]), 0.25)
    result = controller.update(_obs(0.3, hazards=[hazard], objects=[obj]), 0.3)
    assert result is None or result["action"] == "right"

    unknown = {"kind": "approaching_object", "object_id": 999, "ttc_ms": 100}
    result = controller.update(_obs(0.35, hazards=[unknown]), 0.35)
    assert result["action"] == "wait"
    assert result["reason"] == "immediate_unknown_threat"


def test_episode_change_cancels_the_pending_intent():
    controller = LocalController()
    controller.accept("right", 0.0, episode=4, hold_ms=800)
    assert controller.update(_obs(0.0, episode=4), 0.0)["action"] == "right"
    result = controller.update(_obs(0.05, episode=5), 0.05)
    assert result == {"action": "wait", "hold_ms": 100, "reason": "episode_changed"}


def test_strategic_jump_releases_after_200ms_while_direction_continues():
    controller=LocalController()
    controller.accept('right_jump',0.,1,100)
    assert controller.update(_obs(0.),0.)['action']=='right_jump'
    near_end=controller.update(_obs(.18),.18)
    assert near_end['hold_ms']<=21
    after=controller.update(_obs(.23),.23)
    assert after['action']=='right'


def test_late_strategic_reversal_does_not_reverse_a_committed_jump():
    controller=LocalController();controller.accept('right_jump',0.,1)
    controller.update(_obs(0.),0.)
    controller.accept('left',.1,1)
    airborne=_obs(.12);airborne.player.bbox=(10,70,12,12)
    assert controller.update(airborne,.12)['action']=='right_jump'
    airborne.captured_at=.25
    assert controller.update(airborne,.25)['action']=='right'
    landed=_obs(.4)
    assert controller.update(landed,.4)['action']=='left'


def test_blocked_progress_probe_requires_permission_and_actual_stagnation():
    for enabled,moving in [(True,False),(True,True),(False,False)]:
        controller=LocalController();controller.accept('right',0.,1)
        result=None
        for index,now in enumerate([0.,.1,.2,.3,.4,.5,.6,.71]):
            obs=_obs(now);obs.camera={'valid':True,'dx':0.,'dy':0.}
            if moving:obs.player.bbox=(40+index*2,100,12,20)
            result=controller.update(obs,now,allow_stuck_jump=enabled)
        assert result is not None
        assert (result['action']=='right_jump')==(enabled and not moving)
