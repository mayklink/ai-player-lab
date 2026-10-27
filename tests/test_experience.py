import json

import cv2
import numpy as np

from ai_player_lab.experience import ExperienceMemory


GAME_URL = "https://example.test/game/one"
STAGE = "world-1-2"
CONTEXT = {
    "player_location": [0.42, 0.73],
    "moving_objects": [[0.68, 0.64]],
}
REVIEW = {
    "hypothesis": "A passagem pode exigir um salto; validar em nova tentativa.",
    "confidence": 0.4,
}


def _scene(seed=3):
    rng = np.random.default_rng(seed)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (58, 112, 74)
    # Scene geometry and texture sit below the HUD strip.
    frame[45:205, :] = rng.integers(35, 180, (160, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 190), (319, 239), (65, 80, 90), -1)
    cv2.rectangle(frame, (118, 130), (142, 172), (220, 70, 20), -1)
    return frame


def test_persists_png_and_review_and_reloads_for_same_scene(tmp_path):
    frame = _scene()
    memory = ExperienceMemory(tmp_path, GAME_URL)
    entry = memory.add("event-1", REVIEW, frame, STAGE, CONTEXT)

    assert entry["provenance"] == "review_unverified"
    assert entry["confidence"] == 0.4
    assert (tmp_path / entry["reference"]).is_file()
    assert (tmp_path / "experience-memory.json").is_file()

    reloaded = ExperienceMemory(tmp_path, GAME_URL)
    recalled = reloaded.retrieve(frame.copy(), STAGE, dict(CONTEXT))
    assert len(recalled) == 1
    assert recalled[0]["event_id"] == "event-1"
    assert recalled[0]["hypothesis"] == REVIEW["hypothesis"]
    assert recalled[0]["status"] == "unverified"
    assert recalled[0]["provenance"] == "review_unverified"


def test_duplicate_event_id_does_not_create_a_second_entry(tmp_path):
    memory = ExperienceMemory(tmp_path, GAME_URL)
    memory.add("event-1", REVIEW, _scene(), STAGE, CONTEXT)
    memory.add("event-1", {"hypothesis": "replacement", "confidence": 1}, _scene(4), STAGE, CONTEXT)

    data = json.loads((tmp_path / "experience-memory.json").read_text(encoding="utf-8"))
    assert len(data["entries"]) == 1
    assert memory.retrieve(_scene(), STAGE, CONTEXT)[0]["hypothesis"] == REVIEW["hypothesis"]


def test_exact_game_url_stage_player_and_scene_are_scope_requirements(tmp_path):
    frame = _scene()
    memory = ExperienceMemory(tmp_path, GAME_URL)
    memory.add("event-1", REVIEW, frame, STAGE, CONTEXT)

    assert ExperienceMemory(tmp_path, GAME_URL + "/").retrieve(frame, STAGE, CONTEXT) == []
    assert memory.retrieve(frame, "world-2-1", CONTEXT) == []
    assert memory.retrieve(frame, STAGE, {**CONTEXT, "player_location": [0.9, 0.1]}) == []
    assert memory.retrieve(_scene(91), STAGE, CONTEXT) == []


def test_moving_object_geometry_must_remain_compatible_when_present(tmp_path):
    memory = ExperienceMemory(tmp_path, GAME_URL)
    frame = _scene()
    danger={"player_location":[.42,.73],"moving_objects":[{"label":"enemy_rex","cx":.68,"cy":.64,"vx":-.03,"vy":0.,"age":4}]}
    memory.add("event-1", REVIEW, frame, STAGE, danger)

    changed = {**danger, "moving_objects": [{"label":"enemy_rex","cx":.1,"cy":.1,"vx":-.03,"vy":0.,"age":4}]}
    assert memory.retrieve(frame, STAGE, changed) == []


def test_same_scene_with_opposite_relative_motion_is_not_recalled(tmp_path):
    frame = _scene()
    memory = ExperienceMemory(tmp_path, GAME_URL)
    context = {
        "player_location": [0.42, 0.73],
        "player": {"vx": 0.0, "vy": 0.0, "age": 4},
        "moving_objects": [{"id": "a", "label":"enemy_rex", "cx": 0.62, "cy": 0.70,
                            "vx": -0.03, "vy": 0.0, "age": 4}],
    }
    memory.add("approaching", REVIEW, frame, STAGE, context)
    opposite = {**context, "moving_objects": [{"id": "other", "label":"enemy_rex", "cx": 0.62, "cy": 0.70,
                                                 "vx": 0.03, "vy": 0.0, "age": 4}]}

    assert memory.retrieve(frame, STAGE, opposite) == []


def test_live_pixel_velocities_tolerate_noise_but_reject_reversal(tmp_path):
    frame=_scene()
    memory=ExperienceMemory(tmp_path,GAME_URL)
    context={'width':320,'height':240,
        'player':{'bbox':[120,160,20,24],'vx':0.,'vy':0.,'age':5},
        'objects':[{'bbox':[180,160,24,24],'vx':-80.,'vy':0.,'age':5,'label':'enemy_rex'}]}
    memory.add('live-motion',REVIEW,frame,STAGE,context)
    noisy={**context,'objects':[{**context['objects'][0],'vx':-90.,'id':999}]}
    recalled=memory.retrieve(frame,STAGE,noisy)
    assert recalled[0]['match_evidence']['motion_verified'] is True
    reversed_motion={**context,'objects':[{**context['objects'][0],'vx':80.}]}
    assert memory.retrieve(frame,STAGE,reversed_motion)==[]


def test_relative_object_state_matches_across_ids_and_absolute_positions(tmp_path):
    frame = _scene()
    memory = ExperienceMemory(tmp_path, GAME_URL)
    stored = {"player_location": [0.40, 0.70],
              "moving_objects": [{"id": "first", "label":"enemy_rex", "cx": 0.62, "cy": 0.72}]}
    current = {"player_location": [0.48, 0.70],
               "moving_objects": [{"id": "different", "label":"enemy_rex", "cx": 0.70, "cy": 0.72}]}
    memory.add("relative", REVIEW, frame, STAGE, stored)

    recalled = memory.retrieve(frame, STAGE, current)
    assert recalled[0]["match_kind"] == "scene_and_objects"
    assert recalled[0]["match_evidence"]["motion_verified"] is False


def test_static_scene_without_nearby_objects_is_classified_scene_only(tmp_path):
    frame = _scene()
    memory = ExperienceMemory(tmp_path, GAME_URL)
    context = {"player_location": [0.42, 0.73], "moving_objects": []}
    memory.add("empty-scene", REVIEW, frame, STAGE, context)

    recalled = memory.retrieve(frame, STAGE, context)
    assert recalled[0]["match_kind"] == "scene_only"
    assert recalled[0]["match_evidence"]["motion_verified"] is False


def test_translated_camera_scene_is_recalled_after_geometric_alignment(tmp_path):
    frame = _scene(18)
    transform = np.float32([[1, 0, 12], [0, 1, -4]])
    shifted = cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))
    memory = ExperienceMemory(tmp_path, GAME_URL)
    memory.add("camera-shift", REVIEW, frame, STAGE, CONTEXT)
    shifted_context = {
        "player_location": [0.42 + 12 / 320, 0.73 - 4 / 240],
        "moving_objects": [[0.68 + 12 / 320, 0.64 - 4 / 240]],
    }

    recalled = memory.retrieve(shifted, STAGE, shifted_context)
    assert len(recalled) == 1
    assert recalled[0]["match_evidence"]["scene_method"] == "orb_ransac"


def test_distinct_texture_does_not_match_through_geometric_fallback(tmp_path):
    memory = ExperienceMemory(tmp_path, GAME_URL)
    memory.add("texture-a", REVIEW, _scene(18), STAGE, CONTEXT)

    assert memory.retrieve(_scene(91), STAGE, CONTEXT) == []


def test_malformed_index_fails_closed_without_crashing(tmp_path):
    (tmp_path / "experience-memory.json").write_text("{not json", encoding="utf-8")
    memory = ExperienceMemory(tmp_path, GAME_URL)

    assert memory.retrieve(_scene(), STAGE, CONTEXT) == []
    assert memory.add("event-1", REVIEW, _scene(), STAGE, CONTEXT) is None
    assert (tmp_path / "experience-memory.json").read_text(encoding="utf-8") == "{not json"


def test_followup_death_updates_memory_without_claiming_causality(tmp_path):
    memory=ExperienceMemory(tmp_path,GAME_URL)
    memory.add('first',REVIEW,_scene(),STAGE,CONTEXT)
    memory.record_outcome(['first'],'death','second')
    memory.record_outcome(['first'],'death','second')
    recalled=ExperienceMemory(tmp_path,GAME_URL).retrieve(_scene(),STAGE,CONTEXT)
    assert recalled[0]['status']=='unverified'
    assert recalled[0]['alternative_failed_in_later_attempt'] is False
    assert recalled[0]['subsequent_attempts']==[{
        'event_id':'second','outcome':'death','causal_link':'unverified'}]


SUCCESS_EVIDENCE = {
    "kind": "tracked_object_passed", "track_id": "track-7",
    "start_time": 10.0, "end_time": 11.2,
    "forward_progress_px": 31, "separation_px": 28,
    "survived_s": 1.2, "no_contact": True,
}
SUCCESS_ACTIONS = [
    {"action": "right", "hold_ms": 100, "offset_s": 0},
    {"action": "right_jump", "hold_ms": 120, "offset_s": 0.1},
]
MOTION_CONTEXT = {
    "player_location": [0.42, 0.73],
    "player": {"vx": 0.0, "vy": 0.0, "age": 4},
    "moving_objects": [{"id": "track-7", "label":"enemy_rex", "cx": 0.58, "cy": 0.70,
                        "vx": -0.03, "vy": 0.0, "age": 4}],
}


def test_add_success_persists_local_observation_and_sequence(tmp_path):
    memory = ExperienceMemory(tmp_path, GAME_URL)
    entry = memory.add_success("pass-1", _scene(), STAGE, MOTION_CONTEXT,
                               SUCCESS_ACTIONS, SUCCESS_EVIDENCE)

    assert entry["provenance"] == "tracked_local_observation"
    assert entry["review"]["outcome"] == "success"
    assert entry["review"]["status"] == "observed_success"
    assert entry["review"]["suggested_action"] == "right"
    assert entry["review"]["successful_sequence"] == SUCCESS_ACTIONS
    reloaded = ExperienceMemory(tmp_path, GAME_URL)
    recalled = reloaded.similar_successes(_scene(), STAGE, MOTION_CONTEXT)
    assert len(recalled) == 1
    assert recalled[0]["event_id"] == "pass-1"
    assert recalled[0]["outcome"] == "success"
    assert recalled[0]["status"] == "observed_success"
    assert recalled[0]["reference"] == entry["reference"]
    assert recalled[0]["success_evidence"] == SUCCESS_EVIDENCE
    assert recalled[0]["match_evidence"]["motion_verified"] is True
    assert recalled[0]["action_authorized"] is True
    assert recalled[0]["evidence_status"] == "verified_local_passage"


def test_add_success_rejects_missing_or_weak_tracked_evidence(tmp_path):
    memory = ExperienceMemory(tmp_path, GAME_URL)
    invalid = [
        {**SUCCESS_EVIDENCE, "kind": "review"},
        {**SUCCESS_EVIDENCE, "forward_progress_px": 23.9},
        {**SUCCESS_EVIDENCE, "separation_px": float("nan")},
        {**SUCCESS_EVIDENCE, "survived_s": 0.9},
        {**SUCCESS_EVIDENCE, "no_contact": False},
    ]
    for index, evidence in enumerate(invalid):
        try:
            memory.add_success(f"invalid-{index}", _scene(), STAGE, MOTION_CONTEXT,
                               SUCCESS_ACTIONS, evidence)
        except ValueError:
            pass
        else:
            raise AssertionError("evidência incompleta foi aceita")
    assert not (tmp_path / "experience-memory.json").exists()


def test_success_is_prioritized_only_for_verified_local_motion(tmp_path):
    memory = ExperienceMemory(tmp_path, GAME_URL)
    frame = _scene()
    memory.add("baseline", {"hypothesis": "Baseline review.", "confidence": .4},
               frame, STAGE, MOTION_CONTEXT)
    memory.add_success("pass", frame, STAGE, MOTION_CONTEXT,
                       SUCCESS_ACTIONS, SUCCESS_EVIDENCE)

    recalled = memory.retrieve(frame, STAGE, MOTION_CONTEXT)
    assert recalled[0]["event_id"] == "pass"
    assert memory.similar_successes(frame, STAGE,
                                    {**MOTION_CONTEXT, "moving_objects": []}) == []
    static_memory = ExperienceMemory(tmp_path / "static", GAME_URL)
    static_context = {"player_location": [0.42, 0.73], "moving_objects": [{"cx":.58,"cy":.70}]}
    static_memory.add("baseline", {"hypothesis": "Baseline review.", "confidence": .4},
                      frame, STAGE, static_context)
    static_memory.add_success("pass", frame, STAGE, static_context,
                              SUCCESS_ACTIONS, SUCCESS_EVIDENCE)
    unmatched = static_memory.retrieve(frame, STAGE, static_context)
    assert unmatched[0]["event_id"] == "baseline"


def test_later_death_keeps_success_record_but_removes_success_priority(tmp_path):
    memory = ExperienceMemory(tmp_path, GAME_URL)
    frame = _scene()
    memory.add("baseline", {"hypothesis": "Baseline review.", "confidence": .4},
               frame, STAGE, MOTION_CONTEXT)
    memory.add_success("pass", frame, STAGE, MOTION_CONTEXT,
                       SUCCESS_ACTIONS, SUCCESS_EVIDENCE)
    memory.record_outcome(["pass"], "death", "later-death",attributions=[{
        'event_id':'pass','attribution':'local_action_match_not_causal_proof','action':'right_jump'}])

    recalled = memory.retrieve(frame, STAGE, MOTION_CONTEXT)
    assert [entry["event_id"] for entry in recalled] == ["baseline", "pass"]
    success = recalled[1]
    assert success["outcome"] == "success"
    assert success["status"] == "observed_success"
    assert success["successful_sequence"] == SUCCESS_ACTIONS
    assert success["alternative_failed_in_later_attempt"] is True


def test_scene_only_success_does_not_displace_full_motion_danger(tmp_path):
    memory = ExperienceMemory(tmp_path, GAME_URL)
    frame = _scene()
    static = {"player_location": [0.42, 0.73], "moving_objects": []}
    memory.add_success("scene-only-pass", frame, STAGE, static,
                       SUCCESS_ACTIONS, SUCCESS_EVIDENCE)
    memory.add("motion-danger", {
        "hypothesis": "A tracked object approached during this observation.",
        "confidence": .8, "outcome": "death", "trigger_frame_id": "frame-1",
    }, frame, STAGE, MOTION_CONTEXT)

    recalled = memory.retrieve(frame, STAGE, MOTION_CONTEXT)
    assert recalled[0]["event_id"] == "motion-danger"


def test_observed_prefix_persists_as_context_without_authorizing_replay(tmp_path):
    from ai_player_lab.action_replay import ActionReplay
    memory=ExperienceMemory(tmp_path,GAME_URL);frame=_scene()
    context={'width':320,'height':240,'player':{'bbox':[120,160,20,24]},'objects':[]}
    evidence={'kind':'observed_forward_segment','duration_s':4.,'forward_progress_px':48.,'survived_s':1.,'no_contact':True}
    memory.add_prefix('prefix',frame,STAGE,context,[{'action':'right','hold_ms':100,'offset_s':.1}],evidence)
    recalled=ExperienceMemory(tmp_path,GAME_URL).retrieve(frame,STAGE,context)
    assert recalled[0]['status']=='observed_prefix'
    assert recalled[0]['action_authorized'] is True
    assert recalled[0]['evidence_status']=='verified_local_prefix'
    replay=ActionReplay()
    assert not replay.start(recalled[0],100.,context['player']['bbox'],1)
    assert replay.next(100.11,1,context['player']['bbox'],'gameplay',True,[],[]) is None


def test_retrieve_preloads_decoded_references_and_reuses_them(tmp_path, monkeypatch):
    import ai_player_lab.experience as experience_module
    writer=ExperienceMemory(tmp_path,GAME_URL)
    frame=_scene()
    writer.add('one',REVIEW,frame,STAGE,CONTEXT)
    original=experience_module.cv2.imread
    reads=[]
    def counted(path,*args,**kwargs):
        reads.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(experience_module.cv2,'imread',counted)

    reader=ExperienceMemory(tmp_path,GAME_URL)
    assert len(reads)==1  # PNG decoded once while the index is loaded.
    assert reader.retrieve(frame,STAGE,CONTEXT)
    after_first=len(reads)
    assert reader.retrieve(frame,STAGE,CONTEXT)
    assert len(reads)==after_first

    # A sibling writer changes the manifest. The reader reloads the index and
    # its bounded reference cache, then reuses those decoded images.
    writer.add('two',REVIEW,frame,STAGE,CONTEXT)
    reader.retrieve(frame,STAGE,CONTEXT)
    after_invalidation=len(reads)
    assert after_invalidation-after_first==2
    reader.retrieve(frame,STAGE,CONTEXT)
    assert len(reads)==after_invalidation


def test_revalidate_checks_current_scene_player_and_object_geometry(tmp_path):
    memory=ExperienceMemory(tmp_path,GAME_URL)
    frame=_scene()
    entry=memory.add('advice',REVIEW,frame,STAGE,CONTEXT)
    candidate=[{'event_id':entry['event_id']}]
    assert memory.revalidate(candidate,frame,STAGE,CONTEXT)
    assert memory.revalidate(candidate,_scene(91),STAGE,CONTEXT)==[]
    assert memory.revalidate(candidate,frame,STAGE,
                              {**CONTEXT,'player_location':[.9,.1]})==[]
    # Legacy coordinate-only contexts carry no semantic threat evidence.
    assert memory.revalidate(candidate,frame,STAGE,
                              {**CONTEXT,'moving_objects':[[.1,.1]]})[0]['match_kind']=='scene_only'


def test_unverified_review_stays_hypothesis_and_cannot_authorize_sequence(tmp_path):
    frame=_scene();memory=ExperienceMemory(tmp_path,GAME_URL)
    review={**REVIEW,'outcome':'death','suggested_action':'right_jump',
            'trigger_frame_id':'actionable-frame',
            'successful_sequence':[{'action':'right_jump','hold_ms':100,'offset_s':.1}]}
    memory.add('unverified-death',review,frame,STAGE,CONTEXT)
    result=memory.retrieve(frame,STAGE,CONTEXT)[0]
    assert result['status']=='unverified'
    assert result['action_authorized'] is False
    assert result['evidence_status']=='unverified_model_review'
    assert result['hypothesis_status']=='model_claim_unverified'
    assert result['outcome']=='unverified'
    assert result['reviewed_outcome']=='death'
    assert result['suggested_action'] is None
    assert result['successful_sequence'] is None
    assert result['match_kind']=='scene_only'


def test_model_claimed_success_without_local_evidence_is_not_success_authority(tmp_path):
    frame=_scene();memory=ExperienceMemory(tmp_path,GAME_URL)
    review={'outcome':'success','hypothesis':'model said success','successful_sequence':SUCCESS_ACTIONS,
            'status':'observed_success','confidence':1.0}
    memory.add('claimed-success',review,frame,STAGE,MOTION_CONTEXT)
    result=memory.retrieve(frame,STAGE,MOTION_CONTEXT)[0]
    assert result['outcome']=='unverified'
    assert result['reviewed_outcome']=='success'  # Preserve the historical claim.
    assert result['status']=='unverified'
    assert result['action_authorized'] is False


def test_locally_confirmed_life_loss_is_a_fact_but_never_a_success_action(tmp_path):
    frame=_scene();memory=ExperienceMemory(tmp_path,GAME_URL)
    review={**REVIEW,'outcome':'death','outcome_source':'local_life_counter',
        'trigger_frame_id':'actionable-frame',
        'life_evidence':{'confirmed':True,'status':'life_lost','before_value':3,'after_value':2,'decrement':1}}
    memory.add('local-death',review,frame,STAGE,CONTEXT)
    result=memory.retrieve(frame,STAGE,CONTEXT)[0]
    assert result['outcome']=='death'
    assert result['reviewed_outcome']=='death'
    assert result['evidence_status']=='verified_local_life_loss'
    assert result['local_event_evidence']['decrement']==1
    assert result['action_authorized'] is False
    assert result['successful_sequence'] is None


def test_explicitly_predicted_and_unknown_tracks_are_ignored(tmp_path):
    frame=_scene();memory=ExperienceMemory(tmp_path,GAME_URL)
    context={'width':320,'height':240,'player':{'bbox':[120,160,20,24]},'objects':[
        {'bbox':[180,160,24,24],'vx':-80.,'vy':0.,'age':5,'label':'predicted_enemy_rex'},
        {'bbox':[80,160,24,24],'vx':-80.,'vy':0.,'age':5,'label':'unknown_motion'}]}
    memory.add('legacy-semantics',REVIEW,frame,STAGE,context)
    recalled=memory.retrieve(frame,STAGE,context)[0]
    assert recalled['match_kind']=='scene_only'
    assert recalled['match_evidence']['nearby_objects']==0
    assert recalled['action_authorized'] is False
