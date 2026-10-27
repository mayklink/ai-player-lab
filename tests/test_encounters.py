import numpy as np
from ai_player_lab.encounters import EncounterRecorder,PrefixRecorder,review_frames


def test_review_always_contains_actual_post_event_frame_between_samples():
    history=[{'frame_hash':str(index),'scene':'gameplay'} for index in range(40)]
    current={'frame_hash':'new-map','scene':'map'}
    selected=review_frames(history,current)
    assert selected[-1] is current
    assert len(selected)<=8
    assert history[-1]['scene']=='gameplay'
from ai_player_lab.types import Observation,Track


def observation(index,missing=False,contact=False,label='enemy_rex',grounded=True,scene='gameplay'):
    player=Track(0,(100.,160.,20.,20.),0.,0.,.9,age=5)
    obj=Track(7,(160.-5*index,120. if not contact else 160.,20.,20.),-50.,0.,.9,age=5,label=label)
    obs=Observation(index,1.+index*.1,'frame','scene',256,224,
        player,[] if missing else [obj],{'valid':True,'dx':-4.,'dy':0.})
    obs.character_state={'grounded':grounded,'jump_phase':'grounded' if grounded else 'rising'}
    obs.hypotheses=[{'source':'local_vision','scene':scene}]
    return obs


def test_record_requires_crossing_and_survival_and_forward_progress():
    recorder=EncounterRecorder();frame=np.zeros((224,256,3),np.uint8)
    actions=[{'action':'right_jump','hold_ms':200,'executed_at':1.1}]
    events=[]
    for index in range(34):events+=recorder.update(frame,observation(index),actions)
    assert len(events)==1
    assert events[0]['evidence']['survived_s']>=1
    assert events[0]['evidence']['forward_progress_px']>=24
    assert events[0]['evidence']['separation_px']>=24
    assert events[0]['actions'][0]['action']=='right_jump'


def test_airborne_pass_candidate_waits_for_confirmed_landing():
    recorder=EncounterRecorder();frame=np.zeros((224,256,3),np.uint8)
    for index in range(34):
        airborne=index>=24
        events=recorder.update(frame,observation(index,grounded=not airborne),[
            {'action':'right_jump','hold_ms':200,'executed_at':1.1}])
        assert events==[]
    assert recorder.active
    landed=recorder.update(frame,observation(34,grounded=True),[
        {'action':'right_jump','hold_ms':200,'executed_at':1.1}])
    assert len(landed)==1
    assert landed[0]['evidence']['survived_s']>=1


def test_unknown_ground_state_and_terminal_scene_cannot_certify_passage():
    for grounded,scene in ((None,'gameplay'),(True,'death')):
        recorder=EncounterRecorder();frame=np.zeros((224,256,3),np.uint8)
        events=[]
        for index in range(36):
            events+=recorder.update(frame,observation(index,grounded=grounded,scene=scene),[
                {'action':'right_jump','hold_ms':200,'executed_at':1.1}])
        assert events==[]
        if scene=='death':assert not recorder.active


def test_disappearing_object_is_not_a_success():
    recorder=EncounterRecorder();frame=np.zeros((224,256,3),np.uint8)
    for index in range(10):
        assert recorder.update(frame,observation(index,missing=index>3),[])==[]
    assert not recorder.active


def test_brief_occlusion_requires_reobserved_pass_and_full_survival_window():
    recorder=EncounterRecorder();frame=np.zeros((224,256,3),np.uint8)
    events=[]
    for index in range(41):
        events+=recorder.update(frame,observation(index,missing=index==25),[
            {'action':'right_jump','hold_ms':200,'executed_at':1.1}])
        if index<=35:assert events==[]
    assert len(events)==1
    assert events[0]['evidence']['survived_s']>=1


def test_background_unknown_and_predicted_tracks_never_start_pass_memory():
    for label in ('background','unknown_motion','predicted_enemy_rex','static_prop'):
        recorder=EncounterRecorder();frame=np.zeros((224,256,3),np.uint8)
        for index in range(40):
            assert recorder.update(frame,observation(index,label=label),[])==[]
        assert not recorder.active


def test_contact_cancels_a_candidate():
    recorder=EncounterRecorder();frame=np.zeros((224,256,3),np.uint8)
    for index in range(12):recorder.update(frame,observation(index),[])
    assert recorder.active
    assert recorder.update(frame,observation(12,contact=True),[])==[]
    assert not recorder.active


def test_elapsed_time_without_motion_is_not_a_success():
    recorder=EncounterRecorder();frame=np.zeros((224,256,3),np.uint8)
    for index in range(34):
        obs=observation(index);obs.camera['dx']=0.
        assert recorder.update(frame,obs,[{'action':'wait','hold_ms':100,'executed_at':1.1}])==[]


def test_forward_prefix_requires_advancing_before_survival_window():
    recorder=PrefixRecorder();frame=np.zeros((224,256,3),np.uint8);result=None
    actions=[{'action':'right','hold_ms':100,'executed_at':1.1}]
    for index in range(52):
        obs=observation(index);obs.objects=[];obs.hazards=[]
        result=recorder.update(frame,obs,actions) or result
    assert result and result['evidence']['duration_s']==4
    assert result['evidence']['survived_s']>=1
    recorder=PrefixRecorder();result=None
    for index in range(52):
        obs=observation(index);obs.objects=[];obs.hazards=[];obs.camera['dx']=0. if index<=40 else -4.
        result=recorder.update(frame,obs,actions) or result
    assert result is None


def test_prefix_waits_for_ground_confirmation_and_resets_on_terminal_scene():
    recorder=PrefixRecorder();frame=np.zeros((224,256,3),np.uint8)
    actions=[{'action':'right','hold_ms':100,'executed_at':1.1}]
    for index in range(52):
        obs=observation(index,grounded=index<40);obs.objects=[];obs.hazards=[]
        assert recorder.update(frame,obs,actions) is None
    assert recorder.start is not None
    landed=observation(52,grounded=True);landed.objects=[];landed.hazards=[]
    result=recorder.update(frame,landed,actions)
    assert result is not None
    assert result['evidence']['survived_s']>=1

    recorder=PrefixRecorder();result=None
    for index in range(52):
        obs=observation(index,grounded=True,scene='death' if index>=45 else 'gameplay')
        obs.objects=[];obs.hazards=[]
        result=recorder.update(frame,obs,actions) or result
    assert result is None
    assert recorder.start is None
