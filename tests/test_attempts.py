import copy
import numpy as np
from ai_player_lab.attempts import AttemptCycle

FRAME=np.full((224,256,3),80,np.uint8)
BOX=[20,170,20,24]

def recorded(root):
    cycle=AttemptCycle(root);cycle.begin(FRAME,BOX,10.,'1','first')
    for i in range(60):
        t=i*.05
        cycle.record('right',100,10+t,[20+t*25,170,20,24],source='local_controller')
        cycle.observe(10+t,[20+t*25,170,20,24])
    return cycle

def finish_prefix_before_patch(cycle,box=BOX):
    if cycle._resume_visual_patch_after_prefix:
        cycle.next(cycle.started_at+cycle._awaiting_end+.01,box)

def review(action='right_jump',at=2.5):
    return {'outcome':'death','confidence':.9,'outcome_confidence':.99,'outcome_source':'local_life_counter',
        'life_evidence':{'confirmed':True,'status':'life_lost','before_value':3,'after_value':2,'decrement':1},'trigger_frame_id':'before',
        'retry_plan':{'at_s':at,'action':action,'hold_ms':200}}

PATCH_CONTEXT={'width':256,'height':224,'player':{'bbox':BOX},
    'character_state':{'grounded':True,'grounded_evidence':'visual_estimate','can_start_jump':True},'objects':[]}
FRAMES=[{'frame_hash':'before','captured_at':12.6,'scene':'gameplay',
    'image':FRAME,'context':PATCH_CONTEXT}]


def test_visual_retry_waits_for_evidence_then_runs_only_short_patch(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    assert cycle.begin(FRAME,BOX,20.,'1','retry')
    assert cycle.mode=='prefix'
    finish_prefix_before_patch(cycle)
    assert cycle.mode=='awaiting_visual_patch' and not cycle.active
    noisy=np.full_like(FRAME,88)
    assert not cycle.arm_visual_patch(noisy,[120,170,20,24],20.79,can_jump=True,context=PATCH_CONTEXT)
    assert cycle.patch_gate['reason']=='player_anchor_mismatch'
    assert not cycle.arm_visual_patch(FRAME,BOX,20.79,can_jump=False,context=PATCH_CONTEXT)
    assert cycle.patch_gate['reason']=='retry_too_early'
    assert not cycle.arm_visual_patch(FRAME,BOX,20.79,can_jump=True,context=PATCH_CONTEXT)
    assert cycle.patch_gate['reason']=='retry_too_early'
    cycle._progress_observed=True;cycle._progress=cycle._visual_trigger_progress_px
    assert cycle.arm_visual_patch(FRAME,BOX,22.61,can_jump=True,context=PATCH_CONTEXT)
    assert cycle.plan['nominal_at_s']==2.5
    assert len(cycle.schedule)==1
    patch=cycle.next(22.62,BOX)
    assert patch['patch'] and patch['action']=='right_jump'
    cycle.record(patch['action'],patch['hold_ms'],22.62,BOX,source=patch['source'])
    assert cycle.patch_executed


def test_retry_requires_scene_and_player_even_when_elapsed_time_and_progress_differ(tmp_path):
    cycle=recorded(tmp_path)
    frames=FRAMES+[{'frame_hash':'death','captured_at':13.5,'scene':'gameplay',
        'image':FRAME,'context':PATCH_CONTEXT}]
    cycle.finish({**review(at=3.109),'trigger_frame_id':'death'},frames)
    cycle.data['stages']['1'].pop('prefix',None)
    candidate_context={**PATCH_CONTEXT,'objects':[{'label':'enemy_rex','bbox':[90,170,20,20]}]}
    cycle.data['stages']['1']['candidate']['trigger_threats']=[{'kind':'enemy_rex','relative':[.25,0.0]}]
    cycle._pending_test_context=candidate_context
    assert cycle.begin(FRAME,BOX,20.,'1','retry-early')
    finish_prefix_before_patch(cycle)
    early_time=cycle.started_at+.79
    # Typed, position-matched threat evidence can make the visual anchor primary.
    assert cycle.arm_visual_patch(FRAME,BOX,early_time,can_jump=True,context=candidate_context)
    assert cycle.patch_gate['accepted']


def test_unanchored_plan_is_evidence_pending_not_timed_replay(tmp_path):
    cycle=recorded(tmp_path)
    result=cycle.finish(review(),[{'frame_hash':'before','captured_at':13.,'scene':'gameplay'}])
    assert result['reason']=='patch_missing_visual_anchor'
    assert cycle.data['stages']['1']['candidate']['status']=='pending_evidence'
    assert cycle.begin(FRAME,BOX,20.,'1','retry')
    assert cycle.active and cycle.mode=='prefix'
    assert cycle.plan is None
    assert all(step['action']!='right_jump' for step in cycle.schedule)


def test_pending_retry_replays_only_bounded_validated_prefix(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    stored=cycle.data['stages']['1']
    original=[{'offset_s':i*.5,'action':'right','hold_ms':500,
        'player_box':[20+i,170,20,24],'source':'local_controller'} for i in range(16)]
    stored['prefix']={'until_s':9.,'sequence':original,'status':'observed_once'}
    candidate={'status':'pending_evidence','plan':{'at_s':8.,'action':'right_jump','hold_ms':100},
        'unrelated':'leave-me'}
    stored['candidate']=candidate.copy()
    original_candidate=stored['candidate'].copy()

    assert cycle.begin(FRAME,BOX,20.,'1','pending-prefix')
    assert cycle.mode=='prefix' and cycle.reason=='evidence_pending_replaying_observed_prefix'
    assert cycle.plan is None and cycle._awaiting_end==4.
    assert stored['candidate']==original_candidate
    assert all(step['action']!='right_jump' for step in cycle.schedule)
    assert max(step['offset_s']+step['hold_ms']/1000 for step in cycle.schedule)<=4.
    assert cycle.next(20.01,BOX)['source']=='timed_prefix'
    assert not cycle.arm_visual_patch(FRAME,BOX,20.02,can_jump=True,context=PATCH_CONTEXT)
    assert cycle.patch_gate['reason']=='not_waiting_for_visual_patch'
    assert stored['candidate']['status']=='pending_evidence'


def test_pending_retry_prefix_stops_half_second_before_candidate(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    stored=cycle.data['stages']['1']
    stored['prefix']={'until_s':8.,'sequence':[
        {'offset_s':0.,'action':'right','hold_ms':500,'player_box':BOX},
        {'offset_s':.5,'action':'right','hold_ms':500,'player_box':BOX},
        {'offset_s':1.,'action':'right','hold_ms':500,'player_box':BOX},
        {'offset_s':1.5,'action':'right','hold_ms':500,'player_box':BOX},
        {'offset_s':2.,'action':'right','hold_ms':500,'player_box':BOX},
        {'offset_s':2.5,'action':'right','hold_ms':500,'player_box':BOX},
        {'offset_s':3.,'action':'right','hold_ms':500,'player_box':BOX}], 'status':'observed_once'}
    stored['candidate']={'status':'needs_new_plan','plan':{'at_s':3.,'action':'right_jump','hold_ms':100}}

    assert cycle.begin(FRAME,BOX,20.,'1','pending-prefix-short')
    assert cycle._awaiting_end==2.5
    assert max(step['offset_s']+step['hold_ms']/1000 for step in cycle.schedule)<=2.5
    assert all(step['action']!='right_jump' for step in cycle.schedule)


def test_pending_candidate_from_other_strategy_does_not_cut_validated_jump(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    stored=cycle.data['stages']['1']
    prefix={'until_s':4.,'status':'observed_once','phase_id':'validated-phase',
        'sequence':[
            {'offset_s':0.,'action':'right','hold_ms':500,'player_box':BOX},
            {'offset_s':.5,'action':'wait','hold_ms':500,'player_box':BOX},
            {'offset_s':1.,'action':'wait','hold_ms':250,'player_box':BOX},
            {'offset_s':1.25,'action':'right_jump','hold_ms':100,'player_box':BOX},
            {'offset_s':1.35,'action':'right','hold_ms':250,'player_box':BOX},
        ]}
    stored['prefix']=prefix
    candidate={'status':'pending_evidence','plan':{'at_s':1.546,'action':'right_jump','hold_ms':100},
        'failure_at_s':1.,'from_phase_id':'other-phase'}
    stored['candidate']=candidate.copy()

    assert cycle.begin(FRAME,BOX,20.,'1','different-strategy')
    assert cycle.reason=='observed_prefix_strategy_differs_from_candidate'
    assert cycle.mode=='prefix' and cycle.plan is None
    assert any(step['action']=='right_jump' and step['offset_s']==1.25 for step in cycle.schedule)
    assert cycle._awaiting_end==4.
    assert stored['candidate']==candidate


def test_ready_retry_replays_prefix_then_resumes_same_visual_patch(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    stored=cycle.data['stages']['1']
    anchor=dict(stored['candidate'])
    prefix=dict(stored['prefix'])

    assert cycle.begin(FRAME,BOX,20.,'1','ready-prefix')
    assert cycle.mode=='prefix' and cycle.plan['action']=='right_jump'
    assert cycle._resume_visual_patch_after_prefix
    assert cycle._awaiting_end<=min(4.,prefix['until_s'],anchor['plan']['at_s']-.5)
    assert stored['candidate']['trigger_signature']==anchor['trigger_signature']
    assert all(step['action']!='right_jump' for step in cycle.schedule)
    assert cycle.next(20.+cycle._awaiting_end+.01,BOX) is None
    assert cycle.mode=='awaiting_visual_patch'
    assert cycle.reason=='observed_prefix_completed_waiting_for_patch'
    assert stored['candidate']['status']=='pending'
    assert stored['candidate']['trigger_signature']==anchor['trigger_signature']
    assert not any(t.get('outcome')=='patch_not_reached' for t in stored['trials'])


def test_jev_death_does_not_truncate_unreplayed_prefix(tmp_path):
    cycle=AttemptCycle(tmp_path)
    cycle.begin(FRAME,BOX,10.,'1','fresh-jev')
    for i in range(60):
        t=i*.05
        cycle.record('right',100,10+t,[20+t*25,170,20,24],source='local_controller')
        cycle.observe(10+t,[20+t*25,170,20,24])
    stored=cycle.data['stages'].setdefault('1',{'trials':[]})
    stored['prefix']={'until_s':8.,'sequence':[
        {'offset_s':0.,'action':'right','hold_ms':500,'player_box':BOX},
        {'offset_s':7.5,'action':'right','hold_ms':500,'player_box':BOX}],
        'status':'observed_once','evidence':'prior_validated','phase_id':'older'}
    original=copy.deepcopy(stored['prefix'])
    result=cycle.finish(review(),FRAMES)
    assert result is not None
    assert stored['prefix']==original
    assert stored.get('prefix_history',[])==[]


def test_model_only_death_never_creates_retry_candidate(tmp_path):
    cycle=recorded(tmp_path)
    hypothesis={**review(),'outcome_source':'model_review','life_evidence':{'confirmed':False}}
    assert cycle.finish(hypothesis,FRAMES) is None
    assert cycle.patch_gate['reason']=='local_death_unconfirmed'
    assert cycle.data['stages']=={}


def test_jump_patch_requires_confirmed_ground_at_arm_time(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    cycle.begin(FRAME,BOX,20.,'1','retry')
    finish_prefix_before_patch(cycle)
    airborne={**PATCH_CONTEXT,'character_state':{'grounded':None,'can_start_jump':False}}
    assert not cycle.arm_visual_patch(FRAME,BOX,23.,can_jump=True,context=airborne)
    assert cycle.patch_gate['reason']=='ground_not_confirmed'


def _context_at(box,*,grounded,threat=False):
    context={'width':256,'height':224,'player':{'bbox':box},
        'character_state':({'grounded':True,'grounded_evidence':'visual_estimate','can_start_jump':True}
            if grounded else {'grounded':False,'grounded_evidence':'airborne','can_start_jump':False}),
        'objects':[]}
    if threat:
        context['objects']=[{'label':'enemy_rex','bbox':[90,170,20,20]}]
    return context


def test_airborne_retry_origin_aligns_to_nearby_preceding_ground_threat(tmp_path):
    cycle=recorded(tmp_path)
    frames=[
        {'frame_hash':'ground-threat','captured_at':11.5,'scene':'gameplay','image':FRAME,
         'context':_context_at(BOX,grounded=True,threat=True)},
        {'frame_hash':'origin-airborne','captured_at':13.,'scene':'gameplay','image':FRAME,
         'context':_context_at(BOX,grounded=False)},
        {'frame_hash':'death','captured_at':14.5,'scene':'gameplay','image':FRAME,
         'context':_context_at(BOX,grounded=False)},
    ]
    result=cycle.finish({**review(at=3.),'trigger_frame_id':'death'},frames)
    candidate=cycle.data['stages']['1']['candidate']
    plan=candidate['plan']
    assert result['reason']=='retry_aligned_to_preceding_ground_threat'
    assert plan['at_s']==1.5 and plan['requested_at_s']==3.
    assert plan['ground_alignment']['kind']=='preceding_ground_known_threat'
    assert plan['ground_alignment']['offset_s']==-1.5
    assert candidate['trigger_frame_id']=='ground-threat'


def test_airborne_retry_does_not_reschedule_without_nearby_known_threat(tmp_path):
    cases=[('no-threat',12.,False),('too-distant',10.5,True)]
    for name,ground_at,threat in cases:
        cycle=recorded(tmp_path/name)
        frames=[
            {'frame_hash':'ground','captured_at':ground_at,'scene':'gameplay','image':FRAME,
             'context':_context_at(BOX,grounded=True,threat=threat)},
            {'frame_hash':'origin-airborne','captured_at':13.,'scene':'gameplay','image':FRAME,
             'context':_context_at(BOX,grounded=False)},
            {'frame_hash':'death','captured_at':14.5,'scene':'gameplay','image':FRAME,
             'context':_context_at(BOX,grounded=False)},
        ]
        result=cycle.finish({**review(at=3.),'trigger_frame_id':'death'},frames)
        candidate=cycle.data['stages']['1']['candidate']
        assert result['reason']=='patch_missing_ground_anchor'
        assert candidate['status']=='pending_evidence'
        assert candidate['plan']['at_s']==3. and 'ground_alignment' not in candidate['plan']


def test_relative_threat_context_must_still_match_when_anchor_has_threat(tmp_path):
    cycle=recorded(tmp_path)
    danger_context={**PATCH_CONTEXT,'objects':[{'label':'enemy_rex','bbox':[90,170,20,20]}]}
    frame={**FRAMES[0],'context':danger_context}
    cycle.finish(review(),[frame]);cycle.begin(FRAME,BOX,20.,'1','retry')
    finish_prefix_before_patch(cycle)
    changed={**PATCH_CONTEXT,'objects':[{'label':'enemy_rex','bbox':[35,170,20,20]}]}
    assert not cycle.arm_visual_patch(FRAME,BOX,23.,can_jump=True,context=changed)
    assert cycle.patch_gate['reason']=='relative_threat_mismatch'


def test_scene_signature_masks_player_pose_but_retains_surrounding_scene():
    from ai_player_lab.attempts import visual_signature
    base=np.full((224,256,3),80,np.uint8)
    alternate=base.copy();alternate[170:194,20:40]=(0,0,255)
    assert visual_signature(base,BOX)==visual_signature(alternate,BOX)
    changed=base.copy();changed[160:200,60:100]=(255,0,0)
    assert visual_signature(base,BOX)!=visual_signature(changed,BOX)


def test_legacy_unanchored_candidate_is_quarantined(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    candidate=cycle.data['stages']['1']['candidate'];candidate.pop('trigger_signature');cycle.save()
    assert not cycle.begin(FRAME,BOX,20.,'1','retry')
    assert candidate['status']=='pending_evidence'


def test_patch_not_reached_is_bounded_and_does_not_replace_prefix(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    prefix=cycle.data['stages']['1']['prefix'].copy()
    for i in range(2):
        cycle.begin(FRAME,BOX,20.,'1',f'retry-{i}')
        cycle.cancel('tracking_lost');cycle.cancel('tracking_lost')
    trials=cycle.data['stages']['1']['trials']
    assert len([t for t in trials if t['outcome']=='patch_not_reached'])==2
    assert not cycle.begin(FRAME,BOX,30.,'1','blocked')
    assert cycle.reason=='patch_not_reached_limit'
    assert cycle.data['stages']['1']['prefix']==prefix


def test_tracking_loss_and_skipped_patch_abort(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    cycle.begin(FRAME,BOX,20.,'1','retry');cycle.arm_visual_patch(FRAME,BOX,21.,can_jump=True,context=PATCH_CONTEXT)
    cycle.next(21.02,None,valid=False);cycle.next(21.25,None,valid=False)
    assert not cycle.active and cycle.reason=='tracking_lost'
    cycle.begin(FRAME,BOX,30.,'1','retry2');cycle.arm_visual_patch(FRAME,BOX,31.,can_jump=True,context=PATCH_CONTEXT)
    assert cycle.next(31.18,BOX) is None and cycle.reason=='schedule_late'


def test_uncertain_death_and_bad_identity_never_create_script(tmp_path):
    cycle=recorded(tmp_path)
    assert cycle.finish({**review(),'outcome':'uncertain'},FRAMES) is None
    assert cycle.data['stages']=={}
    cycle.actions[12]['player_box']=[220,170,20,24]
    assert cycle.finish(review(),FRAMES)['rejected']=='visual_identity_unreliable'


def test_failed_executed_patch_is_not_reproposed(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    cycle.begin(FRAME,BOX,10.,'1','retry');cycle.arm_visual_patch(FRAME,BOX,12.49,can_jump=True,context=PATCH_CONTEXT)
    cycle.record('right_jump',200,12.5,BOX,source='timed_experiment')
    assert cycle.patch_executed
    assert cycle.finish(review(),FRAMES)['rejected']=='already_failed_plan'


def test_visual_alignment_does_not_change_plan_identity(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    cycle.begin(FRAME,BOX,10.,'1','retry');finish_prefix_before_patch(cycle)
    cycle.arm_visual_patch(FRAME,BOX,15.,can_jump=True,context=PATCH_CONTEXT)
    cycle.record('right_jump',200,15.01,BOX,source='timed_experiment')
    assert cycle.patch_executed
    assert cycle.finish(review(),FRAMES)['rejected']=='already_failed_plan'


def test_patch_survival_can_promote_camera_relative_progress(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(),FRAMES)
    cycle.begin(FRAME,BOX,10.,'1','retry');cycle.arm_visual_patch(FRAME,BOX,12.49,can_jump=True,context=PATCH_CONTEXT)
    cycle.record('right_jump',200,12.5,BOX,source='timed_experiment')
    assert not cycle.observe(13.2,BOX,{'valid':True,'dx':-40})
    assert cycle.observe(14.2,BOX,{'valid':True,'dx':-100})
    assert cycle.data['stages']['1']['candidate']['status']=='passed_local_window'


def test_visual_retry_uses_precise_scene_instead_of_previous_run_clock(tmp_path):
    cycle=recorded(tmp_path)
    context=copy.deepcopy(PATCH_CONTEXT)
    context['objects']=[]
    cycle.finish(review(),[{**FRAMES[0],'context':context}])
    cycle.begin(FRAME,BOX,20.,'1','retry')
    finish_prefix_before_patch(cycle)
    cycle._progress_observed=True;cycle._progress=200
    different=np.full_like(FRAME,220)
    assert not cycle.arm_visual_patch(different,BOX,20.9,can_jump=True,context=context)
    assert cycle.patch_gate['reason']=='local_scene_mismatch'
    assert not cycle.arm_visual_patch(FRAME,BOX,20.9,can_jump=True,context=PATCH_CONTEXT)
    assert cycle.patch_gate['reason']=='retry_too_early'
    cycle._progress_observed=True;cycle._progress=cycle._visual_trigger_progress_px
    assert cycle.arm_visual_patch(FRAME,BOX,23.0,can_jump=True,context=PATCH_CONTEXT)
    assert cycle.plan['alignment']=='visual_scene'


def test_noisy_scene_uses_time_and_progress_as_bounded_fallback(tmp_path):
    cycle=recorded(tmp_path)
    frames=FRAMES+[{'frame_hash':'death','captured_at':13.5,'scene':'gameplay',
        'image':FRAME,'context':PATCH_CONTEXT}]
    cycle.finish({**review(at=3.109),'trigger_frame_id':'death'},frames)
    cycle.data['stages']['1'].pop('prefix',None)
    candidate_context={**PATCH_CONTEXT,'objects':[{'label':'enemy_rex','bbox':[90,170,20,20]}]}
    cycle.data['stages']['1']['candidate']['trigger_threats']=[{'kind':'enemy_rex','relative':[.25,0.0]}]
    cycle.begin(FRAME,BOX,20.,'1','retry')
    finish_prefix_before_patch(cycle)
    cycle._progress_observed=True;cycle._progress=1000
    noisy=np.full_like(FRAME,88)
    assert not cycle.arm_visual_patch(noisy,BOX,20.79,can_jump=True,context=candidate_context)
    assert cycle.patch_gate['reason']=='retry_too_early'
    assert not cycle.arm_visual_patch(noisy,BOX,23.0,can_jump=True,context=candidate_context)
    assert cycle.patch_gate['reason']=='retry_progress_mismatch'


def test_moving_threat_can_reanchor_noisy_scene_only_with_relative_match(tmp_path):
    cycle=recorded(tmp_path)
    trigger_context={**PATCH_CONTEXT,'objects':[{'label':'enemy_projectile_large','bbox':[90,170,20,20]}]}
    frames=[{'frame_hash':'before','captured_at':13.5,'scene':'gameplay',
        'image':FRAME,'context':trigger_context}]
    cycle.finish({**review(at=3.109),'trigger_frame_id':'before'},frames)
    cycle.begin(FRAME,BOX,20.,'1','moving-threat')
    finish_prefix_before_patch(cycle)
    cycle._progress_observed=True
    cycle._progress=cycle._visual_trigger_progress_px
    noisy=np.full_like(FRAME,220)
    matching={**PATCH_CONTEXT,'objects':[{'label':'enemy_projectile_large','bbox':[94,170,20,20]}]}
    assert cycle.arm_visual_patch(noisy,BOX,23.0,can_jump=True,context=matching)
    assert cycle.plan['alignment']=='visual_scene'
    assert cycle.plan['visual_error']>.12

    cycle.begin(FRAME,BOX,30.,'1','wrong-threat-position')
    finish_prefix_before_patch(cycle)
    cycle._progress_observed=True
    cycle._progress=cycle._visual_trigger_progress_px
    mismatched={**PATCH_CONTEXT,'objects':[{'label':'enemy_projectile_large','bbox':[150,170,20,20]}]}
    assert not cycle.arm_visual_patch(noisy,BOX,33.0,can_jump=True,context=mismatched)
    assert cycle.patch_gate['reason']=='relative_threat_mismatch'



def test_exact_player_relative_scene_can_match_after_camera_reframes_player(tmp_path):
    cycle=recorded(tmp_path)
    cycle.finish(review(),FRAMES)
    cycle.begin(FRAME,BOX,20.,'1','camera-reframed')
    finish_prefix_before_patch(cycle)
    cycle._progress_observed=True;cycle._progress=cycle._visual_trigger_progress_px
    assert not cycle.arm_visual_patch(FRAME,[140,170,20,24],23.0,can_jump=True,context=PATCH_CONTEXT)
    assert cycle.patch_gate['reason']=='player_anchor_mismatch'


def test_ready_candidate_replays_its_observed_route_before_anchor(tmp_path):
    cycle=recorded(tmp_path);cycle.finish(review(at=2.5),FRAMES)
    stored=cycle.data['stages']['1']
    candidate=stored['candidate']
    candidate['sequence']=[{'offset_s':i*.1,'action':'right','hold_ms':100,
        'player_box':[20+i,170,20,24],'source':'local_controller'} for i in range(20)]
    stored['prefix']={'until_s':4.,'sequence':[{'offset_s':0.,'action':'wait','hold_ms':100,
        'player_box':BOX,'source':'local_controller'}],'status':'observed_once'}
    assert cycle.begin(FRAME,BOX,20.,'1','retry-route')
    assert cycle.mode=='prefix'
    assert cycle.schedule and all(step['action']=='right' for step in cycle.schedule)
    assert cycle._resume_visual_patch_after_prefix
