import copy
import json
import httpx
import numpy as np
from ai_player_lab.attempts import AttemptCycle
from ai_player_lab.providers import JevPolicy
from ai_player_lab.types import Observation,Track
from ai_player_lab.experience import _success_sequence
from ai_player_lab.local_controller import LocalController,reactive_replay_guard

FRAME=np.full((224,256,3),80,np.uint8)
BOX=[20,170,20,24]


def test_current_rex_takes_control_from_replay_and_reactive_controller_can_jump():
    rex=Track(7,(65,170,16,28),vx=-30,confidence=.9,age=5,label='enemy_rex')
    obs=Observation(1,1.,'f','s',256,224,Track(0,tuple(BOX),vx=80),[rex],{},
        hazards=[{'kind':'approaching_object','object_id':7,'ttc_ms':300}],
        character_state={'grounded':True,'can_start_jump':True})
    assert reactive_replay_guard(obs) is not None
    controller=LocalController();controller.accept('right',.9,0,100)
    controller.accept('left',1.,0,100,bounded=True,replay=True)
    controller.resume_reactive(1.02,0)
    command=controller.update(obs,1.03)
    assert command['action']=='right_jump'
    assert command['reason']=='ground_threat_jump'


def test_reactive_jev_intent_does_not_expire_between_quarter_second_decisions():
    controller=LocalController();controller.accept('right',1.,0,100)
    obs=Observation(1,1.2,'f','s',256,224,Track(0,tuple(BOX)),[],{},
        character_state={'grounded':True,'can_start_jump':True})
    assert controller.update(obs,1.21)['action']=='right'


def test_replay_takeover_does_not_extend_recorded_input_without_current_policy():
    controller=LocalController();controller.accept('left',1.,0,100,bounded=True,replay=True)
    controller.resume_reactive(1.02,0)
    obs=Observation(1,1.03,'f','s',256,224,Track(0,tuple(BOX)),[],{},
        character_state={'grounded':True,'can_start_jump':True})
    assert controller.update(obs,1.04)['action']=='wait'


def test_safe_segment_keeps_full_duration_and_real_action_changes_after_many_renewals():
    actions=[{'action':'right' if i<100 else 'right_jump','hold_ms':100,'offset_s':round(i*.02,3)} for i in range(196)]
    compact,_=_success_sequence(actions)
    assert len(compact)<64
    assert compact[0]['offset_s']==0
    assert abs(compact[-1]['offset_s']+compact[-1]['hold_ms']/1000-4.)<.001
    assert abs(sum(s['hold_ms'] for s in compact if s['action']=='right')-2000)<=1
    assert min(s['offset_s'] for s in compact if s['action']=='right_jump')==2.


def context(now,box=BOX):
    return {'captured_at':now,'width':256,'height':224,'player':{'bbox':list(box)},'objects':[],
        'character_state':{'grounded':True,'grounded_evidence':'visual_estimate','can_start_jump':True}}


def learned_cycle(root,*,proven=True):
    cycle=AttemptCycle(root);cycle.begin(FRAME,BOX,10.,'1','first')
    for i in range(120):
        box=[20+i*.3,170,20,24]
        cycle.record('right',100,10+i*.1,box,source='local_controller')
        cycle.observe(10+i*.1,box)
    review={'outcome':'death','confidence':.9,'outcome_confidence':.99,'outcome_source':'local_life_counter',
        'life_evidence':{'confirmed':True,'status':'life_lost','before_value':3,'after_value':2,'decrement':1},
        'trigger_frame_id':'before','hypothesis':'The jump hit the overhead projectile; try waiting.',
        'retry_plan':{'at_s':9.5,'action':'wait','hold_ms':100}}
    frames=[{'frame_hash':key,'captured_at':at,'scene':'gameplay','image':FRAME,'context':context(at,[46,170,20,24])}
        for key,at in [('approach',18.6),('before',19.5)]]
    cycle.finish(review,frames)
    if proven:
        cycle.data['stages']['1']['prefix'].update(status='tracked_passage',evidence={
            'kind':'tracked_object_passed','track_id':7,'no_contact':True,
            'survived_s':1.2,'forward_progress_px':48,'separation_px':30})
    return cycle


def test_full_route_replay_then_early_jev_handoff_without_automatic_patch(tmp_path):
    cycle=learned_cycle(tmp_path)
    assert cycle.begin(FRAME,BOX,30.,'1','retry',anticipatory=True,lead_s=.8)
    assert cycle._awaiting_end==8.7 and cycle._awaiting_end>4
    assert max(s['offset_s'] for s in cycle.schedule)>8
    cycle.next(38.71,[46,170,20,24],native=True)
    guidance=cycle.policy_handoff(FRAME,[46,170,20,24],38.72,context(38.72))
    assert guidance['experiment_authorized']
    assert guidance['remaining_to_plan_ms']>=700
    assert guidance['hypothesis'].startswith('The jump hit')
    assert cycle.mode=='jev' and not cycle.active
    assert not any(s.get('patch') for s in cycle.schedule)
    assert cycle.policy_handoff(FRAME,BOX,38.73,context(38.73)) is None
    assert not cycle.patch_executed
    assert cycle.accept_correction('wait',38.8,guidance['decision_id'])
    assert cycle.record('wait',100,38.81,[46,170,20,24],source='local_controller')
    assert cycle.patch_executed


def test_replay_divergence_does_not_block_correction_for_current_matching_scene(tmp_path):
    cycle=learned_cycle(tmp_path)
    cycle.begin(FRAME,BOX,30.,'1','diverged',anticipatory=True)
    cycle.cancel('position_diverged')
    guidance=cycle.policy_handoff(FRAME,[200,170,20,24],35.,context(35.))
    assert guidance['experiment_authorized'] and not guidance['route_aligned']
    assert guidance['handoff_reason']=='replay_or_scene_diverged'
    assert cycle.accept_correction('wait',35.1,guidance['decision_id'])


def test_jev_alternative_does_not_replace_or_fail_the_unexecuted_review_plan(tmp_path):
    cycle=learned_cycle(tmp_path,proven=False)
    stored=cycle.data['stages']['1']
    stored['candidate']['plan']={'at_s':2.5,'action':'jump','hold_ms':100}
    cycle.begin(FRAME,BOX,30.,'1','alternative',anticipatory=True)
    guidance=cycle.policy_handoff(FRAME,BOX,31.8,context(31.8))
    assert guidance and guidance['proposed_plan']['action']=='jump'
    assert cycle.accept_correction('right',31.9,guidance['decision_id'])
    assert cycle.record('right',100,32.2,BOX,source='local_controller')
    assert cycle.plan['action']=='jump'
    assert cycle._correction['matches_proposal'] is False

    review={'outcome':'death','confidence':.9,'outcome_confidence':.99,'outcome_source':'local_life_counter',
        'life_evidence':{'confirmed':True,'status':'life_lost','before_value':3,'after_value':2,'decrement':1},
        'trigger_frame_id':'retry-death','hypothesis':'The alternative did not pass.',
        'retry_plan':{'at_s':3.2,'action':'right_jump','hold_ms':100}}
    frames=[
        {'frame_hash':'retry-approach','captured_at':32.1,'scene':'gameplay','image':FRAME,'context':context(32.1)},
        {'frame_hash':'retry-death','captured_at':33.5,'scene':'gameplay','image':FRAME,'context':context(33.5)},
    ]
    cycle.finish(review,frames)
    history=stored['candidate_history']
    assert history[-1]['candidate']['plan']['action']=='jump'
    assert history[-1]['outcome']=='proposal_not_executed'
    assert history[-1]['actual_plan']['action']=='right'
    trial=stored['trials'][-1]
    assert trial['plan']['action']=='right'
    assert trial['proposed_plan']['action']=='jump'
    assert trial['proposal_executed'] is False


def test_missing_or_wrong_visual_anchor_cannot_authorize_experiment(tmp_path):
    cycle=learned_cycle(tmp_path)
    candidate=cycle.data['stages']['1']['candidate']
    candidate.pop('approach_anchor',None);candidate.pop('trigger_signature',None)
    cycle.begin(FRAME,BOX,30.,'1','missing-anchor',anticipatory=True)
    cycle.next(38.71,BOX,native=True)
    guidance=cycle.policy_handoff(FRAME,BOX,38.72,context(38.72))
    assert guidance['route_aligned'] and not guidance['experiment_authorized']
    assert not cycle.accept_correction('wait',38.8,guidance['decision_id'])


def test_longer_survival_only_route_cannot_delay_or_veto_new_correction(tmp_path):
    cycle=learned_cycle(tmp_path,proven=False)
    stored=cycle.data['stages']['1']
    longer=copy.deepcopy(stored['prefix']);longer['until_s']=14.;longer['phase_id']='old-unproven-route'
    longer['sequence']=copy.deepcopy(cycle.actions)
    stored['prefix_history']=[longer];stored['best_prefix']=longer
    stored['candidate']['failure_at_s']=3.;stored['candidate']['plan']['at_s']=2.5
    assert not cycle.begin(FRAME,BOX,30.,'1','reactive-retry',anticipatory=True)
    assert not cycle.active and not cycle.schedule
    assert cycle.plan['at_s']==2.5
    assert cycle.policy_handoff(FRAME,BOX,30.1,context(30.1)) is None
    guidance=cycle.policy_handoff(FRAME,BOX,31.8,context(31.8))
    assert guidance['experiment_authorized']
    assert guidance['proposed_plan']['at_s']==2.5
    assert guidance.get('proposal_applicable') is not False


def test_unproven_candidate_sequence_is_never_replayed(tmp_path):
    cycle=learned_cycle(tmp_path,proven=False)
    assert cycle.data['stages']['1']['candidate']['sequence']
    assert not cycle.begin(FRAME,BOX,30.,'1','no-proof',anticipatory=True)
    assert not cycle.schedule and cycle.mode=='jev'


def test_new_stage_cannot_reuse_a_previous_stage_handoff(tmp_path):
    cycle=learned_cycle(tmp_path);cycle.begin(FRAME,BOX,30.,'1','first-retry',anticipatory=True)
    assert not cycle.begin(FRAME,BOX,40.,'2','different-stage',anticipatory=True)
    assert cycle.policy_handoff(FRAME,BOX,40.1,context(40.1)) is None


def test_actual_key_intervals_replace_overlapping_logical_leases(tmp_path):
    cycle=AttemptCycle(tmp_path);cycle.begin(FRAME,BOX,10.,'1','keys')
    for at in [10.,10.25,10.5,10.75]:cycle.record('right_jump',250,at,BOX,source='local_controller')
    for key,down,at in [('ArrowRight',True,10.1),('x',True,10.3),('x',False,10.55),('ArrowRight',False,10.8)]:
        cycle.record_key_event({'key':key,'down':down,'sent_at':at,'acknowledged_at':at+.002})
    sequence=cycle.replay_sequence()
    duration=lambda action:sum(s['hold_ms'] for s in sequence if s['action']==action)
    assert duration('right_jump')==250
    assert duration('right')==450
    assert duration('wait')==300
    assert all(s['source']=='observed_key_events' for s in sequence)


def test_current_ground_and_actors_are_rechecked_after_jev_response(tmp_path):
    cycle=learned_cycle(tmp_path);cycle.begin(FRAME,BOX,30.,'1','reply',anticipatory=True)
    cycle.next(38.71,BOX,native=True)
    cycle.policy_handoff(FRAME,BOX,38.72,context(38.72))
    old=context(38.72);current=context(38.8)
    assert cycle.validate_correction_reply('wait',38.81,old,current) is None
    airborne=copy.deepcopy(current);airborne['character_state']['can_start_jump']=False
    assert cycle.validate_correction_reply('right_jump',38.81,old,airborne)=='retry_jump_not_available'
    assert cycle.validate_correction_reply('wait',39.1,old,current)=='retry_current_observation_stale'
    assert cycle.validate_correction_reply('wait',41.,old,current)=='retry_guidance_expired'


def test_jev_receives_authorized_experiment_and_previous_failed_inputs(tmp_path):
    cycle=learned_cycle(tmp_path);cycle.begin(FRAME,BOX,30.,'1','prompt',anticipatory=True)
    cycle.next(38.71,BOX,native=True)
    guidance=cycle.policy_handoff(FRAME,BOX,38.72,context(38.72))
    obs=Observation(1,38.72,'frame','scene',256,224,Track(0,tuple(BOX)),[],{},hypotheses=[guidance])
    bodies=[]
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200,json={'answers':{'action':{'type':'choice','choice':'wait','confidence':.8}}})
    policy=JevPolicy(api_key='test',transport=httpx.MockTransport(handle))
    try:assert policy(obs).action=='wait'
    finally:policy.close()
    sent=bodies[0]['state']['hypotheses'][0]
    assert sent['proposed_plan']['action']=='wait' and sent['failed_actions']
    assert 'explicitly authorizes ONE bounded experiment' in bodies[0]['questions']['action']['instructions']
