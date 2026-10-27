import json
import httpx
import pytest
from ai_player_lab.death_review import DeathReviewer
from ai_player_lab.experience import ExperienceMemory
from ai_player_lab.providers import JevPolicy
from ai_player_lab.diagnostics import fixture_frame
from ai_player_lab.perception import VisualPipeline


def test_death_review_sends_chronological_pixels_and_executed_actions():
    requests=[]
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'outcome':'death','hypothesis':'Jump may have caused contact; test right.',
            'confidence':.8,'evidence_frame_ids':['before','after'],'trigger_frame_id':'before','suggested_action':'right'})}}]})
    reviewer=DeathReviewer(base_url='https://example.test/v1',model='gpt-6-luna',api_key='fake',transport=httpx.MockTransport(handle))
    try:
        result=reviewer({'game_rules':'Duck only when overhead clearance is sufficient.',
            'available_actions':['wait','right','right_jump'],'states':[{'action':'right_jump','hold_ms':200}],
            'frames':[{'frame_hash':key,'captured_at':index,'jpeg':b'image'} for index,key in enumerate(['before','after'])]})
        assert result['outcome']=='death' and result['status']=='unverified'
        content=requests[0]['messages'][0]['content']
        assert 'right_jump' in content[0]['text']
        assert 'Duck only when overhead clearance is sufficient.' in content[0]['text']
        assert 'available_actions' in content[0]['text']
        assert len([x for x in content if x['type']=='image_url'])==2
    finally:reviewer.close()


def test_invented_evidence_cannot_be_saved_as_a_death_review():
    def handle(request):
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'outcome':'death','hypothesis':'Claim','confidence':.9,
            'evidence_frame_ids':['invented'],'suggested_action':'right'})}}]})
    reviewer=DeathReviewer(base_url='https://example.test/v1',model='test',api_key='fake',transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(ValueError):reviewer({'states':[],'frames':[{'frame_hash':'real','captured_at':1,'jpeg':b'image'}]})
    finally:reviewer.close()


def test_dense_event_can_cite_all_twelve_actual_frames():
    ids=[str(i) for i in range(12)]
    def handle(request):
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'outcome':'death','hypothesis':'Contact observed; test earlier jump.','confidence':.8,
            'evidence_frame_ids':ids,'trigger_frame_id':'1','suggested_action':'jump'})}}]})
    reviewer=DeathReviewer(base_url='https://example.test/v1',model='test',api_key='fake',transport=httpx.MockTransport(handle))
    try:
        result=reviewer({'states':[],'frames':[{'frame_hash':key,'captured_at':i,'jpeg':b'image'} for i,key in enumerate(ids)]})
        assert result['evidence_frame_ids']==ids
    finally:reviewer.close()


def test_timed_retry_uses_phase_origin_and_receives_failed_plan_history():
    requests=[]
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'outcome':'death','hypothesis':'The prior right jump was late; jump earlier.',
            'confidence':.9,'evidence_frame_ids':['trigger'],'trigger_frame_id':'trigger',
            'suggested_action':'right_jump','retry_plan':{'at_s':2.5,'action':'right_jump','hold_ms':200}})}}]})
    reviewer=DeathReviewer(base_url='https://example.test/v1',model='test',api_key='fake',transport=httpx.MockTransport(handle))
    try:
        job={'states':[],'frames':[{'frame_hash':'trigger','captured_at':13.,'jpeg':b'image'}],
            'available_actions':['right','right_jump'],'attempt_context':{'phase_started_at':10.,
                'previous_trials':[{'plan':{'at_s':2.8,'action':'right_jump','hold_ms':200},'outcome':'death_after_executed_patch'}]}}
        result=reviewer(job)
        assert result['retry_plan']['at_s']==2.5
        assert 'death_after_executed_patch' in requests[0]['messages'][0]['content'][0]['text']
        job['frames'][0]['captured_at']=12.
        rejected=reviewer(job)
        assert 'retry_plan' not in rejected and 'retry_plan_error' in rejected
        assert rejected['outcome']=='death'
    finally:reviewer.close()


def test_unverified_episode_is_context_only_even_when_scene_and_motion_match(tmp_path):
    frame,box=fixture_frame(0)
    obs=VisualPipeline(frame,box).process(frame,1)
    memory=ExperienceMemory(tmp_path,'https://game.test/mario')
    memory.add('death-1',{'hypothesis':'Prior jump failed; test staying on ground.',
        'suggested_action':'right','confidence':.8,'status':'unverified'},frame,'1',obs.to_dict())
    recalled=ExperienceMemory(tmp_path,'https://game.test/mario').retrieve(frame,'1',obs.to_dict())
    assert recalled and recalled[0]['suggested_action'] is None
    assert recalled[0]['action_authorized'] is False
    obs.hypotheses=[{'source':'episodic_experience',**entry} for entry in recalled]
    bodies=[]
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200,json={'answers':{'action':{'type':'choice','choice':'right','confidence':.8}}})
    policy=JevPolicy(api_key='fake',transport=httpx.MockTransport(handle))
    try:
        policy(obs)
        experience=bodies[0]['state']['hypotheses'][0]
        assert experience['event_id']=='death-1'
        assert experience['status']=='unverified'
        assert experience['suggested_action'] is None
        assert 'HISTORICAL CONTEXT, NOT AN ACTION RULE' in bodies[0]['questions']['action']['instructions']
        assert 'priority over the default' not in bodies[0]['questions']['action']['instructions']
        obs.hypotheses[0]['match_kind']='scene_and_objects'
        obs.hypotheses[0]['match_evidence']={'motion_verified':True}
        policy(obs)
        assert 'HISTORICAL CONTEXT, NOT AN ACTION RULE' in bodies[1]['questions']['action']['instructions']
        assert 'priority over the default' not in bodies[1]['questions']['action']['instructions']
        obs.hypotheses[0]['alternative_failed_in_later_attempt']=True
        policy(obs)
        assert 'HISTORICAL CONTEXT, NOT AN ACTION RULE' in bodies[2]['questions']['action']['instructions']
        assert 'priority over the default' not in bodies[2]['questions']['action']['instructions']
        baseline={**obs.hypotheses[0],'event_id':'passed-1','outcome':'success',
            'action_authorized':True,'evidence_status':'verified_local_passage',
            'alternative_failed_in_later_attempt':False,'status':'observed_success',
            'successful_sequence':[{'action':'right_jump','hold_ms':200,'offset_s':.1}]}
        obs.hypotheses.append(baseline)
        policy(obs)
        assert 'OBSERVED SUCCESS BASELINE' in bodies[3]['questions']['action']['instructions']
        assert 'right_jump' in bodies[3]['questions']['action']['instructions']
    finally:policy.close()
