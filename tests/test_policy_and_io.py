import json
import time
import httpx
import pytest
from ai_player_lab.types import Observation, Track, Decision
from ai_player_lab.providers import JevPolicy, MultimodalReviewer, parse_review, choose_action
from ai_player_lab.workers import LatestWorker
from ai_player_lab.control import SafeKeyboard, DryController
from ai_player_lab.runtime import validate_config


def observation():
    return Observation(1,10,'hash','0000000000000000',320,180,Track(0,(20,20,10,20),confidence=.9),[],{})


def test_stale_decision_is_not_used():
    obs=observation(); old=Decision('right',9,1)
    chosen,reason=choose_action(obs,old,10.1,250)
    assert chosen.action=='wait' and reason=='stale_decision'


def test_previous_episode_is_not_used():
    obs=observation(); obs.episode=2
    chosen,reason=choose_action(obs,Decision('right',10,1,episode=1),10.1)
    assert chosen.action=='wait' and reason=='wrong_episode'


def test_missing_player_stops():
    obs=observation(); obs.player=None
    assert choose_action(obs,Decision('right',10,1),10.1)[0].action=='wait'


def test_low_confidence_stops():
    assert choose_action(observation(),Decision('right',10,1,.2),10.1)[1]=='low_confidence'


def test_gap_reflex_vetoes_forward_motion():
    obs=observation(); obs.hazards=[{'kind':'gap_candidate','distance_px':5}]
    assert choose_action(obs,Decision('right',10,1),10.1)[1]=='local_gap_veto'


def test_future_timestamp_rejected_at_gate():
    assert choose_action(observation(),Decision('right',12,1),10.1)[1]=='stale_decision'


def test_jev_contract_and_no_image_upload():
    seen={}
    def handler(request):
        seen.update(json.loads(request.content))
        assert request.url.path=='/v1/systemone'
        assert request.headers['authorization']=='Bearer test-only'
        return httpx.Response(200,json={'model':'jev-test','answers':{'action':{'type':'choice','choice':'right_jump','confidence':.8}},'usage':{'input_tokens':100}})
    client=JevPolicy(api_key='test-only',transport=httpx.MockTransport(handler))
    try:
        d=client(observation())
        assert d.action=='right_jump'; assert d.observed_at==10
        assert 'assets' not in seen
        assert seen['questions']['action']['type']=='choice'
    finally: client.close()


def test_jev_budget_stops_calls():
    c=JevPolicy(api_key='test-only',max_calls=0,transport=httpx.MockTransport(lambda r: pytest.fail('No request expected')))
    with pytest.raises(RuntimeError): c(observation())
    c.close()


def test_jev_invalid_action_response_is_rejected():
    c=JevPolicy(api_key='test',transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'answers':{'action':{'type':'choice','choice':'shell','confidence':1}}})))
    with pytest.raises(ValueError): c(observation())
    c.close()


def test_jev_timeout_is_reported_not_silently_accepted():
    def fail(request): raise httpx.ReadTimeout('synthetic timeout')
    c=JevPolicy(api_key='test',transport=httpx.MockTransport(fail))
    with pytest.raises(httpx.ReadTimeout): c(observation())
    c.close()


def review_text(**changes):
    r={'hypothesis':'Possible fall after moving right. Cause not yet proven.','confidence':.4,'evidence_frame_ids':['abc'],'suggested_action':'jump'}
    r.update(changes); return json.dumps(r)


def test_valid_review_is_always_unverified():
    result=parse_review(review_text(),{'abc'})
    assert result['status']=='unverified'


@pytest.mark.parametrize('patch',[
    {'confidence':float('nan')}, {'confidence':1.2}, {'confidence':True},
    {'evidence_frame_ids':['invented']}, {'suggested_action':'execute_code'}, {'hypothesis':''}
])
def test_bad_review_rejected(patch):
    with pytest.raises(ValueError): parse_review(review_text(**patch),{'abc'})


def test_latest_worker_is_nonblocking_and_bounded():
    def slow(value): time.sleep(.05); return value
    worker=LatestWorker(slow,'test-worker')
    try:
        start=time.monotonic()
        for i in range(100): worker.submit(i)
        assert time.monotonic()-start < .1
        assert worker.pending.qsize()<=1
        deadline=time.monotonic()+1; last=None
        while time.monotonic()<deadline:
            result=worker.poll()
            if result: last=result.value
            if last==99: break
            time.sleep(.005)
        assert last==99
        assert worker.dropped>0
    finally: worker.close()


def test_worker_exception_is_contained():
    def fail(_): raise RuntimeError('do not leak response body')
    worker=LatestWorker(fail,'exception-worker'); worker.submit(1)
    deadline=time.monotonic()+1; result=None
    while time.monotonic()<deadline:
        result=worker.poll()
        if result: break
        time.sleep(.005)
    worker.close()
    assert result.error=='RuntimeError' and result.value is None


class FakeKeyboard:
    test_backend=True
    def __init__(self): self.held=set()
    def press(self,key): self.held.add(key)
    def release(self,key): self.held.discard(key)


def test_input_defaults_disarmed_and_expires():
    clock=[10.]; focus=['Mario game']; keyboard=FakeKeyboard()
    c=SafeKeyboard('Mario',keyboard=keyboard,focus=lambda:focus[0],clock=lambda:clock[0])
    try:
        d=Decision('right_jump',10,1,hold_ms=100)
        c.apply(d); assert keyboard.held==set()
        assert c.arm()
        c.apply(d); assert keyboard.held=={'right','space'}
        clock[0]=10.2; c.tick(); assert keyboard.held==set()
    finally: c.close()


def test_focus_loss_disarms_and_releases():
    focus=['Mario']; keyboard=FakeKeyboard()
    c=SafeKeyboard('Mario',keyboard=keyboard,focus=lambda:focus[0])
    try:
        c.arm(); c.apply(Decision('right',time.monotonic(),1))
        focus[0]='Email'; c.tick()
        assert not c.armed and keyboard.held==set()
    finally: c.close()


def test_dry_run_never_arms():
    controller=DryController(); assert not controller.arm()
    controller.apply(Decision('right',1,1)); assert controller.armed is False


def test_config_rejects_path_traversal_game_id():
    cfg={'capture_region':[0,0,320,180],'game_id':'../../bad'}
    with pytest.raises(ValueError): validate_config(cfg)


def test_vlm_contract_uses_cropped_frames_and_checks_evidence():
    captured={}
    def handle(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':review_text()}}]})
    reviewer=MultimodalReviewer("https://vision.test/v1", "test-model", api_key="test-key", transport=httpx.MockTransport(handle))
    job={'event_id':'event-1','event':'death_marked','frames':[{'frame_hash':'abc','captured_at':1.0,'jpeg':b'fake_jpeg_for_contract_test'}],'states':[]}
    result=reviewer(job); reviewer.close()
    assert result['event_id']=='event-1'
    assert captured['max_tokens']==320
    assert captured['messages'][0]['content'][-1]['image_url']['url'].startswith('data:image/jpeg;base64,')
