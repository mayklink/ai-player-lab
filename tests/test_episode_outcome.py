from pathlib import Path
import cv2
from ai_player_lab.life_evidence import LifeCounter,life_loss_evidence
from ai_player_lab.episode_outcome import review_episode

def historical_job():
    root=Path(__file__).resolve().parents[1]/'runs/browser/20261024-232700/episodes/20261024-232700-1'
    frames=[{'image':cv2.imread(str(root/f'{i:02d}.png')),'scene':'gameplay' if i<10 else 'map','frame_hash':str(i)} for i in [6,7,10,11]]
    if any(f['image'] is None for f in frames):return None
    return {'life_evidence':life_loss_evidence(frames,LifeCounter()),'candidates':frames}

def review(outcome,plan=None):
    result={'outcome':outcome,'confidence':.94,'hypothesis':'No life was lost, continue right.',
        'trigger_frame_id':'6','suggested_action':'right','status':'unverified'}
    if plan:result['retry_plan']=plan
    return result

def test_local_counter_overrules_two_incorrect_model_replies():
    evidence={'confirmed':True,'before_value':5,'after_value':4,'evidence_frame_ids':['a','b']}
    calls=[]
    def wrong(job):calls.append(job);return review('not_death')
    result=review_episode(wrong,{'life_evidence':evidence})
    assert len(calls)==2 and calls[1]['confirmed_outcome']==evidence
    assert result['outcome']=='death' and result['model_outcome']=='not_death'
    assert result['outcome_source']=='local_life_counter' and not result['actionable']
    assert 'retry_plan' not in result

def test_confirmed_death_requests_new_action_and_keeps_valid_timed_plan():
    calls=[]
    plan={'at_s':2.5,'action':'right_jump','hold_ms':200}
    def corrected(job):
        calls.append(job)
        return review('not_death') if len(calls)==1 else review('death',plan)
    result=review_episode(corrected,{'life_evidence':{'confirmed':True,'before_value':5,'after_value':4}})
    assert result['outcome']=='death' and result['actionable']
    assert result['retry_plan']==plan and result['previous_model_outcome']=='not_death'

def test_model_failure_cannot_erase_confirmed_death():
    def unavailable(job):raise TimeoutError()
    result=review_episode(unavailable,{'life_evidence':{'confirmed':True,'before_value':5,'after_value':4},'candidates':[]})
    assert result['outcome']=='death' and not result['actionable']
    assert result['outcome_source']=='local_life_counter'

def test_unknown_life_reading_does_not_override_model():
    result=review_episode(lambda job:review('not_death'),{'life_evidence':{'confirmed':False,'status':'unknown'}})
    assert result['outcome']=='not_death'
