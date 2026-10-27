"""A model may explain a death but cannot veto verified HUD evidence."""
def reconcile_outcome(review,job):
    evidence=job.get('life_evidence') or {}
    if not evidence.get('confirmed'):return review
    result=dict(review)
    result['model_outcome']=review.get('outcome')
    result['outcome']='death';result['outcome_source']='local_life_counter'
    result['outcome_confidence']=.99;result['life_evidence']=evidence
    result['actionable']=review.get('outcome')=='death' and bool(review.get('retry_plan'))
    if review.get('outcome')!='death':
        result.pop('retry_plan',None)
        result['hypothesis']=('HUD confirms life loss '+str(evidence['before_value'])+' -> '+str(evidence['after_value'])+
            '. The model disagreed; its explanation is not accepted as a successful passage. Cause remains uncertain.')
    return result


def review_episode(reviewer,job):
    evidence=job.get('life_evidence') or {}
    try:first=reviewer(job)
    except Exception as exc:
        if not evidence.get('confirmed'):raise
        playable=[f for f in job.get('candidates',[]) if f.get('scene')=='gameplay']
        trigger=playable[0]['frame_hash'] if playable else None
        first={'outcome':'uncertain','hypothesis':'Life loss confirmed locally; model analysis unavailable.',
            'confidence':0.,'evidence_frame_ids':evidence.get('evidence_frame_ids',[]),
            'trigger_frame_id':trigger,'suggested_action':'wait','status':'unverified','planning_error':type(exc).__name__}
    if evidence.get('confirmed') and (first.get('outcome')!='death' or not first.get('retry_plan')):
        correction={**job,'previous_review':first,'confirmed_outcome':evidence}
        try:
            final=reviewer(correction)
            result=reconcile_outcome(final,job)
            result['previous_model_outcome']=first.get('outcome');result['outcome_reconsidered']=True
            return result
        except Exception as exc:
            result=reconcile_outcome(first,job);result['planning_error']=type(exc).__name__;return result
    return reconcile_outcome(first,job)
