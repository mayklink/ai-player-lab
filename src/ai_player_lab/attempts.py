"""Executed-action prefixes and explicitly unverified, timed retry experiments.

No model or visual memory search runs in this scheduler. Each session is
anchored to a playable level entry, and only actual input leases are recorded.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import cv2
import numpy as np

ACTIONS={'wait','left','right','jump','left_jump','right_jump'}


def _masked_player_frame(frame,box):
    if not isinstance(box,(list,tuple)) or len(box)<4:return frame
    image=np.asarray(frame).copy();h,w=image.shape[:2]
    try:x,y,bw,bh=(int(round(float(v))) for v in box[:4])
    except (TypeError,ValueError,OverflowError):return image
    x0=max(0,x-2);y0=max(40,y-2);x1=min(w,x+bw+2);y1=min(h,y+bh+2)
    if x1<=x0 or y1<=y0:return image
    # Fill the identity crop with the local border median so pose/color changes
    # do not move the scene anchor while nearby hazards and geometry remain.
    border=np.concatenate([image[max(40,y0-2):y0,x0:x1].reshape(-1,image.shape[2] if image.ndim==3 else 1),
                           image[y1:min(h,y1+2),x0:x1].reshape(-1,image.shape[2] if image.ndim==3 else 1),
                           image[y0:y1,max(0,x0-2):x0].reshape(-1,image.shape[2] if image.ndim==3 else 1),
                           image[y0:y1,x1:min(w,x1+2)].reshape(-1,image.shape[2] if image.ndim==3 else 1)])
    if border.size:
        fill=np.median(border,axis=0).astype(image.dtype)
        image[y0:y1,x0:x1]=fill if image.ndim==3 else fill[0]
    return image

def _align_player(frame,box):
    if not isinstance(box,(list,tuple)) or len(box)<4:return frame,box
    h,w=frame.shape[:2]
    try:x,y,bw,bh=(float(v) for v in box[:4])
    except (TypeError,ValueError,OverflowError):return frame,box
    dx=w*.5-(x+bw*.5);dy=h*.76-(y+bh*.5)
    aligned=cv2.warpAffine(frame,np.float32([[1,0,dx],[0,1,dy]]),(w,h),flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REPLICATE)
    return aligned,[x+dx,y+dy,bw,bh]

def signature(frame,box=None):
    aligned,aligned_box=_align_player(frame,box)
    masked=_masked_player_frame(aligned,aligned_box)
    gray=cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY) if masked.ndim==3 else masked
    return cv2.resize(gray[40:],(32,20),interpolation=cv2.INTER_AREA).tolist()

def visual_signature(frame,box=None):
    if not isinstance(box,(list,tuple)) or len(box)<4:
        return cv2.resize(frame[40:],(24,16),interpolation=cv2.INTER_AREA).tolist()
    aligned,aligned_box=_align_player(frame,box)
    return cv2.resize(_masked_player_frame(aligned,aligned_box)[40:],(24,16),interpolation=cv2.INTER_AREA).tolist()

def _ground_confirmed(context):
    state=(context or {}).get('character_state',context or {})
    return (state.get('grounded') is True and bool(state.get('grounded_evidence'))
            and state.get('can_start_jump') is True)

def _threat_kind(item):
    label=str(item.get('label','')).lower()
    return label if label.startswith('enemy_') else 'independent_motion'

def _relative_threats(context):
    if not isinstance(context,dict):return []
    player=(context or {}).get('player') or {}
    if not isinstance(player,dict) or not isinstance(player.get('bbox'),(list,tuple)):
        return []
    size=(context or {}).get('frame_size') or {}
    if isinstance(size,(list,tuple)) and len(size)>=2:size={'width':size[0],'height':size[1]}
    if not isinstance(size,dict):size={}
    width=(context or {}).get('width') or size.get('width')
    height=(context or {}).get('height') or size.get('height')
    if not width or not height:return []
    try:
        px=(float(player['bbox'][0])+float(player['bbox'][2])/2)/float(width)
        py=(float(player['bbox'][1])+float(player['bbox'][3])/2)/float(height)
    except (TypeError,ValueError,OverflowError,ZeroDivisionError):return []
    result=[]
    for item in (context or {}).get('objects',[]) or []:
        if not isinstance(item,dict):continue
        label=str(item.get('label','')).lower()
        if (item.get('predicted') is True or item.get('is_prediction') is True or item.get('missed',0) not in (0,None)
                or label.startswith(('predicted','unknown','background','static'))
                or label.endswith('_predicted')
                or not (label.startswith('enemy_') or label=='independent_motion' or item.get('independent_motion') is True)):
            continue
        box=item.get('bbox')
        if not isinstance(box,(list,tuple)) or len(box)<4:continue
        try:
            ox=(float(box[0])+float(box[2])/2)/float(width)
            oy=(float(box[1])+float(box[3])/2)/float(height)
        except (TypeError,ValueError,OverflowError,ZeroDivisionError):continue
        if not (-.10<=ox-px<=.45 and abs(oy-py)<=.35):continue
        result.append({'kind':_threat_kind(item),'relative':[round(ox-px,4),round(oy-py,4)]})
    return result

def _retry_threats_compatible(context,reference):
    if not isinstance(context,dict):return False
    current=_relative_threats(context)
    if not reference:return True
    if len(reference)!=len(current):return False
    unmatched=list(current)
    for old in reference:
        candidates=[(math.dist(old['relative'],item['relative']),index) for index,item in enumerate(unmatched)
                    if item['kind']==old['kind']]
        if not candidates or min(candidates)[0]>.12:return False
        unmatched.pop(min(candidates)[1])
    return True


def validate_plan(plan, duration, allowed=ACTIONS):
    if not isinstance(plan,dict):raise ValueError('Retry plan must be an object')
    at=plan.get('at_s');hold=plan.get('hold_ms');action=plan.get('action')
    if (isinstance(at,bool) or not isinstance(at,(int,float)) or not math.isfinite(at)
            or not 0<=at<min(duration,120) or action not in allowed
            or isinstance(hold,bool) or not isinstance(hold,int) or not 20<=hold<=250):
        raise ValueError('Invalid bounded retry plan')
    return {'at_s':round(float(at),3),'action':action,'hold_ms':hold}


def plan_key(plan):
    return (plan['action'],round(plan.get('requested_at_s',plan.get('nominal_at_s',plan['at_s'])),1),plan['hold_ms'])


def _same_plan_action(first,second):
    return (isinstance(first,dict) and isinstance(second,dict)
        and isinstance(first.get('action'),str) and first.get('action')==second.get('action'))


def plausible_tracking(sequence):
    for before,after in zip(sequence,sequence[1:]):
        a=before.get('player_box');b=after.get('player_box')
        dt=max(0,after['offset_s']-before['offset_s'])
        if a and b and math.hypot(a[0]-b[0],a[1]-b[1])>max(40,600*dt):return False
    return True


def proven_passage_prefix(prefix):
    if not isinstance(prefix,dict) or prefix.get('status')!='tracked_passage':return False
    evidence=prefix.get('evidence')
    if not isinstance(evidence,dict):return False
    try:
        return (evidence.get('kind')=='tracked_object_passed' and evidence.get('no_contact') is True
            and evidence.get('track_id') is not None and float(evidence.get('survived_s',0))>=1
            and float(evidence.get('forward_progress_px',0))>=24 and float(evidence.get('separation_px',0))>=24
            and bool(prefix.get('sequence')) and plausible_tracking(prefix['sequence']))
    except (TypeError,ValueError,KeyError,OverflowError):return False


def _bounded_prefix(prefix,plan_at,*,strict_plan_cutoff=False,max_duration=4.,lead_s=.5):
    """Return a short prefix that ends before any unresolved plan boundary."""
    try:
        until=float(prefix.get('until_s',0));plan_at=float(plan_at)
        if not math.isfinite(until) or not math.isfinite(plan_at):return [],0.
    except (TypeError,ValueError,OverflowError):return [],0.
    bound=min(float(max_duration),until)
    plan_bound=plan_at-lead_s
    if strict_plan_cutoff:
        if plan_bound<=0:return [],0.
        bound=min(bound,plan_bound)
    elif plan_bound>0:
        bound=min(bound,plan_bound)
    if bound<=0:return [],0.
    bounded=[]
    sequence=prefix.get('sequence')
    if not isinstance(sequence,list):return [],0.
    for item in sequence:
        try:
            start=float(item['offset_s']);hold=int(item['hold_ms'])
            if not math.isfinite(start) or hold<=0:return [],0.
            end=start+hold/1000
            if start>=bound:break
            step=dict(item)
            step['hold_ms']=min(hold,max(0,int(math.floor((bound-start)*1000))))
            if step['hold_ms']>0:bounded.append(step)
            if end>=bound:break
        except (KeyError,TypeError,ValueError,OverflowError):return [],0.
    if not bounded or not plausible_tracking(bounded):return [],0.
    return bounded,bound


class AttemptCycle:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'attempt-cycle.json'
        self.data={'version':1,'stages':{}}
        if self.path.exists():
            self.data=json.loads(self.path.read_text(encoding='utf-8'))
            if self.data.get('version')!=1 or not isinstance(self.data.get('stages'),dict):
                raise ValueError('Invalid attempt cycle storage')
        self.clear()

    def clear(self):
        self.started_at=None;self.stage=None;self.actions=[];self.schedule=[]
        self.mode='idle';self.reason='idle';self.plan=None;self.patch_executed=False
        self.start_signature=None;self.start_box=None;self._last_step=-1
        self._awaiting_end=0.;self._lost_since=None;self._promoted=False
        self.max_progress=0.;self.phase_id=None
        self._progress=0.;self._previous_box=None;self._direction=1;self._direction_set=False
        self._progress_observed=False
        self.trusted_entry=False
        self.replay_aborted=False;self._not_reached_recorded=False
        self._replaying_prefix=False;self._resume_visual_patch_after_prefix=False
        self._visual_nominal_at=None;self._visual_trigger_progress_px=None
        self.patch_gate={'accepted':False,'reason':'not_checked'}
        self.key_events=[];self.initial_keys=[];self._anticipatory=False
        self._handoff_sent=False;self._guidance=None;self._correction=None
        self._candidate=None;self._route_strategy_changed=False;self._handoff_lead=.8

    def save(self):
        temp=self.path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(self.data,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(self.path)

    @staticmethod
    def _archive_prefix(stored,prefix):
        if float(prefix.get('until_s',0))>float((stored.get('best_prefix') or {}).get('until_s',0)):
            stored['best_prefix']=copy.deepcopy(prefix)
        history=stored.get('prefix_history')
        if not isinstance(history,list):history=[]
        history.append(copy.deepcopy(prefix))
        stored['prefix_history']=history[-4:]

    @property
    def active(self):return self.mode in {'prefix','experiment'}

    def begin(self,frame,box,now,stage,phase_id,*,allow_replay=True,trusted_entry=True,
              anticipatory=False,lead_s=.8,initial_keys=()):
        self.clear();self.started_at=now;self.stage=str(stage);self.phase_id=phase_id
        self.initial_keys=sorted(initial_keys);self._anticipatory=anticipatory
        self.start_signature=signature(frame,box);self.start_box=list(box)
        self._previous_box=list(box)
        self.trusted_entry=trusted_entry
        self.mode='jev';self.reason='new_phase'
        stored=self.data['stages'].get(self.stage,{})
        anchor=stored.get('anchor')
        if not trusted_entry or not allow_replay or not anchor:return False
        if anticipatory and (stored.get('candidate') or {}).get('status') in {'pending','pending_evidence','needs_new_plan','failed'}:
            self._candidate=copy.deepcopy(stored['candidate']);self.plan=dict(self._candidate['plan'])
        current_signature=self.start_signature if anchor.get('signature_version')==2 else signature(frame)
        delta=float(np.mean(np.abs(np.array(anchor['signature'],dtype=float)-np.array(current_signature,dtype=float))))/255
        drift=math.hypot(box[0]-anchor['box'][0],box[1]-anchor['box'][1])
        if delta>.12 or drift>20:
            self.replay_aborted=True
            self.reason='entry_mismatch';return False
        candidate=stored.get('candidate')
        prefix=stored.get('prefix',{})
        if anticipatory:
            # Failed routes and survival-only prefixes remain diagnostic data.
            # Only a locally tracked passage can authorize replayed inputs.
            proven=[p for p in [prefix,stored.get('best_prefix'),*stored.get('prefix_history',[])]
                if proven_passage_prefix(p)]
            route=max(proven,key=lambda p:float(p['until_s']),default=None)
            self._handoff_lead=max(.6,min(1.5,float(lead_s)))
            if candidate and candidate.get('status') in {'pending','pending_evidence','needs_new_plan','failed'}:
                self._candidate=copy.deepcopy(candidate);self.plan=dict(candidate['plan'])
            self.mode='jev';self.reason='reactive_until_retry_boundary' if self.plan else 'reactive_no_proven_replay'
            if route:
                boundary=self.plan['at_s'] if self.plan else 0
                bounded,bound=_bounded_prefix(route,boundary,strict_plan_cutoff=bool(self.plan),
                    max_duration=120.,lead_s=self._handoff_lead)
                if bounded:
                    self.schedule=bounded;self.mode='prefix';self._awaiting_end=bound
                    self._replaying_prefix=True;self._resume_visual_patch_after_prefix=bool(self.plan)
                    self.reason='proven_passage_replay_with_reactive_guard'
                    return True
            return False
        if candidate and candidate.get('status')=='pending' and any(
                item.get('kind')=='enemy' for item in candidate.get('trigger_threats',[])):
            candidate['status']='pending_evidence'
            candidate['blocked_reason']='legacy_untyped_threat_anchor'
            self.save()
        if candidate and candidate.get('status') in {'pending_evidence','needs_new_plan'}:
            # Missing evidence blocks the proposed action, not an independently
            # observed prefix. Replay only a short, tracking-consistent segment
            # and hand control back before the unresolved plan's trigger.
            candidate_plan=candidate.get('plan') or {}
            plan_at=candidate_plan.get('at_s',0)
            prefix_phase=prefix.get('phase_id')
            candidate_phase=candidate.get('from_phase_id')
            try:
                candidate_boundary=float(candidate.get('failure_at_s',plan_at))+.5
                strategy_differs=(bool(prefix_phase and candidate_phase and prefix_phase!=candidate_phase)
                    and float(prefix.get('until_s',0))>candidate_boundary)
            except (TypeError,ValueError,OverflowError):
                strategy_differs=False
            bounded,bound=_bounded_prefix(prefix,0 if strategy_differs else plan_at)
            if bounded:
                self.schedule=bounded;self.plan=None;self.mode='prefix';self._awaiting_end=bound
                self._replaying_prefix=True
                self.reason=('observed_prefix_strategy_differs_from_candidate' if strategy_differs
                    else 'evidence_pending_replaying_observed_prefix')
                return True
            self.reason='patch_missing_visual_anchor' if candidate['status']=='pending_evidence' else 'patch_not_reached_limit'
            return False
        if candidate and candidate.get('status')=='pending':
            if not candidate.get('trigger_signature') or candidate.get('trigger_signature_version')!=2:
                candidate['status']='pending_evidence';self.save()
                self.reason='patch_missing_visual_anchor';return False
            missed=sum(t.get('outcome')=='patch_not_reached' and
                plan_key(t['plan'])==plan_key(candidate['plan']) for t in stored.get('trials',[]))
            if missed>=2:
                candidate['status']='needs_new_plan';self.save()
                self.reason='patch_not_reached_limit';return False
            self.plan=dict(candidate['plan']);self.schedule=[]
            if candidate.get('trigger_signature'):
                self._visual_trigger=candidate['trigger_signature']
                self._visual_box=candidate.get('trigger_box')
                self._visual_threats=candidate.get('trigger_threats',[])
                self._visual_nominal_at=float(self.plan.get('at_s',0))
                self._visual_trigger_progress_px=candidate.get('trigger_progress_px')
                self.mode='awaiting_visual_patch';self.reason='waiting_for_failure_scene'
                # Recreate the observed route that led to this exact failure
                # scene; a generic short prefix can hand control to the policy
                # on a different route before the learned anchor is reachable.
                candidate_prefix={'until_s':float(self._visual_nominal_at),
                    'sequence':candidate.get('sequence',[])}
                bounded,bound=_bounded_prefix(candidate_prefix,self._visual_nominal_at,strict_plan_cutoff=True)
                if not bounded:
                    bounded,bound=_bounded_prefix(prefix,self._visual_nominal_at,strict_plan_cutoff=True)
                if bounded:
                    self.schedule=bounded;self.mode='prefix';self._awaiting_end=bound
                    self._replaying_prefix=True;self._resume_visual_patch_after_prefix=True
                    self.reason='replaying_prefix_before_visual_patch'
                    return True
                return False
        elif prefix.get('sequence'):
            self.schedule=list(prefix['sequence']);self.mode='prefix'
            self._awaiting_end=prefix['until_s']
        else:return False
        if not plausible_tracking(self.schedule):
            stored.pop('prefix',None)
            if stored.get('candidate'):stored['candidate']['status']='invalid_tracking'
            self.save();self.schedule=[];self.plan=None;self.mode='jev';self.reason='recorded_tracking_discontinuity';return False
        if self.mode=='prefix':self._replaying_prefix=True
        self.reason='matched_phase_entry';return True

    def arm_visual_patch(self,frame,box,now,*,can_jump=None,context=None):
        def reject(reason):
            self.patch_gate={'accepted':False,'reason':reason}
            return False
        if self.mode!='awaiting_visual_patch':return reject('not_waiting_for_visual_patch')
        at=now-self.started_at+.01
        if getattr(self,'_visual_threats',[]) and context is None:
            return reject('relative_threat_context_missing')
        error=float(np.mean(np.abs(np.asarray(visual_signature(frame,box),dtype=float)-np.asarray(self._visual_trigger,dtype=float))))/255
        threats=getattr(self,'_visual_threats',[])
        compatible_threats=None
        if context is not None:
            try:
                compatible_threats=_retry_threats_compatible(context,threats)
            except (TypeError,ValueError,KeyError,OverflowError):
                compatible_threats=False
            if not compatible_threats:return reject('relative_threat_mismatch')
        # Moving hazards can dominate the pixel difference even when the
        # tracked hazard is still at the same player-relative position. In
        # that case use the typed threat anchor, while retaining the time,
        # progress, and player-position gates below. A scene with no typed
        # threat still needs a close image match.
        if error>.12 and not (threats and compatible_threats):return reject('local_scene_mismatch')
        if error>.025 and not threats and error>.055:return reject('local_scene_mismatch')
        # A near-exact visual match is the primary trigger. Slightly changed
        # animation/cloud pixels can still match at the planned time, with a
        # bounded progress tolerance; neither historical clock nor progress
        # alone can execute an action.
        strong_visual_match=bool(threats) and error<=.025
        target=self._visual_box
        if (not strong_visual_match and target
                and math.hypot(box[0]-target[0],box[1]-target[1])>28):
            return reject('player_anchor_mismatch')
        earliest=self._visual_nominal_at
        if not strong_visual_match and earliest is not None and at<max(0.,float(earliest)-.5):
            return reject('retry_too_early')
        try:
            ref_progress=float(self._visual_trigger_progress_px)
            if (not strong_visual_match and self._progress_observed and math.isfinite(ref_progress)
                    and abs(float(self._progress)-ref_progress)>96):
                return reject('retry_progress_mismatch')
        except (TypeError,ValueError,OverflowError):
            pass
        if 'jump' in self.plan['action']:
            if can_jump is not True:return reject('jump_not_available')
            if not _ground_confirmed(context):
                return reject('ground_not_confirmed')
        nominal_at=self.plan.get('requested_at_s',self.plan.get('nominal_at_s',self.plan['at_s']))
        self.plan={**self.plan,'nominal_at_s':nominal_at,'at_s':round(at,4),'alignment':'visual_scene','visual_error':round(error,4)}
        self.schedule=[{'offset_s':self.plan['at_s'],'action':self.plan['action'],'hold_ms':self.plan['hold_ms'],
            'patch':True,'player_box':list(box)}]
        self._awaiting_end=self.plan['at_s']+self.plan['hold_ms']/1000
        self.mode='experiment';self.reason='visual_failure_scene_matched'
        self.patch_gate={'accepted':True,'reason':'visual_anchor_and_ground_confirmed' if 'jump' in self.plan['action']
                         else 'visual_anchor_and_context_confirmed','visual_error':round(error,4)}
        return True

    def context(self):
        stored=self.data['stages'].get(self.stage,{})
        timeline=[]
        for action in self.actions:
            end=action['offset_s']+action['hold_ms']/1000
            if timeline and timeline[-1]['action']==action['action'] and action['offset_s']<=timeline[-1]['end_s']+.02:
                timeline[-1]['end_s']=round(max(end,timeline[-1]['end_s']),4)
                timeline[-1]['leases']+=1
            else:timeline.append({'action':action['action'],'at_s':action['offset_s'],
                'end_s':round(end,4),'leases':1})
        return {'phase_started_at':self.started_at,'phase_id':self.phase_id,'stage':self.stage,
            'elapsed_s':self.actions[-1]['offset_s'] if self.actions else 0,
            'executed_timeline':timeline[:240],'timeline_truncated':len(timeline)>240,'retry_plan':self.plan,
            'retry_patch_executed':self.patch_executed,
            'decision_handoff':copy.deepcopy(self._guidance),
            'correction_decision':copy.deepcopy(self._correction),
            'key_events_recorded':len(self.key_events),
            'tracking_continuity_valid':plausible_tracking(self.actions),
            'previous_trials':stored.get('trials',[])[-6:],
            'validated_prefix_until_s':stored.get('prefix',{}).get('until_s',0)}

    def record(self,action,hold_ms,now,box,*,source):
        if self.started_at is None:return
        offset=now-self.started_at
        if not 0<=offset<=120 or len(self.actions)>=3600:return
        self.actions.append({'action':action,'hold_ms':int(hold_ms),'offset_s':round(offset,4),
            'player_box':list(box),'source':source,'progress_px':round(self._progress,2)})
        if not self._direction_set and action.startswith(('left','right')):
            self._direction=-1 if action.startswith('left') else 1;self._direction_set=True
        if source=='timed_experiment' and self.plan and action==self.plan['action'] and abs(offset-self.plan['at_s'])<.16:
            self.patch_executed=True
        correction=self._correction
        if (correction and not correction.get('executed') and action==correction['action']
                and now<=correction['valid_until'] and (not self._anticipatory
                    or offset>=self._candidate['plan']['at_s']-.35
                    or action==self._candidate['plan']['action'])):
            correction.update(executed=True,executed_at_s=round(offset,4),hold_ms=int(hold_ms))
            correction['executed_plan']={'at_s':round(offset,3),'action':action,'hold_ms':int(hold_ms)}
            correction['matches_proposal']=_same_plan_action(correction.get('proposal'),correction['executed_plan'])
            self.patch_executed=True
            self.actions[-1]['source']='learned_correction' if correction['matches_proposal'] else 'learning_alternative'
            self.actions[-1]['decision_id']=correction['decision_id']
            return True
        return False

    def record_key_event(self,event):
        if self.started_at is None:return
        offset=float(event['sent_at'])-self.started_at
        if 0<=offset<=120:
            self.key_events.append({**event,'offset_s':round(offset,6),'phase_id':self.phase_id})

    def replay_sequence(self):
        if not self.key_events or not self.actions:return [dict(a) for a in self.actions]
        held=set(self.initial_keys);previous=0.;sequence=[]
        end=max(a['offset_s']+a['hold_ms']/1000 for a in self.actions)
        events=sorted(self.key_events,key=lambda e:e['offset_s'])+[{'offset_s':end,'key':None}]
        for event in events:
            until=min(end,event['offset_s'])
            if until>previous:
                direction='right' if 'ArrowRight' in held and 'ArrowLeft' not in held else 'left' if 'ArrowLeft' in held and 'ArrowRight' not in held else ''
                action=(direction+'_jump' if direction else 'jump') if 'x' in held else direction or 'wait'
                while previous<until-.0005:
                    hold=min(250,int(round((until-previous)*1000)))
                    if hold<=0:break
                    sample=min(self.actions,key=lambda a:abs(a['offset_s']-previous))
                    sequence.append({**sample,'action':action,'hold_ms':hold,'offset_s':round(previous,6),
                        'keys':sorted(held),'source':'observed_key_events'})
                    previous+=hold/1000
            if event.get('key') is not None:
                if event.get('down'):held.add(event['key'])
                else:held.discard(event['key'])
            if until>=end:break
        return sequence or [dict(a) for a in self.actions]

    def policy_handoff(self,frame,box,now,context):
        """Give the policy one explicit branch point, never schedule a correction."""
        if (not self._anticipatory or self._handoff_sent or self.active
                or not getattr(self,'_candidate',None) or box is None):return None
        candidate=self._candidate;review=candidate.get('review',{});archive={}
        if not review and candidate.get('from_phase_id'):
            filename=hashlib.sha256(str(candidate['from_phase_id']).encode()).hexdigest()[:20]+'.json'
            try:
                archive=json.loads((self.root/filename).read_text(encoding='utf-8'))
                review=archive.get('review',{})
            except (OSError,ValueError):review={}
        local_death=(review.get('outcome_source')=='local_life_counter'
            and (review.get('life_evidence') or {}).get('confirmed') is True)
        expected=self.schedule[-1].get('player_box') if self.schedule else None
        drift=math.dist(box[:2],expected[:2]) if expected else None
        aligned=not self.replay_aborted and (drift is None or drift<=48)
        anchor=candidate.get('approach_anchor')
        if not anchor and candidate.get('trigger_signature'):
            anchor={'signature':candidate['trigger_signature'],'threats':candidate.get('trigger_threats',[])}
        visual_error=None;visual_aligned=False
        if anchor:
            visual_error=float(np.mean(np.abs(np.asarray(visual_signature(frame,box),dtype=float)
                -np.asarray(anchor['signature'],dtype=float))))/255
            visual_aligned=(visual_error<=.12 or bool(anchor.get('threats'))
                and _retry_threats_compatible(context,anchor['threats']))
        elapsed=now-self.started_at
        typed_match=bool(anchor and anchor.get('threats')) and visual_aligned
        if not self.replay_aborted and elapsed<self.plan['at_s']-self._handoff_lead and not typed_match:
            return None
        failed_actions=candidate.get('failed_actions') or [a for a in archive.get('actions',[])
            if self.plan['at_s']-1<=a['offset_s']<=candidate.get('failure_at_s',self.plan['at_s'])+.2][-16:]
        self._handoff_sent=True;self.mode='jev';self.reason='anticipatory_policy_handoff'
        self._guidance={'source':'retry_experiment','decision_id':self.phase_id+':retry',
            'prior_phase_id':candidate.get('from_phase_id'),'status':'authorized_bounded_experiment',
            'experiment_authorized':bool(local_death and visual_aligned),'route_aligned':aligned,
            'visual_aligned':bool(visual_aligned),
            'handoff_reason':self.reason if aligned else 'replay_or_scene_diverged',
            'player_drift_px':round(drift,2) if drift is not None else None,
            'visual_error':round(visual_error,4) if visual_error is not None else None,
            'hypothesis':review.get('hypothesis','Prior attempt ended in locally confirmed life loss.'),
            'proposed_plan':dict(self.plan),'phase_elapsed_s':round(now-self.started_at,4),
            'failed_actions':failed_actions,
            'previous_trials':copy.deepcopy(self.data['stages'].get(self.stage,{}).get('trials',[])[-6:]),
            'remaining_to_plan_ms':round((self.plan['at_s']-(now-self.started_at))*1000),
            'issued_at':now,'valid_until':now+max(2.,min(6.,self.plan['at_s']-(now-self.started_at)+.8))}
        return copy.deepcopy(self._guidance)

    def guidance(self,now):
        if self._guidance and now<=self._guidance['valid_until'] and not self.patch_executed:
            return copy.deepcopy(self._guidance)
        return None

    def accept_correction(self,action,now,decision_id):
        guidance=self.guidance(now)
        if not guidance or guidance['decision_id']!=decision_id or not guidance['experiment_authorized']:return False
        proposal=copy.deepcopy(guidance.get('proposed_plan') or self.plan)
        self._correction={'action':action,'decision_id':decision_id,'proposal':proposal,
            'matches_proposal':_same_plan_action({'action':action},proposal),
            'decided_at_s':round(now-self.started_at,4),'valid_until':now+.5,'executed':False}
        return True

    def validate_correction_reply(self,action,now,pending,current):
        guidance=self.guidance(now)
        if not guidance:return 'retry_guidance_expired'
        if not current or not current.get('player'):return 'retry_current_player_missing'
        if now-float(current.get('captured_at',0))>.2:return 'retry_current_observation_stale'
        old=(pending.get('player') or {}).get('bbox');new=current['player'].get('bbox')
        if not old or not new or math.dist(old[:2],new[:2])>48:return 'retry_player_moved'
        if not _retry_threats_compatible(current,_relative_threats(pending)):return 'retry_actors_changed'
        if 'jump' in action and not _ground_confirmed(current):return 'retry_jump_not_available'
        return None

    def next(self,now,box,*,valid=True,native=False):
        if not self.active:return None
        if not valid or box is None:
            if self._lost_since is None:self._lost_since=now
            if now-self._lost_since>.20:self.cancel('tracking_lost')
            return None
        self._lost_since=None
        elapsed=now-self.started_at
        if elapsed>=self._awaiting_end:
            if self._resume_visual_patch_after_prefix:
                self._resume_visual_patch_after_prefix=False
                self.mode='awaiting_visual_patch'
                self._last_step=-1
                self.reason='observed_prefix_completed_waiting_for_patch'
                return None
            self.cancel('handoff_to_jev');return None
        due=[(i,s) for i,s in enumerate(self.schedule) if s['offset_s']<=elapsed]
        if not due:return None
        index,step=due[-1]
        if native:
            expected=step.get('player_box')
            if expected and math.hypot(box[0]-expected[0],box[1]-expected[1])>48:self.cancel('position_diverged')
            return None
        if index==self._last_step:return None
        # Never skip a scheduled patch or burst expired inputs to catch up.
        previous_action=self.schedule[self._last_step]['action'] if self._last_step>=0 else None
        skipped=[s for i,s in due if self._last_step<i<index and (s.get('patch') or s['action']!=previous_action)]
        if skipped or elapsed-step['offset_s']>.15:
            self.cancel('schedule_late');return None
        expected=step.get('player_box')
        if expected and math.hypot(box[0]-expected[0],box[1]-expected[1])>48:
            self.cancel('position_diverged');return None
        self._last_step=index
        remaining=step['hold_ms']-int(max(0,elapsed-step['offset_s'])*1000)
        if remaining<=0:return None
        self.reason='timed_patch' if step.get('patch') else 'executed_prefix'
        return {'action':step['action'],'hold_ms':min(250,remaining),
            'source':'timed_experiment' if self.mode=='experiment' else 'timed_prefix',
            'patch':bool(step.get('patch')),'offset_s':step['offset_s']}

    def cancel(self,reason):
        if self.active and reason not in {'handoff_to_jev','episode_ended'}:
            self.replay_aborted=True
        if self.plan and not self.patch_executed and not self._not_reached_recorded:
            stored=self.data['stages'].get(self.stage)
            if stored is not None:
                stored.setdefault('trials',[]).append({'plan':self.plan,'outcome':'patch_not_reached',
                    'reason':reason,'phase_id':self.phase_id})
                stored['trials']=stored['trials'][-20:];self.save()
            self._not_reached_recorded=True
        self.mode='jev';self.reason=reason

    def observe(self,now,box,camera=None):
        if self.started_at is None or box is None:return False
        camera=camera or {}
        self._progress+=(box[0]-self._previous_box[0]-(camera.get('dx',0) if camera.get('valid') else 0))*self._direction
        self._previous_box=list(box)
        self._progress_observed=True
        self.max_progress=max(self.max_progress,self._progress)
        if self._anticipatory:return False
        stored=self.data['stages'].get(self.stage,{})
        candidate=stored.get('candidate')
        if (self._promoted or not candidate or candidate.get('status')!='pending'
                or not self.patch_executed):return False
        elapsed=now-self.started_at
        if elapsed<candidate.get('survival_until_s',candidate['failure_at_s']+1) or self.max_progress<24:return False
        if self._progress<candidate.get('failure_progress_px',0)+24:return False
        # Survival past the former failure is a local observation, not level completion.
        until=min(elapsed-.3,120)
        sequence=[dict(a) for a in self.actions if a['offset_s']+a['hold_ms']/1000<=until]
        if not sequence:return False
        stored['prefix']={'until_s':until,'sequence':sequence,'status':'observed_once',
            'evidence':'survived_past_prior_failure','phase_id':self.phase_id}
        executed=self._correction.get('executed_plan') if self._correction else None
        matches_proposal=(_same_plan_action(executed,candidate['plan']) if executed else True)
        if matches_proposal:candidate['status']='passed_local_window'
        stored.setdefault('trials',[]).append({'plan':candidate['plan'] if matches_proposal else executed,
            'proposed_plan':candidate['plan'],'proposal_executed':matches_proposal,
            'outcome':'survived_prior_failure_window','phase_id':self.phase_id})
        self._promoted=True;self.save();return True

    def confirm_passage(self,now,evidence):
        if not self.trusted_entry or evidence.get('kind')!='tracked_object_passed' or evidence.get('no_contact') is not True:return False
        until=max(0,now-self.started_at-.2)
        sequence=[a for a in self.replay_sequence() if a['offset_s']+a['hold_ms']/1000<=until]
        if not sequence:return False
        stored=self.data['stages'].setdefault(self.stage,{'trials':[]})
        prefix=stored.get('prefix')
        if prefix and float(prefix.get('until_s',0))>=until:return False
        if prefix:self._archive_prefix(stored,prefix)
        stored['prefix']={'until_s':until,'sequence':sequence,'status':'tracked_passage',
            'evidence':copy.deepcopy(evidence),'phase_id':self.phase_id}
        candidate=stored.get('candidate')
        if (candidate and self.patch_executed and now-self.started_at>candidate['plan']['at_s']
                and self._progress>=candidate.get('failure_progress_px',0)+24):
            executed=self._correction.get('executed_plan') if self._correction else None
            matches_proposal=(_same_plan_action(executed,candidate['plan']) if executed else True)
            if matches_proposal:candidate['status']='passed_local_window'
            stored.setdefault('trials',[]).append({'plan':candidate['plan'] if matches_proposal else executed,
                'proposed_plan':candidate['plan'],'proposal_executed':matches_proposal,
                'outcome':'tracked_passage_after_correction','phase_id':self.phase_id})
        self.save();return True

    def finish(self,review,frames,*,allowed=ACTIONS):
        if self.started_at is None:return None
        ctx=self.context();self.cancel('episode_ended')
        filename=hashlib.sha256(str(self.phase_id).encode()).hexdigest()[:20]+'.json'
        (self.root/filename).write_text(json.dumps({'context':ctx,'actions':self.actions,'key_events':self.key_events,
            'review':review},ensure_ascii=False,allow_nan=False),encoding='utf-8')
        life=review.get('life_evidence') if isinstance(review.get('life_evidence'),dict) else {}
        try:local_life=(review.get('outcome_source')=='local_life_counter' and life.get('confirmed') is True
            and life.get('status')=='life_lost' and float(life.get('decrement'))==1
            and float(life.get('before_value'))-float(life.get('after_value'))==1)
        except (TypeError,ValueError,OverflowError):local_life=False
        if not self.trusted_entry or not local_life or review.get('outcome')!='death' or review.get('outcome_confidence',review.get('confidence',0))<.7:
            reason=('untrusted_entry' if not self.trusted_entry else 'local_death_unconfirmed' if not local_life
                    else 'outcome_not_death' if review.get('outcome')!='death' else 'review_confidence_too_low')
            self.patch_gate={'accepted':False,'reason':reason}
            return None
        trigger=next((f for f in frames if f['frame_hash']==review.get('trigger_frame_id')),None)
        if not trigger or trigger.get('scene')!='gameplay':
            self.patch_gate={'accepted':False,'reason':'pre_failure_trigger_missing'}
            return None
        failure=max(0,trigger['captured_at']-self.started_at)
        episode_end=max([failure]+[a['offset_s']+a['hold_ms']/1000 for a in self.actions])
        if failure>120:return None
        stored=self.data['stages'].setdefault(self.stage,{'trials':[]})
        if not plausible_tracking(self.actions):
            stored.pop('prefix',None)
            if stored.get('candidate'):stored['candidate']['status']='invalid_tracking'
            self.save();self.patch_gate={'accepted':False,'reason':'visual_identity_unreliable'}
            return {'rejected':'visual_identity_unreliable','candidate':None}
        stored.setdefault('anchor',{'signature':self.start_signature,'signature_version':2,'box':self.start_box})
        if self.plan and self.patch_executed:
            correction=self._correction or {}
            executed=correction.get('executed_plan')
            proposal=correction.get('proposal') or self.plan
            matches_proposal=(_same_plan_action(executed,proposal) if executed else True)
            trial_plan=proposal if matches_proposal else executed
            stored['trials'].append({'plan':trial_plan,'proposed_plan':proposal,
                'proposal_executed':matches_proposal,'outcome':'death_after_executed_patch','phase_id':self.phase_id})
            candidate=stored.get('candidate')
            same_candidate=bool(candidate and plan_key(candidate.get('plan',{}))==plan_key(self.plan))
            if same_candidate and matches_proposal:
                candidate['status']='failed'
            elif same_candidate and not matches_proposal:
                history=stored.setdefault('candidate_history',[])
                history.append({'candidate':copy.deepcopy(candidate),'outcome':'proposal_not_executed',
                    'actual_plan':copy.deepcopy(executed),'phase_id':self.phase_id})
                del history[:-12]
        if review.get('actionable') is False:
            stored.pop('prefix',None);self.save()
            self.patch_gate={'accepted':False,'reason':'review_marked_non_actionable'}
            return {'outcome_only':True,'candidate':None,'reason':'local_death_recorded_without_valid_model_plan'}
        # Keep only the actually survived part before the actionable failure window.
        until=max(0,failure-.5)
        executed_sequence=self.replay_sequence()
        safe=[dict(a) for a in executed_sequence if a['offset_s']+a['hold_ms']/1000<=until]
        can_update_prefix=not self.replay_aborted and (not self.plan or self.patch_executed or self._replaying_prefix)
        prior=stored.get('prefix')
        has_prior=bool(prior and prior.get('sequence'))
        if (can_update_prefix and self._replaying_prefix and has_prior and prior.get('status')!='tracked_passage'
                and float(prior.get('until_s',0))>until):
            self._archive_prefix(stored,prior)
            prior['sequence']=[a for a in prior['sequence'] if a['offset_s']+a['hold_ms']/1000<=until]
            prior['until_s']=until
            if not prior['sequence']:stored.pop('prefix',None)
        elif (can_update_prefix and not has_prior and not self._replaying_prefix
              and safe and until>.4 and self.max_progress>=24):
            stored['prefix']={'until_s':until,'sequence':safe,'status':'observed_once',
                'evidence':'executed_surviving_segment_before_failure','phase_id':self.phase_id}
        plan=review.get('retry_plan')
        result={'prefix_until_s':stored.get('prefix',{}).get('until_s',0),'candidate':None}
        if plan is not None:
            try:plan=validate_plan(plan,max(episode_end+.01,.02),allowed)
            except ValueError:
                result['rejected']='invalid_plan';self.save();return result
            repeated=any(t['outcome']=='death_after_executed_patch' and plan_key(t['plan'])==plan_key(plan) for t in stored['trials'])
            missed=sum(t.get('outcome')=='patch_not_reached' and plan_key(t['plan'])==plan_key(plan) for t in stored['trials'])
            if repeated:result['rejected']='already_failed_plan'
            elif missed>=2:result['rejected']='patch_not_reached_limit'
            else:
                stored['candidate']={'status':'pending','plan':plan,'failure_at_s':failure,
                    'review':copy.deepcopy(review),
                    'failed_actions':[dict(a) for a in self.actions if failure-2<=a['offset_s']<=failure+.2],
                    'survival_until_s':max([failure+.5]+[f['captured_at']-self.started_at+.5 for f in frames if f.get('scene')=='gameplay']),
                    'failure_box':(trigger.get('context',{}).get('player') or {}).get('bbox'),
                    'failure_progress_px':next((a.get('progress_px',0) for a in reversed(self.actions) if a['offset_s']<=failure),0),
                    'sequence':[dict(a) for a in executed_sequence if a['offset_s']<plan['at_s']],
                    'from_phase_id':self.phase_id}
                approach_frames=[f for f in frames if f.get('scene')=='gameplay' and f.get('image') is not None
                    and (f.get('context',{}).get('player') or {}).get('bbox')
                    and f['captured_at']<=self.started_at+plan['at_s']-.6]
                if approach_frames:
                    approach_frame=max(approach_frames,key=lambda f:f['captured_at'])
                    approach_context=approach_frame['context'];approach_box=approach_context['player']['bbox']
                    stored['candidate']['approach_anchor']={'signature':visual_signature(approach_frame['image'],approach_box),
                        'at_s':round(approach_frame['captured_at']-self.started_at,4),
                        'box':approach_box,'threats':_relative_threats(approach_context)}
                requested_at_s=plan['at_s']
                origin=self.started_at+requested_at_s
                jump_plan='jump' in plan['action']
                visual_candidates=[];eligible=[]
                trigger_at=float(trigger['captured_at'])
                for frame in frames:
                    ctx=frame.get('context',{})
                    if (frame.get('scene')!='gameplay' or not isinstance(ctx,dict)
                            or not isinstance(ctx.get('player'),dict)
                            or not isinstance(ctx['player'].get('bbox'),(list,tuple))
                            or frame.get('image') is None):continue
                    visual_candidates.append(frame)
                    if float(frame['captured_at'])>trigger_at:continue
                    if jump_plan and not _ground_confirmed(ctx):continue
                    eligible.append(frame)
                reference=min(eligible,key=lambda f:abs(f['captured_at']-origin)) if eligible else None
                reanchored=False
                if jump_plan:
                    requested_frame=min(visual_candidates,key=lambda f:abs(float(f['captured_at'])-origin)) if visual_candidates else None
                    if requested_frame is not None and not _ground_confirmed(requested_frame.get('context')):
                        preceding=[f for f in eligible
                            if float(f['captured_at'])<origin
                            and float(f['captured_at'])>self.started_at
                            and origin-float(f['captured_at'])<=2.0
                            and _relative_threats(f.get('context'))]
                        if preceding:
                            reference=max(preceding,key=lambda f:float(f['captured_at']))
                            reanchored=True
                if reference is not None and (abs(reference['captured_at']-origin)<=.6 or reanchored):
                    ref_context=reference['context']
                    anchor_offset=float(reference['captured_at'])-self.started_at
                    progress_sample=max((a for a in self.actions if a['offset_s']<=anchor_offset),
                        key=lambda a:a['offset_s'],default=None)
                    if progress_sample is not None:
                        stored['candidate']['trigger_progress_px']=progress_sample.get('progress_px')
                    if reanchored:
                        aligned_at=round(float(reference['captured_at'])-self.started_at,3)
                        plan={**plan,'at_s':aligned_at,'requested_at_s':requested_at_s,
                              'nominal_at_s':requested_at_s,
                              'ground_alignment':{'kind':'preceding_ground_known_threat',
                                  'requested_at_s':requested_at_s,'aligned_at_s':aligned_at,
                                  'offset_s':round(aligned_at-requested_at_s,3),
                                  'frame_hash':reference['frame_hash']}}
                        stored['candidate']['plan']=plan
                        stored['candidate']['sequence']=[dict(a) for a in executed_sequence if a['offset_s']<aligned_at]
                        result['reason']='retry_aligned_to_preceding_ground_threat'
                    stored['candidate'].update(trigger_signature=visual_signature(reference['image'],ref_context['player']['bbox']),
                        trigger_signature_version=2,trigger_box=ref_context['player']['bbox'],
                        trigger_threats=_relative_threats(ref_context),trigger_frame_id=reference['frame_hash'])
                else:
                    stored['candidate']['status']='pending_evidence'
                    missing_ground=(jump_plan and visual_candidates and
                        (not eligible or reference is None or abs(float(reference['captured_at'])-origin)>.6))
                    result['reason']='patch_missing_ground_anchor' if missing_ground else 'patch_missing_visual_anchor'
                    self.patch_gate={'accepted':False,'reason':result['reason']}
                result['candidate']=plan
        stored['trials']=stored['trials'][-20:];self.save()
        return result
