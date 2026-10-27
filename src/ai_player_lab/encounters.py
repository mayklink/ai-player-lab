"""Conservative visual evidence that a tracked encounter was crossed alive.

No disappearance-based success and no assumption that every moving object is
an enemy. Each candidate is cancelled on contact, missing player/track or gaps.
"""
from .types import center,iou


def _playable(obs):
    """Fail closed on explicit terminal/local non-gameplay scene evidence."""
    for hypothesis in getattr(obs,'hypotheses',[]) or []:
        if not isinstance(hypothesis,dict) or hypothesis.get('source')!='local_vision':continue
        scene=hypothesis.get('scene')
        if scene is not None and scene!='gameplay':return False
    state=getattr(obs,'character_state',{}) or {}
    return state.get('jump_phase') not in {'death','dead','dying','falling_death'}


def _grounded(obs):
    state=getattr(obs,'character_state',{}) or {}
    return state.get('grounded') is True


def review_frames(history,current):
    """Include the actual post-event screen even between sampling ticks."""
    chosen=list(history)
    chosen=chosen[::max(1,len(chosen)//7)][-7:]
    if not chosen or chosen[-1]['frame_hash']!=current['frame_hash']:chosen.append(current)
    return chosen


class EncounterRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active={};self.previous_player=None;self.last_time=None;self.progress=0.

    def update(self,frame,obs,executed_actions):
        now=obs.captured_at
        if not _playable(obs) or obs.player is None or (self.last_time is not None and not 0<now-self.last_time<=.2):
            self.reset();return []
        player=obs.player;px,py=center(player.bbox)
        if self.previous_player is not None:
            if not obs.camera.get('valid'):
                self.reset();return []
            self.progress+=px-self.previous_player-obs.camera.get('dx',0.)
        self.previous_player=px;self.last_time=now
        tracks={o.id:o for o in obs.objects if o.age>=3 and o.missed==0 and _relevant_threat(o)}
        completed=[]
        for tid,event in list(self.active.items()):
            obj=tracks.get(tid)
            if now-event['start_time']>8:
                del self.active[tid];continue
            if obj is None:
                if event['occluded_since'] is None:event['occluded_since']=now
                if now-event['occluded_since']>.2:del self.active[tid]
                # A brief occlusion preserves a candidate, but never advances
                # its pass or survival evidence while the threat is absent.
                continue
            if iou(player.bbox,obj.bbox)>.02:
                del self.active[tid];continue
            ox,oy=center(obj.bbox)
            relative=(ox-px,oy-py)
            if event['last_relative'] is not None:
                dt=max(0.,now-event['last_seen'])
                if ((relative[0]-event['last_relative'][0])**2+(relative[1]-event['last_relative'][1])**2)**.5>32+250*dt:
                    del self.active[tid];continue
            if event['occluded_since'] is not None:
                # Restart the survival window after re-observing the threat.
                event['passed_at']=None
            event['occluded_since']=None;event['last_seen']=now;event['last_relative']=relative
            separation=px-(obj.bbox[0]+obj.bbox[2])
            if separation>=24:
                if event['passed_at'] is None:event['passed_at']=now
                survived=now-event['passed_at']
                progress=self.progress-event['initial_progress']
                if survived>=1 and progress>=24 and _grounded(obs):
                    evidence={'kind':'tracked_object_passed','track_id':tid,'start_time':event['start_time'],
                        'actor_label':obj.label,
                        'end_time':now,'forward_progress_px':round(progress,2),'separation_px':round(separation,2),
                        'survived_s':round(survived,3),'no_contact':True}
                    actions=[{'action':a['action'],'hold_ms':a['hold_ms'],
                        'offset_s':round(a['executed_at']-event['start_time'],3)} for a in executed_actions
                        if event['start_time']<=a['executed_at']<=now
                        and a['executed_at']-event['start_time']+a['hold_ms']/1000<=8]
                    if actions:completed.append({**event,'evidence':evidence,'actions':actions})
                    del self.active[tid]
            else:event['passed_at']=None
        for tid,obj in tracks.items():
            if tid in self.active:continue
            ox,oy=center(obj.bbox)
            gap=obj.bbox[0]-(player.bbox[0]+player.bbox[2])
            closing=player.vx-obj.vx
            if 24<=gap<=100 and abs(oy-py)<=50 and closing>5 and iou(player.bbox,obj.bbox)==0:
                self.active[tid]={'start_time':now,'initial_progress':self.progress,'passed_at':None,
                    'frame':frame.copy(),'context':obs.to_dict(),'track_id':tid,
                    'last_seen':now,'last_relative':(ox-px,oy-py),'occluded_since':None}
        return completed


def _relevant_threat(track):
    label=str(getattr(track,'label','')).lower()
    if label.startswith(('predicted','unknown','background','static')):return False
    return label.startswith('enemy_') or label=='independent_motion' or getattr(track,'independent_motion',False) is True


class PrefixRecorder:
    """Record short progressing prefixes with an additional survival window."""
    def __init__(self):self.reset()

    def reset(self):
        self.start=None;self.last=None;self.previous_x=None;self.progress=0.;self.prefix_progress=None

    def update(self,frame,obs,actions):
        now=obs.captured_at
        if (not _playable(obs) or obs.player is None or not obs.camera.get('valid')
                or (self.last is not None and now-self.last>.2)):
            self.reset();return None
        if any(h.get('kind') in {'visual_overlap','gap_candidate'} or h.get('ttc_ms',9999)<750 for h in obs.hazards):
            self.reset();return None
        px,py=center(obs.player.bbox)
        if self.start is None:
            # Do not certify an empty-path script in the middle of an encounter.
            if any(-24<center(o.bbox)[0]-px<130 and abs(center(o.bbox)[1]-py)<60 for o in obs.objects):return None
            self.start={'time':now,'frame':frame.copy(),'context':obs.to_dict()}
        if self.previous_x is not None:self.progress+=px-self.previous_x-obs.camera.get('dx',0.)
        self.previous_x=px;self.last=now
        elapsed=now-self.start['time']
        if elapsed>=4 and self.prefix_progress is None:self.prefix_progress=self.progress
        if elapsed<5:return None
        # Keep a qualifying segment pending in the air; only a confirmed
        # landing makes the survival window safe to publish.
        if not _grounded(obs):
            if elapsed>8:self.reset()
            return None
        sequence=[{'action':a['action'],'hold_ms':a['hold_ms'],
            'offset_s':round(a['executed_at']-self.start['time'],3),
            'player_box':(a.get('state',{}).get('player') or {}).get('bbox')}
            for a in actions if self.start['time']<=a['executed_at']
            and a['executed_at']+a['hold_ms']/1000<=self.start['time']+4]
        result=None
        if self.prefix_progress is not None and self.prefix_progress>=24 and sequence:
            result={**self.start,'actions':sequence,'evidence':{'kind':'observed_forward_segment',
                'duration_s':4.,'forward_progress_px':round(self.prefix_progress,2),'survived_s':round(elapsed-4,3),
                'no_contact':True,'start_time':self.start['time'],'end_time':now}}
        self.reset();return result
