"""Single native-input owner with deadlines independent of the vision loop."""
from collections import deque
from concurrent.futures import Future
from threading import Condition,Thread
import json
import time


KEYS={'ArrowLeft':(37,'ArrowLeft'),'ArrowUp':(38,'ArrowUp'),'ArrowRight':(39,'ArrowRight'),
      'ArrowDown':(40,'ArrowDown'),'Enter':(13,'Enter'),**{k:(ord(k.upper()),'Key'+k.upper()) for k in 'xzasv'}}


class CDPKeyboard:
    def __init__(self,url):
        import websocket
        self.ws=websocket.create_connection(url,timeout=.2,suppress_origin=True,http_no_proxy=['127.0.0.1','localhost'])
        self.seq=0

    def event(self,key,down):
        vk,code=KEYS[key];self.seq+=1
        params={'type':'keyDown' if down else 'keyUp','key':key,'code':code,
                'windowsVirtualKeyCode':vk,'nativeVirtualKeyCode':vk}
        if down and len(key)==1:params.update(text=key,unmodifiedText=key)
        self.ws.send(json.dumps({'id':self.seq,'method':'Input.dispatchKeyEvent','params':params}))
        deadline=time.monotonic()+.25
        while time.monotonic()<deadline:
            result=json.loads(self.ws.recv())
            if result.get('id')==self.seq:
                if 'error' in result:raise RuntimeError('Native keyboard command rejected')
                return
        raise TimeoutError('Native keyboard acknowledgement timeout')

    def close(self):self.ws.close()


class TimedInput:
    def __init__(self,sink_factory):
        self.condition=Condition();self.pending=None;self.stopping=False
        self.held=set();self.deadline=0.;self.down_times={};self.timings=deque(maxlen=2048)
        self.error=None;self.ready=Future();self.factory=sink_factory
        self.timeline=None;self.executions=deque(maxlen=4096)
        self.key_events=deque(maxlen=16384)
        self.key_origins={}
        self.thread=Thread(target=self._run,daemon=True,name='native-input-deadline');self.thread.start()

    def set(self,keys,hold_ms=100,*,reason='superseded_by_set'):
        keys=set(keys)
        if not keys<=KEYS.keys() or isinstance(hold_ms,bool) or not isinstance(hold_ms,int) or not 1<=hold_ms<=250:raise ValueError('Invalid bounded key command')
        future=Future()
        with self.condition:
            if self.error or self.stopping:
                future.set_exception(RuntimeError('Native input unavailable'));return future
            if self.pending is not None:self.pending[2].set_exception(RuntimeError('Superseded input command'))
            if self.timeline:
                self.executions.append({'token':self.timeline['token'],
                    'cancelled':reason,'at':time.monotonic(),
                    'replacement_keys':sorted(keys)})
            self.timeline=None
            self.pending=(keys,hold_ms,future);self.condition.notify()
        return future

    def schedule(self,events,*,token,valid_until):
        prepared=[]
        for event in events:
            keys=set(event['keys']);hold=event['hold_ms']
            if not keys<=KEYS.keys() or not isinstance(hold,int) or not 1<=hold<=250:raise ValueError('Invalid timeline event')
            prepared.append({**event,'keys':keys})
        with self.condition:
            self.timeline={'events':deque(sorted(prepared,key=lambda e:e['at'])),
                'token':token,'valid_until':valid_until}
            self.condition.notify()

    def heartbeat(self,token,valid_until):
        with self.condition:
            if self.timeline and self.timeline['token']==token:
                self.timeline['valid_until']=valid_until;self.condition.notify()
                return True
            return False

    def drain_executions(self,token):
        with self.condition:
            found=[e for e in self.executions if e['token']==token]
            self.executions=deque((e for e in self.executions if e['token']!=token),maxlen=4096)
            return found

    def drain_key_events(self):
        with self.condition:
            events=list(self.key_events);self.key_events.clear();return events

    def _run(self):
        sink=None
        def release(keys,reason):
            requested=self.deadline
            for key in tuple(keys):
                sent=time.monotonic();sink.event(key,False);now=time.monotonic()
                origin=self.key_origins.pop(key,{})
                with self.condition:self.key_events.append({**origin,'key':key,'down':False,'sent_at':sent,'acknowledged_at':now,'reason':reason})
                self.timings.append({'key':key,'reason':reason,'down_at':self.down_times.pop(key,now),
                    'release_sent_at':sent,'released_at':now,'deadline':requested,
                    'late_ms':round(max(0.,sent-requested)*1000,3),'ack_late_ms':round(max(0.,now-requested)*1000,3)})
                self.held.discard(key)
        try:
            sink=self.factory();self.ready.set_result(True)
            while True:
                scheduled=None;expired=False
                with self.condition:
                    while self.pending is None and not self.stopping:
                        now=time.monotonic();deadlines=[self.deadline] if self.held else []
                        if self.timeline:
                            deadlines.append(self.timeline['valid_until'])
                            if self.timeline['events']:deadlines.append(self.timeline['events'][0]['at'])
                        remaining=min(deadlines)-now if deadlines else None
                        if remaining is not None and remaining<=0:break
                        self.condition.wait(remaining)
                    if self.stopping:break
                    command=self.pending;self.pending=None
                    if self.timeline and time.monotonic()>=self.timeline['valid_until']:
                        self.executions.append({'token':self.timeline['token'],'cancelled':'visual_heartbeat_expired','at':time.monotonic()})
                        self.timeline=None;expired=True
                    elif self.timeline and self.timeline['events'] and time.monotonic()>=self.timeline['events'][0]['at']:
                        scheduled=self.timeline['events'].popleft();scheduled['token']=self.timeline['token']
                if self.held and time.monotonic()>=self.deadline:release(self.held,'deadline')
                if expired and self.held:release(self.held,'visual_heartbeat_expired')
                if scheduled and command is None:
                    sent=time.monotonic();late_ms=(sent-scheduled['at'])*1000
                    if late_ms>150:
                        with self.condition:
                            self.timeline=None;self.executions.append({'token':scheduled['token'],'cancelled':'schedule_late','at':sent})
                        release(self.held,'schedule_late')
                    else:
                        keys=scheduled['keys'];release(self.held-keys,'timeline_changed')
                        self.deadline=sent+scheduled['hold_ms']/1000
                        for key in keys-self.held:
                            sent_key=time.monotonic();self.down_times[key]=sent_key;self.held.add(key);sink.event(key,True)
                            self.key_origins[key]={'replay_token':scheduled['token'],'input_source':'replay'}
                            with self.condition:self.key_events.append({'key':key,'down':True,'sent_at':sent_key,
                                'acknowledged_at':time.monotonic(),'reason':'replay','replay_token':scheduled['token']})
                        with self.condition:self.executions.append({**scheduled,'keys':sorted(keys),'sent_at':sent,'late_ms':round(late_ms,2)})
                if command:
                    keys,hold_ms,future=command
                    try:
                        release(self.held-keys,'changed')
                        self.deadline=time.monotonic()+hold_ms/1000
                        for key in keys-self.held:
                            sent_key=time.monotonic();self.down_times[key]=sent_key;self.held.add(key);sink.event(key,True)
                            self.key_origins[key]={'input_source':'command'}
                            with self.condition:self.key_events.append({'key':key,'down':True,'sent_at':sent_key,
                                'acknowledged_at':time.monotonic(),'reason':'command'})
                        future.set_result(True)
                    except Exception as exc:
                        future.set_exception(exc);raise
        except Exception as exc:
            self.error=type(exc).__name__
            if not self.ready.done():self.ready.set_exception(exc)
        finally:
            if sink:
                try:release(self.held,'closed')
                except Exception:pass
                sink.close()
            with self.condition:
                if self.pending and not self.pending[2].done():self.pending[2].set_exception(RuntimeError('Input closed'))

    def close(self):
        with self.condition:self.stopping=True;self.condition.notify()
        self.thread.join(timeout=2)
        if self.thread.is_alive():raise RuntimeError('Native input did not stop')
