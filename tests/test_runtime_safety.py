"""Failure paths use fake resources; never touches an actual keyboard or API."""
import json
import time
from pathlib import Path
from threading import Event
import pytest

from ai_player_lab.control import SafeKeyboard
from ai_player_lab.types import Decision
from ai_player_lab.workers import LatestWorker


class Keyboard:
    test_backend = True
    def __init__(self): self.held = set(); self.presses = 0; self.release_attempts = []
    def press(self, key): self.presses += 1; self.held.add(key)
    def release(self, key): self.release_attempts.append(key); self.held.discard(key)


@pytest.mark.parametrize('fail_after_injection', [False, True])
def test_partial_keypress_failure_releases_every_attempted_key(fail_after_injection):
    class BrokenKeyboard(Keyboard):
        def press(self, key):
            if self.presses == 1:
                if fail_after_injection: self.held.add(key)
                raise RuntimeError('simulated injection failure')
            super().press(key)
    keyboard = BrokenKeyboard()
    control = SafeKeyboard('Mario', keyboard=keyboard, focus=lambda:'Mario')
    try:
        assert control.arm()
        with pytest.raises(RuntimeError): control.apply(Decision('right_jump',time.monotonic(),1))
        assert not control.armed
        assert keyboard.held == set() and control.held == set()
    finally: control.close()


def test_release_failure_still_attempts_other_keys_and_remembers_failed_key():
    class BrokenRelease(Keyboard):
        def release(self, key):
            self.release_attempts.append(key)
            if key == 'right': raise RuntimeError('simulated release failure')
            self.held.discard(key)
    keyboard = BrokenRelease()
    control = SafeKeyboard('Mario', keyboard=keyboard, focus=lambda:'Mario')
    control.stop.set(); control.thread.join()  # deterministic fake-only assertions
    control.held = keyboard.held = {'right','space'}
    with pytest.raises(RuntimeError): control.disarm()
    assert set(keyboard.release_attempts) == {'right','space'}
    assert control.held == {'right'} and keyboard.held == {'right'}
    keyboard.release = lambda key: keyboard.held.discard(key)
    control.close()
    assert not control.held and not keyboard.held


def test_worker_close_drops_pending_work_and_closes_provider_after_inflight():
    entered, finish, closed = Event(), Event(), Event()
    calls = []
    def task(value):
        calls.append(value); entered.set(); finish.wait(2); return value
    worker = LatestWorker(task, 'fake-api', on_close=closed.set)
    worker.submit(1); assert entered.wait(1)
    worker.submit(2)
    worker.close()
    assert not closed.is_set(), 'provider must not close under in-flight operation'
    assert not worker.submit(3)
    finish.set(); worker.thread.join(1)
    assert closed.is_set() and calls == [1]
    assert worker.pending.empty()


def test_setup_failure_closes_worker_database_and_video(tmp_path, monkeypatch):
    from ai_player_lab import runtime
    resources = {}
    class Memory:
        def __init__(self, _): self.closed = False; resources['memory'] = self
        def close(self): self.closed = True
    class Video:
        def __init__(self, _): self.closed = False; resources['video'] = self
        def isOpened(self): return False
        def release(self): self.closed = True
    real_worker = runtime.LatestWorker
    def worker(*args, **kwargs):
        resources['worker'] = real_worker(*args, **kwargs)
        return resources['worker']
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime, 'MemoryStore', Memory)
    monkeypatch.setattr(runtime, 'LatestWorker', worker)
    monkeypatch.setattr(runtime.cv2, 'VideoCapture', Video)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'capture_region':[0,0,320,180]}))
    with pytest.raises(ValueError, match='vídeo'):
        runtime.run_session(cfg,video_path=Path('missing.avi'),headless=True)
    assert resources['memory'].closed and resources['video'].closed
    assert not resources['worker'].thread.is_alive()


def test_setup_failure_closes_provider_before_worker_ownership(tmp_path, monkeypatch):
    from ai_player_lab import runtime
    closed = []
    class Provider:
        def __init__(self,*args,**kwargs): pass
        def close(self): closed.append(True)
    monkeypatch.setattr(runtime, 'JevPolicy', Provider)
    monkeypatch.setattr(runtime.MultimodalReviewer,'from_config',lambda _: (_ for _ in ()).throw(ValueError('configuration failure')))
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'capture_region':[0,0,320,180]}))
    with pytest.raises(ValueError, match='configuration'):
        runtime.run_session(cfg,jev=True,vlm=True,headless=True)
    assert closed == [True]


def test_closed_controller_cannot_be_rearmed_without_watchdog():
    keyboard=Keyboard(); control=SafeKeyboard('Mario',keyboard=keyboard,focus=lambda:'Mario')
    control.close()
    assert not control.arm()
    control.apply(Decision('right',time.monotonic(),1))
    assert not keyboard.held and not control.armed


@pytest.mark.parametrize('key', ['alt','ctrl','cmd','f4','esc','enter','left_jump'])
def test_os_shortcut_mapping_rejected(key):
    with pytest.raises(ValueError):
        SafeKeyboard('Mario',{'left':'left','right':key,'jump':'space'},keyboard=Keyboard(),focus=lambda:'Mario')


@pytest.mark.parametrize('events', [['arm','disarm'],['disarm','arm'],['arm','death'],['arm','reseed'],['arm','quit']])
def test_safety_hotkey_dominates_arm_batch(events):
    from ai_player_lab.runtime import apply_input_events
    class Control:
        armed=True
        def arm(self): self.armed=True
        def disarm(self): self.armed=False
    control=Control()
    assert apply_input_events(control,events)
    assert not control.armed


def test_cleanup_continues_when_video_release_fails(tmp_path, monkeypatch):
    from ai_player_lab import runtime
    closed=[]
    class Memory:
        def __init__(self,_): pass
        def close(self): closed.append('memory')
    class Video:
        def __init__(self,_): pass
        def isOpened(self): return False
        def release(self): closed.append('video'); raise RuntimeError('release failed')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime,'MemoryStore',Memory)
    monkeypatch.setattr(runtime.cv2,'VideoCapture',Video)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'capture_region':[0,0,320,180]}))
    with pytest.raises(RuntimeError,match='release failed'):
        runtime.run_session(cfg,video_path=Path('bad.avi'),headless=True)
    assert set(closed)=={'memory','video'}


@pytest.mark.parametrize('hud', [-1, 1.5, True])
def test_invalid_hud_rejected_before_setup(hud):
    from ai_player_lab.runtime import validate_config
    with pytest.raises(ValueError,match='hud_px'):
        validate_config({'capture_region':[0,0,320,180],'hud_px':hud})


def test_worker_shutdown_during_dequeue_never_starts_pending_job(monkeypatch):
    from queue import Queue
    from ai_player_lab import workers
    dequeued, proceed = Event(), Event()
    class GatedQueue(Queue):
        def get(self,block=True,timeout=None):
            item=super().get(block=block,timeout=timeout)
            if block: dequeued.set(); proceed.wait(2)
            return item
    monkeypatch.setattr(workers,'Queue',GatedQueue)
    calls=[];worker=LatestWorker(calls.append,'dequeue-race')
    worker.submit(1); assert dequeued.wait(1)
    worker.close(); proceed.set();worker.thread.join(1)
    assert calls == [] and not worker.thread.is_alive()


def test_slow_event_detection_is_included_in_decision_age(tmp_path,monkeypatch):
    from ai_player_lab import runtime
    from ai_player_lab.diagnostics import fixture_frame
    from ai_player_lab.sources import encode_player_template
    frame,box=fixture_frame(0)
    class Video:
        def __init__(self,_): self.used=False
        def isOpened(self): return True
        def get(self,_): return 30.
        def read(self):
            if self.used:return False,None
            self.used=True;return True,frame
        def release(self):pass
    class Detector:
        def __init__(self,*args):pass
        def detect(self,*args):time.sleep(.05);return False
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime.cv2,'VideoCapture',Video)
    monkeypatch.setattr(runtime,'UpstreamDeathDetector',Detector)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'capture_region':[0,0,320,180],
        'player_box':box,'player_template':encode_player_template(frame,box),
        'death_templates':{'fake':'not-read'},'max_decision_age_ms':'25'}))
    result=runtime.run_session(cfg,video_path=Path('fake.avi'),headless=True,seconds=1)
    row=json.loads((Path(result['output'])/'state.jsonl').read_text().splitlines()[0])
    assert row['proposed']=='wait' and row['gate']=='stale_decision'
