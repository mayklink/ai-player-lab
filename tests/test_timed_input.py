import time
from ai_player_lab.timed_input import TimedInput


class Sink:
    def __init__(self):self.events=[]
    def event(self,key,down):self.events.append((key,down,time.monotonic()))
    def close(self):pass


def test_observed_key_events_record_overlapping_combinations_without_duplicate_downs():
    sink=Sink();worker=TimedInput(lambda:sink)
    try:
        worker.ready.result(timeout=1)
        for keys in [{'ArrowRight'},{'ArrowRight','x'},{'ArrowRight'},set()]:
            worker.set(keys,250).result(timeout=1)
        events=worker.drain_key_events()
        assert [(e['key'],e['down']) for e in events]==[
            ('ArrowRight',True),('x',True),('x',False),('ArrowRight',False)]
        assert all(e['acknowledged_at']>=e['sent_at'] for e in events)
        assert not worker.drain_key_events()
    finally:worker.close()


def test_deadline_survives_a_blocked_main_thread():
    sink=Sink();worker=TimedInput(lambda:sink)
    try:
        worker.ready.result(timeout=1)
        worker.set({'ArrowRight'},100).result(timeout=1)
        time.sleep(.3)
        assert not worker.held
        assert [x[1] for x in sink.events]==[True,False]
        duration=sink.events[1][2]-sink.events[0][2]
        assert .08<=duration<.18
    finally:worker.close()


def test_lease_renewal_is_continuous_and_close_releases():
    sink=Sink();worker=TimedInput(lambda:sink)
    worker.ready.result(timeout=1)
    worker.set({'ArrowRight'},200).result(timeout=1)
    time.sleep(.03)
    worker.set({'ArrowRight'},200).result(timeout=1)
    assert [x[1] for x in sink.events]==[True]
    worker.close()
    assert [x[1] for x in sink.events]==[True,False]


def test_timed_sequence_executes_between_frames_and_stops_on_expired_visual_lease():
    sink=Sink();worker=TimedInput(lambda:sink);worker.ready.result(timeout=1)
    try:
        start=time.monotonic()
        worker.schedule([{'at':start+.03,'keys':{'ArrowRight'},'hold_ms':100,'action':'right'},
            {'at':start+.07,'keys':{'ArrowRight','x'},'hold_ms':80,'action':'right_jump'},
            {'at':start+.24,'keys':{'ArrowLeft'},'hold_ms':100,'action':'left'}],
            token='phase',valid_until=start+.18)
        # No frame loop calls are needed to execute the second action.
        time.sleep(.31)
        executed=worker.drain_executions('phase')
        assert [e['action'] for e in executed if 'action' in e]==['right','right_jump']
        assert any(e.get('cancelled')=='visual_heartbeat_expired' for e in executed)
        assert not worker.held and not any(k=='ArrowLeft' for k,_,_ in sink.events)
    finally:worker.close()


def test_manual_stop_cancels_future_scheduled_inputs():
    sink=Sink();worker=TimedInput(lambda:sink);worker.ready.result(timeout=1)
    try:
        worker.schedule([{'at':time.monotonic()+.08,'keys':{'x'},'hold_ms':100,'action':'jump'}],
            token='phase',valid_until=time.monotonic()+.3)
        worker.set(set(),100).result(timeout=1)
        time.sleep(.13)
        assert sink.events==[] and not worker.heartbeat('phase',time.monotonic()+.1)
        cancelled=worker.drain_executions('phase')
        assert any(e.get('cancelled')=='superseded_by_set' for e in cancelled)
    finally:worker.close()


def test_explicit_stop_reason_is_preserved_without_sending_future_input():
    sink=Sink();worker=TimedInput(lambda:sink);worker.ready.result(timeout=1)
    try:
        worker.schedule([{'at':time.monotonic()+.08,'keys':{'x'},'hold_ms':100,'action':'jump'}],
            token='phase',valid_until=time.monotonic()+.3)
        worker.set(set(),reason='identity_gap').result(timeout=1)
        assert worker.drain_executions('phase')[0]['cancelled']=='identity_gap'
        time.sleep(.12)
        assert not sink.events
    finally:worker.close()
