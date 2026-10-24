import json
import pytest
from ai_player_lab.memory import EpisodeBuffer, MemoryStore
from ai_player_lab.diagnostics import fixture_frame
from ai_player_lab.perception import VisualPipeline
from ai_player_lab.types import Decision


def make_job():
    frame,box=fixture_frame(0)
    pipe=VisualPipeline(frame,box)
    buffer=EpisodeBuffer(seconds=2,sample_hz=4)
    for i in range(9):
        # Keep a stable textured view to test retrieval, not synthetic gameplay learning.
        obs=pipe.process(frame,1+i*.25)
        buffer.add(frame,pipe.last_static,obs,Decision('right',obs.captured_at,obs.seq))
    return buffer.snapshot('death_marked',3),pipe.last_static


def test_event_persists_without_inventing_cause(tmp_path):
    store=MemoryStore(tmp_path/'memory.sqlite3'); job,_=make_job()
    store.add_event(job)
    assert store.summary()=={'events':1,'reviewed_events':0}
    pending=store.pending_jobs()
    assert len(pending)==1 and pending[0]['event_id']==job['event_id']
    store.close()


def test_same_event_is_not_counted_twice(tmp_path):
    store=MemoryStore(tmp_path/'memory.sqlite3'); job,_=make_job()
    store.add_event(job); store.add_event(job)
    assert store.summary()['events']==1
    store.close()


def test_review_attaches_to_original_event_and_recall_checks_geometry(tmp_path):
    store=MemoryStore(tmp_path/'memory.sqlite3'); job,static=make_job(); store.add_event(job)
    review={'hypothesis':'Possible gap; needs verification.','confidence':.4,'suggested_action':'jump','status':'unverified'}
    store.attach_review({'event_id':job['event_id'],'review':review,'model':'mock-vlm'})
    memories=store.recall(static)
    assert memories and memories[0]['place_match']['verified']
    assert memories[0]['confidence']==.4
    assert memories[0]['status']=='unverified_cause'
    assert store.summary()['reviewed_events']==1
    store.close()


def test_missing_event_cannot_receive_review(tmp_path):
    store=MemoryStore(tmp_path/'memory.sqlite3')
    with pytest.raises(ValueError): store.attach_review({'event_id':'missing','review':{},'model':'mock'})
    store.close()


def test_episode_buffer_is_bounded_and_uses_pre_event_anchor():
    job,_=make_job()
    times=[f['captured_at'] for f in job['frames']]
    assert len(job['frames'])<=4
    assert len(job['states'])<=8
    assert times==sorted(times)
