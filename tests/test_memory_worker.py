from threading import Event
import time
import numpy as np
from ai_player_lab.memory_worker import LatestMemory


def test_search_does_not_block_or_queue_old_frames():
    entered=Event();release=Event()
    class Memory:
        def retrieve(self,*args):
            entered.set();release.wait(2)
            return [{'event_id':'learned'}]
    frame=np.zeros((224,256,3),np.uint8)
    context={'width':256,'height':224,'player':{'bbox':[100,160,20,24]},'objects':[]}
    worker=LatestMemory(Memory())
    try:
        assert worker.submit(frame,'1',context,10.,4)
        assert entered.wait(1)
        assert worker.ready(frame,'1',context,10.1,4)==[]
        assert not worker.submit(frame,'1',context,10.6,4)
        release.set();worker.future.result(timeout=2)
        assert worker.ready(frame,'1',context,10.2,4)==[{'event_id':'learned'}]
        assert worker.ready(frame,'1',context,10.5,4)==[]
        assert worker.ready(frame,'1',context,10.2,5)==[]
        assert worker.ready(frame,'2',context,10.2,4)==[]
        moved={**context,'player':{'bbox':[150,160,20,24]}}
        assert worker.ready(frame,'1',moved,10.2,4)==[]
    finally:release.set();worker.close()


def test_slow_search_revalidates_candidate_on_latest_frame_before_adoption():
    frame=np.full((224,256,3),90,np.uint8)
    context={'width':256,'height':224,'player':{'bbox':[100,160,20,24]},
             'player_location':[110/256,172/224],'objects':[]}
    fresh=frame.copy();fresh[100:150,100:160]=(92,91,89)
    class Memory:
        def __init__(self): self.revalidated=None
        def retrieve(self,*args):
            time.sleep(.42)  # Same order as the recorded 400 ms freshness limit.
            return [{'event_id':'historical-advice'}]
        def revalidate(self,matches,current_frame,stage,current_context):
            self.revalidated=(current_frame.copy(),stage,current_context.copy(),matches)
            return [{'event_id':'historical-advice','geometry':'current'}]
    memory=Memory();worker=LatestMemory(memory,max_age=.4)
    try:
        assert worker.submit(frame,'1',context,10.,4)
        worker.future.result(timeout=2)
        # The original retrieval exceeded the action freshness window; it is
        # never returned as if its old geometry were current.
        assert worker.ready(fresh,'1',context,10.5,4)==[]
        worker.future.result(timeout=2)
        adopted=worker.ready(fresh,'1',context,10.55,4)
        assert adopted==[{'event_id':'historical-advice','geometry':'current'}]
        assert np.array_equal(memory.revalidated[0],fresh)
        assert memory.revalidated[1:3]==('1',context)
        assert worker.stats['last_search_ms']>=400
        assert worker.stats['revalidated']==1 and worker.stats['adopted']==1
    finally: worker.close()


def test_stale_result_rejects_changed_scene_position_and_collision_context():
    frame=np.full((224,256,3),90,np.uint8)
    context={'width':256,'height':224,'player':{'bbox':[100,160,20,24]},
             'player_location':[110/256,172/224],
             'objects':[],'moving_objects':[[.55,.72]]}
    variants=[
        ('scene_changed',np.full_like(frame,220),context,context),
        ('player_moved',frame,{**context,'player':{'bbox':[140,160,20,24]},
                               'player_location':[150/256,172/224]},context),
        ('objects_changed',frame,{**context,'moving_objects':[[.85,.72]]},context),
        ('objects_changed',frame,{**context,'moving_objects':[[.55,.72]]},
         {**context,'moving_objects':[]}),
    ]
    class Memory:
        def __init__(self): self.revalidate_calls=0
        def retrieve(self,frame,stage,context):
            self.reference=(frame.copy(),stage,context.copy())
            return [{'event_id':'historical'}]
        def revalidate(self,matches,frame,stage,context):
            self.revalidate_calls+=1
            old_frame,old_stage,old_context=self.reference
            same_frame=np.array_equal(old_frame,frame)
            same_player=old_context['player']==context['player']
            same_objects=old_context.get('moving_objects')==context.get('moving_objects')
            return matches if stage==old_stage and same_frame and same_player and same_objects else []
    for reason,current,changed_context,query_context in variants:
        memory=Memory();worker=LatestMemory(memory,max_age=.4)
        try:
            assert worker.submit(frame,'1',query_context,10.,4)
            worker.future.result(timeout=2)
            assert worker.ready(current,'1',changed_context,10.6,4)==[]
            worker.future.result(timeout=2)
            assert memory.revalidate_calls==1
            assert worker.ready(current,'1',changed_context,10.65,4)==[]
            assert worker.stats['reject_reasons']['no_current_match']==1
            assert worker.stats['revalidation_attempts']==1
        finally: worker.close()


def test_unconsumed_search_result_blocks_new_submit_until_revalidation():
    frame=np.full((224,256,3),90,np.uint8)
    context={'width':256,'height':224,'player':{'bbox':[100,160,20,24]},
             'player_location':[110/256,172/224],'objects':[]}
    release=Event();entered=Event()
    class Memory:
        def retrieve(self,*args):
            entered.set();release.wait(2)
            return [{'event_id':'old-result'}]
        def revalidate(self,matches,*args): return matches
    worker=LatestMemory(Memory())
    try:
        assert worker.submit(frame,'1',context,10.,4)
        assert entered.wait(1)
        release.set();worker.future.result(timeout=2)
        worker.poll()
        pending_result=worker.completed
        assert pending_result and pending_result['kind']=='search'
        # `submit` runs before `ready` in some caller loops. It must not
        # replace the stale result or start another search beside revalidation.
        assert not worker.submit(frame,'1',context,11.,4)
        assert worker.completed is pending_result
        assert worker.ready(frame,'1',context,10.6,4)==[]
        assert worker.stats['submitted']==1
        assert worker.future is not None
        revalidation_future=worker.future
        assert not worker.submit(frame,'1',context,11.2,4)
        assert worker.future is revalidation_future
        assert worker.stats['submitted']==1
    finally:release.set();worker.close()


def test_stale_result_keeps_stage_and_episode_as_hard_revalidation_gates():
    frame=np.full((224,256,3),90,np.uint8)
    context={'width':256,'height':224,'player':{'bbox':[100,160,20,24]},
             'player_location':[110/256,172/224],'objects':[]}
    class Memory:
        def __init__(self): self.revalidate_calls=0
        def retrieve(self,*args): return [{'event_id':'old'}]
        def revalidate(self,*args):
            self.revalidate_calls+=1
            return [{'event_id':'old'}]
    for stage,episode,reason in [('2',4,'stage_changed'),('1',5,'episode_changed')]:
        memory=Memory();worker=LatestMemory(memory)
        try:
            assert worker.submit(frame,'1',context,10.,4)
            worker.future.result(timeout=2)
            assert worker.ready(frame,stage,context,10.6,episode)==[]
            assert worker.stats['reject_reasons'][reason]==1
            assert memory.revalidate_calls==0
        finally:worker.close()


def test_stale_result_with_moved_player_can_adopt_only_fresh_revalidated_geometry():
    old_frame=np.full((224,256,3),90,np.uint8)
    current_frame=old_frame.copy();current_frame[130:190,90:170]=(105,105,105)
    old_context={'width':256,'height':224,'player':{'bbox':[80,160,20,24]},
                 'player_location':[90/256,172/224],'objects':[]}
    current_context={**old_context,'player':{'bbox':[100,160,20,24]},
                     'player_location':[110/256,172/224]}
    class Memory:
        def retrieve(self,*args): return [{'event_id':'old-hypothesis'}]
        def revalidate(self,matches,frame,stage,context):
            assert np.array_equal(frame,current_frame)
            assert context==current_context
            return [{'event_id':'old-hypothesis','player_box':[100,160,20,24]}]
    worker=LatestMemory(Memory())
    try:
        assert worker.submit(old_frame,'1',old_context,10.,4)
        worker.future.result(timeout=2)
        assert worker.ready(current_frame,'1',current_context,10.6,4)==[]
        worker.future.result(timeout=2)
        assert worker.ready(current_frame,'1',current_context,10.65,4)==[
            {'event_id':'old-hypothesis','player_box':[100,160,20,24]}]
        assert worker.stats['revalidated']==1
        assert worker.stats['adopted']==1
    finally: worker.close()
