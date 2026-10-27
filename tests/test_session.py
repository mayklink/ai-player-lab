from ai_player_lab.session import ControlRouter


def screen(router, scene='gameplay', at=10, **changes):
    result={'generation':router.generation,'captured_at':at,'confidence':.9,'scene':scene,
            'summary':'test','game':'Mario','controls':{},'player_box':[2,3,10,12]}
    result.update(changes)
    return result


def test_handoff_requires_two_distinct_observations_and_expires_old_epoch():
    r=ControlRouter()
    old=screen(r)
    assert r.accept(old,10)=='confirming'
    assert r.accept(old,10.1)=='confirming'
    assert r.owner=='Luna'
    assert r.accept(screen(r,at=11),11)=='handoff'
    assert r.owner=='Jev' and r.generation==1
    assert r.accept(old,11)=='stale'
    assert r.lease_valid(15) and not r.lease_valid(18)


def test_map_or_death_takes_control_from_jev_and_invalidates_decisions():
    r=ControlRouter(owner='Jev',generation=4)
    assert r.accept(screen(r,'death'),10)=='takeover'
    assert r.owner=='Luna' and r.generation==5
    assert r.scene=='death'


def test_stale_and_uncertain_screens_cannot_handoff():
    r=ControlRouter()
    assert r.accept(screen(r,at=1),10)=='stale'
    assert r.accept(screen(r,at=11),10)=='stale'
    assert r.accept(screen(r,confidence=.4),10)=='uncertain'
    assert r.owner=='Luna'


def test_gameplay_without_player_returns_control_to_director():
    r=ControlRouter(owner='Jev')
    assert r.accept(screen(r,player_box=None),10)=='takeover'


def test_pause_discards_inflight_confirmation():
    r=ControlRouter(); reply=screen(r)
    r.reset('Paused')
    assert r.accept(reply,10)=='stale'
    assert r.owner=='Luna'


def test_two_locally_verified_frames_can_confirm_one_multiframe_result():
    r=ControlRouter()
    assert r.accept(screen(r),10,temporal_confirmation=True)=='handoff'
    assert r.owner=='Jev' and r.generation==1
    # Local visual confirmation cannot bypass low confidence or stale provenance.
    r.reset('new epoch')
    assert r.accept(screen(r,at=1),10,temporal_confirmation=True)=='stale'
    assert r.accept(screen(r,confidence=.5),10,temporal_confirmation=True)=='uncertain'
