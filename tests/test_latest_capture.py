import asyncio
from ai_player_lab.latest_capture import LatestCapture


def test_slow_consumer_gets_latest_frame_without_queue_and_close_stops_producer():
    async def scenario():
        count=0
        advanced=asyncio.Event()
        async def source():
            nonlocal count
            count+=1
            await asyncio.sleep(.001)
            return count
        def published(packet):
            # Captures sharing a coarse Windows clock tick are deliberately
            # dropped; wait for published packets rather than source calls.
            if packet['seq']>=6:advanced.set()
        pump=LatestCapture(source,fps=1000,on_packet=published);pump.start()
        first=await pump.next()
        await asyncio.wait_for(advanced.wait(),1)
        latest=await pump.next()
        assert latest['seq']>first['seq']+1
        assert pump.dropped>0 and latest['captured_at']<=latest['completed_at']
        await pump.close();assert pump.task.done()
    asyncio.run(scenario())


def test_navigation_invalidates_even_a_capture_that_was_in_flight():
    async def scenario():
        started=asyncio.Event();release=asyncio.Event();calls=0
        async def source():
            nonlocal calls
            calls+=1
            if calls==1:
                started.set();await release.wait();return 'old-page'
            return 'new-page'
        pump=LatestCapture(source,fps=200);pump.start()
        await started.wait();pump.invalidate();release.set()
        frame=await pump.next()
        assert frame['frame']=='new-page'
        await pump.close()
    asyncio.run(scenario())


def test_packet_callback_publishes_capture_before_consumer_and_default_is_unthrottled():
    async def scenario():
        published = []
        ready = asyncio.Event()
        async def source():
            return len(published) + 1
        def on_packet(packet):
            published.append(packet)
            if len(published) >= 5:
                ready.set()
        pump = LatestCapture(source, on_packet=on_packet)
        assert pump.period == 0
        pump.start()
        await asyncio.wait_for(ready.wait(), 1)
        latest = await pump.next()
        assert published[0]['seq'] == 1
        assert published[0]['frame'] == 1
        assert latest['seq'] >= published[0]['seq']
        assert all(item['captured_at'] <= item['completed_at'] for item in published)
        assert all(left['captured_at'] < right['captured_at']
                   for left, right in zip(published, published[1:]))
        await pump.close()
    asyncio.run(scenario())
