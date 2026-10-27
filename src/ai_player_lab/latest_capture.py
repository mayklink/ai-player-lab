"""Async browser producer with one overwrite-only slot, never a frame queue."""
import asyncio
from collections import deque
import time


class LatestCapture:
    def __init__(self, capture, *, fps=None, on_packet=None):
        """Capture newest frames without imposing a rate limit by default.

        ``on_packet`` is called synchronously for every accepted packet, before
        the processing consumer asks for it. It should be a short, nonblocking
        publisher (for example ``LiveServer.update``). Packets contain ``seq``,
        ``frame``, ``captured_at`` and ``completed_at`` from the producer clock.
        Set ``fps`` only when an explicit capture throttle is desired.
        """
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive or None")
        self.capture = capture
        self.period = 0. if fps is None else 1 / fps
        self.on_packet = on_packet
        self.task = None
        self.event = asyncio.Event()
        self.packet = None
        self.seq = 0
        self.consumed = 0
        self.generation = 0
        self.error = None
        self.times = deque(maxlen=60)
        self.dropped = 0

    def start(self):
        self.task = asyncio.create_task(self._run())

    def invalidate(self):
        self.generation += 1
        self.packet = None
        self.event.clear()

    @property
    def fps(self):
        return (len(self.times) - 1) / (self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.

    async def _run(self):
        try:
            while True:
                started = time.monotonic()
                generation = self.generation
                frame = await self.capture()
                completed = time.monotonic()
                # A coarse Windows clock can stamp two captures identically.
                # Wait for a real new tick; never invent future observation time.
                if self.packet is not None and completed <= self.packet['captured_at']:
                    await asyncio.sleep(0)
                    continue
                if frame is not None and generation == self.generation:
                    if self.packet is not None and self.packet['seq'] > self.consumed:
                        self.dropped += 1
                    self.seq += 1
                    self.times.append(completed)
                    # Use completion time as the frame's observation time. The
                    # capture start is kept for latency diagnostics only; a
                    # duplicate start tick must never break downstream strict
                    # timestamp ordering when the producer runs very quickly.
                    captured_at = completed
                    packet = {'seq': self.seq, 'frame': frame,
                              'captured_at': captured_at, 'completed_at': completed,
                              'capture_started_at': started}
                    self.packet = packet
                    self.event.set()
                    if self.on_packet is not None:
                        result = self.on_packet(packet)
                        if hasattr(result, '__await__'):
                            close = getattr(result, 'close', None)
                            if close:
                                close()
                            raise TypeError('on_packet must be a short synchronous callback')
                # No forced millisecond sleep: on Windows that can become a
                # full timer tick. Yield to consumers, throttling only on demand.
                await asyncio.sleep(max(0., self.period - (time.monotonic() - started)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = type(exc).__name__
            self.event.set()

    async def next(self, timeout=.5):
        end = time.monotonic() + timeout
        while True:
            if self.packet is not None and self.packet['seq'] > self.consumed:
                self.consumed = self.packet['seq']
                return self.packet
            if self.error:
                raise RuntimeError('Capture producer failed: ' + self.error)
            self.event.clear()
            try:
                await asyncio.wait_for(self.event.wait(), max(.001, end - time.monotonic()))
            except asyncio.TimeoutError:
                return None

    async def close(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
