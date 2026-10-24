from __future__ import annotations
from dataclasses import dataclass
from queue import Queue, Empty, Full
from threading import Event, Thread, Lock
from typing import Any, Callable
import time


@dataclass
class WorkResult:
    value: Any
    error: str | None
    elapsed_ms: float


class LatestWorker:
    """Um worker, um item pendente, um resultado: nunca acumula backlog infinito.

    O item em execução não é cancelado; consumidores precisam verificar TTL.
    Uma exceção é devolvida como erro, sem encerrar o loop de captura.
    """
    def __init__(self, function: Callable, name: str) -> None:
        self.function = function
        self.pending: Queue = Queue(maxsize=1)
        self.results: Queue = Queue(maxsize=1)
        self.stop_event = Event()
        self.busy = Event()
        self.submission_lock = Lock()
        self.dropped = 0
        self.thread = Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, value: Any, replace: bool = True) -> bool:
        with self.submission_lock:
            if self.stop_event.is_set(): return False
            if not replace and (self.busy.is_set() or not self.pending.empty()): return False
            try:
                self.pending.put_nowait(value)
            except Full:
                if not replace: return False
                try:
                    self.pending.get_nowait(); self.dropped += 1
                except Empty: pass
                try: self.pending.put_nowait(value)
                except Full: return False
            return True

    def poll(self) -> WorkResult | None:
        try: return self.results.get_nowait()
        except Empty: return None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try: item = self.pending.get(timeout=0.05)
            except Empty: continue
            self.busy.set(); start = time.monotonic()
            try:
                result = WorkResult(self.function(item), None, (time.monotonic()-start)*1000)
            except Exception as exc:
                # Do not leak provider response bodies or credentials into logs.
                result = WorkResult(None, type(exc).__name__, (time.monotonic()-start)*1000)
            self.busy.clear()
            try: self.results.put_nowait(result)
            except Full:
                try: self.results.get_nowait()
                except Empty: pass
                try: self.results.put_nowait(result)
                except Full: pass

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=0.2)
