from __future__ import annotations
import ctypes
import platform
import time
from threading import Thread, Event, Lock
from typing import Callable
import string
from .types import Decision


class DryController:
    armed = False
    def __init__(self) -> None:
        self.last_action = "wait"
    def apply(self, decision: Decision) -> None:
        self.last_action = decision.action
    def disarm(self) -> None: self.last_action = "wait"
    def arm(self) -> bool: return False
    def close(self) -> None: self.disarm()


def windows_foreground_title() -> str:
    if platform.system() != 'Windows': return ''
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
    user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length+1)
    user32.GetWindowTextW(hwnd,buffer,length+1)
    return buffer.value


class SafeKeyboard:
    """Teclado só no Windows, opt-in, foco verificado e watchdog independente."""
    def __init__(self, window_title: str, keys: dict[str,str] | None = None,
                 keyboard=None, focus: Callable[[],str] | None = None,
                 clock: Callable[[],float] = time.monotonic) -> None:
        if not window_title.strip(): raise ValueError("Título exato/parcial do jogo é obrigatório")
        self.keys = keys if keys is not None else {"left":"left","right":"right","jump":"space"}
        safe_names = {'left','right','up','down','space'}
        if (not isinstance(self.keys,dict) or set(self.keys) != {'left','right','jump'} or
                any(not isinstance(k,str) or (k not in safe_names and
                    (len(k) != 1 or k not in string.ascii_letters+string.digits))
                    for k in self.keys.values()) or len(set(self.keys.values())) != 3):
            raise ValueError('Controles precisam de três teclas distintas: letras, números, setas ou space; modificadores/atalhos do SO não são permitidos')
        if keyboard is None:
            if platform.system() != 'Windows': raise RuntimeError("Controle real homologável apenas no Windows neste protótipo")
            from pynput.keyboard import Controller
            keyboard = Controller()
        self.keyboard = keyboard; self.focus = focus or windows_foreground_title
        self.window_title = window_title.casefold(); self.clock = clock
        self.held: set = set(); self.armed = False; self.deadline = 0.0
        self.lock = Lock(); self.stop = Event()
        self.thread = Thread(target=self._watchdog,daemon=True,name="input-watchdog"); self.thread.start()

    def _resolve(self, name: str):
        if hasattr(self.keyboard,'test_backend'): return name
        from pynput.keyboard import Key
        return getattr(Key,name) if hasattr(Key,name) else name

    def _focused(self) -> bool:
        return self.window_title in self.focus().casefold()

    def arm(self) -> bool:
        with self.lock:
            if self.stop.is_set(): return False
            self.armed = False
            self._release()
            self.armed = self._focused()
            return self.armed

    def _release(self) -> None:
        first_error = None
        for key in tuple(self.held):
            try:
                self.keyboard.release(self._resolve(key))
                self.held.discard(key)
            except Exception as exc:
                # Keep failed releases tracked for watchdog/close retry, but try every key.
                if first_error is None: first_error = exc
        if first_error is not None: raise first_error

    def disarm(self) -> None:
        with self.lock:
            self.armed = False; self._release()

    def apply(self, decision: Decision) -> None:
        parts = [] if decision.action == 'wait' else decision.action.split('_')
        self.apply_keys({self.keys[p] for p in parts}, decision.hold_ms)

    def apply_keys(self, desired: set[str], hold_ms: int = 100) -> None:
        safe_names = {'left','right','up','down','space','enter','x','z','a','s','v','q','e'}
        if (not isinstance(desired, set) or not desired <= safe_names or
                isinstance(hold_ms, bool) or not isinstance(hold_ms, int) or not 1 <= hold_ms <= 250):
            raise ValueError('Teclas ou duração fora dos limites permitidos')
        with self.lock:
            try:
                if self.stop.is_set() or not self.armed or not self._focused():
                    self.armed = False; self._release(); return
                for key in self.held-desired:
                    self.keyboard.release(self._resolve(key))
                    self.held.discard(key)
                for key in desired-self.held:
                    # A backend can inject a key and THEN raise. Track the attempt first.
                    self.held.add(key)
                    self.keyboard.press(self._resolve(key))
                self.deadline = self.clock()+hold_ms/1000
            except BaseException:
                self.armed = False
                self.deadline = 0.0
                try: self._release()
                except Exception: pass  # failed releases remain tracked for retry
                raise

    def tick(self) -> None:
        with self.lock:
            if not self._focused():
                self.armed = False; self._release()
            elif self.held and self.clock() >= self.deadline:
                self._release()

    def _watchdog(self) -> None:
        while not self.stop.wait(0.01):
            try: self.tick()
            except Exception:
                # Conservative failure: disarm, attempt release and stop this controller.
                self.armed = False
                try:
                    with self.lock: self._release()
                except Exception:
                    pass  # close() can retry; never forget unsuccessfully released keys
                finally: self.stop.set()

    def close(self) -> None:
        self.stop.set(); self.thread.join(timeout=0.2); self.disarm()
