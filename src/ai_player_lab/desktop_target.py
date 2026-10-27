"""Bind game input to the selected screen rectangle and its native window."""
import ctypes
from ctypes import wintypes
import platform


class RegionTarget:
    def __init__(self, region: list[int]) -> None:
        if platform.system() != 'Windows':
            raise RuntimeError('Controle ao vivo requer Windows')
        self.region = region
        self.user32 = ctypes.windll.user32
        self.user32.WindowFromPoint.argtypes = [wintypes.POINT]
        self.user32.WindowFromPoint.restype = wintypes.HWND
        self.user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self.user32.GetAncestor.restype = wintypes.HWND
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        x,y,w,h = region
        child = self.user32.WindowFromPoint(wintypes.POINT(x+w//2,y+h//2))
        self.hwnd = self.user32.GetAncestor(child, 2)
        if not self.hwnd:
            raise ValueError('Seleção não pertence a uma janela válida')
        self.rect = self._rect()

    def _rect(self) -> tuple:
        r = wintypes.RECT()
        if not self.user32.GetWindowRect(self.hwnd, ctypes.byref(r)):
            return ()
        return r.left,r.top,r.right,r.bottom

    def focused(self) -> bool:
        return self.user32.GetForegroundWindow() == self.hwnd and self._rect() == self.rect

    def contains(self, x: float, y: float) -> bool:
        left,top,w,h = self.region
        return left <= x < left+w and top <= y < top+h

    def pointer_inside(self) -> bool:
        p = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(p))
        return self.contains(p.x,p.y)

    def activate(self) -> bool:
        """The Start/Resume opt-in focuses only the previously selected game surface."""
        if self._rect() != self.rect:
            return False
        self.user32.SetForegroundWindow(self.hwnd)
        if not self.focused():
            return False
        from pynput.mouse import Controller, Button
        x,y,w,h=self.region
        pointer=Controller()
        pointer.position=(x+w//2,y+h//2)
        child=self.user32.WindowFromPoint(wintypes.POINT(x+w//2,y+h//2))
        if self.user32.GetAncestor(child,2)!=self.hwnd:
            return False
        pointer.click(Button.left)
        return True
