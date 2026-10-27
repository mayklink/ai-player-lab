"""Small native status panel; capture and game stay visible, with no video overlay."""
from queue import SimpleQueue
import tkinter as tk


class LivePanel:
    def __init__(self, events: SimpleQueue, region: list[int]) -> None:
        self.root = tk.Tk()
        self.root.title('AI Player Lab — controle ao vivo')
        width, height = 360, 210
        x = 20 if region[0] > width + 30 else region[0] + region[2] + 15
        if x + width > self.root.winfo_screenwidth():
            x = 20
        y = 100 if x + width < region[0] else region[1] + region[3] + 15
        if y + height > self.root.winfo_screenheight():
            y = max(0, region[1] - height - 15)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.attributes('-topmost', True)
        self.state = tk.StringVar(value='Preparando captura…')
        self.detail = tk.StringVar(value='Clique no jogo para começar. F10 pausa; Esc encerra.')
        tk.Label(self.root, textvariable=self.state, font=('Segoe UI', 14, 'bold')).pack(pady=8)
        tk.Label(self.root, textvariable=self.detail, wraplength=340, justify='left').pack(padx=10)
        bar = tk.Frame(self.root); bar.pack(pady=12)
        tk.Button(bar, text='Pausar (F10)', command=lambda: events.put('pause')).pack(side='left', padx=4)
        tk.Button(bar, text='Retomar', command=lambda: events.put('resume')).pack(side='left', padx=4)
        tk.Button(bar, text='Encerrar', command=lambda: events.put('quit')).pack(side='left', padx=4)
        self.root.protocol('WM_DELETE_WINDOW', lambda: events.put('quit'))

    def update(self, owner: str, scene: str, armed: bool, summary: str, calls: str) -> None:
        self.state.set(f'{owner} • {scene} • ' + ('AO VIVO' if armed else 'PAUSADO'))
        self.detail.set(summary[:220] + '\n' + calls + '\nF9 retoma no jogo · F10 pausa · Esc encerra')
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        self.root.destroy()
