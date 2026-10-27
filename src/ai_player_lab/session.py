"""Control ownership and freshness independent of model and keyboard implementations."""
from dataclasses import dataclass, field


@dataclass
class ControlRouter:
    generation: int = 0
    owner: str = 'Luna'
    scene: str = 'unknown'
    controls: dict = field(default_factory=dict)
    game: str = ''
    summary: str = ''
    confirmed_at: float = -1e9
    _gameplay_samples: int = 0
    _last_sample: float = -1e9

    def reset(self, reason: str) -> None:
        self.generation += 1
        self.owner = 'Luna'
        self.summary = reason
        self._gameplay_samples = 0
        self._last_sample = -1e9

    def accept(self, result: dict, now: float, max_age: float = 6.0, *, temporal_confirmation: bool = False) -> str:
        age = now - result['captured_at']
        if result['generation'] != self.generation or not 0 <= age <= max_age:
            return 'stale'
        if result['confidence'] < .75:
            self.reset('Luna: tela ainda incerta; teclas soltas')
            return 'uncertain'
        self.scene = result['scene']
        self.summary = result['summary']
        self.game = result['game']
        self.controls.update(result['controls'])
        if self.scene == 'gameplay' and result['player_box'] is not None:
            self.confirmed_at = now
            if self.owner == 'Jev':
                return 'supervised'
            if result['captured_at'] > self._last_sample:
                self._gameplay_samples += 1
                self._last_sample = result['captured_at']
            if temporal_confirmation:
                self._gameplay_samples = 2
            if self._gameplay_samples >= 2:
                self.owner = 'Jev'
                self.generation += 1
                self._gameplay_samples = 0
                return 'handoff'
            return 'confirming'
        if self.owner == 'Jev':
            self.reset('Luna retomou: ' + self.scene)
            return 'takeover'
        self._gameplay_samples = 0
        return 'navigate' if self.scene in {'title','menu','map','results'} else 'wait'

    def lease_valid(self, now: float) -> bool:
        return self.owner == 'Jev' and now - self.confirmed_at <= 6.0
