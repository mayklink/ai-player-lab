from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any
import math

BBox = tuple[float, float, float, float]  # x, y, width, height in working image
ACTIONS = frozenset({"wait", "left", "right", "jump", "left_jump", "right_jump"})


def center(box: BBox) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def iou(a: BBox, b: BBox) -> float:
    x = max(a[0], b[0]); y = max(a[1], b[1])
    r = min(a[0] + a[2], b[0] + b[2]); d = min(a[1] + a[3], b[1] + b[3])
    overlap = max(0.0, r-x) * max(0.0, d-y)
    return overlap / max(1e-9, a[2]*a[3] + b[2]*b[3] - overlap)


@dataclass
class Track:
    id: int
    bbox: BBox
    vx: float = 0.0  # screen px/s, not world coordinates
    vy: float = 0.0
    confidence: float = 0.0  # heuristic score, not calibrated probability
    age: int = 1
    missed: int = 0
    label: str = "unknown_motion"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    seq: int
    captured_at: float  # monotonic seconds; same clock for all live workers
    frame_hash: str
    scene_hash: str
    width: int
    height: int
    player: Track | None
    objects: list[Track]
    camera: dict[str, Any]
    hazards: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    episode: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    action: str
    observed_at: float
    seq: int
    confidence: float = 1.0
    source: str = "local"
    hold_ms: int = 100
    episode: int = 0

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"Ação não permitida: {self.action}")
        if not math.isfinite(self.observed_at) or not math.isfinite(self.confidence):
            raise ValueError("Tempos e confiança precisam ser finitos")
        if not 0 <= self.confidence <= 1 or not 1 <= self.hold_ms <= 250:
            raise ValueError("Decisão fora dos limites de segurança")
