"""Small model-free parser and verifier for simple Mario goals."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
import unicodedata
from typing import Any


_STAGE_NOUN = r"(?:fase|nivel)"
_REACH = re.compile(rf"\b(?:chegar|chegue|chega)\s+(?:(?:na|no|a|ate)\s+)?{_STAGE_NOUN}\s*(\d+)\b")
_COMPLETE = re.compile(rf"\b(?:concluir|conclua|conclui|conclua)\s+(?:a\s+)?{_STAGE_NOUN}\s*(\d+)\b")
_WIN = re.compile(r"\b(?:zerar|zera|zer[e]?|vencer|venca|vencam|ven[cç]a|venc[e]?r)\b")
_TRUSTED_SOURCES = {"local_ocr", "labelled_template"}
_STAGE_SCENES = {"map", "gameplay", "stage_complete"}


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(c))


@dataclass
class GameGoal:
    description: str
    mode: str
    target_stage: int | None = None
    status: str = "pending"
    current_stage: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    _last_timestamp: float | None = field(default=None, repr=False, compare=False)
    _candidate_stage: int | None = field(default=None, repr=False, compare=False)
    _candidate_count: int = field(default=0, repr=False, compare=False)
    _victory_count: int = field(default=0, repr=False, compare=False)
    _entered_target: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def from_text(cls, text: str) -> "GameGoal":
        """Parse a supported Portuguese goal; leave other requests unverified."""
        return parse_goal(text)

    def to_dict(self) -> dict[str, Any]:
        """Return only the public, JSON-friendly goal state."""
        return {
            "description": self.description,
            "mode": self.mode,
            "target_stage": self.target_stage,
            "status": self.status,
            "completed": self.status == 'completed',
            "current_stage": self.current_stage,
            "evidence": self.evidence.copy(),
        }

    @staticmethod
    def _valid_stage(stage: int | None) -> bool:
        return isinstance(stage, int) and not isinstance(stage, bool) and stage > 0

    @staticmethod
    def _trusted(evidence: dict | None) -> bool:
        return isinstance(evidence, dict) and evidence.get("source") in _TRUSTED_SOURCES

    def _record(self, scene: str, stage: int | None, evidence: dict, now: float) -> None:
        observations = self.evidence.setdefault("observations", [])
        item = {"scene": scene, "source": evidence.get("source"), "timestamp": float(now)}
        if stage is not None:
            item["stage"] = stage
        # Keep the serialized state bounded during long-running objectives.
        observations.append(item)
        if len(observations) > 20:
            del observations[:-20]
        self.evidence['observations_verified'] = True

    def observe(self, scene: str, stage: int | None, evidence: dict, now: float) -> bool:
        """Consume one local visual observation and return whether the goal is done.

        A stage is accepted only when its evidence source is local OCR or a
        human-labelled visual template. Confidence scores and model labels are
        deliberately ignored.
        """
        if self.status == "completed":
            return True
        try:
            timestamp = float(now)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(timestamp) or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
            return False
        self._last_timestamp = timestamp
        label = str(scene).strip().lower()
        trusted = self._trusted(evidence)
        valid_stage = self._valid_stage(stage) and trusted and label in _STAGE_SCENES
        if valid_stage:
            self.current_stage = stage
            self._record(label, stage, evidence, timestamp)

        in_gameplay = label == "gameplay" and valid_stage
        if in_gameplay and self.target_stage is not None and stage == self.target_stage:
            self._entered_target = True

        if self.mode == "reach" and in_gameplay:
            if stage >= (self.target_stage or 1):
                if stage == self._candidate_stage:
                    self._candidate_count += 1
                else:
                    self._candidate_stage, self._candidate_count = stage, 1
                if self._candidate_count >= 2:
                    self.status = "completed"
                    self.evidence["completion"] = {"rule": "same_or_later_gameplay_stage_twice", "stage": stage}
            else:
                self._candidate_stage, self._candidate_count = None, 0
        elif self.mode == "reach":
            self._candidate_stage, self._candidate_count = None, 0

        if self.mode == "complete" and self._entered_target and label == "stage_complete" and valid_stage and stage == self.target_stage:
            self.status = "completed"
            self.evidence["completion"] = {"rule": "target_stage_complete_after_entry", "stage": stage}

        if self.mode == "win":
            if label == "victory" and trusted:
                self._record(label, None, evidence, timestamp)
                self._victory_count += 1
                if self._victory_count >= 2:
                    self.status = "completed"
                    self.evidence["completion"] = {"rule": "verified_victory_scene_twice"}
            else:
                self._victory_count = 0

        if self.mode != 'custom' and self.status == "pending" and valid_stage:
            self.status = "running"
        if self.status=='completed':self.evidence['verification']='verified'
        return self.status == "completed"


def parse_goal(text: str) -> GameGoal:
    """Parse common Portuguese Mario objectives without inventing a target."""
    if not isinstance(text,str) or not 3<=len(text.strip())<=1000:
        raise ValueError('Descreva o objetivo em 3 a 1000 caracteres')
    original = text.strip()
    normalized = _fold(original)
    reach = _REACH.search(normalized)
    if reach:
        return GameGoal(original, "reach", int(reach.group(1)), evidence={"verification": "unverified"})
    completion = _COMPLETE.search(normalized)
    if completion:
        return GameGoal(original, "complete", int(completion.group(1)), evidence={"verification": "unverified"})
    if _WIN.search(normalized):
        return GameGoal(original, "win", evidence={"verification": "unverified"})
    return GameGoal(original, "custom", evidence={"verification": "unverified", "reason": "unsupported_goal"})
