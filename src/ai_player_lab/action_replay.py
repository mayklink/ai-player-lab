"""Guarded replay of a short, previously observed successful action sequence.

This module deliberately accepts only auditable experience dictionaries. It
does not infer success from survival or turn an unverified memory hypothesis
into an executable rule.
"""
from __future__ import annotations

import math
from typing import Any


MAX_REPLAY_SECONDS = 8.0
MAX_EXPERIENCE_AGE_SECONDS = 60.0
MAX_PLAYER_DRIFT_PX = 48.0
MAX_ACTION_LATENESS_SECONDS = 0.150
MAX_HOLD_MS = 250
_ACTIONS = {"wait", "left", "right", "jump", "left_jump", "right_jump"}


def _finite_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _player_box(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if (not isinstance(value, (tuple, list)) or len(value) != 4
            or any(not _finite_number(part) for part in value)):
        return None
    x, y, width, height = (float(part) for part in value)
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _sequence(experience: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw = experience.get("sequence", experience.get("actions",experience.get('successful_sequence')))
    if not isinstance(raw, list) or not raw:
        return None
    result: list[dict[str, Any]] = []
    previous_offset = -1.0
    for item in raw:
        if not isinstance(item, dict):
            return None
        action = item.get("action")
        hold_ms = item.get("hold_ms")
        offset = item.get("offset_s", item.get("offset"))
        # Accept explicit millisecond offsets, while keeping the schedule in
        # seconds internally. A bare `offset` is expressed in seconds.
        if "offset_ms" in item:
            offset = item["offset_ms"]
            if not _finite_number(offset):
                return None
            offset = float(offset) / 1000.0
        elif not _finite_number(offset):
            return None
        else:
            offset = float(offset)
        if (not isinstance(action, str) or action not in _ACTIONS or isinstance(hold_ms, bool)
                or not isinstance(hold_ms, int) or not 1 <= hold_ms <= MAX_HOLD_MS
                or offset < 0 or offset < previous_offset or offset > MAX_REPLAY_SECONDS):
            return None
        result.append({"action": action, "hold_ms": hold_ms, "offset_s": offset})
        if _player_box(item.get('player_box')) is not None:result[-1]['player_box']=item['player_box']
        previous_offset = offset
    return result


class ActionReplay:
    """Schedule one bounded action at a time from observed-success memory."""

    def __init__(self) -> None:
        self._active = False
        self._sequence: list[dict[str, Any]] = []
        self._index = 0
        self._started_at = 0.0
        self._episode: Any = None
        self._anchor: tuple[float, float, float, float] | None = None
        self._event_id = ""
        self._prefix = False
        self._expected_hazard_ids: set[Any] = set()

    @property
    def active(self) -> bool:
        return self._active

    def cancel(self) -> None:
        self._active = False
        self._sequence = []
        self._index = 0

    def start(self, experience: Any, now: float, playerbbox: Any, episode: Any) -> bool:
        """Start only a fresh, verified successful experience.

        `now` and an experience timestamp must share the caller's monotonic
        clock. Experiences without a timestamp are allowed; those carrying a
        timestamp older than the bounded freshness window are rejected.
        """
        self.cancel()
        if not _finite_number(now) or not isinstance(experience, dict):
            return False
        experience=dict(experience)
        if experience.get('action_authorized') is False:return False
        if experience.get('alternative_failed_in_later_attempt'):return False
        experience.setdefault('motion_verified',experience.get('match_evidence',{}).get('motion_verified'))
        if experience.get("alternative_failed") not in (None, False, [], {}):
            return False
        outcome = experience.get("outcome")
        prefix = outcome == "safe_prefix"
        if (outcome != "success"
              or experience.get("status") != "observed_success"
              or experience.get("match_kind") != "scene_and_objects"
              or experience.get("motion_verified") is not True
              or experience.get('action_authorized') is not True):
            return False
        evidence=experience.get('success_evidence')
        if (not isinstance(evidence,dict) or evidence.get('kind')!='tracked_object_passed'
                or evidence.get('no_contact') is not True or evidence.get('track_id') is None):return False
        for field,minimum in [('survived_s',1),('forward_progress_px',24),('separation_px',24)]:
            if not _finite_number(evidence.get(field)) or evidence[field]<minimum:return False
        sequence = _sequence(experience)
        anchor = _player_box(playerbbox)
        event_id = experience.get("event_id")
        if sequence is None or anchor is None or not isinstance(event_id, str) or not event_id:
            return False
        if sequence[-1]["offset_s"] > MAX_REPLAY_SECONDS:
            return False
        if sequence[-1]["offset_s"] + sequence[-1]["hold_ms"] / 1000.0 > MAX_REPLAY_SECONDS:
            return False
        # If a source timestamp is provided, reject future and stale records.
        timestamp = next((experience[key] for key in ("observed_at", "captured_at", "created_at", "timestamp")
                          if key in experience), None)
        if timestamp is not None:
            if not _finite_number(timestamp):
                return False
            age = float(now) - float(timestamp)
            if age < 0 or age > MAX_EXPERIENCE_AGE_SECONDS:
                return False
        self._sequence = sequence
        self._index = 0
        self._started_at = float(now)
        self._episode = episode
        self._anchor = anchor
        self._event_id = event_id
        self._prefix = prefix
        evidence = experience.get("match_evidence")
        ids = evidence.get("relative_object_ids", evidence.get("matched_object_ids", [])) if isinstance(evidence, dict) else []
        if not isinstance(ids, (list, tuple, set)):
            ids = []
        self._expected_hazard_ids = {value for value in ids if isinstance(value, (int, str))}
        self._active = True
        return True

    def next(self, now: float, episode: Any, playerbbox: Any, scene: str,
             stable: bool, hazards: Any, objects: Any = None) -> dict[str, Any] | None:
        """Return at most one due step, cancelling on any unsafe divergence."""
        if not self._active:
            return None
        if (not _finite_number(now) or scene != "gameplay" or stable is not True
                or episode != self._episode):
            self.cancel()
            return None
        player = _player_box(playerbbox)
        if player is None or self._anchor is None:
            self.cancel()
            return None
        elapsed = float(now) - self._started_at
        if elapsed < 0 or elapsed > MAX_REPLAY_SECONDS:
            self.cancel()
            return None
        ax, ay = _center(self._anchor)
        checkpoints=[s for s in self._sequence if s.get('player_box') is not None]
        if checkpoints:
            first=_center(checkpoints[0]['player_box'])
            expected=next((s for s in reversed(checkpoints) if s['offset_s']<=elapsed),checkpoints[0])
            target=_center(expected['player_box']);ax+=target[0]-first[0];ay+=target[1]-first[1]
        px, py = _center(player)
        if math.hypot(px - ax, py - ay) > MAX_PLAYER_DRIFT_PX:
            self.cancel()
            return None
        if self._dangerous(hazards, prefix=self._prefix,
                           expected_ids=self._expected_hazard_ids):
            self.cancel()
            return None
        if self._prefix and self._nearby_object(player, objects):
            self.cancel()
            return None
        if self._index >= len(self._sequence):
            self.cancel()
            return None

        step = self._sequence[self._index]
        due = self._started_at + step["offset_s"]
        if float(now) < due:
            return None
        if float(now) - due > MAX_ACTION_LATENESS_SECONDS:
            self.cancel()
            return None

        self._index += 1
        if self._index == len(self._sequence):
            # The returned step remains valid; subsequent calls see inactive.
            self._active = False
        return {"action": step["action"], "hold_ms": step["hold_ms"], "event_id": self._event_id}

    @staticmethod
    def _dangerous(hazards: Any, *, prefix: bool = False,
                   expected_ids: set[Any] | None = None) -> bool:
        if hazards is None:
            return False
        if not isinstance(hazards, list):
            return True
        for hazard in hazards:
            if not isinstance(hazard, dict):
                return True
            kind = str(hazard.get("kind", "")).lower()
            if prefix and kind in {"approaching_object", "visual_overlap", "immediate_overlap", "overlap", "gap", "gap_candidate"}:
                return True
            if kind in {"gap", "gap_candidate", "visual_overlap", "immediate_overlap", "overlap"}:
                return True
            ttc = hazard.get("ttc_ms")
            if ttc is not None:
                if not _finite_number(ttc):
                    return True
                if float(ttc) < 120 and (prefix or hazard.get("object_id") not in (expected_ids or set())):
                    return True
        return False

    def _nearby_object(self, player: tuple[float, float, float, float], objects: Any) -> bool:
        if objects is None:
            return False
        if not isinstance(objects, (list, tuple)):
            return True
        step = self._sequence[self._index] if self._index < len(self._sequence) else None
        direction = -1 if step and step["action"] in {"left", "left_jump"} else 1
        px, py, pw, ph = player
        player_cy = py + ph / 2
        for obj in objects:
            box_value = obj.get("bbox") if isinstance(obj, dict) else getattr(obj, "bbox", None)
            box = _player_box(box_value)
            if box is None:
                # Unknown object geometry cannot safely establish clearance.
                return True
            ox, oy, ow, oh = box
            object_cy = oy + oh / 2
            if abs(object_cy - player_cy) >= 60:
                continue
            gap = (px - (ox + ow)) if direction < 0 else (ox - (px + pw))
            if 0 <= gap <= 130:
                return True
        return False
