"""Conservative pixel-only character motion and jump-state estimates."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import math
import statistics

import cv2
import numpy as np


def _value(item, name, default=None):
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _box(item):
    box = _value(item, "bbox", None) or _value(item, "player_box", None)
    if box is None and isinstance(item, Mapping):
        box = item.get("box")
    if box is None or len(box) < 4:
        return None
    try:
        result = tuple(float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(v) for v in result) and result[2] > 0 and result[3] > 0 else None


class CharacterStateEstimator:
    """Estimate grounded/jump timing from current pixels and observed motion.

    Coordinates and velocities are image-space pixels, so camera movement can
    look like character movement. Ground contact is only a visual edge heuristic.
    """

    def __init__(self, parameters=None):
        self.parameters = dict(parameters or {})
        try:
            inset = float(self.parameters.get("foot_inset_px", 3))
            self.foot_inset_px = max(0.0, inset) if math.isfinite(inset) else 3.0
        except (TypeError, ValueError):
            self.foot_inset_px = 3.0
        self._configured_rise_ms = self._valid_rise(self.parameters.get("jump_rise_ms", 180))
        self._rise_samples = deque(maxlen=5)
        self._rise_sample_count = 0
        self._clear_transient()

    @staticmethod
    def _valid_rise(value):
        try:
            value = float(value)
            return value if math.isfinite(value) and 0 < value <= 3000 else None
        except (TypeError, ValueError):
            return None

    def _clear_transient(self):
        self._previous_box = None
        self._previous_time = None
        self._vy = None
        self._ground_streak = 0
        self._ground_feet = deque(maxlen=5)
        self._was_rising = False
        self._jump_key_was_down = False
        self._press_time = None
        self._press_foot = None
        self._takeoff_time = None
        self._takeoff_foot = None
        self._jump_edge_time = None
        self._jump_edge_foot = None
        self._ascent_samples = 0
        self._flight_confirmed = False

    def reset(self):
        """Clear per-scene motion state while retaining learned rise calibration."""
        self._clear_transient()

    @staticmethod
    def _held(held_keys, key):
        if isinstance(held_keys, Mapping):
            return bool(held_keys.get(key, False))
        if isinstance(held_keys, str):
            return key.casefold() in held_keys.casefold().split()
        try:
            return key in held_keys
        except TypeError:
            return False

    @staticmethod
    def _time(obs):
        for key in ("captured_at", "timestamp", "time_s", "now"):
            try:
                val = float(_value(obs, key))
                if math.isfinite(val):
                    return val
            except (TypeError, ValueError):
                pass
        return None

    def _support(self, frame, box):
        x, y, w, h = box
        height, width = frame.shape[:2]
        left = max(0, min(width, int(math.ceil(x + self.foot_inset_px))))
        right = max(left, min(width, int(math.floor(x + w - self.foot_inset_px))))
        foot = int(round(y + h - self.foot_inset_px))
        top, bottom = max(0, foot), min(height, foot + 5)
        if right <= left or bottom <= top:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if gray.ndim != 2 or gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        edges = cv2.Canny(gray[top:bottom, left:right], 40, 120)
        covered = np.any(edges > 0, axis=0)
        return float(np.count_nonzero(covered)) / max(1, right-left) >= 0.60

    @staticmethod
    def _objects(obs):
        objects = _value(obs, "objects", None)
        if objects is None:
            objects = _value(obs, "tracks", [])
        return list(objects or [])

    def _approaching(self, obs, player_box, delay_ms, rise_ms):
        px, py, pw, ph = player_box
        player_vx = float(_value(_value(obs, "player", obs), "vx", 0.0) or 0.0)
        results = []
        for obj in self._objects(obs):
            label=_value(obj,'label','')
            if _value(obj,'missed',0) or label in {'unknown_motion','background_consistent','predicted_motion'}:
                continue
            obox = _box(obj)
            if obox is None:
                continue
            ox, oy, ow, oh = obox
            vx = float(_value(obj, "vx", 0.0) or 0.0)
            vy = float(_value(obj, "vy", 0.0) or 0.0)
            # Extrapolate both boxes over the measured actuation delay.
            horizon = max(0.0, delay_ms) / 1000.0
            predicted_player = (px + player_vx*horizon, py, pw, ph)
            predicted_obj = (ox + vx*horizon, oy + vy*horizon, ow, oh)
            if min(predicted_player[1]+ph, predicted_obj[1]+oh) <= max(predicted_player[1], predicted_obj[1]):
                continue
            # TTC is measured from the current observation. The prediction is
            # exposed separately; delay is deducted once in the deadline.
            if ox >= px+pw:
                gap, closing = ox-(px+pw), player_vx-vx
            elif px >= ox+ow:
                gap, closing = px-(ox+ow), vx-player_vx
            else:
                gap, closing = 0.0, 0.0
            if closing <= 1:
                continue
            ttc = max(0.0, gap / closing * 1000.0)
            if ttc > 5000:
                continue
            deadline = None if rise_ms is None else ttc - max(0.0, delay_ms) - rise_ms
            can_jump = self._last_can_start_jump
            results.append({
                "kind": "approaching_object", "object_id": _value(obj, "id"),
                "ttc_ms": round(ttc, 1), "predicted_player_box": [round(v, 2) for v in predicted_player],
                "predicted_object_box": [round(v, 2) for v in predicted_obj],
                "jump_deadline_ms": None if deadline is None else round(deadline, 1),
                "jump_deadline_known": deadline is not None,
                "deadline_known": deadline is not None,
                "jump_must_start_this_decision": bool(deadline is not None and 0 <= deadline <= 150 and can_jump is True),
                "too_late": None if deadline is None else deadline < 0,
                "can_start_jump": can_jump,
            })
        return min(results, key=lambda item: item["ttc_ms"]) if results else None

    def update(self, frame, obs, held_keys, actuation_delay_ms=300, jump_key="x"):
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            raise ValueError("frame deve ser um array de imagem")
        delay = max(0.0, float(actuation_delay_ms))
        player = _value(obs, "player", None)
        box = _box(player) if player is not None else _box(obs)
        if box is None:
            self._clear_transient()
            return self._unknown_state()

        now = self._time(obs)
        if now is None:
            now = (self._previous_time + 1/30) if self._previous_time is not None else 0.0
        x, y, w, h = box
        foot = y + h - self.foot_inset_px
        dt = now - self._previous_time if self._previous_time is not None else None
        if self._previous_box is not None and dt is not None and dt > 0:
            self._vy = (foot - self._previous_box[1] - self._previous_box[3] + self.foot_inset_px) / dt
        else:
            self._vy = None
        grounded_support = self._support(frame, box)
        self._ground_feet.append(foot)
        # Pixel sprite boxes can alternate by one pixel while feet remain planted.
        stable_vertical = (self._vy is not None and abs(self._vy) <= 8.0) or (
            len(self._ground_feet)>=4 and max(self._ground_feet)-min(self._ground_feet)<=1)
        if grounded_support and stable_vertical:
            self._ground_streak += 1
        else:
            self._ground_streak = 0
        grounded = self._ground_streak >= 3
        if self._flight_confirmed and self._jump_edge_foot is not None and foot<self._jump_edge_foot-6:
            grounded=False

        key_down = self._held(held_keys, jump_key)
        key_released = not key_down
        rise_ms = statistics.median(self._rise_samples) if self._rise_samples else self._configured_rise_ms
        rise_source = "observed" if self._rise_samples else ("configured_estimate" if rise_ms is not None else "unknown")
        if key_down and not self._jump_key_was_down:
            self._press_time, self._press_foot = now, foot
            self._jump_edge_time,self._jump_edge_foot=now,foot
            self._ascent_samples=0;self._flight_confirmed=False
        armed=self._jump_edge_time is not None and 0<=now-self._jump_edge_time<=1.5
        if armed and self._vy is not None and self._vy < -8 and foot<=self._jump_edge_foot-4:
            self._ascent_samples+=1
            if self._ascent_samples>=2:self._flight_confirmed=True
        if not armed:self._flight_confirmed=False
        if self._flight_confirmed and self._vy is not None and self._vy < -8:
            if self._takeoff_time is None:
                self._takeoff_time, self._takeoff_foot = self._jump_edge_time, self._jump_edge_foot
            if self._press_time is not None and self._press_foot is not None and foot <= self._press_foot - 16:
                measured = (now - self._press_time) * 1000.0
                if 0 < measured <= 3000:
                    self._rise_samples.append(measured)
                    self._rise_sample_count += 1
                    rise_ms = statistics.median(self._rise_samples)
                    rise_source = "observed"
                self._press_time = self._press_foot = None
        if grounded:
            self._takeoff_time = self._takeoff_foot = None
            phase = "grounded"
        elif self._vy is None:
            phase = "unknown"
        elif self._flight_confirmed and self._vy < -8:
            phase, self._was_rising = "rising", True
        elif self._flight_confirmed and self._vy > 8:
            phase, self._was_rising = "falling", False
        elif self._flight_confirmed and self._was_rising:
            phase = "apex"
        else:
            phase = "unknown"

        can_start = True if grounded and key_released else (False if phase in {"rising", "apex", "falling"} or key_down else None)
        if grounded and self._jump_edge_time is not None and now-self._jump_edge_time>.2:
            self._flight_confirmed=False;self._jump_edge_time=None;self._was_rising=False
        air_time = (max(0.0, (now-self._takeoff_time)*1000.0)
                    if self._takeoff_time is not None and phase in {"rising", "apex", "falling"} else None)
        displacement = (max(0.0, self._takeoff_foot-foot)
                        if self._takeoff_foot is not None and phase in {"rising", "apex", "falling"} else None)
        self._last_can_start_jump = can_start
        approaching = self._approaching(obs, box, delay, rise_ms)
        self._previous_box, self._previous_time = box, now
        self._jump_key_was_down = key_down
        return {
            "grounded": grounded if grounded else None,
            "grounded_evidence": "visual_estimate" if grounded else None,
            "ground_support": grounded_support,
            "stable_vertical_frames": self._ground_streak,
            "jump_phase": phase, "can_start_jump": can_start,
            "flight_confirmed":self._flight_confirmed,
            "air_time_ms": None if air_time is None else round(air_time, 1),
            "displacement_since_takeoff_px": None if displacement is None else round(displacement, 2),
            "jump_key_released": key_released,
            "jump_rise_time_ms": None if rise_ms is None else round(rise_ms, 1),
            "jump_rise_time_source": rise_source,
            "rise_samples": self._rise_sample_count,
            "approaching_object": approaching,
            "camera_motion_caveat": "bbox e velocidades estão em pixels da tela; movimento da câmera pode parecer movimento do personagem",
        }

    def _unknown_state(self):
        rise_ms = statistics.median(self._rise_samples) if self._rise_samples else self._configured_rise_ms
        return {"grounded": None, "grounded_evidence": None, "ground_support": False,
                "stable_vertical_frames": 0, "jump_phase": "unknown", "can_start_jump": None,
                "air_time_ms": None, "displacement_since_takeoff_px": None,
                "jump_key_released": None,
                "jump_rise_time_ms": None if rise_ms is None else round(rise_ms, 1),
                "jump_rise_time_source": "observed" if self._rise_samples else ("configured_estimate" if rise_ms is not None else "unknown"),
                "rise_samples": self._rise_sample_count,
                "approaching_object": None,
                "camera_motion_caveat": "bbox e velocidades estão em pixels da tela; movimento da câmera pode parecer movimento do personagem"}
