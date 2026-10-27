from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .types import Track, BBox, center, iou, actionable_actor


class MultiTracker:
    """Associação geométrica experimental, não reidentificação semântica."""
    MAX_PREDICTION_AGE_S = 0.2

    def __init__(self, max_distance: float = 30.0, max_missed: int = 5) -> None:
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.elapsed: dict[int, float] = {}
        self.camera_shift: dict[int, np.ndarray] = {}
        self.relative_velocity: dict[int, np.ndarray] = {}
        self.camera_velocity = np.zeros(2, dtype=float)
        self._clock = 0.0
        self._last_confidence: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}

    def reset(self) -> None:
        self.tracks.clear()
        self.elapsed.clear()
        self.camera_shift.clear()
        self.relative_velocity.clear()
        self._last_confidence.clear()
        self._last_seen.clear()
        self._clock = 0.0

    def update(self, detections: list[BBox], dt: float, camera: dict | None = None, *,
               timestamp: float | None = None, allow_new_objects: bool = True) -> list[Track]:
        dt = max(float(dt), 1e-3)
        if timestamp is not None:
            current = float(timestamp)
            delta = current - self._clock
            dt = delta if delta > 1e-3 else dt
            self._clock = current
        else:
            self._clock += dt
        shift = np.array([camera.get('dx',0),camera.get('dy',0)], dtype=float) if camera else np.zeros(2)
        self.camera_velocity = .4*self.camera_velocity+.6*shift/dt
        ids = list(self.tracks)
        for tid in ids:
            self.elapsed[tid] += dt
            self.camera_shift[tid] += shift
        used: set[int] = set()
        matched: set[int] = set()
        if ids and detections:
            costs = np.full((len(ids), len(detections)), 1e6)
            for i, tid in enumerate(ids):
                t = self.tracks[tid]
                cx, cy = center(t.bbox)
                horizon = self.elapsed[tid]
                px, py = np.array([cx,cy])+self.relative_velocity[tid]*horizon+self.camera_shift[tid]
                for j, box in enumerate(detections):
                    bx, by = center(box)
                    distance = math.hypot(px-bx, py-by)
                    area_ratio = box[2]*box[3]/max(t.bbox[2]*t.bbox[3], 1)
                    if distance <= self.max_distance*(1+0.3*t.missed) and 0.2 < area_ratio < 5:
                        costs[i,j] = distance + 5*(1-iou(t.bbox, box))
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i,j] >= 1e5:
                    continue
                tid = ids[int(i)]; t = self.tracks[tid]; box = detections[int(j)]
                old_x, old_y = center(t.bbox); new_x, new_y = center(box)
                elapsed = self.elapsed[tid]
                self.relative_velocity[tid] = .4*self.relative_velocity[tid]+.6*(np.array([new_x-old_x,new_y-old_y])-self.camera_shift[tid])/elapsed
                t.vx, t.vy = map(float,self.relative_velocity[tid]+self.camera_velocity)
                t.bbox = tuple(float(v) for v in box)
                t.age += 1; t.missed = 0; t.confidence = min(0.9, 0.4+0.05*t.age)
                self._last_confidence[tid] = t.confidence
                self._last_seen[tid] = self._clock
                self.elapsed[tid] = 0.0
                self.camera_shift[tid][:] = 0
                used.add(int(j)); matched.add(tid)
        for tid in ids:
            if tid not in matched:
                self.tracks[tid].missed += 1
                elapsed = self.elapsed[tid]
                self.tracks[tid].confidence = self._last_confidence.get(tid, self.tracks[tid].confidence) * max(
                    0.0, 1.0 - elapsed / self.MAX_PREDICTION_AGE_S)
                expired_prediction = timestamp is not None and elapsed >= self.MAX_PREDICTION_AGE_S
                if self.tracks[tid].missed > self.max_missed or expired_prediction:
                    del self.tracks[tid]
                    del self.elapsed[tid]
                    del self.camera_shift[tid]
                    del self.relative_velocity[tid]
                    self._last_confidence.pop(tid, None)
                    self._last_seen.pop(tid, None)
        for j, box in enumerate(detections):
            if j not in used and allow_new_objects:
                self.tracks[self.next_id] = Track(self.next_id, tuple(float(v) for v in box), confidence=0.4)
                self.elapsed[self.next_id] = 0.0
                self.camera_shift[self.next_id] = np.zeros(2,dtype=float)
                self.relative_velocity[self.next_id] = np.zeros(2,dtype=float)
                self._last_confidence[self.next_id] = 0.4
                self._last_seen[self.next_id] = self._clock
                self.next_id += 1
        result = []
        for tid, t in self.tracks.items():
            if t.missed == 0:
                result.append(Track(**t.to_dict()))
            elif self.elapsed[tid] <= .2:
                predicted = Track(**t.to_dict())
                x,y,w,h = t.bbox
                position = np.array([x,y])+self.relative_velocity[tid]*self.elapsed[tid]+self.camera_shift[tid]
                predicted.bbox = (float(position[0]),float(position[1]),w,h)
                velocity = self.relative_velocity[tid]+self.camera_velocity
                predicted.vx, predicted.vy = map(float,velocity)
                predicted.label = 'predicted_motion'
                predicted.confidence = self.tracks[tid].confidence
                result.append(predicted)
        return result


def collision_estimate(player: Track, obj: Track, *, require_current_evidence=False) -> dict | None:
    """Aproximação linear de contato horizontal; não conhece hitboxes do jogo."""
    if require_current_evidence and not actionable_actor(obj):return None
    px, py, pw, ph = player.bbox; ox, oy, ow, oh = obj.bbox
    predicted_geometry = obj.missed > 0 or obj.label == 'predicted_motion'
    provenance = {'predicted_geometry': predicted_geometry,
                  'geometry_source': 'predicted_motion' if predicted_geometry else 'current_motion',
                  'object_missed': int(obj.missed)}
    if require_current_evidence:
        provenance.update(control_authorized=True,actor_evidence=obj.label)
    if min(py+ph, oy+oh) <= max(py, oy):
        return None
    if ox >= px+pw:
        gap, closing = ox-px-pw, player.vx-obj.vx
    elif px >= ox+ow:
        gap, closing = px-ox-ow, obj.vx-player.vx
    else:
        return {"kind": "visual_overlap", "object_id": obj.id, "ttc_ms": 0.0,
                "confidence": min(player.confidence, obj.confidence), "semantic_danger": "unknown",
                **provenance}
    if closing <= 1 or obj.age < 3:
        return None
    ttc = 1000*gap/closing
    if ttc > 2000:
        return None
    return {"kind": "approaching_object", "object_id": obj.id, "gap_px": round(gap,2),
            "closing_px_s": round(closing,2), "ttc_ms": round(ttc,1),
            "confidence": min(player.confidence, obj.confidence), "semantic_danger": "unknown",
            **provenance}
