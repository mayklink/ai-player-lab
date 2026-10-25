from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .types import Track, BBox, center, iou


class MultiTracker:
    """Associação geométrica experimental, não reidentificação semântica."""
    def __init__(self, max_distance: float = 30.0, max_missed: int = 5) -> None:
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.elapsed: dict[int, float] = {}

    def update(self, detections: list[BBox], dt: float) -> list[Track]:
        dt = max(dt, 1e-3)
        ids = list(self.tracks)
        for tid in ids:
            self.elapsed[tid] += dt
        used: set[int] = set()
        matched: set[int] = set()
        if ids and detections:
            costs = np.full((len(ids), len(detections)), 1e6)
            for i, tid in enumerate(ids):
                t = self.tracks[tid]
                cx, cy = center(t.bbox)
                horizon = self.elapsed[tid]
                px, py = cx+t.vx*horizon, cy+t.vy*horizon
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
                t.vx = 0.4*t.vx + 0.6*(new_x-old_x)/elapsed
                t.vy = 0.4*t.vy + 0.6*(new_y-old_y)/elapsed
                t.bbox = tuple(float(v) for v in box)
                t.age += 1; t.missed = 0; t.confidence = min(0.9, 0.4+0.05*t.age)
                self.elapsed[tid] = 0.0
                used.add(int(j)); matched.add(tid)
        for tid in ids:
            if tid not in matched:
                self.tracks[tid].missed += 1
                self.tracks[tid].confidence *= 0.7
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]
                    del self.elapsed[tid]
        for j, box in enumerate(detections):
            if j not in used:
                self.tracks[self.next_id] = Track(self.next_id, tuple(float(v) for v in box), confidence=0.4)
                self.elapsed[self.next_id] = 0.0
                self.next_id += 1
        result = []
        for tid, t in self.tracks.items():
            if t.missed == 0:
                result.append(Track(**t.to_dict()))
            elif self.elapsed[tid] <= .2:
                predicted = Track(**t.to_dict())
                x,y,w,h = t.bbox
                predicted.bbox = (x+t.vx*self.elapsed[tid], y+t.vy*self.elapsed[tid], w, h)
                predicted.label = 'predicted_motion'
                result.append(predicted)
        return result


def collision_estimate(player: Track, obj: Track) -> dict | None:
    """Aproximação linear de contato horizontal; não conhece hitboxes do jogo."""
    px, py, pw, ph = player.bbox; ox, oy, ow, oh = obj.bbox
    if min(py+ph, oy+oh) <= max(py, oy):
        return None
    if ox >= px+pw:
        gap, closing = ox-px-pw, player.vx-obj.vx
    elif px >= ox+ow:
        gap, closing = px-ox-ow, obj.vx-player.vx
    else:
        return {"kind": "visual_overlap", "object_id": obj.id, "ttc_ms": 0.0,
                "confidence": min(player.confidence, obj.confidence), "semantic_danger": "unknown"}
    if closing <= 1 or obj.age < 3:
        return None
    ttc = 1000*gap/closing
    if ttc > 2000:
        return None
    return {"kind": "approaching_object", "object_id": obj.id, "gap_px": round(gap,2),
            "closing_px_s": round(closing,2), "ttc_ms": round(ttc,1),
            "confidence": min(player.confidence, obj.confidence), "semantic_danger": "unknown"}
