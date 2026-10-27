"""Bounded, event-focused visual evidence selection for asynchronous reviews."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np


def _state(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept both raw observations and EpisodeBuffer state wrappers."""
    value = record.get("context",record.get("state", record))
    if isinstance(value, Mapping) and isinstance(value.get("state"), Mapping):
        value = value["state"]
    return value if isinstance(value, Mapping) else {}


def _timestamp(record: Mapping[str, Any]) -> float:
    try:
        return float(record.get("captured_at", _state(record).get("captured_at", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _hash(record: Mapping[str, Any]) -> str | None:
    value = record.get("frame_hash") or record.get("hash")
    return str(value) if value is not None else None


def _scene(record: Mapping[str, Any]) -> str:
    state = _state(record)
    return str(record.get('scene') or state.get("scene") or state.get("scene_name") or "").lower()


def _is_gameplay(record: Mapping[str, Any]) -> bool:
    state = _state(record)
    scene = _scene(record)
    if scene == "gameplay":
        return True
    # Older observation records do not contain a classified scene. A player
    # track is the strongest available signal that the frame is gameplay.
    return state.get("player") is not None


def _is_transition(record: Mapping[str, Any]) -> bool:
    state = _state(record)
    camera = state.get("camera")
    return bool(state.get("scene_transition") or
                (isinstance(camera, Mapping) and camera.get("scene_transition")) or
                _scene(record) in {"death", "loading", "map", "title"})


def _collision_ttc(record: Mapping[str, Any]) -> float | None:
    state = _state(record)
    hazards = state.get("hazards", ())
    if not isinstance(hazards, Sequence) or isinstance(hazards, (str, bytes)):
        return None
    candidates: list[float] = []
    for hazard in hazards:
        if not isinstance(hazard, Mapping):
            continue
        kind = str(hazard.get("kind", "")).lower()
        try:
            ttc = float(hazard.get("ttc_ms"))
        except (TypeError, ValueError):
            continue
        if kind == "visual_overlap" or (kind in {"approaching_object", "approaching"} and ttc < 300):
            candidates.append(ttc)
    return min(candidates) if candidates else None


def _normalize_history(history: Sequence[Mapping[str, Any]], current: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sort records and collapse repeated frame hashes, favoring the current map."""
    by_hash: dict[str, dict[str, Any]] = {}
    unhashed: list[dict[str, Any]] = []
    for raw in [*history, current]:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        frame_hash = _hash(item)
        if frame_hash is None:
            unhashed.append(item)
        else:
            # Later copies win so the supplied current frame is retained.
            by_hash[frame_hash] = item
    return sorted([*by_hash.values(), *unhashed], key=_timestamp)


def _collision_band(records: list[dict[str, Any]], anchor_index: int, count: int = 6) -> set[int]:
    """Choose a compact contiguous neighborhood around a collision observation."""
    left = right = anchor_index
    while right - left + 1 < count:
        can_left = left > 0 and _timestamp(records[left]) - _timestamp(records[left - 1]) <= 0.251
        can_right = right + 1 < len(records) and _timestamp(records[right + 1]) - _timestamp(records[right]) <= 0.251
        if can_left and (not can_right or anchor_index - left <= right - anchor_index):
            left -= 1
        elif can_right:
            right += 1
        else:
            break
    return set(range(left, right + 1))


def select_event_frames(
    history: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    max_frames: int = 12,
) -> list[dict[str, Any]]:
    """Select bounded chronological evidence around the final gameplay event.

    The first gameplay frame supplies scene context. The ending gameplay and
    transition/current frames are retained, with a dense final two-second band.
    A nearby visual overlap or sub-300 ms approach reserves a six-frame local
    neighborhood when capacity permits.
    """
    if max_frames < 1:
        return []
    if not isinstance(current, Mapping):
        raise TypeError("current precisa ser um mapa de frame")
    records = _normalize_history(history, current)
    if not records:
        return []

    # A same-hash record in history was replaced with current by normalization.
    current_hash = _hash(current)
    current_index = next((i for i, item in enumerate(records)
                          if current_hash is not None and _hash(item) == current_hash), len(records) - 1)
    gameplay = [i for i, item in enumerate(records) if _is_gameplay(item)]
    last_gameplay = gameplay[-1] if gameplay else max(0, current_index - 1)
    first_context = gameplay[0] if gameplay else 0
    end_time = _timestamp(records[last_gameplay])

    selected: set[int] = {current_index, first_context, last_gameplay}
    # Keep explicit terminal state even if capture cadence leaves it outside
    # the dense gameplay interval.
    transitions = [i for i, item in enumerate(records) if _is_transition(item)]
    if transitions:
        selected.add(transitions[-1])

    collision = [(i, _collision_ttc(item)) for i, item in enumerate(records)]
    collision = [(i, ttc) for i, ttc in collision if ttc is not None and
                  _timestamp(records[i]) <= end_time and end_time - _timestamp(records[i]) <= 6.0]
    collision_anchor = min(collision, key=lambda pair: (abs(end_time - _timestamp(records[pair[0]])), pair[1]))[0] if collision else None
    collision_band: set[int] = set()
    if collision_anchor is not None:
        collision_band = _collision_band(records, collision_anchor)
        selected.update(collision_band)

    # Fill from the final dense band first; evenly spread only within that band
    # so earlier startup frames cannot displace event evidence.
    dense = [i for i, item in enumerate(records)
             if end_time - 2.0 <= _timestamp(item) <= _timestamp(records[current_index])]
    candidates = [i for i in dense if i not in selected]
    # Interleave indices across the remaining capacity to cover the full band.
    budget = max(0, max_frames - len(selected))
    if budget and candidates:
        positions = np.linspace(0, len(candidates) - 1, min(budget, len(candidates)), dtype=int)
        selected.update(candidates[int(pos)] for pos in positions)

    # If mandatory anchors exceed the requested cap, preserve current, collision
    # neighborhood, latest gameplay/transition, then early context.
    if len(selected) > max_frames:
        priority: list[int] = [current_index]
        if collision_anchor is not None:
            priority.extend(sorted(collision_band, key=lambda i: abs(i - collision_anchor)))
        priority.extend([last_gameplay, *(reversed(transitions)), first_context])
        priority.extend(sorted(selected, key=lambda i: abs(_timestamp(records[i]) - end_time)))
        kept: list[int] = []
        for i in priority:
            if i in selected and i not in kept and len(kept) < max_frames:
                kept.append(i)
        selected = set(kept)

    return [records[i] for i in sorted(selected, key=lambda i: (_timestamp(records[i]), i))]


def evidence_summary(
    selected: Sequence[Mapping[str, Any]],
    anchor_frame_hash: str | None = None,
) -> dict[str, Any]:
    """Describe temporal coverage and the selected collision anchor."""
    ordered = sorted((item for item in selected if isinstance(item, Mapping)), key=_timestamp)
    times = [_timestamp(item) for item in ordered]
    gaps_ms = [round((b - a) * 1000, 1) for a, b in zip(times, times[1:])]
    collisions = [(i, _collision_ttc(item)) for i, item in enumerate(ordered)]
    collisions = [(i, ttc) for i, ttc in collisions if ttc is not None]
    anchor = (ordered[min(collisions, key=lambda pair: pair[1])[0]] if collisions else None)
    hashes = [_hash(item) for item in ordered]
    local_times=[_timestamp(item) for item in ordered if anchor is not None and abs(_timestamp(item)-_timestamp(anchor))<=.6]
    return {
        "frame_count": len(ordered),
        "max_gap_ms": max(gaps_ms, default=0.0),
        "collision_band_max_gap_ms":max([round((b-a)*1000,1) for a,b in zip(local_times,local_times[1:])],default=0.),
        "anchor_frame_hash": anchor_frame_hash or (_hash(anchor) if anchor else None),
        "anchor_ttc_ms": _collision_ttc(anchor) if anchor else None,
        "evidence_frame_ids": [item.get("frame_id") or _hash(item) for item in ordered
                               if item.get("frame_id") is not None or _hash(item) is not None],
        "frame_hashes": [value for value in hashes if value is not None],
    }


def enlarge_hud(jpeg: bytes | bytearray | memoryview) -> bytes | None:
    """Decode a JPEG, enlarge its top HUD strip 3x with nearest-neighbor pixels."""
    if not isinstance(jpeg, (bytes, bytearray, memoryview)) or not jpeg:
        return None
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        return None
    crop = image[:min(40, image.shape[0]), :, :]
    enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    ok, encoded = cv2.imencode(".jpg", enlarged, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return encoded.tobytes() if ok else None
