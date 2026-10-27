"""Conservative matching of local actions to a reviewed death trigger.

Matches are evidence associations only. They do not establish that an action
caused a death, and review prose is never treated as causal evidence.
"""

from __future__ import annotations

import math
from typing import Any


ATTRIBUTION_LABEL = "local_action_match_not_causal_proof"


def attribute_review(
    review: dict[str, Any],
    candidates: list[dict[str, Any]],
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Associate a tightly timed, spatially aligned local action with a death.

    A candidate is the captured trigger frame (``frame_id`` or
    ``frame_hash``), with ``scene='gameplay'`` and a player bounding box in
    ``context.player.bbox``. State entries contain the executed action and
    retrieved episodic hypotheses. Malformed or incomplete evidence is ignored.
    """
    if not isinstance(review, dict) or review.get("outcome") != "death":
        return []
    confidence = _number(review.get("confidence"))
    trigger_id = review.get("trigger_frame_id")
    if confidence is None or confidence < 0.7 or not isinstance(trigger_id, str) or not trigger_id:
        return []
    if not isinstance(candidates, list) or not isinstance(states, list):
        return []

    trigger = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        frame_id = candidate.get("frame_id", candidate.get("frame_hash"))
        if frame_id == trigger_id and candidate.get("scene") == "gameplay":
            timestamp = _number(candidate.get("captured_at"))
            player = _player_bbox(candidate)
            if timestamp is not None and player is not None:
                trigger = (timestamp, player, candidate.get('context',{}))
                break
    if trigger is None:
        return []
    trigger_time, trigger_box, trigger_context = trigger
    trigger_objects=_nearby_ids(trigger_context)
    if not trigger_objects:return []

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state in states:
        if not isinstance(state, dict):
            continue
        observation = state.get("state") if isinstance(state.get("state"), dict) else state
        executed_at = _number(state.get("executed_at"))
        captured_at = _number(observation.get("captured_at"))
        action = state.get("action")
        hold_ms = _number(state.get("hold_ms"))
        state_box = _player_bbox(observation)
        if (executed_at is None or captured_at is None or hold_ms is None or hold_ms < 0
                or not isinstance(action, str)
                or not action or state_box is None):
            continue
        if not -0.25 <= executed_at - trigger_time <= 0.6:
            continue
        if abs(captured_at - trigger_time) > 0.4:
            continue
        if _center_distance(trigger_box, state_box) > 24:
            continue
        if not trigger_objects.intersection(_nearby_ids(observation)):
            continue

        hypotheses = observation.get("hypotheses")
        if not isinstance(hypotheses, list):
            continue
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            if hypothesis.get("source") != "episodic_experience":
                continue
            if hypothesis.get("match_kind") != "scene_and_objects":
                continue
            evidence = hypothesis.get("match_evidence")
            if not isinstance(evidence, dict) or evidence.get("motion_verified") is not True:
                continue
            if hypothesis.get("suggested_action") != action:
                continue
            event_id = hypothesis.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in seen:
                continue
            seen.add(event_id)
            matches.append({
                "event_id": event_id,
                "action": action,
                "executed_at": executed_at,
                "trigger_time": trigger_time,
                "attribution": ATTRIBUTION_LABEL,
            })
    return matches


def _nearby_ids(observation):
    player=_player_bbox(observation)
    if player is None:return set()
    px,py,pw,ph=player;result=set()
    for obj in observation.get('objects',[]):
        if not isinstance(obj,dict):continue
        box=_player_bbox({'player':obj})
        if box is None or obj.get('age',0)<3 or obj.get('missed',0):continue
        x,y,w,h=box
        if -24<=x+w/2-(px+pw/2)<=110 and abs(y+h/2-(py+ph/2))<=60:
            if isinstance(obj.get('id'),(str,int)):result.add(obj['id'])
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _player_bbox(value: dict[str, Any]) -> tuple[float, float, float, float] | None:
    context = value.get("context")
    player = context.get("player") if isinstance(context, dict) else value.get("player")
    if not isinstance(player, dict):
        return None
    bbox = player.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    numbers = tuple(_number(item) for item in bbox)
    if any(item is None for item in numbers):
        return None
    x, y, width, height = numbers
    if width < 0 or height < 0:
        return None
    return x, y, width, height


def _center_distance(a: tuple[float, float, float, float],
                     b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return math.hypot((ax + aw / 2) - (bx + bw / 2),
                      (ay + ah / 2) - (by + bh / 2))
