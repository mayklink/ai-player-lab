"""Small, persistent, pixel-only scene memory for repeated game situations.

This module stores reviewed hypotheses as unverified notes. It does not train a
model or infer object semantics from pixels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import hashlib
import tempfile
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np


_FORMAT_VERSION = 1
_MAX_RECALL = 2
_SIGNATURE_LIMIT = 0.075
_PIXEL_LIMIT = 0.08
_SUCCESS_ACTIONS = {"wait", "left", "right", "jump", "left_jump", "right_jump"}
_REFERENCE_CACHE_LIMIT = 64


class ExperienceMemory:
    """Persist and retrieve visually similar experiences for one game URL.

    ``root`` is a directory owned by the caller. The exact ``game_url`` and
    ``stage`` are hard scope boundaries; visual similarity never crosses them.
    Frames are stored as PNG references and a compact grayscale play-area
    signature is used for matching.
    """

    def __init__(self, root: str | Path, game_url: str) -> None:
        self.root = Path(root)
        self.game_url = str(game_url)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "experience-memory.json"
        self._records, self._readable = self._read_index()
        self._index_token = self._current_index_token()
        self._reference_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._preload_references(self._records)

    def add(
        self,
        event_id: str,
        review: dict[str, Any],
        frame: np.ndarray,
        stage: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Save an event once, returning its record or None on corrupt storage."""
        if not self._readable:
            return None
        if not isinstance(event_id, str) or not event_id or not isinstance(stage, str) or not stage:
            raise ValueError("event_id e stage precisam ser strings não vazias")
        if not isinstance(review, dict) or not isinstance(context, dict):
            raise ValueError("review e context precisam ser objetos")
        existing = next((r for r in self._records if r.get("event_id") == event_id), None)
        if existing is not None:
            return existing

        image = _normalize_frame(frame)
        signature = _signature(image)
        filename = hashlib.sha256(event_id.encode("utf-8")).hexdigest() + ".png"
        encoded_ok, encoded = cv2.imencode(".png", image)
        if not encoded_ok:
            raise ValueError("Falha ao codificar frame")
        _atomic_write(self.root / filename, encoded.tobytes())

        safe_review = _json_copy(review)
        entry = {
            "event_id": event_id,
            "game_url": self.game_url,
            "stage": stage,
            "reference": filename,
            "signature": signature,
            "context": _json_copy(context),
            "review": safe_review,
            "provenance": "review_unverified",
            "confidence": _confidence(safe_review),
        }
        self._records.append(entry)
        try:
            self._write_index()
        except Exception:
            self._records.pop()
            raise
        return entry

    def add_success(
        self,
        event_id: str,
        frame: np.ndarray,
        stage: str,
        context: dict[str, Any],
        actions: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Store a bounded successful sequence backed by tracked local evidence.

        This records an observed pass, not a generally valid game strategy.
        The evidence format is intentionally strict so a caller cannot promote
        an ordinary review or an unverified claim into a success record.
        """
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id precisa ser uma string não vazia")
        existing = next((r for r in self._records if r.get("event_id") == event_id), None)
        if existing is not None:
            return existing
        _validate_success_evidence(evidence)
        sequence, suggested = _success_sequence(actions)
        review = {
            "outcome": "success",
            "status": "observed_success",
            "hypothesis": "Neste contexto observado, esta sequência precedeu uma passagem rastreada; resultado local, não garantia geral.",
            "suggested_action": suggested,
            "successful_sequence": sequence,
            "confidence": 1.0,
        }
        entry = self.add(event_id, review, frame, stage, context)
        if entry is not None:
            entry["provenance"] = "tracked_local_observation"
            entry["success_evidence"] = _json_copy(evidence)
            self._records = [entry if r.get("event_id") == event_id else r for r in self._records]
            self._write_index()
        return entry

    def add_prefix(self,event_id,frame,stage,context,actions,evidence):
        if evidence.get('kind')!='observed_forward_segment' or evidence.get('no_contact') is not True:
            raise ValueError('Prefix needs local evidence')
        if not 0<float(evidence.get('duration_s',0))<=8 or float(evidence.get('forward_progress_px',0))<24 or float(evidence.get('survived_s',0))<1:
            raise ValueError('Prefix evidence insufficient')
        if not all(np.isfinite(float(evidence[k])) for k in ['duration_s','forward_progress_px','survived_s']):raise ValueError('Invalid prefix evidence')
        sequence,suggested=_success_sequence(actions)
        entry=self.add(event_id,{'outcome':'safe_prefix','hypothesis':'Prefixo observado com avanço e continuidade; repetir somente na mesma situação.',
            'status':'observed_prefix','confidence':1.,'suggested_action':suggested,'successful_sequence':sequence},frame,stage,context)
        if entry:
            entry['success_evidence']=_json_copy(evidence);entry['provenance']='tracked_prefix_observation'
            self._write_index()
        return entry

    def retrieve(
        self, frame: np.ndarray, stage: str, context: dict[str, Any], *, success_only: bool = False,
        _event_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return at most two supported scene hypotheses, or [] on bad data."""
        if not isinstance(stage, str) or not isinstance(context, dict):
            return []
        # Stat the compact manifest on each call, but parse it and preload PNGs
        # only when it changed. This lets sibling instances publish records.
        self._refresh_index()
        if not self._readable:
            return []
        try:
            signature = np.asarray(_signature(_normalize_frame(frame)),dtype=np.float32)
        except (TypeError, ValueError, cv2.error):
            return []
        try:
            current_player = _player_location(context)
        except (TypeError, ValueError, KeyError, OverflowError):
            return []
        if current_player is None:
            return []

        current_frame = _normalize_frame(frame)
        matches: list[tuple[float, dict[str, Any]]] = []
        current_features: list[Any] = [None]
        for record in self._records:
            if _event_ids is not None and record.get("event_id") not in _event_ids:
                continue
            if record.get('disabled'):
                continue
            if not isinstance(record.get('review'),dict):continue
            if success_only and record['review'].get('outcome')!='success':continue
            if record['review'].get('outcome')=='death' and not record['review'].get('trigger_frame_id'):
                continue  # Legacy reviews did not identify an actionable trigger.
            if record.get("game_url") != self.game_url or record.get("stage") != stage:
                continue
            try:
                ref_player = _player_location(record.get("context", {}))
                if ref_player is None:
                    continue
                old_sig = np.asarray(record["signature"], dtype=np.float32)
                if old_sig.shape != signature.shape or not np.isfinite(old_sig).all():
                    continue
                difference = float(np.mean(np.abs(signature - old_sig)))
                if Path(record['reference']).name != record['reference']:
                    continue
                cached = self._get_reference(record['reference'])
                if cached is None:
                    continue
                reference = cached['image']
                if reference.shape != current_frame.shape:
                    continue
                scene = _compare_scene(reference, current_frame, ref_player, current_player,
                                       record.get("context", {}), context, difference,
                                       reference_features=cached.get('features'),
                                       reference_cache=cached,
                                       current_features=current_features)
                if scene is None:
                    continue
                object_match = _objects_compatible(context, record.get("context", {}))
                if object_match is None:
                    continue
                review = record.get("review")
                if not isinstance(review, dict):
                    continue
                hypothesis = review.get("hypothesis")
                if not isinstance(hypothesis, str) or not hypothesis.strip():
                    continue
                outcome = review.get("outcome", "review")
                evidence_status, action_authorized = _record_authority(record, outcome)
                local_death = evidence_status == "verified_local_life_loss"
                recalled_outcome = (outcome if action_authorized or local_death else "unverified")
                recalled_status = ("observed_success" if action_authorized and outcome == "success"
                                   else "observed_prefix" if action_authorized and outcome == "safe_prefix"
                                   else "unverified")
                attempts = record.get("subsequent_attempts", [])
                later_death = any(isinstance(a, dict) and a.get("outcome") == "death"
                                  and a.get('attribution')=='local_action_match_not_causal_proof'
                                  for a in attempts)
                motion_verified = object_match["nearby_count"] > 0 and \
                                  object_match["motion_compared"] == object_match["nearby_count"]
                matches.append((scene["score"], {
                    "event_id": record["event_id"],
                    "reference": record["reference"],
                    "hypothesis": hypothesis,
                    "suggested_action": review.get('suggested_action','wait') if action_authorized else None,
                    "successful_sequence": review.get("successful_sequence") if action_authorized else None,
                    "success_evidence": record.get("success_evidence"),
                    "outcome": recalled_outcome,
                    "reviewed_outcome": outcome,
                    "status": recalled_status,
                    "evidence_status": evidence_status,
                    "action_authorized": action_authorized,
                    "hypothesis_status": "model_claim_unverified" if outcome == "death" else None,
                    "local_event_evidence": review.get("life_evidence") if local_death else None,
                    "match_kind": "scene_and_objects" if object_match["nearby_count"] else "scene_only",
                    "match_evidence": {
                        "scene_method": scene["method"],
                        "pixel_residual": round(scene["residual"], 4),
                        "signature_difference": round(difference, 4),
                        "player_alignment_error": round(scene["player_error"], 4),
                        "nearby_objects": object_match["nearby_count"],
                        "relative_motion_compared": object_match["motion_compared"],
                        "motion_verified": motion_verified,
                    },
                    "confidence": record.get("confidence", 0.0),
                    "provenance": record.get('provenance','review_unverified'),
                    "alternative_failed_in_later_attempt": later_death,
                    "subsequent_attempts": attempts[-3:] if isinstance(attempts, list) else [],
                    "similarity": round(max(0.0, 1.0 - scene["score"]), 4),
                    "_match_tier": 1 if motion_verified else 0,
                    "_priority": (2 if outcome == "success" and motion_verified and not later_death
                                  else 0 if later_death or outcome == "death" else 1),
                }))
            except (KeyError, TypeError, ValueError, OSError, cv2.error):
                # Individual malformed records are ignored fail-closed.
                continue
        # A success may lead only when this retrieval verified the same local
        # relative motion and no later attempt contradicted it. Other records
        # retain ordinary visual-similarity order and remain visible as options.
        matches.sort(key=lambda item: (-item[1]["_match_tier"],
                                       -item[1]["_priority"], item[0]))
        result = [value for _, value in matches[:_MAX_RECALL]]
        for value in result:
            value.pop("_priority", None)
            value.pop("_match_tier", None)
        return result

    def revalidate(self, matches: list[dict[str, Any]], frame: np.ndarray, stage: str,
                   context: dict[str, Any]) -> list[dict[str, Any]]:
        """Re-check old strategic notes against a fresh frame and context.

        Only IDs from the earlier result are considered. Current geometry,
        object motion, stage, exact game URL, and scene pixels are all checked
        again by ``retrieve``; historical matches alone never authorize action.
        """
        event_ids = {item.get('event_id') for item in matches
                     if isinstance(item, dict) and isinstance(item.get('event_id'), str)}
        if not event_ids:
            return []
        return self.retrieve(frame, stage, context, _event_ids=event_ids)

    def similar_successes(
        self, frame: np.ndarray, stage: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Return only successful observations supported for this scene."""
        return self.retrieve(frame, stage, context, success_only=True)

    def record_outcome(self,event_ids: list[str],outcome: str,event_id: str,*,attributions=None) -> None:
        """An episode outcome is evidence, not proof the earlier advice caused it."""
        if outcome not in {'death','not_death','uncertain'}:raise ValueError('Invalid outcome')
        records,readable=self._read_index()
        if not readable:return
        for entry in records:
            if entry.get('game_url')==self.game_url and entry.get('event_id') in event_ids:
                attempts=entry.setdefault('subsequent_attempts',[])
                if not any(a['event_id']==event_id for a in attempts):
                    attribution=next((a for a in (attributions or []) if a.get('event_id')==entry.get('event_id')),None)
                    item={'event_id':event_id,'outcome':outcome,'causal_link':'unverified'}
                    if attribution:item.update({k:v for k,v in attribution.items() if k!='event_id'})
                    attempts.append(item)
                entry['subsequent_attempts']=attempts[-20:]
        self._records=records;self._write_index()

    def _current_index_token(self):
        try:
            stat = self._index_path.stat()
            return (stat.st_size, stat.st_mtime_ns, getattr(stat, 'st_ino', 0))
        except FileNotFoundError:
            return None

    def _refresh_index(self) -> None:
        token = self._current_index_token()
        if token == self._index_token:
            return
        self._records, self._readable = self._read_index()
        self._index_token = token
        self._reference_cache.clear()
        if self._readable:
            self._preload_references(self._records)

    def _preload_references(self, records: list[dict[str, Any]]) -> None:
        """Decode a bounded working set once when the manifest changes."""
        for record in records:
            if len(self._reference_cache) >= _REFERENCE_CACHE_LIMIT:
                break
            name = record.get('reference')
            if record.get('game_url') != self.game_url or not isinstance(name, str):
                continue
            self._get_reference(name)

    def _get_reference(self, name: str) -> dict[str, Any] | None:
        if not isinstance(name, str) or Path(name).name != name:
            return None
        path = self.root / name
        try:
            stat = path.stat()
            token = (stat.st_size, stat.st_mtime_ns, getattr(stat, 'st_ino', 0))
        except OSError:
            self._reference_cache.pop(name, None)
            return None
        cached = self._reference_cache.get(name)
        if cached is not None and cached['token'] == token:
            self._reference_cache.move_to_end(name)
            return cached
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self._reference_cache.pop(name, None)
            return None
        cached = {'token': token, 'image': image, 'features': None}
        self._reference_cache[name] = cached
        self._reference_cache.move_to_end(name)
        while len(self._reference_cache) > _REFERENCE_CACHE_LIMIT:
            self._reference_cache.popitem(last=False)
        return cached

    def _read_index(self) -> tuple[list[dict[str, Any]], bool]:
        if not self._index_path.exists():
            return [], True
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            if data.get("version") != _FORMAT_VERSION or not isinstance(data.get("entries"), list):
                return [], False
            if any(not isinstance(entry, dict) for entry in data["entries"]):
                return [], False
            return data["entries"], True
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError):
            return [], False

    def _write_index(self) -> None:
        data = {"version": _FORMAT_VERSION, "entries": self._records}
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        _atomic_write(self._index_path, payload.encode("utf-8"))
        self._index_token = self._current_index_token()


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame inválido")
    if frame.ndim == 2:
        color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        color = frame
    elif frame.ndim == 3 and frame.shape[2] == 4:
        color = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError("frame precisa ser grayscale, BGR ou BGRA")
    if color.dtype != np.uint8:
        color = np.clip(color, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(color)


def _validate_success_evidence(evidence: Any) -> None:
    """Require a concrete tracked-object pass before accepting success."""
    if not isinstance(evidence, dict) or evidence.get("kind") != "tracked_object_passed":
        raise ValueError("success exige evidência tracked_object_passed")
    track_id = evidence.get("track_id")
    if not isinstance(track_id, (str, int)) or isinstance(track_id, bool) or str(track_id).strip() == "":
        raise ValueError("track_id inválido")
    start, end = _finite_number(evidence.get("start_time")), _finite_number(evidence.get("end_time"))
    progress = _finite_number(evidence.get("forward_progress_px"))
    separation = _finite_number(evidence.get("separation_px"))
    survived = _finite_number(evidence.get("survived_s"))
    if end <= start or end - start < 1 or progress < 24 or separation < 24 or survived < 1:
        raise ValueError("evidência não atende aos limites mínimos de passagem")
    if evidence.get("no_contact") is not True:
        raise ValueError("passagem exige no_contact=true")
    # Ensure persistence cannot fail later due to NaN, custom objects or cycles.
    _json_copy(evidence)


def _finite_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("evidência numérica inválida")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("evidência numérica inválida") from exc
    if not np.isfinite(number):
        raise ValueError("evidência numérica não finita")
    return number


def _record_authority(record: dict[str, Any], outcome: Any) -> tuple[str, bool]:
    """Keep model reviews as hypotheses; only locally validated evidence authorizes replay."""
    evidence = record.get("success_evidence")
    review = record.get("review") if isinstance(record.get("review"),dict) else {}
    try:
        if (outcome == "success" and record.get("provenance") == "tracked_local_observation"):
            _validate_success_evidence(evidence)
            return "verified_local_passage", True
        if (outcome == "safe_prefix" and record.get("provenance") == "tracked_prefix_observation"
                and isinstance(evidence, dict) and evidence.get("kind") == "observed_forward_segment"
                and evidence.get("no_contact") is True
                and 0 < _finite_number(evidence.get("duration_s")) <= 8
                and _finite_number(evidence.get("forward_progress_px")) >= 24
                and _finite_number(evidence.get("survived_s")) >= 1):
            return "verified_local_prefix", True
        life=review.get("life_evidence") if isinstance(review.get("life_evidence"),dict) else {}
        if (outcome == "death" and review.get("outcome_source") == "local_life_counter"
                and life.get("confirmed") is True and life.get("status") == "life_lost"
                and _finite_number(life.get("decrement")) == 1
                and _finite_number(life.get("before_value")) - _finite_number(life.get("after_value")) == 1):
            return "verified_local_life_loss", False
    except (TypeError, ValueError, OverflowError):
        pass
    if outcome == "death":
        return "unverified_model_review", False
    return "no_action_evidence", False


def _success_sequence(actions: Any) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(actions, list) or not actions or len(actions) > 4096:
        raise ValueError("actions precisa conter de 1 a 4096 eventos")
    sequence: list[dict[str, Any]] = []
    next_offset = 0
    for raw in actions:
        if not isinstance(raw, dict):
            raise ValueError("cada ação precisa ser um objeto")
        action = raw.get("action")
        if action not in _SUCCESS_ACTIONS:
            raise ValueError("ação não permitida")
        hold = raw.get("hold_ms")
        if isinstance(hold, bool) or not isinstance(hold, (int, float)) or not np.isfinite(hold):
            raise ValueError("hold_ms inválido")
        hold = int(hold)
        if hold < 1 or hold > 250:
            raise ValueError("hold_ms fora dos limites")
        offset = raw.get("offset_s", next_offset / 1000)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not np.isfinite(offset):
            raise ValueError("offset_s inválido")
        offset = float(offset)
        if offset < 0 or offset > 8 or offset + hold / 1000 > 8:
            raise ValueError("sequência excede 8 segundos")
        sequence.append({"action": action, "hold_ms": hold, "offset_s": round(offset, 3)})
        if isinstance(raw.get('player_box'),(list,tuple)) and len(raw['player_box'])==4:
            sequence[-1]['player_box']=list(raw['player_box'])
        next_offset = int(offset * 1000) + hold
    # Leases renewed every frame are not separate key presses. Preserve the
    # complete observed interval, including supersession by a different action.
    intervals=[]
    for step in sequence:
        start=step['offset_s'];end=start+step['hold_ms']/1000
        if intervals and start<intervals[-1]['start']:raise ValueError('ações fora de ordem')
        if intervals and intervals[-1]['action']==step['action'] and start<=intervals[-1]['end']+.001:
            intervals[-1]['end']=max(intervals[-1]['end'],end)
        else:
            if intervals:intervals[-1]['end']=min(intervals[-1]['end'],start)
            intervals.append({'action':step['action'],'start':start,'end':end})
    compact=[]
    for interval in intervals:
        at=interval['start']
        while at<interval['end']-.0005:
            hold=min(250,int(round((interval['end']-at)*1000)))
            if hold<=0:break
            item={'action':interval['action'],'hold_ms':hold,'offset_s':round(at,3)}
            sample=min(sequence,key=lambda s:abs(s['offset_s']-at))
            if sample.get('player_box'):item['player_box']=sample['player_box']
            compact.append(item);at+=hold/1000
    if not compact or len(compact)>512:raise ValueError('sequência compactada fora dos limites')
    return compact,compact[0]['action']


def _play_area(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    # The upper 14% and lower 4% commonly contain HUD/status strips.
    top, bottom = int(height * 0.14), max(int(height * 0.96), 1)
    crop = frame[top:bottom, :]
    if crop.size == 0 or width == 0:
        raise ValueError("frame sem área de jogo")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA)


def _signature(frame: np.ndarray) -> list[float]:
    return (_play_area(frame).astype(np.float32) / 255.0).reshape(-1).tolist()


def _player_location(context: dict[str, Any]) -> tuple[float, float] | None:
    """Read normalized center from common detector-context shapes."""
    value = context.get("player_location", context.get("player"))
    if isinstance(value, dict):
        if "bbox" in value:
            box = value["bbox"]
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                return None
            x, y = float(box[0]) + float(box[2]) / 2, float(box[1]) + float(box[3]) / 2
        elif "center_x" in value and "center_y" in value:
            x, y = value["center_x"], value["center_y"]
        elif "cx" in value and "cy" in value:
            x, y = value["cx"], value["cy"]
        elif all(k in value for k in ("x", "y", "width", "height")):
            x, y = float(value["x"]) + float(value["width"]) / 2, float(value["y"]) + float(value["height"]) / 2
        else:
            return None
        normalized = bool(value.get("normalized", False))
        if not normalized:
            dimensions = context.get("frame_size")
            if dimensions is None and "width" in context and "height" in context:
                dimensions = {"width": context["width"], "height": context["height"]}
            if isinstance(dimensions, dict):
                width, height = dimensions.get("width"), dimensions.get("height")
            elif isinstance(dimensions, (list, tuple)) and len(dimensions) >= 2:
                width, height = dimensions[0], dimensions[1]
            else:
                # Detector-provided normalized centers are preferred. Pixel
                # coordinates need explicit frame dimensions to compare safely.
                return None
            if not width or not height:
                return None
            x, y = float(x) / float(width), float(y) / float(height)
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
    else:
        return None
    try:
        result = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(result)) or not (0 <= result[0] <= 1 and 0 <= result[1] <= 1):
        return None
    return result


def _objects_compatible(current: dict[str, Any], stored: dict[str, Any]) -> dict[str, int] | None:
    """Compare nearby actor state relative to the player, ignoring detector IDs."""
    current_items = current.get("moving_objects", current.get("objects"))
    stored_items = stored.get("moving_objects", stored.get("objects"))
    if stored_items is None:
        # Older records without object evidence can only support a scene recall.
        return {"nearby_count": 0, "motion_compared": 0}
    if current_items is None:
        return None
    if not isinstance(current_items, list) or not isinstance(stored_items, list):
        return None
    current_player, stored_player = _player_location(current), _player_location(stored)
    if current_player is None or stored_player is None:
        return None

    def nearby(items: list[Any], context: dict[str, Any], player: tuple[float, float]):
        result = []
        for item in items:
            if not _is_relevant_threat(item):
                continue
            parsed = _object_state(item, context)
            if parsed is None:
                return None
            point, velocity = parsed
            relative = (point[0] - player[0], point[1] - player[1])
            # Objects behind the player, on the far side of the screen, or in
            # the HUD do not turn harmless scene changes into danger changes.
            if -0.10 <= relative[0] <= 0.42 and abs(relative[1]) <= 0.30:
                player_velocity = _velocity(context.get("player"), context)
                relative_velocity = (velocity[0] - player_velocity[0],
                                     velocity[1] - player_velocity[1]) if velocity and player_velocity else None
                result.append((relative, relative_velocity, _threat_kind(item)))
        return result

    a, b = nearby(current_items, current, current_player), nearby(stored_items, stored, stored_player)
    if a is None or b is None:
        return None
    # Any stored nearby hazard must have a compatible current counterpart;
    # extra/far objects are tolerated and IDs are deliberately ignored.
    if not b:
        return {"nearby_count": 0, "motion_compared": 0}
    unmatched = list(a)
    compared = 0
    for old_pos, old_velocity, old_kind in b:
        candidates = sorted((_distance(old_pos, new_pos), index, new_velocity, new_kind)
                            for index, (new_pos, new_velocity, new_kind) in enumerate(unmatched)
                            if new_kind == old_kind)
        if not candidates or candidates[0][0] > 0.10:
            return None
        distance, index, new_velocity, _ = candidates[0]
        if old_velocity is not None and new_velocity is not None:
            # Camera translation cancels in object velocity minus player
            # velocity. Opposite/meaningfully different motion is a new state.
            if any(a*b<0 and min(abs(a),abs(b))>.02 for a,b in zip(old_velocity,new_velocity)):
                return None
            if _distance(old_velocity, new_velocity) > 0.18:
                return None
            compared += 1
        elif (old_velocity is None) != (new_velocity is None):
            # Missing motion is unknown, so do not reuse a directional warning.
            return None
        unmatched.pop(index)
    return {"nearby_count": len(b), "motion_compared": compared}


def _is_relevant_threat(item: Any) -> bool:
    """Legacy coordinates have no semantics; require an explicit current threat label."""
    if not isinstance(item, dict):
        return False
    label = str(item.get("label", "")).lower()
    if (item.get("predicted") is True or item.get("is_prediction") is True
            or item.get("background") is True or item.get("missed", 0) not in (0, None)
            or label.startswith(("predicted", "unknown", "background", "static"))):
        return False
    return label.startswith("enemy_") or label == "independent_motion" or item.get("independent_motion") is True


def _threat_kind(item: dict[str, Any]) -> str:
    label = str(item.get("label", "")).lower()
    return "enemy" if label.startswith("enemy_") else "independent_motion"


def _object_state(item: Any, context: dict[str, Any]):
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        point = (float(item[0]), float(item[1]))
        velocity = None
    elif isinstance(item, dict):
        if "bbox" in item:
            box = item["bbox"]
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                return None
            width, height = _frame_dimensions(context)
            if not width or not height:
                return None
            point = ((float(box[0]) + float(box[2]) / 2) / width,
                     (float(box[1]) + float(box[3]) / 2) / height)
        else:
            point = (float(item.get("cx", item.get("x"))),
                     float(item.get("cy", item.get("y"))))
        velocity = _velocity(item, context)
    else:
        return None
    if not np.isfinite(point).all() or not all(0 <= value <= 1 for value in point):
        return None
    return point, velocity


def _frame_dimensions(context: dict[str, Any]):
    size = context.get("frame_size")
    if isinstance(size, dict):
        return size.get("width"), size.get("height")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        return size[0], size[1]
    return context.get("width"), context.get("height")


def _velocity(value: Any, context: dict[str, Any]) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("velocity")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        vx, vy = raw[:2]
    elif "vx" in value and "vy" in value:
        vx, vy = value["vx"], value["vy"]
    else:
        return None
    try:
        if int(value.get("age", value.get("age_frames", value.get("observations", 0)))) < 3:
            return None
        vx, vy = float(vx), float(vy)
        if value.get("velocity_pixels", 'bbox' in value and not value.get('normalized',False)):
            width, height = _frame_dimensions(context)
            vx, vy = vx / float(width), vy / float(height)
        if not np.isfinite((vx, vy)).all():
            return None
        return vx, vy
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _compare_scene(reference: np.ndarray, current: np.ndarray,
                   ref_player: tuple[float, float], current_player: tuple[float, float],
                   ref_context: dict[str, Any], current_context: dict[str, Any],
                   signature_difference: float, *, reference_features=None,
                   reference_cache: dict[str, Any] | None = None,
                   current_features: list[Any] | None = None) -> dict[str, Any] | None:
    """Use direct pixels first, then a conservative ORB/RANSAC camera alignment."""
    direct_error = _distance(ref_player, current_player)
    if signature_difference <= _SIGNATURE_LIMIT and direct_error <= 0.18:
        residual = _masked_residual(reference, current, None, ref_context, current_context)
        if residual is not None and residual <= _PIXEL_LIMIT:
            return {"method": "direct", "residual": residual,
                    "player_error": direct_error, "score": signature_difference + residual}

    if reference_features is None:
        reference_features = (reference_cache.get('features')
                              if reference_cache is not None else None)
        if reference_features is None:
            reference_features = _orb_features(reference)
            if reference_cache is not None:
                reference_cache['features'] = reference_features
    if current_features is not None:
        if current_features[0] is None:
            current_features[0] = _orb_features(current)
        current_features_value = current_features[0]
    else:
        current_features_value = _orb_features(current)
    ref_gray, kp_ref, des_ref = reference_features
    cur_gray, kp_cur, des_cur = current_features_value
    y0, y1 = int(reference.shape[0] * .14), int(reference.shape[0] * .96)
    mask = np.zeros_like(ref_gray)
    mask[y0:y1, :] = 255
    if des_ref is None or des_cur is None or len(kp_ref) < 12 or len(kp_cur) < 12:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_ref, des_cur, k=2)
    good = [first for first, second in pairs if first.distance < .72 * second.distance]
    # Repeated tiles yield many plausible local matches. Require broad spatial
    # coverage on both images plus a strong, non-degenerate RANSAC consensus.
    if len(good) < 12:
        return None
    unique = {}
    for match in good:
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    if len(good) < 12:
        return None
    src = np.float32([kp_ref[m.queryIdx].pt for m in good])
    dst = np.float32([kp_cur[m.trainIdx].pt for m in good])
    transform, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if transform is None or inliers is None:
        return None
    inlier_mask = inliers.ravel().astype(bool)
    inlier_src, inlier_dst = src[inlier_mask], dst[inlier_mask]
    count = len(inlier_src)
    if count < 10 or count / len(good) < .45:
        return None
    if _tile_coverage(inlier_src, reference.shape[1], reference.shape[0]) < 5 or \
       _tile_coverage(inlier_dst, current.shape[1], current.shape[0]) < 5:
        return None
    projected = cv2.perspectiveTransform(inlier_src.reshape(-1, 1, 2), transform).reshape(-1, 2)
    reprojection = np.linalg.norm(projected - inlier_dst, axis=1)
    if float(np.median(reprojection)) > 2.0:
        return None
    player_pixel = np.float32([[[ref_player[0] * reference.shape[1],
                                 ref_player[1] * reference.shape[0]]]])
    mapped = cv2.perspectiveTransform(player_pixel, transform)[0, 0]
    mapped_player = (float(mapped[0] / current.shape[1]), float(mapped[1] / current.shape[0]))
    player_error = _distance(mapped_player, current_player)
    if player_error > 0.09:
        return None
    warped = cv2.warpPerspective(reference, transform, (current.shape[1], current.shape[0]))
    valid = cv2.warpPerspective(np.ones(reference.shape[:2], np.uint8) * 255, transform,
                                (current.shape[1], current.shape[0]))
    overlap = np.count_nonzero((valid > 0) & (mask > 0)) / max(1, np.count_nonzero(mask))
    if overlap < .62:
        return None
    residual = _masked_residual(warped, current, valid, ref_context, current_context)
    if residual is None or residual > _PIXEL_LIMIT:
        return None
    return {"method": "orb_ransac", "residual": residual,
            "player_error": player_error, "score": residual + player_error + (1 - overlap) * .1}


def _orb_features(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    mask = np.zeros_like(gray)
    mask[int(height * .14):int(height * .96), :] = 255
    orb = cv2.ORB_create(nfeatures=1400, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=15, fastThreshold=12)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    return gray, keypoints, descriptors


def _tile_coverage(points: np.ndarray, width: int, height: int) -> int:
    return len({(min(3, int(x / width * 4)), min(3, int(y / height * 4)))
                for x, y in points})


def _masked_residual(reference: np.ndarray, current: np.ndarray, valid: np.ndarray | None,
                     ref_context: dict[str, Any], current_context: dict[str, Any]) -> float | None:
    if reference.shape != current.shape:
        return None
    height, width = current.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(height * .14):max(int(height * .96), 1), :] = 255
    if valid is not None:
        mask[valid == 0] = 0
    for context in (ref_context, current_context):
        player = context.get("player")
        if isinstance(player, dict) and isinstance(player.get("bbox"), (list, tuple)):
            _erase_box(mask, player["bbox"], width, height, normalized=player.get("normalized", False))
        for item in context.get("moving_objects", context.get("objects", [])) or []:
            if isinstance(item, dict) and isinstance(item.get("bbox"), (list, tuple)):
                _erase_box(mask, item["bbox"], width, height, normalized=item.get("normalized", False))
    if np.count_nonzero(mask) < height * width * .35:
        return None
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ref_gray, cur_gray)
    return float(np.mean(diff[mask > 0]) / 255.0)


def _erase_box(mask: np.ndarray, box: Any, width: int, height: int, normalized: bool = False):
    try:
        x, y, w, h = map(float, box[:4])
        if normalized:
            x, w, y, h = x * width, w * width, y * height, h * height
        left, top = max(0, int(x) - 3), max(0, int(y) - 3)
        right, bottom = min(width, int(x + w) + 3), min(height, int(y + h) + 3)
        mask[top:bottom, left:right] = 0
    except (TypeError, ValueError, IndexError):
        pass


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _confidence(review: dict[str, Any]) -> float:
    try:
        value = float(review.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) and 0 <= value <= 1 else 0.0


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _atomic_write(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
