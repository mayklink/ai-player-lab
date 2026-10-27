"""Small persistent, mask-aware appearance bank and conservative identity track.

This module is deliberately generic: callers provide a candidate box and an
explicitly verified positive example. Palette similarity alone never creates
an identity or a reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import uuid

import cv2
import numpy as np


_MAX_REFERENCES = 24
_MIN_MASK_PIXELS = 8
_POSITIVE_THRESHOLD = 0.67
_NEGATIVE_MARGIN = 0.04
_AUTO_ADD_THRESHOLD = 0.86
_MAX_PREDICTION_AGE = 0.2
_MAX_SPEED = 600.0
_MAX_LEARNED_REFERENCES = 16
_LEARNED_THROTTLE_S = 0.75
_NEAR_FULL_MASK_COVERAGE = 0.98


def _frame(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame precisa ser um numpy.ndarray BGR com 3 canais")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _box(box) -> tuple[int, int, int, int]:
    if box is None or len(box) < 4:
        raise ValueError("box precisa conter x, y, largura e altura")
    values = [float(v) for v in box[:4]]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("box deve conter coordenadas finitas")
    x, y, w, h = (int(round(v)) for v in values)
    if w < 2 or h < 2:
        raise ValueError("box precisa ter largura e altura maiores que 1")
    return x, y, w, h


def _crop(image: np.ndarray, box) -> np.ndarray:
    x, y, w, h = _box(box)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(image.shape[1], x+w), min(image.shape[0], y+h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box fica fora do frame")
    return image[y0:y1, x0:x1].copy()


def _foreground_mask(crop: np.ndarray, supplied: np.ndarray | None = None) -> np.ndarray:
    """Estimate foreground from border color, retaining the largest component.

    A supplied mask is preferred for difficult scenes. The fallback compares
    Lab color to a robust border estimate, then connects tiny pixel-art gaps.
    It does not assume a particular game palette.
    """
    h, w = crop.shape[:2]
    if supplied is not None:
        if not isinstance(supplied, np.ndarray) or supplied.ndim not in (2, 3):
            raise ValueError("mask deve ser um array grayscale ou BGR")
        m = supplied
        if m.ndim == 3:
            m = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (m > 0).astype(np.uint8)
    else:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
        # Median is robust to a sprite touching a small portion of the border.
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab - background, axis=2)
        # Small crops need a lower threshold to retain their few sprite pixels.
        mask = (distance > 13.0).astype(np.uint8)
        if w >= 8 and h >= 8:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count > 1:
            # Preserve disjoint sprite colors when close; discard isolated noise.
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = int(np.argmax(areas)) + 1
            keep = np.zeros_like(mask)
            for idx in range(1, count):
                if idx == largest or stats[idx, cv2.CC_STAT_AREA] >= max(2, int(areas.max() * 0.12)):
                    keep[labels == idx] = 1
            mask = keep
    if int(mask.sum()) < _MIN_MASK_PIXELS:
        return np.zeros((h, w), dtype=np.uint8)
    # Reject near-uniform crops; a rectangle of scenery is not a foreground mask.
    if float(np.std(crop.astype(np.float32), axis=(0, 1)).mean()) < 2.0:
        return np.zeros((h, w), dtype=np.uint8)
    return mask.astype(np.uint8)


@dataclass
class _Reference:
    image: np.ndarray
    mask: np.ndarray
    label: str
    source: str
    positive: bool
    key: str
    learned: bool = False


class AppearanceBank:
    """Persistent bank capped at 24 explicitly labelled sprite references.

    Any label other than ``player`` is treated as a negative example. With
    ``root=None`` the bank is memory-only; with a root, references are stored
    below ``<root>/appearance_bank`` as PNG, mask PNG, and JSON metadata.
    """

    def __init__(self, root: str | Path | None = None, *, online_learning: bool = True) -> None:
        self.root = Path(root) / "appearance_bank" if root is not None else None
        # Offline tools retain the historical learning behavior by default.
        # Interactive callers can freeze the persisted bank for a session.
        self.online_learning = bool(online_learning)
        self._references: list[_Reference] = []
        if self.root is not None:
            self._load()

    def _load(self) -> None:
        assert self.root is not None
        manifest = self.root / "references.json"
        if not manifest.exists():
            return
        try:
            rows = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        for row in rows:
            try:
                if not isinstance(row, dict):
                    continue
                key = str(row["key"])
                if re.fullmatch(r"[0-9a-f]{32}", key) is None:
                    continue
                image = cv2.imread(str(self.root / f"{key}.png"), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(self.root / f"{key}.mask.png"), cv2.IMREAD_GRAYSCALE)
                if image is None or mask is None or image.shape[:2] != mask.shape:
                    continue
                self._references.append(_Reference(image, (mask > 0).astype(np.uint8),
                    str(row["label"]), str(row["source"]), bool(row["positive"]), key,
                    bool(row.get("learned", False))))
            except (KeyError, OSError, ValueError):
                continue
        self._references = self._references[-_MAX_REFERENCES:]

    def _save(self) -> None:
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        rows = []
        for ref in self._references:
            image_path = self.root / f"{ref.key}.png"
            mask_path = self.root / f"{ref.key}.mask.png"
            if not image_path.exists():
                cv2.imwrite(str(image_path), ref.image)
            if not mask_path.exists():
                cv2.imwrite(str(mask_path), ref.mask * 255)
            rows.append({"key": ref.key, "label": ref.label, "source": ref.source,
                         "positive": ref.positive, "learned": ref.learned})
        live = {ref.key for ref in self._references}
        for path in self.root.glob("*.png"):
            match = re.fullmatch(r"([0-9a-f]{32})(?:\.mask)?\.png", path.name)
            if match and match.group(1) not in live:
                path.unlink()
        manifest = self.root / "references.json"
        temporary = self.root / "references.json.tmp"
        temporary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        temporary.replace(manifest)

    @property
    def reference_count(self) -> int:
        return len(self._references)

    @staticmethod
    def _quarantined(ref: _Reference) -> bool:
        """Quarantine near-full learned masks; dense non-full poses are unlabelled."""
        return bool(ref.learned and ref.mask.size and
                    float(np.count_nonzero(ref.mask)) / ref.mask.size >= _NEAR_FULL_MASK_COVERAGE)

    def _usable_references(self):
        return [ref for ref in self._references if not self._quarantined(ref)]

    def add(self, frame: np.ndarray, box, label: str = "player", mask: np.ndarray | None = None,
            source: str = "verified_fixture") -> bool:
        """Add an explicit positive (``player``) or negative (other label).

        Returns False for an unusable/flat crop. ``source`` is recorded for
        auditability; it does not grant additional trust.
        """
        image = _frame(frame)
        crop = _crop(image, box)
        foreground = _foreground_mask(crop, mask)
        if int(foreground.sum()) < _MIN_MASK_PIXELS:
            return False
        clean_label = str(label).strip().lower()
        if not clean_label:
            raise ValueError("label não pode ser vazio")
        positive = clean_label == "player"
        key = uuid.uuid4().hex
        self._references.append(_Reference(crop, foreground, clean_label, str(source)[:120], positive, key))
        if len(self._references) > _MAX_REFERENCES:
            evicted = next((i for i, ref in enumerate(self._references) if ref.learned), 0)
            self._references.pop(evicted)
        # A frozen interactive session may seed its in-memory bank, but must
        # not persist seeds or otherwise mutate the saved manifest.
        if self.online_learning:
            self._save()
        return True

    def _add_learned(self, frame: np.ndarray, box, mask: np.ndarray) -> bool:
        """Append a novel high quality pose without evicting explicit seeds."""
        image = _frame(frame)
        crop = _crop(image, box)
        foreground = _foreground_mask(crop, mask)
        if int(foreground.sum()) < _MIN_MASK_PIXELS:
            return False
        coverage = float(np.count_nonzero(foreground)) / foreground.size
        # Full-crop masks are proven contaminated; dense non-full poses are not.
        if coverage >= _NEAR_FULL_MASK_COVERAGE:
            return False
        usable = self._usable_references()
        positive = [self._similarity(crop, ref) for ref in usable if ref.positive]
        negative = [self._similarity(crop, ref) for ref in usable if not ref.positive]
        best_positive, best_negative = max(positive, default=0.0), max(negative, default=0.0)
        if not (_AUTO_ADD_THRESHOLD <= best_positive <= 0.98 and
                best_positive - best_negative >= 0.12 and best_positive < 0.97):
            return False
        if sum(ref.learned for ref in self._references) >= _MAX_LEARNED_REFERENCES:
            oldest = next((i for i, ref in enumerate(self._references) if ref.learned), None)
            if oldest is not None:
                self._references.pop(oldest)
        key = uuid.uuid4().hex
        self._references.append(_Reference(crop, foreground, "player", "confirmed_tracker", True, key, True))
        if len(self._references) > _MAX_REFERENCES:
            oldest = next((i for i, ref in enumerate(self._references) if ref.learned), None)
            if oldest is not None:
                self._references.pop(oldest)
        self._save()
        return True

    @staticmethod
    def _similarity(image: np.ndarray, ref: _Reference,partial=False) -> float:
        candidate = cv2.resize(image, (ref.image.shape[1], ref.image.shape[0]), interpolation=cv2.INTER_AREA)
        mask = ref.mask.astype(bool)
        if int(mask.sum()) < _MIN_MASK_PIXELS:
            return 0.0
        a = candidate.astype(np.float32)[mask]
        b = ref.image.astype(np.float32)[mask]
        # Color and luminance agreement; masked background pixels never count.
        visible=slice(None)
        if partial:
            # Permit a bounded occluded fraction only on an existing track.
            # Acquisition always compares the full masked appearance.
            errors=np.mean(np.abs(a-b),axis=1)
            visible=np.argsort(errors)[:max(_MIN_MASK_PIXELS,int(len(errors)*.75))]
        mae = float(np.mean(np.abs(a - b)[visible])) / 255.0
        gray_a = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)[mask]
        gray_b = cv2.cvtColor(ref.image, cv2.COLOR_BGR2GRAY).astype(np.float32)[mask]
        gray_mae = float(np.mean(np.abs(gray_a - gray_b)[visible])) / 255.0
        return float(np.clip(1.0 - (0.8 * mae + 0.2 * gray_mae), 0.0, 1.0))

    def score(self, frame: np.ndarray, box,*,partial=False) -> dict:
        """Compare the candidate crop only at each reference's foreground mask."""
        image = _frame(frame)
        try:
            crop = _crop(image, box)
        except ValueError:
            return {"positive_score": 0.0, "negative_score": 0.0, "accepted": False}
        if float(np.std(crop.astype(np.float32), axis=(0, 1)).mean()) < 2.0:
            return {"positive_score": 0.0, "negative_score": 0.0, "accepted": False}
        usable = self._usable_references()
        positive = [(self._similarity(crop, r,partial), r) for r in usable if r.positive]
        negative = [(self._similarity(crop, r,partial), r) for r in usable if not r.positive]
        pos, pos_ref = max(positive, key=lambda item: item[0], default=(0.0, None))
        neg, neg_ref = max(negative, key=lambda item: item[0], default=(0.0, None))
        accepted = bool(positive and pos >= (.82 if partial else _POSITIVE_THRESHOLD) and pos - neg >= (.06 if partial else _NEGATIVE_MARGIN))
        return {"positive_score": float(pos), "negative_score": float(neg), "accepted": accepted,
                "partial_match":partial,
                "positive_ref": ({"key": pos_ref.key, "source": pos_ref.source, "learned": pos_ref.learned} if pos_ref else None),
                "negative_ref": ({"key": neg_ref.key, "source": neg_ref.source, "learned": neg_ref.learned} if neg_ref else None)}


class IdentityTracker:
    """Conservative current-frame identity tracking backed by an appearance bank."""

    def __init__(self, bank: AppearanceBank,validator=None) -> None:
        self.bank = bank
        self.validator=validator
        self._box: tuple[float, float, float, float] | None = None
        self._seen_at: float | None = None
        self._velocity = (0.0, 0.0)
        self._streak = 0
        self._camera_offset_value = (0.0, 0.0)
        self._last_auto_add_at: float | None = None

    def reset(self) -> None:
        self._box = None
        self._seen_at = None
        self._velocity = (0.0, 0.0)
        self._streak = 0
        self._camera_offset_value = (0.0, 0.0)
        self._last_auto_add_at = None

    @staticmethod
    def _center(box):
        return (box[0] + box[2] / 2, box[1] + box[3] / 2)

    @staticmethod
    def _camera_offset(camera):
        if isinstance(camera, dict):
            try:
                dx, dy = float(camera.get("dx", 0)), float(camera.get("dy", 0))
                if math.isfinite(dx) and math.isfinite(dy):
                    return dx, dy
            except (TypeError, ValueError):
                pass
        return 0.0, 0.0

    def _predict(self, now: float, camera_offset):
        if self._box is None or self._seen_at is None:
            return None, None
        age = float(now) - self._seen_at
        if not 0 <= age <= _MAX_PREDICTION_AGE:
            return None, age
        dx, dy = self._velocity
        old_x, old_y = self._camera_offset_value
        shift_x, shift_y = camera_offset[0] - old_x, camera_offset[1] - old_y
        # Prediction is diagnostic only and is never returned as confirmed box.
        pred = (self._box[0] + dx * age + shift_x, self._box[1] + dy * age + shift_y,
                self._box[2], self._box[3])
        return [int(round(v)) for v in pred], age

    def _search(self, frame: np.ndarray, now: float, camera_offset,partial=False):
        if self._box is None or self._seen_at is None:
            return None
        age = float(now) - self._seen_at
        if age < 0 or age > _MAX_PREDICTION_AGE:
            return None
        x, y, w, h = (int(round(v)) for v in self._box)
        old_x, old_y = self._camera_offset_value
        vx, vy = self._velocity
        x = int(round(x + vx * age + camera_offset[0] - old_x))
        y = int(round(y + vy * age + camera_offset[1] - old_y))
        radius = max(2, min(40, int(_MAX_SPEED * max(age, 0.03)+8)))
        # Include both the last sighting and its prediction so a direction
        # reversal does not push the real sprite outside the local search.
        old_box_x,old_box_y=int(self._box[0]),int(self._box[1])
        left, top = max(0,min(x,old_box_x)-radius),max(0,min(y,old_box_y)-radius)
        right,bottom=min(frame.shape[1],max(x,old_box_x)+w+radius),min(frame.shape[0],max(y,old_box_y)+h+radius)
        roi = frame[top:bottom, left:right]
        if roi.shape[0] < h or roi.shape[1] < w:
            return None
        # Keep explicit seeds in the local search regardless of how many
        # learned poses were appended later. The bank cap bounds the work.
        refs = [r for r in self.bank._usable_references() if r.positive and r.image.shape[:2] == (h, w)]
        proposals = []
        for ref in refs:
            if int(ref.mask.sum()) < _MIN_MASK_PIXELS:
                continue
            mask = np.repeat((ref.mask * 255)[:, :, None], 3, axis=2).astype(np.uint8)
            if partial:mask[h//2:]=0
            try:
                scores = cv2.matchTemplate(roi, ref.image, cv2.TM_SQDIFF_NORMED, mask=mask)
            except cv2.error:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_ref = cv2.cvtColor(ref.image, cv2.COLOR_BGR2GRAY)
                scores = cv2.matchTemplate(gray_roi, gray_ref, cv2.TM_SQDIFF_NORMED,
                                           mask=ref.mask * 255)
            scores = np.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=1.0)
            local = scores.copy()
            for _ in range(2):
                value, _, location, _ = cv2.minMaxLoc(local)
                quality = 1.0 - math.sqrt(max(0.0, min(1.0, float(value))))
                proposals.append(([left + location[0], top + location[1], w, h], quality))
                px, py = location
                local[max(0, py-2):py+3, max(0, px-2):px+3] = 1.0
        candidates, seen = [], set()
        for candidate, _ in sorted(proposals, key=lambda item: item[1], reverse=True):
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            if self.validator is not None and not self.validator(frame,candidate):continue
            result = self.bank.score(frame, candidate,partial=partial)
            if result["accepted"]:
                candidates.append((candidate, result))
        if candidates:return max(candidates,key=lambda item:item[1]['positive_score'])
        return self._search(frame,now,camera_offset,partial=True) if not partial else None

    def update(self, frame: np.ndarray, now: float, candidate_box, camera=None) -> dict:
        image = _frame(frame)
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now precisa ser finito")
        camera_offset = self._camera_offset(camera)

        # Once the short prediction window expires, a fresh appearance match
        # may establish a new current sighting without inheriting stale motion.
        if self._seen_at is not None and not (0 <= now - self._seen_at <= _MAX_PREDICTION_AGE):
            self._box = None
            self._seen_at = None
            self._velocity = (0.0, 0.0)
            self._streak = 0
            self._camera_offset_value = camera_offset

        selected = None
        result = {"positive_score": 0.0, "negative_score": 0.0, "accepted": False}
        source = "candidate"
        if candidate_box is not None:
            try:
                candidate = list(_box(candidate_box))
                result = self.bank.score(image, candidate)
                if result["accepted"] and (self.validator is None or self.validator(image,candidate)):
                    selected = candidate
            except ValueError:
                result = {"positive_score": 0.0, "negative_score": 0.0, "accepted": False}
        if selected is None and self._box is not None:
            match = self._search(image, now, camera_offset)
            if match is not None:
                selected, result = match
                source = "masked_partial_search" if result.get('partial_match') else "masked_local_search"

        predicted, age = self._predict(now, camera_offset)
        if selected is None:
            self._streak = 0
            status = "rejected" if candidate_box is not None else "missing"
            return {"box": None, "confirmed_box": None, "predicted_box": predicted,
                    "status": status, "score": result, "mask": None, "source": source,
                    "prediction_age": age, "reference": result.get("positive_ref")}

        current = tuple(float(v) for v in selected)
        continuous = True
        if self._box is not None and self._seen_at is not None:
            dt = now - self._seen_at
            old_cx, old_cy = self._center(self._box)
            new_cx, new_cy = self._center(current)
            old_x, old_y = self._camera_offset_value
            camera_dx, camera_dy = camera_offset[0] - old_x, camera_offset[1] - old_y
            distance = math.hypot(new_cx-old_cx-camera_dx, new_cy-old_cy-camera_dy)
            continuous = 0 < dt <= _MAX_PREDICTION_AGE and distance <= _MAX_SPEED * dt + 8
        if self._box is not None and not continuous:
            # An appearance match alone cannot teleport an established identity.
            self._streak = 1
            return {"box": None, "confirmed_box": None, "predicted_box": predicted,
                    "status": "identity_jump_rejected", "score": result, "mask": None, "source": source,
                    "prediction_age": age, "reference": result.get("positive_ref")}

        if self._box is not None and self._seen_at is not None:
            dt = now - self._seen_at
            ox, oy = self._center(self._box)
            nx, ny = self._center(current)
            old_x, old_y = self._camera_offset_value
            instant = ((nx-ox-(camera_offset[0]-old_x))/dt,
                       (ny-oy-(camera_offset[1]-old_y))/dt)
            self._velocity = (float(np.clip(instant[0], -_MAX_SPEED, _MAX_SPEED)),
                              float(np.clip(instant[1], -_MAX_SPEED, _MAX_SPEED)))
            self._streak += 1
        else:
            self._streak = 1
        self._box, self._seen_at = current, now
        self._camera_offset_value = camera_offset
        mask = _foreground_mask(_crop(image, current))
        if (self.bank.online_learning and not result.get('partial_match') and self._streak >= 3 and _AUTO_ADD_THRESHOLD <= result["positive_score"] <= 0.98 and
                result["positive_score"] - result["negative_score"] >= 0.12 and
                (self._last_auto_add_at is None or now - self._last_auto_add_at >= _LEARNED_THROTTLE_S)):
            if self.bank._add_learned(image, current, mask):
                self._last_auto_add_at = now
        return {"box": [int(v) for v in selected], "confirmed_box": [int(v) for v in selected],
                "predicted_box": None, "status": "confirmed", "score": result,
                "mask": mask, "source": source,
                "reference": result.get("positive_ref")}
