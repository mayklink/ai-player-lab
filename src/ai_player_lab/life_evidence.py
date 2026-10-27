"""Conservative Super Mario World HUD life-counter evidence.

The reader only recognizes glyphs explicitly supplied as labelled examples.
It looks in fixed, scene-specific HUD boxes and never treats a map, darkness,
or a scene transition as a life loss by itself.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np


# (x, y, width, height) on a normalized 256x224 frame. The gameplay box starts
# below the "MARIO" label; the map uses its separate x4 HUD location.
LIFE_DIGIT_BOXES: dict[str, tuple[int, int, int, int]] = {
    "gameplay": (40, 24, 9, 10),
    "map": (64, 30, 9, 10),
}
_NORMAL_SIZE = (256, 224)
_DARK_THRESHOLD = 24
_MIN_SCORE = 0.88
_MIN_MARGIN = 0.10
_SEARCH_RADIUS = 2
_SAMPLE_NAME = re.compile(r"^(gameplay|map)__([0-9])(?:__.*)?\.png$", re.I)


def _normalize(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame precisa ser um numpy.ndarray BGR com 3 canais")
    h, w = frame.shape[:2]
    if (w, h) not in {_NORMAL_SIZE, (512, 448)}:
        raise ValueError("frame deve medir 256x224 ou 512x448")
    frame = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
    if (w, h) == (512, 448):
        frame = cv2.resize(frame, _NORMAL_SIZE, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)


def _glyph_mask(image: np.ndarray, scene: str) -> np.ndarray:
    """Return a compact dark-ink mask; color and HUD fill are not relied on."""
    if not isinstance(image, np.ndarray):
        raise ValueError("exemplar precisa ser um numpy.ndarray")
    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.astype(np.uint8)
    else:
        raise ValueError("exemplar deve ser uma imagem grayscale ou BGR")
    h, w = gray.shape
    if (w, h) in {_NORMAL_SIZE, (512, 448)}:
        full = _normalize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        x, y, bw, bh = LIFE_DIGIT_BOXES[scene]
        gray = cv2.cvtColor(full[y:y + bh, x:x + bw], cv2.COLOR_BGR2GRAY)
    else:
        _, _, box_w, box_h = LIFE_DIGIT_BOXES[scene]
        if (w, h) != (box_w, box_h):
            gray = cv2.resize(gray, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
    # SNES HUD digits have a dark outline. Thresholding the outline keeps the
    # representation stable across the blue gameplay and beige map backdrops.
    mask = (gray < _DARK_THRESHOLD).astype(np.uint8)
    if int(mask.sum()) < 5:
        raise ValueError("exemplar não contém um contorno de dígito legível")
    return mask


class LifeCounter:
    """Read a one-digit life counter from explicitly labelled scene examples.

    ``samples`` maps a scene name to ``[(digit, image), ...]``. Images may be
    the digit crop or a normalized/full-size BGR frame. When omitted, samples
    are loaded from ``profiles/mario/life-glyphs`` relative to the package
    root, using names such as ``gameplay__5.png`` and ``map__4.png``.
    """

    def __init__(
        self,
        samples: Mapping[str, Sequence[tuple[int, np.ndarray]]] | None = None,
        profile_dir: str | Path | None = None,
    ) -> None:
        self._templates: dict[str, list[tuple[int, np.ndarray]]] = {"gameplay": [], "map": []}
        if samples is not None:
            for scene, examples in samples.items():
                normalized_scene = str(scene).lower()
                if normalized_scene not in self._templates:
                    continue
                for digit, image in examples:
                    if isinstance(digit, (int, np.integer)) and 0 <= int(digit) <= 9:
                        self._templates[normalized_scene].append((int(digit), _glyph_mask(image, normalized_scene)))
        else:
            root = Path(profile_dir) if profile_dir is not None else Path(__file__).parent / "profiles" / "mario" / "life-glyphs"
            if root.is_dir():
                for path in sorted(root.glob("*.png")):
                    match = _SAMPLE_NAME.fullmatch(path.name)
                    if not match:
                        continue
                    sample = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if sample is None:
                        continue
                    self._templates[match.group(1).lower()].append(
                        (int(match.group(2)), _glyph_mask(sample, match.group(1).lower()))
                    )

    def read(self, frame: np.ndarray, scene: str) -> dict[str, int | float | None]:
        """Return value, best template score, and gap to the next digit score."""
        scene = str(scene).lower()
        if scene not in LIFE_DIGIT_BOXES:
            return {"value": None, "score": 0.0, "margin": 0.0}
        templates = self._templates.get(scene, [])
        if not templates:
            return {"value": None, "score": 0.0, "margin": 0.0}
        image = _normalize(frame)
        x, y, w, h = LIFE_DIGIT_BOXES[scene]
        # HUD glyph placement varies by a pixel or two between gameplay/map
        # captures. Search only this small, declared neighborhood; each class
        # still needs the original strict score and inter-class margin.
        by_digit: dict[int, tuple[float, tuple[int, int]]] = {}
        for dy in range(-_SEARCH_RADIUS, _SEARCH_RADIUS + 1):
            for dx in range(-_SEARCH_RADIUS, _SEARCH_RADIUS + 1):
                left, top = x + dx, y + dy
                if left < 0 or top < 0 or left + w > image.shape[1] or top + h > image.shape[0]:
                    continue
                observed = (cv2.cvtColor(image[top:top + h, left:left + w], cv2.COLOR_BGR2GRAY)
                            < _DARK_THRESHOLD).astype(np.uint8)
                for digit, template in templates:
                    # Dice overlap compares the dark outline rather than raw colors.
                    intersection = int(np.count_nonzero(observed & template))
                    denominator = int(observed.sum() + template.sum())
                    score = (2.0 * intersection / denominator) if denominator else 0.0
                    if score > by_digit.get(digit, (-1.0, (0, 0)))[0]:
                        by_digit[digit] = (score, (dx, dy))
        if not by_digit:
            return {"value": None, "score": 0.0, "margin": 0.0, "offset": (0, 0)}
        # Multiple labelled views and nearby crops may support a digit, but a
        # runner-up class is scored over the same search window.
        ordered = sorted(((score, digit, offset) for digit, (score, offset) in by_digit.items()),
                         key=lambda item: (-item[0], item[1]))
        best_score, best_digit, best_offset = ordered[0]
        margin = best_score - (ordered[1][0] if len(ordered) > 1 else 0.0)
        accepted = best_score >= _MIN_SCORE and margin >= _MIN_MARGIN
        return {
            "value": best_digit if accepted else None,
            "score": round(float(best_score), 4),
            "margin": round(float(margin), 4),
            "offset": best_offset,
        }


def _frame_record(item: Any, index: int) -> tuple[np.ndarray | None, str, Any]:
    if isinstance(item, np.ndarray):
        return item, "", index
    if not isinstance(item, Mapping):
        return None, "", index
    frame = next((item.get(key) for key in ("frame", "image", "pixels")
                  if isinstance(item.get(key), np.ndarray)), None)
    state = item.get("state") if isinstance(item.get("state"), Mapping) else item
    scene = str(state.get("scene") or state.get("scene_name") or "").lower()
    frame_id = item.get("frame_id", item.get("frame_hash",item.get("id", index)))
    return frame, scene, frame_id


def life_loss_evidence(
    frames: Iterable[Any],
    counter: LifeCounter | None = None,
) -> dict[str, Any]:
    """Confirm a one-life decrement from stable gameplay then map readings.

    At least two valid, agreeing observations are required on each side of a
    gameplay-to-map boundary. This reports counter evidence only; it does not
    infer a death from a map screen or a model's scene label.
    """
    counter = counter or LifeCounter()
    observations: list[dict[str, Any]] = []
    for index, item in enumerate(frames):
        frame, scene, frame_id = _frame_record(item, index)
        if frame is None or scene not in LIFE_DIGIT_BOXES:
            continue
        try:
            result = counter.read(frame, scene)
        except (TypeError, ValueError):
            continue
        observations.append({"scene": scene, "frame_id": frame_id, **result})

    # The transition is strictly gameplay -> map. Any other locale or missing
    # half yields an unknown result rather than a guessed event.
    first_map = next((i for i, obs in enumerate(observations) if obs["scene"] == "map"), None)
    if first_map is None:
        return _unknown_evidence("missing_map_observations", observations)
    before = [obs for obs in observations[:first_map] if obs["scene"] == "gameplay" and obs["value"] is not None]
    after = [obs for obs in observations[first_map:] if obs["scene"] == "map" and obs["value"] is not None]
    if len(before) < 2 or len(after) < 2:
        return _unknown_evidence("insufficient_stable_readings", observations)
    before_pair, after_pair = before[-2:], after[:2]
    if before_pair[0]["value"] != before_pair[1]["value"] or after_pair[0]["value"] != after_pair[1]["value"]:
        return _unknown_evidence("unstable_readings", observations)
    before_value, after_value = int(before_pair[-1]["value"]), int(after_pair[-1]["value"])
    decrement = before_value - after_value
    confirmed = decrement == 1
    return {
        "confirmed": confirmed,
        "status": "life_lost" if confirmed else "no_life_loss",
        "before_value": before_value,
        "after_value": after_value,
        "decrement": decrement,
        "evidence_frame_ids": [obs["frame_id"] for obs in (*before_pair, *after_pair)],
        "reason": "stable_counter_decrement" if confirmed else "counter_did_not_decrease_by_one",
        "observations": observations,
    }


def _unknown_evidence(reason: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "confirmed": False,
        "status": "unknown",
        "before_value": None,
        "after_value": None,
        "decrement": None,
        "evidence_frame_ids": [],
        "reason": reason,
        "observations": observations,
    }
