"""Local pixel detector for the manually verified Rex poses.

The references are small, masked sprite crops from the repository's captured
fixtures. This detector makes no claim about other moving objects or enemy
semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class _Template:
    image: np.ndarray
    mask: np.ndarray
    pose: str = "upright"


class RexDetector:
    """Find current Rex pixels in a BGR frame, independent of motion history."""

    # The caller supplies the game's normalized 256x224 image; sprite scale is
    # fixed. Direction can change, so search the reference and its horizontal flip.
    _SCALES = (1.0,)
    _MIN_CONFIDENCE = 0.72
    _NMS_IOU = 0.25
    _BLUE_HUE = (110, 135)
    _MAX_COMPONENTS = 12
    _COMPONENT_AREA = (8, 600)
    _MIN_MASKED_COLOR_MATCH = 0.78

    def __init__(self, reference_dir: str | Path | None = None) -> None:
        root = (Path(reference_dir) if reference_dir is not None else
                Path(__file__).resolve().parent / "profiles" / "mario" / "enemy-seeds")
        self._templates: list[_Template] = []
        for stem, pose in (("rex-upright", "upright"),
                           ("rex-flattened", "flattened"),
                           ("rex-flattened-alt", "flattened")):
            image_path = root / f"{stem}.png"
            mask_path = root / f"{stem}.mask.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape[:2] != mask.shape:
                raise ValueError(f"Referência Rex inválida ou ausente: {stem}")
            binary = np.where(mask > 0, 255, 0).astype(np.uint8)
            if int(np.count_nonzero(binary)) < 12:
                raise ValueError(f"Máscara Rex insuficiente: {stem}")
            self._templates.append(_Template(image, binary, pose))

    @staticmethod
    def _nms_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        x, y, w, h = a
        bx, by, bw, bh = b
        left, top = max(x, bx), max(y, by)
        right, bottom = min(x + w, bx + bw), min(y + h, by + bh)
        overlap = max(0, right - left) * max(0, bottom - top)
        return overlap / max(1, w * h + bw * bh - overlap)

    @staticmethod
    def _rex_palette_visible(frame: np.ndarray, bbox: tuple[int, int, int, int],
                             reference: np.ndarray, mask: np.ndarray,
                             player_box=None) -> bool:
        x, y, width, height = bbox
        crop = frame[y:y + height, x:x + width]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        visible = mask > 0
        if player_box is not None:
            px, py, pw, ph = map(int, player_box[:4])
            left, top = max(x, px), max(y, py)
            right, bottom = min(x + width, px + pw), min(y + height, py + ph)
            if right > left and bottom > top:
                visible[top - y:bottom - y, left - x:right - x] = False
        if not np.any(visible):
            return False
        blue = visible & (hsv[:, :, 0] >= 110) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 60)
        orange = visible & (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] > 90)
        color_match = 1.0 - float(np.mean(np.abs(
            crop.astype(np.float32)[visible] - reference.astype(np.float32)[visible]))) / 255.0
        # Both distinctive colors must remain visible; a fully occluded actor
        # is never returned as a current detection. Masked color agreement
        # separates Rex's actual palette from map scenery with similar shapes.
        return (int(np.count_nonzero(blue)) >= 8 and int(np.count_nonzero(orange)) >= 5 and
                color_match >= RexDetector._MIN_MASKED_COLOR_MATCH)

    @staticmethod
    def _flattened_rex_visible(frame: np.ndarray, bbox: tuple[int, int, int, int],
                              reference: np.ndarray, mask: np.ndarray,
                              player_box=None) -> bool:
        """Verify the orange flattened pose without requiring hidden blue pixels."""
        x, y, width, height = bbox
        visible = mask > 0
        visible = visible.copy()
        if player_box is not None:
            px, py, pw, ph = map(int, player_box[:4])
            left, top = max(x, px), max(y, py)
            right, bottom = min(x + width, px + pw), min(y + height, py + ph)
            if right > left and bottom > top:
                visible[top - y:bottom - y, left - x:right - x] = False
        if int(np.count_nonzero(visible)) < 24:
            return False
        crop = frame[y:y + height, x:x + width]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        orange = visible & (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] > 90)
        if int(np.count_nonzero(orange)) < max(20, int(np.count_nonzero(visible) * .30)):
            return False
        color_match = 1.0 - float(np.mean(np.abs(
            crop.astype(np.float32)[visible] - reference.astype(np.float32)[visible]))) / 255.0
        return color_match >= RexDetector._MIN_MASKED_COLOR_MATCH

    @classmethod
    def _blue_rois(cls, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Use distinctive Rex blue pixels to bound expensive template scans."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue = (((hsv[:, :, 0] >= cls._BLUE_HUE[0]) &
                 (hsv[:, :, 0] <= cls._BLUE_HUE[1]) &
                 (hsv[:, :, 1] > 60)).astype(np.uint8) * 255)
        blue[:35] = 0
        count, _, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
        height, width = frame.shape[:2]
        components = []
        min_area, max_area = cls._COMPONENT_AREA
        for x, y, component_width, component_height, area in stats[1:count]:
            x, y, component_width, component_height, area = map(
                int, (x, y, component_width, component_height, area))
            if not min_area <= area <= max_area:
                continue
            if component_width > 40 or component_height > 40:
                continue
            left = max(0, x - 18)
            top = max(35, y - 15)
            right = min(width, x + component_width + 18)
            bottom = min(height, y + component_height + 30)
            components.append((area, (left, top, right, bottom)))
        # The bounded workload is deterministic and favors the strongest
        # compact blue components when scenery contains many small details.
        components.sort(key=lambda item: item[0], reverse=True)
        return [roi for _, roi in components[:cls._MAX_COMPONENTS]]

    def _candidates(self, frame: np.ndarray, player_box=None) -> list[dict]:
        candidates: list[dict] = []
        rois = self._blue_rois(frame)
        if not rois:
            return candidates
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for reference in self._templates:
            for scale in self._SCALES:
                width = max(5, int(round(reference.image.shape[1] * scale)))
                height = max(5, int(round(reference.image.shape[0] * scale)))
                for flip in (False, True):
                    image = cv2.resize(reference.image, (width, height),
                                       interpolation=cv2.INTER_NEAREST)
                    mask = cv2.resize(reference.mask, (width, height),
                                      interpolation=cv2.INTER_NEAREST)
                    if flip:
                        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
                    if height > frame.shape[0] or width > frame.shape[1]:
                        continue
                    template_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    for left, top, right, bottom in rois:
                        search = gray[top:bottom, left:right]
                        if search.shape[0] < height or search.shape[1] < width:
                            continue
                        scores = cv2.matchTemplate(search, template_gray,
                                                   cv2.TM_SQDIFF_NORMED, mask=mask)
                        scores = np.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=1.0)
                        local = scores.copy()
                        # Repeated local minima permit multiple actors in one ROI.
                        for _ in range(3):
                            min_score, _, location, _ = cv2.minMaxLoc(local)
                            confidence = float(1.0 - max(0.0, min(1.0, min_score)))
                            bbox = (left + int(location[0]), top + int(location[1]), width, height)
                            visible = (
                                self._rex_palette_visible(frame, bbox, image, mask, player_box)
                                if reference.pose == "upright"
                                else self._flattened_rex_visible(frame, bbox, image, mask, player_box)
                            )
                            min_confidence = self._MIN_CONFIDENCE if reference.pose == "upright" else .93
                            if confidence >= min_confidence and visible:
                                candidates.append({
                                    "bbox": bbox,
                                    "confidence": confidence,
                                    "label": "enemy_rex",
                                })
                            x, y = location
                            local[max(0, y - height // 2):min(local.shape[0], y + height // 2 + 1),
                                  max(0, x - width // 2):min(local.shape[1], x + width // 2 + 1)] = 1.0
        return candidates

    def detect(self, frame: np.ndarray, player_box=None) -> list[dict]:
        """Return Rex detections as ``{bbox, confidence, label}`` dictionaries.

        ``frame`` must be a BGR image. Pixels inside a supplied player box do
        not count toward the Rex-color check, allowing visible contact while
        omitting candidates with too little independently visible sprite.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame deve ser uma imagem BGR HxWx3")
        if frame.dtype != np.uint8 or not frame.size:
            raise ValueError("frame deve conter pixels uint8")
        if player_box is not None and len(player_box) < 4:
            raise ValueError("player_box deve conter x, y, largura e altura")

        ranked = sorted(self._candidates(frame, player_box), key=lambda item: item["confidence"],
                        reverse=True)
        kept: list[dict] = []
        for candidate in ranked:
            box = candidate["bbox"]
            # The HUD contains small red/white pixel art; do not report it.
            if box[1] < 35:
                continue
            if any(self._nms_iou(box, other["bbox"]) >= self._NMS_IOU for other in kept):
                continue
            kept.append(candidate)
        return kept
