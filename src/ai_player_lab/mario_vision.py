"""Conservative, local-only visual helpers for the Mario World canvas.

Scene labels are learned only from explicitly labelled templates supplied by a
human. The generic rules intentionally recognize only an almost-black loading
frame; they do not infer gameplay from a blue background or from motion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import json

import cv2
import numpy as np


SCENES = {
    "title", "file_menu", "player_menu", "intro", "map", "hub",
    "gameplay", "death", "loading", "unknown",
}
_TEMPLATE_SCENE = re.compile(r"^(title|file_menu|player_menu|intro|map|hub|gameplay|death|loading)__.+\.png$", re.I)
_SIZE = (256, 224)
_PLAYER_MATCH_GRACE_S = 0.2
_PLAYER_MATCH_RADIUS = 32.0
_PLAYER_MATCH_MIN_SCORE = 0.72


def _cap_shape(red):
    rows=np.count_nonzero(red[:10],axis=1)
    return bool(rows.size and ((rows.max()>=4 and rows.sum()>=12)
        or (rows[:4].max()>=3 and rows[:4].sum()>=5)))


def _normalize(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame precisa ser um numpy.ndarray BGR com 3 canais")
    h, w = frame.shape[:2]
    if (w, h) not in {_SIZE, (512, 448)}:
        raise ValueError("frame deve medir 256x224 ou 512x448")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if (w, h) == (512, 448):
        frame = cv2.resize(frame, _SIZE, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)


@dataclass
class _Template:
    scene: str
    name: str
    image: np.ndarray
    mask: np.ndarray


def _load_template(path: Path, scene: str | None = None) -> _Template | None:
    if path.name.endswith('.mask.png'):
        return None
    match = _TEMPLATE_SCENE.match(path.name)
    if not match:
        return None
    label = (scene or match.group(1)).lower()
    if label not in SCENES - {"unknown"}:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    try:
        image = _normalize(image)
    except ValueError:
        return None
    mask_path = path.with_name(path.stem + ".mask.png")
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape == (448, 512):
            mask = cv2.resize(mask, _SIZE, interpolation=cv2.INTER_NEAREST)
        if mask.shape != (224, 256):
            return None
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = np.ones((224, 256), np.uint8)
    return _Template(label, path.stem, image, mask)


def detect_player(frame: np.ndarray, previous_box: list[float] | tuple[float, ...] | None = None, *, all_candidates=False) -> list[int] | None:
    """Find a small Mario-like sprite using red, skin and blue palette pixels.

    Coordinates are ``[x, y, width, height]`` in a normalized 256x224 frame.
    This is a palette candidate detector, not an identity guarantee.
    """
    image = _normalize(frame)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Keep the red interval narrow so orange fruit does not dominate candidates.
    red = (((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 174)) &
           (hsv[:, :, 1] >= 105) & (hsv[:, :, 2] >= 75))
    skin = ((hsv[:, :, 0] >= 4) & (hsv[:, :, 0] <= 25) &
            (hsv[:, :, 1] >= 25) & (hsv[:, :, 1] <= 185) & (hsv[:, :, 2] >= 120))
    blue = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 132) &
            (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 55))
    # Seed from the red clothing, not the blue sky (which connects to blue overalls).
    red[:35] = False
    palette = red.astype(np.uint8)
    # Connect adjacent pixel-art clothing/face regions without joining fruit rows.
    palette = cv2.morphologyEx(palette, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(palette, 8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prior = tuple(map(float, previous_box[:4])) if previous_box is not None and len(previous_box) >= 4 else None
    candidates: list[tuple[float, list[int]]] = []
    for idx in range(1, count):
        x, y, w, h, area = map(int, stats[idx])
        if not (3 <= w <= 20 and 2 <= h <= 25 and 3 <= area <= 180):
            continue
        component_x, component_y, component_w, component_h = x, y, w, h
        x=max(0,x-2);y=max(35,y-1);w=min(20,image.shape[1]-x);h=min(24,image.shape[0]-y)
        r = int(np.count_nonzero(red[y:y+h, x:x+w]))
        s = int(np.count_nonzero(skin[y:y+h, x:x+w]))
        b = int(np.count_nonzero(blue[y:y+h, x:x+w]))
        # Small Mario can be almost entirely red/black in a browser capture:
        # his blue clothing may shrink to one pixel after scaling. Keep the
        # original three-color route, and allow a compact red silhouette only
        # when the same small component also contains a strong dark outline.
        # This avoids treating an isolated red fruit as the player.
        # A red seed in an enemy's legs plus blue scenery is not Mario.
        # Require a compact horizontal red cap near the TOP of the candidate,
        # instead of counting these colors anywhere in a large rectangle.
        cap=red[y:min(y+10,image.shape[0]),x:x+w]
        cap_shape=_cap_shape(cap)
        palette_match = r >= 3 and s >= 2 and b >= 2 and cap_shape
        dark = int(np.count_nonzero(gray[max(35, component_y-1):min(gray.shape[0], component_y+component_h+1),
                                         max(0, component_x-1):min(gray.shape[1], component_x+component_w+1)] <= 75))
        component_red = int(np.count_nonzero(red[component_y:component_y+component_h,
                                                  component_x:component_x+component_w]))
        compact_red_match = (
            8 <= component_w <= 16 and 8 <= component_h <= 20 and r >= 18 and dark >= 12
            and component_red / max(1, component_w * component_h) <= 0.42
        )
        if not (palette_match or compact_red_match):
            continue
        # Prefer sprites with all three palette regions and reasonable proportions.
        score = min(r, 9) * 0.11 + min(s, 8) * 0.12 + min(b, 8) * 0.10
        if compact_red_match and not palette_match:
            score = min(r, 12) * 0.11 + min(dark, 16) * 0.08
        score -= abs((h / max(w, 1)) - 1.6) * 0.12
        box = [x, y, w, h]
        if prior is not None:
            pcx, pcy = prior[0] + prior[2]/2, prior[1] + prior[3]/2
            distance = np.hypot(x + w/2 - pcx, y + h/2 - pcy)
            if distance>_PLAYER_MATCH_RADIUS:continue
            score += max(0.0, 0.8 - distance / 35.0)
        candidates.append((score, box))
    if all_candidates:return [box for score,box in sorted(candidates,key=lambda item:item[0],reverse=True)]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


class LocalMarioVision:
    """Pixel-only scene classifier with optional explicitly labelled examples."""

    def __init__(self, profile_dir: Path | None = None, *, appearance_root: Path | None = None,
                 online_learning: bool = True) -> None:
        self.profile_dir = Path(profile_dir) if profile_dir is not None else None
        self._templates: list[_Template] = []
        self._last_scene: str | None = None
        self._streak = 0
        self._player_box = None
        self._player_direct_box: list[int] | None = None
        self._player_patch: np.ndarray | None = None
        self._player_seen_at: float | None = None
        self._player_scene: str | None = None
        self.timer_reference = None
        self.identity_tracker=None;self.player_mask=None;self.identity_state={}
        seeds=self.profile_dir/'identity-seeds'/'seeds.json' if self.profile_dir else None
        if appearance_root is not None or (seeds and seeds.exists()):
            from .appearance import AppearanceBank,IdentityTracker
            bank=AppearanceBank(appearance_root, online_learning=online_learning)
            if seeds and seeds.exists():
                for seed in json.loads(seeds.read_text(encoding='utf-8')):
                    if any(r.source==seed['source'] for r in bank._references):continue
                    image=cv2.imread(str(seeds.parent/seed['image']))
                    mask=cv2.imread(str(seeds.parent/seed['mask']),cv2.IMREAD_GRAYSCALE)
                    if image is not None and mask is not None:
                        bank.add(image,[0,0,image.shape[1],image.shape[0]],label=seed['label'],mask=mask,source=seed['source'])
            self.identity_tracker=IdentityTracker(bank,validator=self._valid_player_crop)
        if self.profile_dir:
            self.timer_reference=cv2.imread(str(self.profile_dir/'hud_time.png')) if (self.profile_dir/'hud_time.png').exists() else None
        self.timer_references=([cv2.imread(str(p)) for p in self.profile_dir.glob('hud_time*.png')]
                               if self.profile_dir and self.profile_dir.exists() else [])
        if self.profile_dir and self.profile_dir.exists():
            for path in sorted(self.profile_dir.glob("*.png")):
                template = _load_template(path)
                if template is not None:
                    self._templates.append(template)

    def register_template(self, scene: str, frame: np.ndarray, name: str, mask: np.ndarray | None = None) -> Path:
        """Save one human-labelled example (and optional nonzero=include mask)."""
        label = str(scene).lower()
        if label not in SCENES - {"unknown"}:
            raise ValueError(f"scene precisa ser um rótulo conhecido: {sorted(SCENES - {'unknown'})}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("name aceita somente letras, números, hífen e underscore")
        if self.profile_dir is None:
            raise ValueError("profile_dir é necessário para gravar templates")
        image = _normalize(frame)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        path = self.profile_dir / f"{label}__{name}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"não foi possível gravar template: {path}")
        mask_u8 = np.ones((224, 256), np.uint8) * 255
        if mask is not None:
            if not isinstance(mask, np.ndarray) or mask.ndim not in (2, 3):
                raise ValueError("mask deve ser um array grayscale ou BGR")
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if mask.shape == (448, 512):
                mask = cv2.resize(mask, _SIZE, interpolation=cv2.INTER_NEAREST)
            if mask.shape != (224, 256):
                raise ValueError("mask deve medir 256x224 ou 512x448")
            mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
        mask_path = path.with_name(path.stem + ".mask.png")
        if mask is not None and not cv2.imwrite(str(mask_path), mask_u8):
            raise OSError(f"não foi possível gravar máscara: {mask_path}")
        if mask is None and mask_path.exists():
            mask_path.unlink()
        # Replace a prior in-memory template with the same name.
        self._templates = [t for t in self._templates if t.name != path.stem]
        self._templates.append(_Template(label, path.stem, image, (mask_u8 > 0).astype(np.uint8)))
        return path

    @staticmethod
    def _black_loading(frame: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        very_dark = float(np.mean(gray < 22))
        avg = float(gray.mean())
        # Sparse cursor/noise is tolerated, but fades are kept below confidence.
        return very_dark >= 0.985 and avg < 12, float(min(0.995, 0.80 + very_dark * 0.19))

    @staticmethod
    def _template_score(frame: np.ndarray, template: _Template) -> float:
        mask = template.mask.astype(bool)
        if int(mask.sum()) < frame.shape[0] * frame.shape[1] * 0.12:
            return 0.0
        a = frame.astype(np.float32)[mask]
        b = template.image.astype(np.float32)[mask]
        # Per-channel absolute pixel agreement, robust to small color/encode noise.
        error = float(np.mean(np.abs(a - b))) / 255.0
        return max(0.0, 1.0 - error)

    @staticmethod
    def _red_pixels(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return (((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 174)) &
                (hsv[:, :, 1] >= 105) & (hsv[:, :, 2] >= 75))

    def _valid_player_crop(self,frame,box):
        x,y,w,h=map(int,box)
        if x<0 or y<35 or x+w>frame.shape[1] or y+h>frame.shape[0]:return False
        crop=frame[y:y+h,x:x+w]
        red=self._red_pixels(crop)
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        bright_red=red & (crop[...,2]>=190)
        # Compression can reduce a tiny cap to a few pixels. Appearance and
        # negatives still decide identity; this gate prevents colorless clouds
        # or sky from becoming the sprite through correlation alone.
        return np.count_nonzero(bright_red[:12])>=2 and np.count_nonzero(red)>=3 and np.count_nonzero(red[:16])>=2 and np.count_nonzero(gray<80)>=4

    def _clear_player_track(self,reset_identity=True) -> None:
        self._player_box = None
        self._player_direct_box = None
        self._player_patch = None
        self._player_seen_at = None
        self._player_scene = None
        self.player_mask=None;self.identity_state={}
        if reset_identity and self.identity_tracker is not None:self.identity_tracker.reset()

    def _remember_direct_player(self, image: np.ndarray, box: list[int], now: float, scene: str) -> None:
        x, y, w, h = box
        patch = image[y:y+h, x:x+w]
        if patch.shape[:2] != (h, w) or w < 8 or h < 8:
            self._clear_player_track()
            return
        self._player_box = list(box)
        self._player_direct_box = list(box)
        self._player_patch = patch.copy()
        self._player_seen_at = float(now)
        self._player_scene = scene

    def _match_recent_player(self, image: np.ndarray, now: float, scene: str) -> list[int] | None:
        """Bridge one palette miss only with a nearby, current visual match."""
        if (scene != "gameplay" or self._player_scene != scene or
                self._player_direct_box is None or self._player_patch is None or
                self._player_seen_at is None):
            return None
        age = float(now) - self._player_seen_at
        if age < 0 or age > _PLAYER_MATCH_GRACE_S:
            return None

        x, y, w, h = self._player_direct_box
        pad = int(_PLAYER_MATCH_RADIUS)
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
        search = image[top:bottom, left:right]
        template = self._player_patch
        if search.shape[0] < h or search.shape[1] < w:
            return None
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        candidate = [left + location[0], top + location[1], w, h]
        old_cx, old_cy = x + w / 2, y + h / 2
        new_cx, new_cy = candidate[0] + w / 2, candidate[1] + h / 2
        distance = float(np.hypot(new_cx - old_cx, new_cy - old_cy))
        if score < _PLAYER_MATCH_MIN_SCORE or distance > _PLAYER_MATCH_RADIUS:
            return None

        # Correlation alone can match scenery. Require fresh red clothing in
        # the matched patch; this check does not renew the direct-sighting age.
        patch_red = self._red_pixels(image[candidate[1]:candidate[1]+h,
                                           candidate[0]:candidate[0]+w])
        if int(np.count_nonzero(patch_red)) < 3 or not _cap_shape(patch_red):
            return None
        return candidate

    def observe(self, frame: np.ndarray, now: float) -> dict:
        image = _normalize(frame)
        scene = "unknown"
        confidence = 0.0
        evidence: dict = {}
        black, black_confidence = self._black_loading(image)
        if black:
            self._clear_player_track()
            scene, confidence = "loading", black_confidence
            evidence = {"source": "pixel_rule", "rule": "near_black_frame", "dark_fraction": round(float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 22)), 4)}
        else:
            positive_timer=False
            if self.timer_reference is not None:
                def ink(im):
                    # JPEG chroma subsampling changes the tiny yellow HUD glyphs
                    # as the scenery moves. Keep the spatial glyph checks strict,
                    # but include their darker/compressed edge pixels in BOTH
                    # the reference and live masks.
                    return ((im[:,:,2]>130)&(im[:,:,1]>100)&(im[:,:,0]<175))
                digits=ink(image[24:30,154:168])
                current=ink(image[13:22,150:180]);label_score=0.
                for reference_image in self.timer_references:
                    if reference_image is None:continue
                    reference=ink(reference_image)
                    union=np.count_nonzero(current|reference)
                    score=np.count_nonzero(current&reference)/max(1,union)
                    containment=np.count_nonzero(current&reference)/max(1,np.count_nonzero(reference))
                    if containment>=.80 and score>=.55 and np.count_nonzero(current)<=2*np.count_nonzero(reference):
                        positive_timer=np.count_nonzero(digits)>=3
                        label_score=max(label_score,float(score))
            # A positive HUD already determines the gameplay scene below;
            # comparing all menu templates here would produce discarded work.
            matches = [] if positive_timer else [(self._template_score(image, t), t) for t in self._templates]
            if any(score>=.97 and t.scene in {'file_menu','player_menu'} for score,t in matches):
                matches=[(score,t) for score,t in matches if t.scene!='title']
            if matches:
                matches.sort(key=lambda item: item[0], reverse=True)
                score, template = matches[0]
                ambiguous = (len(matches) > 1 and matches[1][1].scene != template.scene
                             and matches[1][0] >= 0.92 and score - matches[1][0] < 0.025)
                # Agreement and separation both matter when labels conflict.
                if score >= 0.92 and not ambiguous:
                    scene, confidence = template.scene, score
                    evidence = {"source": "labelled_template", "template": template.name, "pixel_similarity": round(score, 4)}
            if positive_timer:
                scene,confidence='gameplay',.96
                evidence={'source':'local_ocr','rule':'mario_hud_positive_timer','label_overlap':round(label_score,3)}
        if scene == "gameplay" and evidence.get("source") not in {"labelled_template","local_ocr"}:
            scene, confidence, evidence = "unknown", 0.0, {}
        if scene == "unknown":
            self._last_scene, self._streak = None, 0
            self._clear_player_track(reset_identity=False)
            return {"scene": "unknown", "confidence": 0.0, "player_box": None, "evidence": {}, "stable": False}
        if self._last_scene is not None and scene != self._last_scene:
            self._clear_player_track()
        if scene == self._last_scene:
            self._streak += 1
        else:
            self._last_scene, self._streak = scene, 1
        player_box = None
        if scene in {"hub", "gameplay", "map", "intro", "death"}:
            prior=self._player_box
            if prior is None and self._player_seen_at is not None and 0<=now-self._player_seen_at<=_PLAYER_MATCH_GRACE_S:
                prior=self._player_direct_box
            direct_box = detect_player(image, prior)
            if self.identity_tracker is not None and scene in {'gameplay','hub'}:
                identity=None
                if self.identity_tracker._box is not None:
                    identity=self.identity_tracker.update(image,now,direct_box)
                    # If local search misses inside the prediction window,
                    # scan detector proposals broadly but only near the
                    # predicted position. IdentityTracker repeats appearance,
                    # semantic and continuity checks before confirmation.
                    if identity['box'] is None and identity.get('predicted_box') is not None:
                        predicted=identity['predicted_box']
                        age=float(identity.get('prediction_age') or 0.0)
                        radius=min(40.0,600.0*max(age,0.03)+8.0)
                        pcx,pcy=predicted[0]+predicted[2]/2,predicted[1]+predicted[3]/2
                        proposals=detect_player(image,all_candidates=True) or []
                        nearby=[]
                        for proposal in proposals:
                            cx=proposal[0]+proposal[2]/2;cy=proposal[1]+proposal[3]/2
                            if float(np.hypot(cx-pcx,cy-pcy))<=radius:
                                nearby.append(proposal)
                        ranked=[(self.identity_tracker.bank.score(image,b),b) for b in nearby]
                        accepted=[(s,b) for s,b in ranked if s['accepted'] and self._valid_player_crop(image,b)]
                        if accepted:
                            reacquired=list(max(accepted,key=lambda item:item[0]['positive_score']-item[0]['negative_score'])[1])
                            identity=self.identity_tracker.update(image,now,reacquired)
                            if identity['box'] is not None:
                                identity['source']='bounded_predicted_reacquisition'
                if identity is None or (identity['box'] is None and identity['predicted_box'] is None):
                    options=detect_player(image,prior,all_candidates=True) or []
                    # Only acquisition needs broad proposals. Established tracks
                    # use masked search near their last confirmed position.
                    options=list({(max(0,b[0]+dx),max(35,b[1]+dy),b[2],b[3])
                        for b in options[:6] for dx in (0,-4,4,-8) for dy in (0,-4,-8,-10,-12)})
                    ranked=[(self.identity_tracker.bank.score(image,b),b) for b in options]
                    accepted=[(s,b) for s,b in ranked if s['accepted'] and self._valid_player_crop(image,b)]
                    direct_box=list(max(accepted,key=lambda item:item[0]['positive_score']-item[0]['negative_score'])[1]) if accepted else None
                    candidate=direct_box or self._match_recent_player(image,now,scene)
                    identity=self.identity_tracker.update(image,now,candidate)
                player_box=identity['box'];self.player_mask=identity['mask']
                self.identity_state={k:v for k,v in identity.items() if k!='mask'}
                self.identity_state['quarantined_learned_refs']=sum(
                    1 for ref in self.identity_tracker.bank._references
                    if self.identity_tracker.bank._quarantined(ref))
                if player_box is not None and direct_box is not None and player_box==direct_box:
                    self._remember_direct_player(image,player_box,now,scene)
                else:self._player_box=player_box
            elif direct_box is not None:
                self._remember_direct_player(image, direct_box, now, scene)
                player_box = direct_box
            elif scene == "gameplay":
                player_box = self._match_recent_player(image, now, scene)
                self._player_box = list(player_box) if player_box is not None else None
            else:
                self._clear_player_track()
        else:
            self._clear_player_track()
        return {"scene": scene, "confidence": float(confidence), "player_box": player_box,
                "identity":self.identity_state,
                "evidence": evidence, "stable": self._streak >= 2}
