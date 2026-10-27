from __future__ import annotations
import math
import cv2
import numpy as np
from .types import Track, Observation, BBox, center, iou
from .tracking import MultiTracker, collision_estimate
from .features import exact_hash, phash, static_view


def estimate_camera(previous: np.ndarray, current: np.ndarray, excluded: list[BBox]) -> dict:
    mask = np.full(previous.shape, 255, np.uint8)
    for x,y,w,h in excluded:
        cv2.rectangle(mask, (max(0,int(x)-4), max(0,int(y)-4)),
                      (int(x+w)+4,int(y+h)+4), 0, -1)
    points = cv2.goodFeaturesToTrack(previous, maxCorners=200, qualityLevel=0.02, minDistance=7, mask=mask)
    empty = {"dx": 0.0, "dy": 0.0, "valid": False, "inliers": 0}
    if points is None or len(points) < 10:
        return empty
    new, st, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(15,15), maxLevel=2)
    if new is None or st is None:
        return empty
    good = st.ravel().astype(bool)
    delta = new.reshape(-1,2)[good] - points.reshape(-1,2)[good]
    if len(delta) < 10:
        return empty
    median = np.median(delta, axis=0)
    inliers = np.linalg.norm(delta-median, axis=1) < 1.7
    if inliers.sum() < 10 or inliers.mean() < 0.5 or np.linalg.norm(median) > 30:
        return empty
    dx,dy = np.median(delta[inliers], axis=0)
    return {"dx": float(dx), "dy": float(dy), "valid": True, "inliers": int(inliers.sum())}


class SeededPlayer:
    """Aparências fixas e recuperação curta; nunca aprende pixels sem confirmação humana."""
    def __init__(self, frame: np.ndarray, bbox: BBox, threshold: float = 0.7,
                 template: np.ndarray | None = None) -> None:
        x,y,w,h = map(int,bbox)
        if min(x,y) < 0 or min(w,h) < 4 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            raise ValueError("ROI do personagem fora da imagem ou menor que 4x4")
        self.anchor = (cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
                       if template is None else template).copy()
        if self.anchor.dtype != np.uint8 or self.anchor.shape != (h, w):
            raise ValueError("Template do personagem incompatível com a calibração")
        if float(self.anchor.std()) < 5:
            raise ValueError("ROI sem textura; inclua o personagem completo e uma pequena borda")
        self.templates = (self.anchor, cv2.flip(self.anchor, 1))
        self.track = Track(0, (float(x),float(y),float(w),float(h)), confidence=0.95, label="player_seeded")
        self.threshold = threshold
        self.missing = 0
        self.elapsed = 0.0
        self.camera_offset = np.zeros(2, dtype=float)
        self.relative_velocity = np.zeros(2, dtype=float)
        self.camera_velocity = np.zeros(2, dtype=float)
        self.pending: BBox | None = None
        self.pending_velocity = np.zeros(2, dtype=float)
        self.requires_reseed = False

    def _lost(self) -> None:
        self.missing += 1
        self.pending = None
        return None

    def update(self, gray: np.ndarray, dt: float, camera: dict) -> Track | None:
        t = self.track; x,y,w,h = t.bbox
        self.camera_velocity = .4*self.camera_velocity+.6*np.array([camera['dx'],camera['dy']])/max(dt,1e-3)
        self.elapsed += max(dt, 1e-3)
        self.camera_offset += (camera['dx'], camera['dy'])
        # Recovery is bounded in time and space; a long loss needs a human reseed.
        if self.requires_reseed or self.elapsed > 1.5:
            self.requires_reseed = True
            return None
        radius = 24 + min(self.missing,4)*5
        px,py = np.array([x,y])+self.camera_offset+self.relative_velocity*self.elapsed
        x1,y1 = max(0,int(px)-radius), max(0,int(py)-radius)
        x2,y2 = min(gray.shape[1],int(px+w)+radius), min(gray.shape[0],int(py+h)+radius)
        if x2-x1 < w or y2-y1 < h:
            return self._lost()
        result = np.maximum.reduce([cv2.matchTemplate(gray[y1:y2,x1:x2], anchor,
                                     cv2.TM_CCOEFF_NORMED) for anchor in self.templates])
        _, score, _, loc = cv2.minMaxLoc(result)
        if not np.isfinite(score) or score < self.threshold:
            return self._lost()
        # Suppress the same spatial peak (including its flipped-pose match), not competitors.
        competitors = result.copy()
        rx, ry = max(3, int(w*.6)), max(3, int(h*.6))
        competitors[max(0,loc[1]-ry):loc[1]+ry+1, max(0,loc[0]-rx):loc[0]+rx+1] = -1
        second = float(competitors.max())
        if second >= self.threshold and score-second < .06:
            return self._lost()
        box = (float(x1+loc[0]),float(y1+loc[1]),w,h)
        if np.hypot(box[0]-px, box[1]-py) > max(12., 400*self.elapsed):
            return self._lost()
        if self.missing:
            expected = None if self.pending is None else (
                self.pending[0]+camera['dx']+self.pending_velocity[0]*dt,
                self.pending[1]+camera['dy']+self.pending_velocity[1]*dt, w, h)
            if expected is None or iou(expected, box) < .5:
                self.pending = box
                self.pending_velocity = np.clip((np.array(box[:2])-np.array(t.bbox[:2])-self.camera_offset)/self.elapsed,-400,400)
                return None
        old = center(t.bbox); new = center(box)
        elapsed = max(self.elapsed,1e-3)
        self.relative_velocity = .4*self.relative_velocity+.6*(np.array(new)-np.array(old)-self.camera_offset)/elapsed
        screen_velocity = self.relative_velocity+self.camera_velocity
        self.track = Track(0, box, float(screen_velocity[0]), float(screen_velocity[1]),
                           min(.95,float(score)), t.age+1, 0, "player_seeded")
        self.missing = 0
        self.elapsed = 0.0
        self.camera_offset[:] = 0
        self.pending = None
        return Track(**self.track.to_dict())


def floor_gaps(gray: np.ndarray, player: Track | None, floor_y: int | None) -> list[dict]:
    """Hipótese restrita: chão horizontal em linha calibrada. Não mapa genérico."""
    if player is None or floor_y is None or not 4 < floor_y < gray.shape[0]-4:
        return []
    edges = cv2.Canny(gray, 50, 130)
    strip = edges[floor_y-3:floor_y+4]
    support = (strip.max(axis=0) > 0).astype(np.uint8)[None,:]
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((1,5),np.uint8))[0]
    px,py,pw,ph = player.bbox
    if abs(py+ph-floor_y) > max(12,ph):
        return []
    foot = int(px+pw/2)
    if not 2 < foot < gray.shape[1]-2 or not np.any(support[max(0,foot-8):foot+8]):
        return []
    start = min(gray.shape[1]-1,int(px+pw))
    regions = []; run = None
    end = min(gray.shape[1], start+120)
    for i in range(start, end):
        if not support[i] and run is None:
            run = i
        if support[i] and run is not None:
            if i-run >= 10 and run > 0 and support[max(0,run-6):run].any():
                regions.append({"kind":"gap_candidate", "x":run, "width":i-run,
                                "distance_px":max(0,run-(px+pw)), "confidence":0.45,
                                "semantic_danger":"unverified", "calibrated_floor_y":floor_y,
                                "right_boundary_visible":True, "width_is_lower_bound":False})
            run = None
    if run is not None and end-run >= 10 and run > 0 and support[max(0,run-6):run].any():
        regions.append({"kind":"gap_candidate", "x":run, "width":end-run,
                        "distance_px":max(0,run-(px+pw)), "confidence":0.35,
                        "semantic_danger":"unverified", "calibrated_floor_y":floor_y,
                        "right_boundary_visible":False, "width_is_lower_bound":True})
    return regions


class VisualPipeline:
    def __init__(self, first_frame: np.ndarray, player_box: BBox | None = None,
                 floor_y: int | None = None, hud_px: int = 0, episode: int = 0,
                 player_template: np.ndarray | None = None, object_detector=None,
                 classify_motion: bool = False) -> None:
        self.previous = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        if type(hud_px) is not int or not 0 <= hud_px < first_frame.shape[0]:
            raise ValueError('hud_px deve ser inteiro entre 0 e altura do frame menos 1')
        self.player_tracker = SeededPlayer(first_frame, player_box, template=player_template) if player_box is not None else None
        self.tracker = MultiTracker()
        self.floor_y = floor_y
        self.hud_px = hud_px
        self.last_time: float | None = None
        self.seq = 0
        self.episode = episode
        self.last_static = first_frame.copy()
        self.last_external_player=player_box
        self._object_appearance: dict[int, dict] = {}
        self.object_detector=object_detector
        self.enemy_tracker=MultiTracker(max_missed=12)
        self._enemy_labels={}
        self.classify_motion=classify_motion
        self._independent_evidence={}
        self._independent_ids=set()

    def _classify_objects(self,objects,camera,motion_boxes):
        """Separate camera-consistent scenery from repeated independent motion."""
        for obj in objects:
            if obj.missed:
                obj.label='independent_motion_predicted' if obj.id in self._independent_ids else 'predicted_motion'
                continue
            relative=self.tracker.relative_velocity.get(obj.id,np.zeros(2))
            speed=float(np.linalg.norm(relative))
            previous,count=self._independent_evidence.get(obj.id,(relative,0))
            coherent=float(np.dot(relative,previous))>=0
            current_residual=any(iou(obj.bbox,b)>.15 for b in motion_boxes)
            if camera.get('valid') and speed>=25 and coherent and current_residual:
                count=min(count+1,5)
            else:count=max(0,count-1)
            self._independent_evidence[obj.id]=(relative.copy(),count)
            if count>=3:
                obj.label='independent_motion';obj.confidence=min(.85,.45+.08*count)
                self._independent_ids.add(obj.id)
            elif camera.get('valid') and obj.age>=3 and speed<12:
                obj.label='background_consistent';obj.confidence=.3
            else:
                obj.label='unknown_motion';obj.confidence=min(.5,obj.confidence)
        live=set(self.tracker.tracks)
        self._independent_evidence={k:v for k,v in self._independent_evidence.items() if k in live}
        self._independent_ids.intersection_update(live)
        return objects

    @staticmethod
    def _masked_similarity(reference: np.ndarray, current: np.ndarray, mask: np.ndarray) -> float:
        visible = mask.astype(bool)
        if np.count_nonzero(visible) < 8:
            return -1.0
        a = reference[visible].astype(np.float32)
        b = current[visible].astype(np.float32)
        a -= float(a.mean()); b -= float(b.mean())
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator < 1e-6:
            return -1.0
        return float(np.dot(a, b) / denominator)

    @staticmethod
    def _foreground_mask(crop: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
        distance = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
        mask = (distance > 14.0).astype(np.uint8)
        if min(crop.shape[:2]) >= 8:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            keep = np.zeros_like(mask)
            for index in range(1, count):
                if int(stats[index, cv2.CC_STAT_AREA]) >= max(2, int(largest * .10)):
                    keep[labels == index] = 1
            mask = keep
        return mask

    @staticmethod
    def _hue_histogram(crop: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        if int(np.count_nonzero(mask)) < 8:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0], (mask > 0).astype(np.uint8) * 255,
                                 [12], [0, 180])
        cv2.normalize(histogram, histogram)
        return histogram

    def _appearance_matches(self, gray: np.ndarray, detections: list[BBox],
                            timestamp: float, camera: dict,
                            player_box: BBox | None) -> list[BBox]:
        """Confirm previously observed motion objects on current visible pixels."""
        if not self._object_appearance:
            return []
        result: list[BBox] = []
        used: set[int] = set()
        for tid, track in list(self.tracker.tracks.items()):
            anchor = self._object_appearance.get(tid)
            if anchor is None:
                continue
            elapsed = max(0.0, timestamp - self.tracker._last_seen.get(tid, timestamp))
            if elapsed > self.tracker.MAX_PREDICTION_AGE_S:
                self._object_appearance.pop(tid, None)
                continue
            template, base_mask = anchor['gray'], anchor['mask']
            h, w = template.shape
            if h < 3 or w < 3:
                continue
            # Velocity already describes screen-space motion; adding this
            # frame's camera delta to the center would double-count scroll.
            px = track.bbox[0] + track.vx * elapsed
            py = track.bbox[1] + track.vy * elapsed
            radius = min(24, max(6, int(4 + max(abs(track.vx), abs(track.vy)) * elapsed
                                   + max(abs(camera.get('dx', 0)), abs(camera.get('dy', 0))))))
            x0, y0 = max(0, int(round(px)) - radius), max(0, int(round(py)) - radius)
            x1 = min(gray.shape[1], int(round(px)) + w + radius)
            y1 = min(gray.shape[0], int(round(py)) + h + radius)
            search = gray[y0:y1, x0:x1]
            if search.shape[0] < h or search.shape[1] < w:
                continue
            try:
                scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
            local = scores.copy()
            best = None
            # Check several nearby peaks because player occlusion can lower
            # the unmasked score of the real actor below a background patch.
            for _ in range(8):
                _, score, _, location = cv2.minMaxLoc(local)
                cx, cy = x0 + location[0], y0 + location[1]
                visible = base_mask.copy()
                if player_box is not None:
                    ex, ey, ew, eh = map(int, player_box)
                    lx0, ly0 = max(0, ex - cx), max(0, ey - cy)
                    lx1, ly1 = min(w, ex + ew - cx), min(h, ey + eh - cy)
                    if lx1 > lx0 and ly1 > ly0:
                        visible[ly0:ly1, lx0:lx1] = 0
                visible_count = int(np.count_nonzero(visible))
                # A partially visible actor can still be checked from a
                # bounded distinctive slice; fewer than 30% is too ambiguous.
                if visible_count >= max(8, int(np.count_nonzero(base_mask) * .30)):
                    similarity = self._masked_similarity(template, gray[cy:cy+h, cx:cx+w], visible)
                    if best is None or similarity > best[0]:
                        best = (similarity, cx, cy)
                lx, ly = location
                local[max(0, ly-3):ly+4, max(0, lx-3):lx+4] = -1.0
            if best is None or best[0] < .84:
                continue
            result.append((float(best[1]), float(best[2]), float(w), float(h)))
            used.add(tid)
        self._appearance_confirmed_ids = used
        return result

    def _appearance_variant_matches(self, frame: np.ndarray, detections: list[BBox],
                                    timestamp: float, camera: dict,
                                    player_box: BBox | None) -> list[tuple[int, BBox, dict]]:
        """Confirm a bounded changed pose from current foreground colors.

        This supports a known actor changing height/pose while retaining its
        track ID. It compares only a current crop against that track's local
        anchor; it never assigns a class such as Rex.
        """
        used = getattr(self, '_appearance_confirmed_ids', set())
        variants = []
        for box in detections:
            bx, by = center(box)
            bw, bh = box[2], box[3]
            if min(bw, bh) < 4:
                continue
            for tid, track in self.tracker.tracks.items():
                anchor = self._object_appearance.get(tid)
                if anchor is None or tid in used:
                    continue
                elapsed = max(0.0, timestamp - self.tracker._last_seen.get(tid, timestamp))
                if elapsed > self.tracker.MAX_PREDICTION_AGE_S:
                    continue
                expected_x = track.bbox[0] + track.vx * elapsed + track.bbox[2] / 2
                expected_y = track.bbox[1] + track.vy * elapsed + track.bbox[3] / 2
                if math.hypot(bx-expected_x, by-expected_y) > self.tracker.max_distance + 12:
                    continue
                area_ratio = (bw * bh) / max(track.bbox[2] * track.bbox[3], 1.0)
                if not .35 <= area_ratio <= 2.8:
                    continue
                x, y, w, h = map(int, box)
                if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                    continue
                crop = frame[y:y+h, x:x+w]
                mask = self._foreground_mask(crop)
                if player_box is not None:
                    px, py, pw, ph = map(int, player_box)
                    left, top = max(x, px), max(y, py)
                    right, bottom = min(x+w, px+pw), min(y+h, py+ph)
                    if right > left and bottom > top:
                        mask[top-y:bottom-y, left-x:right-x] = 0
                count = int(np.count_nonzero(mask))
                if count < 8 or count < int(anchor['mask'].sum() * .20):
                    continue
                histogram = self._hue_histogram(crop, mask)
                if histogram is None:
                    continue
                distance = cv2.compareHist(anchor['histogram'], histogram,
                                           cv2.HISTCMP_BHATTACHARYYA)
                if not np.isfinite(distance) or distance > .30:
                    continue
                new_anchor = {'gray': cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).copy(),
                              'mask': mask.copy(), 'histogram': histogram}
                variants.append((tid, box, new_anchor))
                used.add(tid)
                break
        self._appearance_confirmed_ids = used
        return variants

    def _remember_motion_appearance(self, track: Track, frame: np.ndarray,
                                    motion_mask: np.ndarray, player_box: BBox | None) -> None:
        if track.id in self._object_appearance:
            return
        x, y, w, h = map(int, track.bbox)
        if min(w, h) < 4 or x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            return
        patch = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        if float(patch.std()) < 4.0:
            return
        crop = frame[y:y+h, x:x+w]
        mask = self._foreground_mask(crop)
        if int(mask.sum()) < 8:
            mask = (motion_mask[y:y+h, x:x+w] > 0).astype(np.uint8)
        if player_box is not None:
            px, py, pw, ph = map(int, player_box)
            left, top = max(x, px), max(y, py)
            right, bottom = min(x+w, px+pw), min(y+h, py+ph)
            if right > left and bottom > top:
                mask[top-y:bottom-y, left-x:right-x] = 0
        count = int(np.count_nonzero(mask))
        if count < max(8, int(mask.size * .03)) or count > int(mask.size * .95):
            return
        histogram = self._hue_histogram(crop, mask)
        if histogram is None:
            return
        # Low-entropy patches (plain blocks, ground fragments) are poor
        # identity anchors: they can make a stationary background patch look
        # like a continuously observed moving actor. Keep identity templates
        # only when their foreground colors are reasonably distinctive.
        probabilities = histogram.ravel().astype(np.float64)
        total = float(probabilities.sum())
        if total <= 0:
            return
        probabilities /= total
        entropy = -float(np.sum(probabilities[probabilities > 0] *
                                np.log2(probabilities[probabilities > 0])))
        if entropy < 1.25:
            return
        self._object_appearance[track.id] = {'gray': patch.copy(), 'mask': mask,
                                             'histogram': histogram}

    def process(self, frame: np.ndarray, timestamp: float, player_bbox: BBox | None = None,
                *, captured_at: float | None = None, allow_new_objects: bool = True) -> Observation:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != self.previous.shape:
            raise ValueError("Tamanho do frame mudou; reinicialize a calibração")
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("Timestamp deve ser estritamente crescente")
        dt = 1/30 if self.last_time is None else timestamp-self.last_time
        excluded = [t.bbox for t in self.tracker.tracks.values()]
        if self.player_tracker:
            excluded.append(self.player_tracker.track.bbox)
        if self.hud_px:
            excluded.append((0, 0, gray.shape[1], self.hud_px))
        if player_bbox is not None:excluded.append(player_bbox)
        previous_external=getattr(self,'last_external_player',None)
        if previous_external is not None:excluded.append(previous_external)
        camera = estimate_camera(self.previous, gray, excluded)
        player = self.player_tracker.update(gray, dt, camera) if self.player_tracker else None
        current_player_box = player_bbox or (player.bbox if player is not None else None)
        aligned = cv2.warpAffine(self.previous, np.float32([[1,0,camera['dx']],[0,1,camera['dy']]]),
                                (gray.shape[1],gray.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        diff = cv2.absdiff(gray, aligned)
        mask = cv2.threshold(diff, 22,255,cv2.THRESH_BINARY)[1]
        player_regions = []
        if current_player_box is not None:
            player_regions.append(current_player_box)
            if previous_external is not None:
                x,y,w,h=previous_external
                player_regions.append((x+camera['dx'],y+camera['dy'],w,h))
            # Erase only the endpoint ROIs. A bounding rectangle swept between
            # a jumping Mario's old and new positions can erase a nearby actor.
            for x,y,w,h in player_regions:
                left=max(0,int(x)-3);top=max(0,int(y)-3)
                right=max(0,min(gray.shape[1],int(x+w)+3));bottom=max(0,min(gray.shape[0],int(y+h)+3))
                if right <= left or bottom <= top:
                    continue
                mask[top:bottom,left:right]=0
        self.last_external_player=current_player_box
        mask[:self.hud_px] = 0
        border = int(max(abs(camera['dx']),abs(camera['dy'])))+3
        mask[:border] = 0; mask[-border:] = 0; mask[:,:border] = 0; mask[:,-border:] = 0
        # Join opposite changed edges of a small moving sprite; nearby actors can still merge.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,13),np.uint8))
        mask = cv2.dilate(mask,np.ones((3,3),np.uint8), iterations=1)
        for x,y,w,h in player_regions:
            left=max(0,int(x)-3);top=max(0,int(y)-3)
            right=max(0,min(gray.shape[1],int(x+w)+3));bottom=max(0,min(gray.shape[0],int(y+h)+3))
            if right <= left or bottom <= top:
                continue
            mask[top:bottom,left:right]=0
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if 12 <= cv2.contourArea(cnt) <= gray.size*0.08 and w >= 3 and h >= 3:
                box = (float(x),float(y),float(w),float(h))
                if player is None or iou(box,player.bbox) < 0.08:
                    boxes.append(box)
        # Large residual changes indicate transitions, not dozens of new enemies.
        transition = float(np.count_nonzero(mask))/mask.size > 0.3
        appearance_boxes = [] if transition else self._appearance_matches(
            gray, boxes, timestamp, camera, current_player_box)
        variant_matches = [] if transition else self._appearance_variant_matches(
            frame, boxes, timestamp, camera, current_player_box)
        variant_boxes = [box for _, box, _ in variant_matches]
        confirmed_appearance_ids = getattr(self, '_appearance_confirmed_ids', set())
        motion_boxes = []
        for box in boxes:
            bx, by = center(box)
            matching_anchor = None
            for tid, track in self.tracker.tracks.items():
                if tid not in self._object_appearance:
                    continue
                elapsed = max(0.0, timestamp - self.tracker._last_seen.get(tid, timestamp))
                px = track.bbox[0] + track.vx*elapsed + float(camera.get('dx', 0)) + track.bbox[2]/2
                py = track.bbox[1] + track.vy*elapsed + float(camera.get('dy', 0)) + track.bbox[3]/2
                if math.hypot(bx-px, by-py) <= self.tracker.max_distance + 12:
                    matching_anchor = tid
                    break
            # A known track's contour can be the empty background left after
            # the actor vanished. Let only a current pixel template match
            # refresh that track; otherwise it stays predicted and expires.
            if matching_anchor is None:
                motion_boxes.append(box)
        semantic = (self.object_detector.detect(frame,player_box=current_player_box)
                    if self.object_detector is not None and not transition else [])
        semantic_boxes=[tuple(item['bbox']) for item in semantic]
        enemy_objects=(self.enemy_tracker.update(semantic_boxes,dt,camera=camera,timestamp=timestamp,
                        allow_new_objects=allow_new_objects) if self.object_detector is not None and not transition else [])
        for enemy in enemy_objects:
            match=next((item for item in semantic if iou(enemy.bbox,tuple(item['bbox']))>.5),None)
            if match is not None and not enemy.missed:self._enemy_labels[enemy.id]=match['label']
            label=self._enemy_labels.get(enemy.id,'enemy_unknown')
            enemy.id+=100000
            enemy.label=label+'_predicted' if enemy.missed else label
            if match is not None and not enemy.missed:enemy.confidence=float(match['confidence'])
        self._enemy_labels={k:v for k,v in self._enemy_labels.items() if k in self.enemy_tracker.tracks}
        # A verified sprite sighting owns its geometry. Residual motion around
        # that sprite must not create a second, larger object at the same spot.
        detection_boxes = [b for b in motion_boxes+appearance_boxes+variant_boxes
                           if not any(iou(b,e.bbox)>.12 for e in enemy_objects)]
        if transition:
            self.tracker.reset()
            self._object_appearance.clear()
            self.enemy_tracker.reset()
            self._enemy_labels.clear()
            self._independent_evidence.clear();self._independent_ids.clear()
            objects = []
        else:
            objects = self.tracker.update(detection_boxes, dt, camera=camera, timestamp=timestamp,
                                          allow_new_objects=allow_new_objects)
            for obj in objects:
                if obj.id in getattr(self, '_appearance_confirmed_ids', set()) and obj.missed == 0:
                    obj.label = 'appearance_confirmed_motion'
            for tid, candidate, anchor in variant_matches:
                tracked = self.tracker.tracks.get(tid)
                if tracked is not None and tracked.missed == 0 and \
                        math.hypot(center(tracked.bbox)[0]-center(candidate)[0],
                                   center(tracked.bbox)[1]-center(candidate)[1]) <= 3:
                    self._object_appearance[tid] = anchor
            # Only current motion contours can seed a visual anchor. No class
            # name or persistent enemy model is inferred.
            for track in self.tracker.tracks.values():
                if track.missed != 0 or track.id in self._object_appearance:
                    continue
                x0, y0 = center(track.bbox)
                if not any(math.hypot(center(box)[0]-x0, center(box)[1]-y0) <= 3.0
                           for box in boxes):
                    continue
                if player_bbox is not None and iou(track.bbox, tuple(player_bbox)) > .12:
                    continue
                self._remember_motion_appearance(track, frame, mask, current_player_box)
            live_ids = set(self.tracker.tracks)
            self._object_appearance = {tid: item for tid, item in self._object_appearance.items()
                                       if tid in live_ids}
            if self.classify_motion:
                objects=self._classify_objects(objects,camera,boxes)
                objects=[o for o in objects if o.label!='background_consistent']
            objects=[obj for obj in objects if not any(iou(obj.bbox,e.bbox)>.12 for e in enemy_objects)]+enemy_objects
        hazards = floor_gaps(gray,player,self.floor_y)
        if player:
            for obj in objects:
                hazard = collision_estimate(player,obj,require_current_evidence=self.classify_motion)
                if hazard: hazards.append(hazard)
        if transition:
            camera['scene_transition'] = True
            player = None
            hazards = []
        self.last_static = static_view(frame,[o.bbox for o in objects]+([player.bbox] if player else []),self.hud_px)
        self.seq += 1
        obs = Observation(self.seq, timestamp if captured_at is None else captured_at,
                          exact_hash(frame), phash(self.last_static), frame.shape[1], frame.shape[0],
                          player, objects, camera, hazards, episode=self.episode,
                          media_time=timestamp if captured_at is not None else None)
        self.previous = gray
        self.last_time = timestamp
        return obs
