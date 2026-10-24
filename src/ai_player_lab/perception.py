from __future__ import annotations
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
    """Uma marcação inicial; rastreia aparência local. Recalibrar após perda."""
    def __init__(self, frame: np.ndarray, bbox: BBox, threshold: float = 0.7) -> None:
        x,y,w,h = map(int,bbox)
        if min(x,y) < 0 or min(w,h) < 4 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            raise ValueError("ROI do personagem fora da imagem ou menor que 4x4")
        self.anchor = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY).copy()
        if float(self.anchor.std()) < 5:
            raise ValueError("ROI sem textura; inclua o personagem completo e uma pequena borda")
        self.track = Track(0, (float(x),float(y),float(w),float(h)), confidence=1.0, label="player_seeded")
        self.threshold = threshold
        self.missing = 0

    def update(self, gray: np.ndarray, dt: float, camera: dict) -> Track | None:
        t = self.track; x,y,w,h = t.bbox
        # Do not turn a lost seed into a confident guess elsewhere on the screen.
        if self.missing > 8:
            return None
        radius = 24 + min(self.missing,4)*5
        px,py = x+camera["dx"], y+camera["dy"]
        x1,y1 = max(0,int(px)-radius), max(0,int(py)-radius)
        x2,y2 = min(gray.shape[1],int(px+w)+radius), min(gray.shape[0],int(py+h)+radius)
        if x2-x1 < w or y2-y1 < h:
            self.missing += 1; return None
        result = cv2.matchTemplate(gray[y1:y2,x1:x2], self.anchor, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if not np.isfinite(score) or score < self.threshold:
            self.missing += 1; return None
        box = (float(x1+loc[0]),float(y1+loc[1]),w,h)
        old = center(t.bbox); new = center(box)
        elapsed = max(dt*(self.missing+1),1e-3)
        self.track = Track(0, box, 0.4*t.vx+0.6*(new[0]-old[0])/elapsed,
                           0.4*t.vy+0.6*(new[1]-old[1])/elapsed,
                           float(score), t.age+1, 0, "player_seeded")
        self.missing = 0
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
    for i in range(start, min(gray.shape[1], start+120)):
        if not support[i] and run is None:
            run = i
        if support[i] and run is not None:
            if i-run >= 10 and run > 0 and support[max(0,run-6):run].any():
                regions.append({"kind":"gap_candidate", "x":run, "width":i-run,
                                "distance_px":max(0,run-(px+pw)), "confidence":0.45,
                                "semantic_danger":"unverified", "calibrated_floor_y":floor_y})
            run = None
    return regions


class VisualPipeline:
    def __init__(self, first_frame: np.ndarray, player_box: BBox | None = None,
                 floor_y: int | None = None, hud_px: int = 0, episode: int = 0) -> None:
        self.previous = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.player_tracker = SeededPlayer(first_frame, player_box) if player_box is not None else None
        self.tracker = MultiTracker()
        self.floor_y = floor_y
        self.hud_px = hud_px
        self.last_time: float | None = None
        self.seq = 0
        self.episode = episode
        self.last_static = first_frame.copy()

    def process(self, frame: np.ndarray, timestamp: float) -> Observation:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != self.previous.shape:
            raise ValueError("Tamanho do frame mudou; reinicialize a calibração")
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("Timestamp deve ser estritamente crescente")
        dt = 1/30 if self.last_time is None else timestamp-self.last_time
        excluded = [t.bbox for t in self.tracker.tracks.values()]
        if self.player_tracker:
            excluded.append(self.player_tracker.track.bbox)
        camera = estimate_camera(self.previous, gray, excluded)
        player = self.player_tracker.update(gray, dt, camera) if self.player_tracker else None
        aligned = cv2.warpAffine(self.previous, np.float32([[1,0,camera['dx']],[0,1,camera['dy']]]),
                                (gray.shape[1],gray.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        diff = cv2.absdiff(gray, aligned)
        mask = cv2.threshold(diff, 22,255,cv2.THRESH_BINARY)[1]
        mask[:self.hud_px] = 0
        border = int(max(abs(camera['dx']),abs(camera['dy'])))+3
        mask[:border] = 0; mask[-border:] = 0; mask[:,:border] = 0; mask[:,-border:] = 0
        # Join opposite changed edges of a small moving sprite; nearby actors can still merge.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,13),np.uint8))
        mask = cv2.dilate(mask,np.ones((3,3),np.uint8), iterations=1)
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
        objects = self.tracker.update([] if transition else boxes, dt)
        hazards = floor_gaps(gray,player,self.floor_y)
        if player:
            for obj in objects:
                hazard = collision_estimate(player,obj)
                if hazard: hazards.append(hazard)
        if transition:
            camera['scene_transition'] = True
            player = None
            hazards = []
        self.last_static = static_view(frame,[o.bbox for o in objects]+([player.bbox] if player else []),self.hud_px)
        self.seq += 1
        obs = Observation(self.seq,timestamp,exact_hash(frame),phash(self.last_static),frame.shape[1],frame.shape[0],
                          player,objects,camera,hazards,episode=self.episode)
        self.previous = gray
        self.last_time = timestamp
        return obs
