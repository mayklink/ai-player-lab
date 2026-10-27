from __future__ import annotations
import hashlib
import cv2
import numpy as np
from .types import BBox


def exact_hash(frame: np.ndarray) -> str:
    """Auditoria de igualdade; NÃO é identidade semântica de uma situação."""
    return hashlib.sha256(str((frame.shape, frame.dtype)).encode() + frame.tobytes()).hexdigest()


def phash(frame: np.ndarray) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    coeff = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].flatten()
    bits = coeff > np.median(coeff[1:])
    bits[0] = False
    return f"{sum(int(b) << i for i, b in enumerate(bits)):016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def static_view(frame: np.ndarray, boxes: list[BBox], hud_px: int = 0) -> np.ndarray:
    """Mascara atores conhecidos. Não garante invariância à rolagem/paralaxe."""
    out = frame.copy()
    h, w = out.shape[:2]
    out[:max(0, min(hud_px, h))] = 0
    for x, y, bw, bh in boxes:
        x1, y1 = max(0, int(x)-2), max(0, int(y)-2)
        x2, y2 = min(w, int(x+bw)+2), min(h, int(y+bh)+2)
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = 0
    return out


def geometric_match(a: np.ndarray, b: np.ndarray) -> dict:
    """Verificação ORB + RANSAC de trechos; resultado insuficiente permanece incerto."""
    orb = cv2.ORB_create(nfeatures=350, edgeThreshold=8, fastThreshold=12)
    ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return {"verified": False, "inliers": 0, "reason": "few_features"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.72*p[1].distance]
    if len(good) < 8:
        return {"verified": False, "inliers": 0, "reason": "few_matches"}
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if matrix is None or inliers is None:
        return {"verified": False, "inliers": 0, "reason": "no_transform"}
    count = int(inliers.sum())
    scale = float(np.hypot(matrix[0,0], matrix[1,0]))
    selected = src[inliers.ravel().astype(bool)]
    spread = float(np.ptp(selected[:,0])) if len(selected) else 0.0
    return {"verified": bool(count >= 10 and count/len(good) >= 0.6 and 0.9 <= scale <= 1.1 and spread >= a.shape[1]*0.18),
            "inliers": count, "matches": len(good), "transform": matrix.tolist()}
