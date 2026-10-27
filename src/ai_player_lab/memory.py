from __future__ import annotations
from collections import deque
from pathlib import Path
import json
import sqlite3
import time
import uuid
import cv2
import numpy as np
from .features import phash, hamming, geometric_match
from .types import Observation, Decision


class EpisodeBuffer:
    def __init__(self, seconds: float = 4.0, sample_hz: float = 4.0) -> None:
        self.items: deque = deque(maxlen=max(2,int(seconds*sample_hz)))
        self.interval = 1/sample_hz; self.last = -1e99

    def add(self, frame: np.ndarray, static: np.ndarray, obs: Observation, action: Decision) -> None:
        if obs.captured_at-self.last < self.interval: return
        ok,jpeg = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok: return
        self.items.append({"frame_hash":obs.frame_hash,"captured_at":obs.captured_at,"jpeg":jpeg.tobytes(),
                           "static":static.copy(),"state":obs.to_dict(),"action":action.action})
        self.last = obs.captured_at

    def snapshot(self, event: str, now: float) -> dict:
        if not self.items: raise ValueError("Sem frames no buffer")
        records = list(self.items)
        # Pick context BEFORE the event; the death screen alone is a poor place key.
        prior = [r for r in records if r['captured_at'] <= now-0.75]
        anchor = prior[-1] if prior else records[0]
        indexes = sorted(set(np.linspace(0,len(records)-1,min(3,len(records)),dtype=int).tolist()))
        return {"event_id":uuid.uuid4().hex, "event":event,
                "anchor":anchor['static'], "anchor_frame_hash":anchor['frame_hash'],
                "frames":[{k:records[i][k] for k in ('frame_hash','captured_at','jpeg')} for i in indexes],
                "states":[{"state":r['state'],"action":r['action']} for r in records]}


class MemoryStore:
    """Memória externa auditável. Não altera pesos de modelo."""
    def __init__(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY, created REAL NOT NULL, event TEXT NOT NULL,
            scene_hash TEXT NOT NULL, anchor BLOB NOT NULL, anchor_frame_hash TEXT NOT NULL,
            states TEXT NOT NULL, review TEXT, model TEXT)''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS evidence (
            event_id TEXT NOT NULL, frame_hash TEXT NOT NULL, captured_at REAL NOT NULL,
            jpeg BLOB NOT NULL, PRIMARY KEY(event_id,frame_hash))''')
        self.db.commit()

    def add_event(self, job: dict) -> None:
        ok,png = cv2.imencode('.png',job['anchor'])
        if not ok: raise ValueError("Falha ao codificar âncora")
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,NULL,NULL)',
                (job['event_id'],time.time(),job['event'],phash(job['anchor']),png.tobytes(),job['anchor_frame_hash'],
                 json.dumps(job['states'],ensure_ascii=False)))
            for f in job['frames']:
                self.db.execute('INSERT OR IGNORE INTO evidence VALUES (?,?,?,?)',
                                (job['event_id'],f['frame_hash'],f['captured_at'],f['jpeg']))

    def attach_review(self, result: dict) -> None:
        # Event id is captured by the caller, never supplied by the model's text.
        with self.db:
            cursor = self.db.execute('UPDATE events SET review=?, model=? WHERE id=?',
                (json.dumps(result['review'],ensure_ascii=False),result['model'],result['event_id']))
            if cursor.rowcount != 1: raise ValueError("Evento de origem não encontrado")

    def recall(self, static: np.ndarray, limit: int = 3) -> list[dict]:
        current = phash(static)
        rows = self.db.execute('SELECT id,event,scene_hash,anchor,review FROM events WHERE review IS NOT NULL ORDER BY created DESC LIMIT 100').fetchall()
        ordered = sorted(rows,key=lambda r:hamming(current,r[2]))[:12]
        memories = []
        for event_id,event,scene_hash,png,review_json in ordered:
            anchor = cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)
            match = geometric_match(anchor,static)
            # Similar hash alone is not enough to assert it's the same place.
            if not match['verified']: continue
            review = json.loads(review_json)
            memories.append({"event_id":event_id,"event":event,"hypothesis":review['hypothesis'],
                             "confidence":review['confidence'],"suggested_action":review['suggested_action'],
                             "status":"unverified_cause", "place_match":match,
                             "note":"Same-looking area, not a proven causal rule; no automatic confidence increase."})
            if len(memories)>=limit: break
        return memories

    def summary(self) -> dict:
        count, reviewed = self.db.execute('SELECT COUNT(*),COUNT(review) FROM events').fetchone()
        return {"events":count,"reviewed_events":reviewed}

    def close(self) -> None: self.db.close()

    def pending_jobs(self, limit: int = 5) -> list[dict]:
        rows = self.db.execute('SELECT id,event,anchor,anchor_frame_hash,states FROM events WHERE review IS NULL ORDER BY created LIMIT ?', (limit,)).fetchall()
        jobs=[]
        for event_id,event,png,anchor_hash,states in rows:
            records=self.db.execute('SELECT frame_hash,captured_at,jpeg FROM evidence WHERE event_id=? ORDER BY captured_at',(event_id,)).fetchall()
            jobs.append({'event_id':event_id,'event':event,'anchor':cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR),
                         'anchor_frame_hash':anchor_hash,'states':json.loads(states),
                         'frames':[{'frame_hash':h,'captured_at':t,'jpeg':j} for h,t,j in records]})
        return jobs
