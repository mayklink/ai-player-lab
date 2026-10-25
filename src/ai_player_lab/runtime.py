from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue, Empty
import json
import math
import re
import time
import cv2
import numpy as np
from .types import Decision
from .perception import VisualPipeline, SeededPlayer
from .sources import load_upstream_capture, resize_frame, UpstreamDeathDetector, decode_player_template
from .memory import MemoryStore, EpisodeBuffer
from .providers import JevPolicy, MultimodalReviewer, local_policy, choose_action
from .workers import LatestWorker
from .control import DryController, SafeKeyboard
from .diagnostics import draw_overlay, machine_report
from .features import hamming


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg.get('capture_region'),list) or len(cfg['capture_region']) != 4:
        raise ValueError('capture_region deve ser [x,y,w,h]; execute calibrate')
    if cfg['capture_region'][2] <= 0 or cfg['capture_region'][3] <= 0:
        raise ValueError('Região de captura precisa ter tamanho positivo')
    if not 128 <= int(cfg.get('work_width',320)) <= 1280:
        raise ValueError('work_width fora de 128..1280')
    if not 1 <= float(cfg.get('target_fps',30)) <= 120:
        raise ValueError('target_fps fora de 1..120')
    if not 0 < float(cfg.get('jev_hz',3)) <= 10:
        raise ValueError('jev_hz fora de 0..10')
    if not 1 <= float(cfg.get('max_decision_age_ms',250)) <= 1000:
        raise ValueError('max_decision_age_ms fora de 1..1000')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,60}',cfg.get('game_id','mario-world-v1')):
        raise ValueError('game_id inválido')


def run_session(config_path: Path, *, seconds: float = 120, jev: bool = False, vlm: bool = False,
                enable_input: bool = False, video_path: Path | None = None, headless: bool = False) -> dict:
    cfg=json.loads(config_path.read_text(encoding='utf-8')); validate_config(cfg)
    # Reject legacy/malformed seeds before creating clients, workers or a controller.
    player_template = decode_player_template(cfg)
    if seconds <= 0: raise ValueError('seconds precisa ser positivo')
    if enable_input and video_path: raise ValueError('Replay nunca injeta teclas')
    if enable_input and headless: raise ValueError('Controle real exige visualização e hotkeys')
    if enable_input and not cfg.get('window_title'): raise ValueError('Configure window_title antes de habilitar teclado')
    jev_provider=JevPolicy(cfg.get('jev_model','jev-1.14.0'),max_calls=int(cfg.get('max_jev_calls',120))) if jev else None
    try:
        reviewer=MultimodalReviewer.from_config(cfg) if vlm else None
    except Exception:
        if jev_provider: jev_provider.close()
        raise
    game_id=cfg.get('game_id','mario-world-v1')
    work=Path('runs')/game_id; work.mkdir(parents=True,exist_ok=True)
    out=work/datetime.now().strftime('%Y%m%d-%H%M%S-%f'); out.mkdir()
    memory_path=work/'memory.sqlite3'
    memory=MemoryStore(memory_path)
    # Each worker owns its DB connection; no cross-thread sqlite object sharing.
    def recall_task(job):
        reader=MemoryStore(memory_path)
        try: return {'memories':reader.recall(job['image']),'hash':job['hash'],'time':job['time'],'episode':job['episode']}
        finally: reader.close()
    memory_worker=LatestWorker(recall_task,'memory-recall')
    jev_worker=LatestWorker(jev_provider,'jev-policy') if jev_provider else None
    review_worker=LatestWorker(reviewer,'multimodal-review') if reviewer else None
    controller=SafeKeyboard(cfg['window_title'],cfg.get('keys')) if enable_input else DryController()
    hotkeys=SimpleQueue(); listener=None
    if not headless:
        try:
            from pynput.keyboard import Listener, Key
            keymap={Key.f6:'reseed',Key.f8:'death',Key.f9:'arm',Key.f10:'disarm',Key.esc:'quit'}
            def on_press(key):
                if key in keymap: hotkeys.put(keymap[key])
            listener=Listener(on_press=on_press); listener.start()
        except (ImportError,RuntimeError):
            if enable_input: raise RuntimeError('Hotkeys indisponíveis; controle de teclado não será ativado')
    upstream=Path(cfg.get('upstream_path','vendor/pixel-to-action'))
    capture=None; video=None
    if video_path:
        video=cv2.VideoCapture(str(video_path))
        if not video.isOpened(): raise ValueError('Não foi possível abrir vídeo')
        def get_frame():
            ok,frame=video.read()
            return frame if ok else None
        fps=video.get(cv2.CAP_PROP_FPS) or float(cfg.get('target_fps',30))
        if not math.isfinite(fps) or fps <= 0:
            fps=float(cfg.get('target_fps',30))
    else:
        capture=load_upstream_capture(upstream,tuple(cfg['capture_region']))
        def get_frame(): return capture.get_frame()
        fps=float(cfg.get('target_fps',30))
    death_detector=UpstreamDeathDetector(upstream,cfg['death_templates']) if cfg.get('death_templates') else None
    buffer=EpisodeBuffer(); pipeline=None; decision=None; cached_memory=None
    start=time.monotonic(); last_jev=last_review=last_recall=-1e99; cooldown=0.0
    timings=defaultdict(list); counters=defaultdict(int); episode=0; log=(out/'state.jsonl').open('w',encoding='utf-8')
    # Hard cap prevents accidental indefinite collection or API spending.
    seconds=min(seconds,3600)
    print('Modo observação. F6 remarca jogador; F8 marca morte; F9 arma (opt-in); F10 desarma; Esc encerra.')
    try:
        while time.monotonic()-start < seconds:
            loop_start=time.monotonic(); captured_at=loop_start; raw=get_frame()
            if raw is None: break
            frame=resize_frame(raw,int(cfg.get('work_width',320)))
            timings['capture_resize_ms'].append((time.monotonic()-loop_start)*1000)
            if pipeline is None:
                decode_player_template(cfg, frame)
                pipeline=VisualPipeline(frame,tuple(cfg['player_box']) if cfg.get('player_box') else None,
                                        cfg.get('floor_y'),cfg.get('hud_px',0),episode,
                                        player_template=player_template)
            # Video kinematics use source time; decision expiry still uses real capture time.
            t=time.monotonic()
            obs=pipeline.process(frame,counters['frames']/fps,captured_at=captured_at) if video_path else pipeline.process(frame,captured_at)
            timings['vision_ms'].append((time.monotonic()-t)*1000)
            counters['frames']+=1; counters['player_missing']+=int(obs.player is None)
            response=memory_worker.poll()
            if response:
                if response.error: counters['memory_errors']+=1
                else: cached_memory=response.value
            if cached_memory and cached_memory['episode']==episode and captured_at-cached_memory['time']<0.5 and hamming(obs.scene_hash,cached_memory['hash'])<=4:
                obs.hypotheses=cached_memory['memories']
            if captured_at-last_recall >= 0.5:
                memory_worker.submit({'image':pipeline.last_static.copy(),'hash':obs.scene_hash,'time':captured_at,'episode':episode})
                last_recall=captured_at
            if jev_worker:
                response=jev_worker.poll()
                if response:
                    timings['jev_rtt_ms'].append(response.elapsed_ms)
                    if response.error:
                        counters['jev_errors']+=1; decision=None; cooldown=captured_at+1.0
                    else: decision=response.value
                if captured_at>=cooldown and captured_at-last_jev>=1/float(cfg.get('jev_hz',3)) and jev_provider.calls<jev_provider.max_calls:
                    jev_worker.submit(obs); last_jev=captured_at
            else: decision=local_policy(obs)
            chosen,reason=choose_action(obs,decision,time.monotonic(),cfg.get('max_decision_age_ms',250))
            counters[reason]+=1
            controller.apply(chosen)
            buffer.add(frame,pipeline.last_static,obs,chosen)
            if review_worker:
                result=review_worker.poll()
                if result:
                    timings['vlm_rtt_ms'].append(result.elapsed_ms)
                    if result.error: counters['vlm_errors']+=1
                    else: memory.attach_review(result.value); counters['reviews_completed']+=1
            events=[]
            while True:
                try: events.append(hotkeys.get_nowait())
                except Empty: break
            if death_detector and death_detector.detect(frame,captured_at): events.append('death')
            if 'quit' in events: break
            if 'disarm' in events: controller.disarm()
            if 'arm' in events: controller.arm()
            if 'death' in events and buffer.items:
                controller.disarm()
                job=buffer.snapshot('death_marked',captured_at); memory.add_event(job)
                counters['deaths_marked']+=1
                if review_worker:
                    if reviewer.calls >= reviewer.max_calls or not review_worker.submit(job,replace=False):
                        counters['reviews_deferred']+=1
                episode+=1
                pipeline=VisualPipeline(frame,None,cfg.get('floor_y'),cfg.get('hud_px',0),episode)
                buffer=EpisodeBuffer(); decision=None; cached_memory=None
                print('Morte registrada. Após reaparecer, F6 para remarcar o personagem. Nenhuma causa foi presumida.')
            if 'reseed' in events and not headless:
                controller.disarm()
                box=tuple(map(int,cv2.selectROI('Remarque o jogador',frame,False)))
                cv2.destroyWindow('Remarque o jogador')
                if box[2]>0 and box[3]>0:
                    episode+=1
                    pipeline=VisualPipeline(frame,box,cfg.get('floor_y'),cfg.get('hud_px',0),episode)
                    buffer=EpisodeBuffer(); decision=None; cached_memory=None
            if review_worker and reviewer.calls < reviewer.max_calls and captured_at-last_review>=float(cfg.get('review_interval_s',10)) and len(buffer.items)>=4:
                job=buffer.snapshot('observation',captured_at)
                if review_worker.submit(job,replace=False):
                    memory.add_event(job); last_review=captured_at
            log.write(json.dumps({'observation':obs.to_dict(),'proposed':chosen.action,'input_armed':controller.armed,'gate':reason})+'\n')
            if not headless:
                overlay=draw_overlay(frame,obs,chosen.action)
                cv2.putText(overlay,'ARMED' if controller.armed else 'OBSERVE',(5,frame.shape[0]-6),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
                cv2.imshow('AI Player Lab',overlay)
                key=cv2.waitKey(1)&0xff
                if key in (27,ord('q')): break
                if key==ord('d'): hotkeys.put('death')
                if key==ord('r'): hotkeys.put('reseed')
            timings['loop_work_ms'].append((time.monotonic()-loop_start)*1000)
            time.sleep(max(0,1/fps-(time.monotonic()-loop_start)))
    finally:
        controller.close()
        if listener: listener.stop()
        if capture: capture.close()
        if video: video.release()
        if not headless: cv2.destroyAllWindows()
        for worker in (jev_worker,review_worker,memory_worker):
            if worker: worker.close()
        # Workers may be inside bounded HTTP calls; do not close their clients under them.
        log.close()
        summary={'kind':'live' if not video_path else 'replay','counters':dict(counters),
                 'metrics':{name:{'p50':round(float(np.percentile(values,50)),3),'p95':round(float(np.percentile(values,95)),3)}
                            for name,values in timings.items() if values},
                 'memory':memory.summary(),'machine':machine_report(),'output':str(out),
                 'jev_calls':jev_provider.calls if jev_provider else 0,
                 'vlm_calls':reviewer.calls if reviewer else 0,
                 'real_input_enabled':enable_input,'input_starts_disarmed':True}
        memory.close()
        (out/'session.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
