"""One visible game, a screenshot director for navigation, and Jev for gameplay."""
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from queue import Empty, SimpleQueue
import json
import time
import cv2
import numpy as np

from .control import DryController, SafeKeyboard
from .desktop_target import RegionTarget
from .director import LunaDirector
from .features import exact_hash
from .live_panel import LivePanel
from .memory import EpisodeBuffer, MemoryStore
from .perception import VisualPipeline
from .providers import JevPolicy, choose_action
from .runtime import validate_config
from .session import ControlRouter
from .sources import calibrate, load_upstream_capture, resize_frame
from .types import Decision
from .workers import LatestWorker


def seed_current(source: np.ndarray, current: np.ndarray, box: list, generation: int) -> VisualPipeline:
    """A delayed bbox describes the submitted image, never the latest image."""
    pipeline = VisualPipeline(source, tuple(box), episode=generation)
    obs = pipeline.process(current, time.monotonic())
    if obs.player is None or obs.player.confidence < .7:
        raise ValueError('Mario mudou desde a captura; Luna precisa localizar novamente')
    return VisualPipeline(current,obs.player.bbox,episode=generation)


def run_play(config_path: Path, *, seconds: float = 180, enable_input: bool = False,
             reuse_region: bool = False) -> dict:
    if not 1 <= seconds <= 3600:
        raise ValueError('seconds deve estar entre 1 e 3600')
    if not reuse_region:
        calibrate(config_path, region_only=True)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(cfg)
    luna_limit = int(cfg.get('max_director_calls', 40))
    jev_limit = int(cfg.get('max_jev_calls', 180))
    if not 1 <= luna_limit <= 300 or not 1 <= jev_limit <= 1000:
        raise ValueError('Limites: Luna 1..300, Jev 1..1000 chamadas por sessão')
    luna = LunaDirector(base_url=cfg.get('vlm_base_url'),model=cfg.get('vlm_model'),
                        timeout_s=8,max_calls=luna_limit)
    try:
        jev = JevPolicy(cfg.get('jev_model','jev-1.14.0'),max_calls=jev_limit)
    except Exception:
        luna.close()
        raise
    events = SimpleQueue()
    capture = controller = panel = key_listener = mouse_listener = None
    luna_worker = jev_worker = None
    memory = trace = video = None
    counters = Counter(); timings = {'luna_ms': [], 'jev_ms': []}
    router = ControlRouter()
    router.controls.update(cfg.get('nav_keys', {}))
    router.controls.update(cfg.get('keys', {}))
    goal = cfg.get('goal','Play the selected game. Start a single-player game, enter the first available stage, and progress through stages.')
    output = Path('runs')/cfg.get('game_id','mario-world-v1')/datetime.now().strftime('play-%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    status = 'starting'; armed_requested = enable_input; paused = False
    generation_started = time.monotonic()
    recent_actions = deque(maxlen=8); frames = deque(maxlen=3); source_frames = {}
    pending_nav = None; nav_until = 0.; last_sample = last_luna = last_jev = last_panel = -1e9
    last_luna_done = -1e9; last_luna_error = ''; consecutive_errors = 0; last_evidence = -1e9
    pipeline = None; decision = None; buffer = EpisodeBuffer(); last_obs = None
    gameplay_capture_started = None; target = None
    def record(kind: str, **data) -> None:
        trace.write(json.dumps({'time': time.monotonic(), 'kind': kind, 'generation':router.generation,
                                'owner':router.owner, **data},ensure_ascii=False)+'\n')
        trace.flush()
    try:
        capture = load_upstream_capture(Path(cfg.get('upstream_path','vendor/pixel-to-action')),tuple(cfg['capture_region']))
        if enable_input:
            target = RegionTarget(cfg['capture_region'])
            controller = SafeKeyboard('selected-game-region',focus=lambda:'selected-game-region' if target.focused() else '')
        else:
            controller = DryController()
        panel = LivePanel(events,cfg['capture_region'])
        from pynput import keyboard, mouse
        def key_pressed(key):
            if key == keyboard.Key.f9: events.put('resume')
            elif key == keyboard.Key.f10: controller.disarm(); events.put('pause')
            elif key == keyboard.Key.esc: controller.disarm(); events.put('quit')
            elif key in {keyboard.Key.ctrl_l,keyboard.Key.ctrl_r,keyboard.Key.alt_l,keyboard.Key.alt_r,
                         keyboard.Key.cmd_l,keyboard.Key.cmd_r}:
                controller.disarm(); events.put('focus_lost')
        def clicked(x,y,button,pressed):
            if not pressed or not target: return
            if target.contains(x,y): events.put('game_clicked')
            else: controller.disarm(); events.put('focus_lost')
        def scrolled(*args):
            if controller.armed: controller.disarm(); events.put('focus_lost')
        key_listener = keyboard.Listener(on_press=key_pressed); key_listener.start()
        mouse_listener = mouse.Listener(on_click=clicked,on_scroll=scrolled); mouse_listener.start()
        luna_worker = LatestWorker(luna,'luna-director')
        jev_worker = LatestWorker(jev,'jev-gameplay')
        memory = MemoryStore(output.parent/'memory.sqlite3')
        trace = (output/'trace.jsonl').open('w',encoding='utf-8')
        status = 'running'; start = time.monotonic()
        record('start',input_enabled=enable_input,region=cfg['capture_region'],goal=goal)
        # --enable-input is the explicit Start opt-in; restore the selected game focus.
        if enable_input:
            events.put('resume')
        while time.monotonic()-start < seconds:
            tick = now = time.monotonic()
            commands = []
            while True:
                try: commands.append(events.get_nowait())
                except Empty: break
            if 'quit' in commands:
                status='stopped_by_user'; break
            if 'pause' in commands or 'focus_lost' in commands:
                controller.disarm(); paused=True; armed_requested=False
                router.reset('Pausado. Clique Retomar e depois no jogo, ou F9 sobre o jogo.')
                pipeline=None; decision=None; frames.clear(); pending_nav=None
                generation_started=now
                record('pause',reason='button_or_focus')
            if 'resume' in commands:
                armed_requested=True
                if not enable_input or target.activate():
                    events.put('game_clicked')
            if 'game_clicked' in commands and armed_requested:
                if not enable_input or (target.focused() and controller.arm()):
                    paused=False; armed_requested=False
                    router.reset('Reconhecendo a tela do jogo…'); generation_started=now
                    pipeline=None; decision=None; frames.clear(); pending_nav=None
                    record('resume')
            if enable_input and not controller.armed:
                # Initial click arms; after any focus loss an explicit resume is required.
                if not armed_requested: paused=True
            active = not paused and (not enable_input or controller.armed)
            frame = resize_frame(capture.get_frame(),int(cfg.get('play_work_width',640)))
            now=time.monotonic(); counters['frames']+=1
            if active and now-last_sample>=.25:
                ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
                if ok:
                    h=exact_hash(frame)
                    frames.append({'frame_hash':h,'captured_at':now,'jpeg':jpeg.tobytes()})
                    source_frames.pop(h,None)
                    source_frames[h]=frame.copy()
                    if len(source_frames)>32: source_frames.pop(next(iter(source_frames)))
                    last_sample=now
            result=luna_worker.poll()
            if result:
                timings['luna_ms'].append(result.elapsed_ms); last_luna_done=now
                if result.error:
                    counters['luna_errors']+=1; consecutive_errors+=1; last_luna_error=result.error
                    if active:
                        controller.apply(Decision('wait',now,0)); router.reset('Luna: falha na análise; aguardando nova captura')
                        pipeline=None; decision=None; generation_started=now; frames.clear()
                    record('luna_error',error=result.error,validation=getattr(luna,'last_validation_error',''))
                    if consecutive_errors>=3: status='luna_failed'; break
                elif active:
                    consecutive_errors=0; last_luna_error=''
                    reply=result.value
                    prepared=None
                    evidence=reply.get('evidence',[])
                    if (router.owner=='Luna' and reply['scene']=='gameplay' and reply['player_box']
                            and len(evidence)>=2 and evidence[-1]['captured_at']>evidence[-2]['captured_at']):
                        source=source_frames.get(reply['frame_hash'])
                        previous=source_frames.get(evidence[-2]['frame_hash'])
                        if source is not None and previous is not None:
                            try:
                                seed_current(source,previous,reply['player_box'],router.generation+1)
                                prepared=seed_current(source,frame,reply['player_box'],router.generation+1)
                            except ValueError:pass
                    change=router.accept(reply,now,temporal_confirmation=prepared is not None)
                    record('luna_result',result=reply,transition=change)
                    counters[change]+=1
                    if change=='handoff':
                        source=source_frames.get(reply['frame_hash'])
                        try:
                            if source is None: raise ValueError('Frame de origem expirou')
                            pipeline=prepared or seed_current(source,frame,reply['player_box'],router.generation)
                            if not {'left','right','jump'} <= router.controls.keys():
                                raise ValueError('Controles de movimento/pulo ainda desconhecidos')
                            controller.keys={k:router.controls[k] for k in ('left','right','jump')}
                            generation_started=now; decision=None; frames.clear()
                            buffer=EpisodeBuffer(); gameplay_capture_started=now
                            cv2.imwrite(str(output/'handoff.png'),frame)
                            if video is None:
                                video=cv2.VideoWriter(str(output/'gameplay.avi'),cv2.VideoWriter_fourcc(*'MJPG'),
                                                     float(cfg.get('target_fps',30)),(frame.shape[1],frame.shape[0]))
                            record('handoff',player_box=reply['player_box'],scene=router.scene)
                            counters['handoffs_completed']+=1
                        except ValueError:
                            counters['handoffs_failed']+=1
                            router.reset('Luna: localizar personagem e controles novamente')
                            pipeline=None; generation_started=now; frames.clear()
                    elif change in {'takeover','uncertain'}:
                        controller.apply(Decision('wait',now,0)); pipeline=None; decision=None
                        pending_nav=None; frames.clear(); generation_started=now
                    elif change=='navigate' and reply['action']!='wait':
                        key=router.controls.get(reply['action'])
                        if key:
                            pending_nav=key; nav_until=now+reply['hold_ms']/1000
                            recent_actions.append({'scene':reply['scene'],'action':reply['action'],'key':key})
                            record('navigation',action=reply['action'],key=key,hold_ms=reply['hold_ms'])
                            frames.clear()
                        else:
                            router.summary='Luna: falta identificar tecla para '+reply['action']
            if active:
                if router.owner=='Jev' and not router.lease_valid(now):
                    controller.apply(Decision('wait',now,0)); router.reset('Luna: confirmar se a fase continua')
                    pipeline=None; decision=None; frames.clear(); generation_started=now
                if router.owner=='Jev' and pipeline:
                    obs=pipeline.process(frame,now); last_obs=obs
                    counters['gameplay_frames']+=1
                    if video is not None:video.write(frame)
                    if now-last_evidence>=1:
                        cv2.imwrite(str(output/'latest-gameplay.png'),frame); last_evidence=now
                    if obs.player is None:
                        counters['player_lost']+=1
                        controller.apply(Decision('wait',now,0))
                        if buffer.items:
                            memory.add_event(buffer.snapshot('player_lost_or_transition',now))
                        router.reset('Luna retomou: personagem perdido ou mudança de tela')
                        pipeline=None; decision=None; frames.clear(); generation_started=now
                    else:
                        response=jev_worker.poll()
                        if response:
                            timings['jev_ms'].append(response.elapsed_ms)
                            if response.error:
                                counters['jev_errors']+=1; decision=None
                                record('jev_error',error=response.error)
                            elif response.value.episode==router.generation:
                                decision=response.value
                        if now-last_jev>=1/float(cfg.get('jev_hz',3)) and jev.calls<jev.max_calls:
                            obs.hypotheses=[{'source':'Luna','goal':goal,'summary':router.summary,
                                             'status':'visual_hypothesis'}]+memory.recall(pipeline.last_static,limit=2)
                            if jev_worker.submit(obs,replace=False): last_jev=now
                        chosen,reason=choose_action(obs,decision,time.monotonic(),cfg.get('play_decision_age_ms',450))
                        controller.apply(chosen); counters[reason]+=1
                        buffer.add(frame,pipeline.last_static,obs,chosen)
                        record('gameplay',observation=obs.to_dict(),action=chosen.action,gate=reason,armed=controller.armed)
                        if jev.calls>=jev.max_calls and not jev_worker.busy.is_set() and jev_worker.pending.empty():
                            status='jev_budget_reached'; break
                elif pending_nav and now<nav_until:
                    if enable_input: controller.apply_keys({pending_nav},max(1,min(250,int((nav_until-now)*1000))))
                else:
                    controller.apply(Decision('wait',now,0)); pending_nav=None
                interval=2.0 if router.owner=='Jev' else .7
                if (frames and now-last_luna>=interval and now>=nav_until+.25
                        and luna.calls<luna.max_calls and now-last_luna_done>.15):
                    job={'frames':list(frames),'width':frame.shape[1],'height':frame.shape[0],
                         'generation':router.generation,'context':{'goal':goal,'mode':router.owner,
                         'previous_scene':router.scene,'recent_actions':list(recent_actions),
                         'known_controls':router.controls}}
                    if luna_worker.submit(job,replace=False): last_luna=now
                if luna.calls>=luna.max_calls and not luna_worker.busy.is_set() and luna_worker.pending.empty():
                    status='luna_budget_reached'; break
            if now-last_panel>.15:
                panel.update(router.owner,router.scene,controller.armed,
                    router.summary if active or paused else 'Clique na imagem do jogo para começar.',
                    f'Luna {luna.calls}/{luna.max_calls} · Jev {jev.calls}/{jev.max_calls}')
                (output/'live.json').write_text(json.dumps({'owner':router.owner,'scene':router.scene,
                    'armed':controller.armed,'summary':router.summary,'counters':dict(counters),
                    'luna_calls':luna.calls,'jev_calls':jev.calls},ensure_ascii=False),encoding='utf-8')
                last_panel=now
            time.sleep(max(0,1/float(cfg.get('target_fps',30))-(time.monotonic()-tick)))
        if status=='running':status='time_limit'
    except Exception as exc:
        status='failed_'+type(exc).__name__
        if trace:record('session_error',error=type(exc).__name__)
        raise
    finally:
        if controller: controller.close()
        if key_listener:key_listener.stop()
        if mouse_listener:mouse_listener.stop()
        if capture:capture.close()
        if video is not None:video.release()
        if panel:panel.close()
        for worker in (luna_worker,jev_worker):
            if worker:worker.close(timeout=9)
        # Do not close a client beneath a still-running bounded network call.
        if not luna_worker or not luna_worker.thread.is_alive():luna.close()
        if not jev_worker or not jev_worker.thread.is_alive():jev.close()
        report={'status':status,'counters':dict(counters),'luna_calls':luna.calls,'jev_calls':jev.calls,
                'input_enabled':enable_input,'last_owner':router.owner,'last_scene':router.scene,
                'last_luna_error':last_luna_error,'gameplay_capture_started':gameplay_capture_started,
                'memory':memory.summary() if memory else {},'output':str(output),
                'metrics':{k:{'p50':round(float(np.percentile(v,50)),2),'p95':round(float(np.percentile(v,95)),2)}
                           for k,v in timings.items() if v}}
        (output/'session.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if memory:memory.close()
        if trace:trace.close()
    return report
