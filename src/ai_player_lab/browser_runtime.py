"""Visible browser, local Mario screen profile, Jev gameplay and bounded unknown-screen help."""
import asyncio
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from queue import Empty
import json
import hashlib
import time
import cv2
import numpy as np

from .browser_game import BrowserGame
from .browser_server import LiveServer
from .director import LunaDirector
from .features import exact_hash
from .mario_vision import LocalMarioVision, detect_player
from .perception import VisualPipeline
from .providers import JevPolicy
from .types import Track,center,iou,separate_actor
from .tracking import collision_estimate
from .goal import parse_goal
from .game_profiles import GameProfiles,available_actions,action_keys
from .character_state import CharacterStateEstimator
from .detections import DetectionOverlay
from .game_object_vision import MarioObjectDetector
from .death_review import DeathReviewer
from .experience import ExperienceMemory
from .encounters import EncounterRecorder,PrefixRecorder,review_frames
from .attribution import attribute_review
from .memory_worker import LatestMemory
from .action_replay import ActionReplay
from .local_controller import LocalController,reactive_replay_guard
from .event_evidence import select_event_frames,evidence_summary
from .attempts import AttemptCycle
from .latest_capture import LatestCapture
from .tracking_grace import recent_visual_lease
from .life_evidence import LifeCounter,life_loss_evidence
from .episode_outcome import review_episode


def review_with_baselines(reviewer,job,memory,mode):
    baselines=[]
    if mode=='influence':
        for candidate in job['candidates']:
            if candidate.get('context'):
                for baseline in memory.similar_successes(candidate['image'],str(job['stage']),candidate['context']):
                    if baseline['event_id'] not in {b['event_id'] for b in baselines}:baselines.append(baseline)
    job['successful_baselines']=baselines[:2];job['baseline_frames']=[]
    for baseline in baselines[:2]:
        for filename in [baseline.get('reference',''),(baseline.get('success_evidence') or {}).get('after_reference','')]:
            if filename and Path(filename).name==filename:
                image=cv2.imread(str(memory.root/filename))
                if image is not None:
                    ok,jpeg=cv2.imencode('.jpg',image)
                    if ok:job['baseline_frames'].append({'event_id':baseline['event_id'],'jpeg':jpeg.tobytes()})
    return {'review':review_episode(reviewer,job),'baselines':baselines[:2]}


async def run_browser(*,url:str,port:int=8765,seconds:float=120,enable_input:bool=False,headed:bool=True,
                      memory_mode:str='influence',memory_root:Path=Path('runs/learning'),memory_baseline:Path|None=None,game_id:str|None=None,game_name:str|None=None):
    if memory_mode not in {'observe','influence'}:raise ValueError('Invalid memory mode')
    if not 1<=seconds<=600:raise ValueError('Uma partida deve durar entre 1 e 600 segundos')
    root=Path('runs/browser')/datetime.now().strftime('%Y%m%d-%H%M%S')
    root.mkdir(parents=True,exist_ok=True)
    profiles=GameProfiles()
    game_profile=profiles.resolve(url,game_id,game_name)
    profiles.migrate_mario_legacy(game_profile,memory_root)
    controls=game_profile['controls']
    profile=Path(__file__).parent/'profiles'/game_profile['visual_adapter']
    character=CharacterStateEstimator(game_profile['learned_parameters'])
    character_status={};latencies=deque(maxlen=20);saved_samples=0
    base_samples=game_profile['learned_parameters'].get('jump_rise_samples',0)
    vision=LocalMarioVision(profile,appearance_root=profiles.directory(game_profile['game_id']),online_learning=False)
    game=BrowserGame(url,headed=headed);server=LiveServer(port)
    log=(root/'trace.jsonl').open('w',encoding='utf-8')
    server.start()
    jev=luna=None;jev_task=luna_task=None
    paused=True;shutdown=False;deadline=0.;key_deadline=0.;generation=0
    objective='Jogue Super Mario World e chegue Ã  fase 3'
    goal=parse_goal(objective)
    rules=game_profile['rules']
    selected_stage=None
    last_player=None;last_player_time=None;player_age=0;player_vx=player_vy=0.
    last_prepare=-1e9
    pipeline=None;last_scene='unknown';last_jev=last_luna=unknown_since=last_nav=last_profile=-1e9
    samples=deque(maxlen=3);counters=Counter();last_action='';last_effect='';last_summary=''
    pending_effect=None;last_frame=None;frame_times=deque(maxlen=30);luna_calls=jev_calls=0
    overlay_detector=DetectionOverlay();last_disk=-1e9
    memory=ExperienceMemory(profiles.learning_root(game_profile['game_id']),'game://'+game_profile['game_id'])
    recall_memory=ExperienceMemory(memory_baseline,'game://'+game_profile['game_id']) if memory_baseline else memory
    memory_worker=LatestMemory(recall_memory)
    episode_frames=deque(maxlen=384);episode_states=deque(maxlen=320)
    last_episode_sample=-1e9;phase_seen=False;review_task=None;review_job=None
    death_reviewer=None;episode_index=0;last_memory_ids=();latest_state={};episode_memory_ids=set()
    pending_experiences=[];pending_state={}
    encounters=EncounterRecorder();success_index=0
    prefixes=PrefixRecorder();replay=ActionReplay();replayed_ids=set()
    controller=LocalController();intent_action=None;intent_hypotheses=[]
    control_owned=False
    attempts=AttemptCycle(profiles.directory(game_profile['game_id'])/'attempts')
    level_entry_ready=False
    def publish_capture(packet):
        ok,encoded=cv2.imencode('.jpg',packet['frame'],[cv2.IMWRITE_JPEG_QUALITY,90])
        if ok:
            packet['jpeg']=encoded
            server.update(frame=encoded.tobytes(),frame_id=packet['seq'],frame_time=packet['captured_at'])
    capture=LatestCapture(game.capture,on_packet=publish_capture)
    last_control_obs=None
    life_counter=LifeCounter();life_frames=deque(maxlen=192)
    last_life_sample=-1e9;map_review_started=None
    native_replay_token=None
    def record(kind,**data):
        log.write(json.dumps({'time':time.monotonic(),'kind':kind,'generation':generation,**data},ensure_ascii=False)+'\n');log.flush()
    async def send_keys(keys,hold_ms=None,*,reason='policy_command'):
        record('input_command',keys=sorted(keys),hold_ms=hold_ms,reason=reason,
            replay_token=native_replay_token)
        await game.set_keys(keys,hold_ms,reason=reason)
    identity_gap_started=None;last_identity_sample=-1e9;last_metrics_sample=-1e9
    last_processed_capture=-1e9;last_object_sample=-1e9;last_patch_gate=None
    source_root=Path(__file__).resolve().parent
    source_files=[*source_root.glob('*.py'),source_root/'web'/'live.html']
    provenance={'source_root':str(source_root),'working_directory':str(Path.cwd()),
        'files':{str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in source_files if p.is_file()}}
    (root/'source-manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    record('source_configuration',source_root=str(source_root),working_directory=str(Path.cwd()))
    record('memory_configuration',mode=memory_mode,write_root=str(memory_root),baseline=str(memory_baseline) if memory_baseline else None)
    server.update(url=url,objective=objective,rules=rules,goal=goal.to_dict(),
        game_profile=game_profile,character_state=character_status)
    def release_reason(reason):
        nonlocal paused,pipeline,generation,last_summary
        paused=True;pipeline=None;generation+=1;last_summary=reason
        if review_task is None:attempts.clear()
        record('paused',reason=reason)
    try:
        await game.open()
        capture.start()
        record('browser_open',url=url)
        while not shutdown:
            tick=time.monotonic()
            commands=[]
            while True:
                try:commands.append(server.commands.get_nowait())
                except Empty:break
            for cmd in commands:
                action=cmd['action']
                if action=='shutdown':shutdown=True;break
                if action in {'pause','stop'}:
                    await send_keys(set());release_reason('Pausado pelo usuÃ¡rio' if action=='pause' else 'SessÃ£o encerrada pelo usuÃ¡rio')
                elif action=='start':
                    if not enable_input:
                        last_summary='Reinicie com --enable-input para permitir as teclas';continue
                    try:
                        new_goal=parse_goal(cmd.get('objective',objective))
                        new_url=cmd.get('url',game.url)
                        new_profile=profiles.resolve(new_url,cmd.get('game_id'),cmd.get('game_name'))
                        if review_task is not None:
                            raise ValueError('Aguarde a anÃ¡lise do episÃ³dio antes de iniciar outra tentativa')
                        new_rules=cmd.get('rules',new_profile['rules'])
                        if not isinstance(new_rules,str) or len(new_rules)>6000:raise ValueError('InstruÃ§Ãµes do jogo muito longas')
                        new_profile['rules']=new_rules.strip() or new_profile['rules']
                        profiles.save(new_profile)
                        if new_url!=game.url:
                            capture.invalidate();await game.navigate(new_url);capture.invalidate()
                        if new_profile['game_id']!=game_profile['game_id']:
                            await asyncio.to_thread(memory_worker.close)
                            memory=ExperienceMemory(profiles.learning_root(new_profile['game_id']),'game://'+new_profile['game_id'])
                            recall_memory=ExperienceMemory(memory_baseline,'game://'+new_profile['game_id']) if memory_baseline else memory
                            memory_worker=LatestMemory(recall_memory)
                            episode_frames.clear();episode_states.clear();phase_seen=False
                            character=CharacterStateEstimator(new_profile['learned_parameters']);saved_samples=0
                            base_samples=new_profile['learned_parameters'].get('jump_rise_samples',0)
                            selected_stage=None;last_memory_ids=();replayed_ids.clear()
                            life_frames.clear();last_control_obs=None
                            attempts=AttemptCycle(profiles.directory(new_profile['game_id'])/'attempts');level_entry_ready=False
                        character.reset();character_status={}
                        game_profile=new_profile;controls=game_profile['controls']
                        profile=Path(__file__).parent/'profiles'/game_profile['visual_adapter']
                        vision=LocalMarioVision(profile,appearance_root=profiles.directory(game_profile['game_id']),online_learning=False)
                        objective=new_goal.description;goal=new_goal;rules=game_profile['rules']
                    except (ValueError,TypeError) as exc:
                        last_summary=str(exc);continue
                    if jev is None:jev=JevPolicy(max_calls=max(180,int(seconds*3.0)),game_rules=rules)
                    else:jev.game_rules=rules
                    jev.available_actions=available_actions(controls)
                    if luna is None:
                        try:luna=LunaDirector(timeout_s=6,max_calls=3)
                        except ValueError:luna=None
                    await send_keys(set());await game.prepare()
                    paused=False;generation+=1;pipeline=None;deadline=time.monotonic()+seconds
                    attempts.clear()
                    life_frames.clear();last_control_obs=None
                    last_summary='VisÃ£o local reconhecendo o jogo';record('started',duration=seconds,memory_mode=memory_mode,game_id=game_profile['game_id'],memory_root=str(memory.root))
                elif action=='key' and paused:
                    before=await game.capture()
                    await game.pulse(cmd['key'],cmd.get('hold_ms',100))
                    last_action=cmd['key'];record('manual_key_sent',key=cmd['key'],hold_ms=cmd.get('hold_ms',100))
                    if before is not None:pending_effect=(before,time.monotonic(),last_action)
                elif action=='reset' and paused:
                    await send_keys(set())
                    capture.invalidate()
                    await game.page.get_by_role('button',name='Restart',exact=True).click(timeout=3000)
                    capture.invalidate()
                    record('reset_game')
                elif action=='inspect':
                    info=await game.discover();server.update(inspect=info)
                    record('dom_inspect',info=info)
            if shutdown:break
            now=time.monotonic()
            for key_event in game.drain_key_events():
                attempts.record_key_event(key_event)
                record('key_event',event=key_event,phase_id=attempts.phase_id)
            if not game.native_input and now>=key_deadline and game.held:await send_keys(set());record('keys_released',reason='hold_deadline')
            if not paused and now>=deadline:
                await send_keys(set());release_reason('Prazo de teste concluÃ­do')
            packet=await capture.next()
            frame=packet['frame'] if packet else None
            if frame is None:
                if now-last_prepare>2:
                    await game.prepare();last_prepare=now
                server.update(summary='Localizando a Ã¡rea do jogo e iniciando o emuladorâ€¦',paused=paused,running=not paused,
                              url=game.url,objective=objective,goal=goal.to_dict())
                await asyncio.sleep(.2);continue
            now=packet['captured_at'];last_frame=frame;obs=None
            if now<=last_processed_capture:
                counters['duplicate_capture_timestamp']+=1
                continue
            last_processed_capture=now
            jpeg=packet.get('jpeg')
            if jpeg is None:
                counters['encode_failed']+=1;continue
            if time.monotonic()-now>.15:
                if game.held or native_replay_token:await send_keys(set(),reason='stale_capture')
                counters['stale_capture_dropped']+=1
                continue
            processing_started=time.monotonic()
            if now-last_disk>=.5:cv2.imwrite(str(root/'snapshot.png'),frame)
            frame_times.append(now);counters['frames']+=1
            if now-last_profile>1:
                signature=tuple((p.name,p.stat().st_mtime_ns) for p in sorted(profile.glob('*.png'))) if profile.exists() else ()
                if signature!=getattr(vision,'signature',None):
                    vision=LocalMarioVision(profile,appearance_root=profiles.directory(game_profile['game_id']),online_learning=False);vision.signature=signature
                last_profile=now
            observed=await asyncio.to_thread(vision.observe,frame,now);scene=observed['scene'];stable=observed['stable']
            vision_finished=time.monotonic()
            confirmed=observed.get('player_box') is not None
            if scene=='gameplay' and not paused:
                counters['gameplay_frames']+=1
                if confirmed:counters['identity_confirmed_frames']+=1
                if not confirmed and identity_gap_started is None:
                    identity_gap_started=now
                    record('identity_lost',seq=packet['seq'],identity=observed.get('identity',{}))
                elif confirmed and identity_gap_started is not None:
                    record('identity_reacquired',seq=packet['seq'],gap_ms=round((now-identity_gap_started)*1000,2),
                        identity=observed.get('identity',{}))
                    identity_gap_started=None
                if now-last_identity_sample>=.20:
                    record('identity_sample',seq=packet['seq'],confirmed=confirmed,
                        identity=observed.get('identity',{}),player_box=observed.get('player_box'))
                    last_identity_sample=now
            elif identity_gap_started is not None:
                record('identity_gap_ended',reason='left_gameplay',gap_ms=round((now-identity_gap_started)*1000,2))
                identity_gap_started=None

            if native_replay_token and game.native_input and observed.get('player_box') is not None and scene=='gameplay' and stable:
                game.native_input.heartbeat(native_replay_token,now+.65)
            if game_profile['visual_adapter']!='mario':
                observed={'scene':'unknown','stable':False,'player_box':None,'evidence':{'reason':'game_visual_adapter_not_configured'}}
                scene='unknown';stable=False
            grace=recent_visual_lease(last_control_obs,time.monotonic(),episode=generation,scene=scene,
                stable=stable,paused=paused,identity=observed.get('identity')) if observed['player_box'] is None else 0
            if paused or scene!='gameplay' or not stable or (observed['player_box'] is None and not grace):
                if attempts.active:
                    attempts.next(now,None,valid=False)
                    if not attempts.active:record('timed_handoff',reason=attempts.reason)
                character.reset();character_status={}
                controller.stop('tracking_or_scene_lost');intent_action=None;intent_hypotheses=[]
                if control_owned and (game.held or native_replay_token):await send_keys(set(),reason='identity_or_scene_gap')
                control_owned=False
                prefixes.reset();encounters.reset()
                if replay.active:
                    replay.cancel();await send_keys(set());record('replay_stopped',reason='tracking_or_scene_lost')
            if scene=='map' and stable:
                level_entry_ready=True
                label=observed['evidence'].get('template','')
                import re
                stage_match=re.search(r'stage[_-]?(\d+)',label)
                if stage_match:selected_stage=int(stage_match.group(1))
            if stable and goal.observe(scene,selected_stage,observed['evidence'],now):
                await send_keys(set());release_reason('Objetivo alcanÃ§ado com evidÃªncia visual local')
                record('goal_completed',goal=goal.to_dict())
            if scene!=last_scene:
                record('local_scene',result=observed)
                generation+=1;pipeline=None;await send_keys(set(),reason='scene_changed');last_scene=scene
                control_owned=False
                last_player=None;last_player_time=None
                player_age=0;player_vx=player_vy=0.
                encounters.reset()
                character.reset();character_status={}
                controller.reset();intent_action=None;intent_hypotheses=[]
                prefixes.reset();replay.cancel()
                if scene in {'map','loading'}:replayed_ids.clear()
                unknown_since=now if scene=='unknown' else -1e9
            samples.append({'frame_hash':exact_hash(frame),'captured_at':now,'jpeg':jpeg.tobytes()})
            if not paused and game_profile['visual_adapter']=='mario' and scene in {'gameplay','map'} and now-last_life_sample>=.06:
                life_frames.append({**samples[-1],'frame_id':samples[-1]['frame_hash'],'scene':scene,'image':frame})
                last_life_sample=now
            if not paused and now-last_episode_sample>=.06:
                episode_frames.append({**samples[-1],'scene':scene,'image':frame.copy()})
                last_episode_sample=now
            if not paused and scene=='gameplay' and stable:phase_seen=True
            # Map return is a candidate, never proof of death or a learned rule.
            # Pause navigation while the reviewer distinguishes death/completion.
            if scene!='map':map_review_started=None
            map_review_pending=not paused and phase_seen and scene=='map' and stable and review_task is None
            if map_review_pending and map_review_started is None:
                map_review_started=now
            if map_review_pending:
                last_summary='Confirmando contador de vidas antes de avaliar a tentativa.'
            if map_review_pending and now-map_review_started>=.6:
                phase_seen=False;episode_index+=1
                event_id=root.name+'-'+str(episode_index)
                prior=[f for f in episode_frames if f['scene']=='gameplay']
                chosen=select_event_frames(episode_frames,{**samples[-1],'scene':scene,'image':frame.copy()},max_frames=12)
                if prior and chosen:
                    life_proof=await asyncio.to_thread(life_loss_evidence,list(life_frames),life_counter)
                    required_ids=set(life_proof.get('evidence_frame_ids',[]))
                    required=[f for f in life_frames if f['frame_hash'] in required_ids]
                    grounded=[f for f in episode_frames if f.get('scene')=='gameplay'
                        and f.get('context',{}).get('player')
                        and f.get('context',{}).get('character_state',{}).get('grounded') is True
                        and now-f['captured_at']<=12]
                    ground_refs=[]
                    for item in reversed(grounded):
                        if not ground_refs or ground_refs[-1]['captured_at']-item['captured_at']>=.2:
                            ground_refs.append(item)
                        if len(ground_refs)>=6:break
                    # Keep both HUD readings and the latest collision window.
                    chosen=sorted({f['frame_hash']:f for f in [*chosen[-8:],*required,*ground_refs]}.values(),key=lambda f:f['captured_at'])
                    life_proof={k:v for k,v in life_proof.items() if k!='observations'}
                    record('life_counter_evidence',event_id=event_id,evidence=life_proof)
                    event_root=root/'episodes'/event_id;event_root.mkdir(parents=True,exist_ok=True)
                    for index,item in enumerate(chosen):cv2.imwrite(str(event_root/f'{index:02d}.png'),item['image'])
                    review_job={'event_id':event_id,'event':'suspected_death_map_return',
                        'frames':[{k:f[k] for k in ('frame_hash','captured_at','jpeg')} for f in chosen],
                        'states':list(episode_states),'stage':selected_stage,
                        'attempt_context':attempts.context(),
                        'life_evidence':life_proof,
                        'game_rules':rules,'available_actions':available_actions(controls),
                        'used_experiences':sorted(episode_memory_ids),
                        'candidates':chosen,
                        'reference':prior[-min(8,len(prior))]}
                    phase_origin=review_job['attempt_context'].get('phase_started_at')
                    review_job['attempt_context']['anchor_candidates']=[{
                        'frame_id':f['frame_hash'],'at_s':round(f['captured_at']-phase_origin,3),
                        'grounded':f.get('context',{}).get('character_state',{}).get('grounded'),
                        'can_start_jump':f.get('context',{}).get('character_state',{}).get('can_start_jump')}
                        for f in chosen if phase_origin is not None and f['captured_at']>=phase_origin
                        and f.get('context',{}).get('player')]
                    (event_root/'episode.json').write_text(json.dumps({
                        'event_id':event_id,'event':review_job['event'],'stage':selected_stage,
                        'frames':[{k:f[k] for k in ('frame_hash','captured_at','scene')} for f in chosen],
                        'states':review_job['states']},ensure_ascii=False,indent=2),encoding='utf-8')
                    await send_keys(set());generation+=1;pipeline=None
                    try:
                        if death_reviewer is None:death_reviewer=DeathReviewer(timeout_s=12,max_calls=8)
                        review_task=asyncio.create_task(asyncio.to_thread(review_with_baselines,death_reviewer,review_job,memory,memory_mode))
                        record('death_review_requested',event_id=event_id,frames=len(chosen),states=len(episode_states))
                        record('event_evidence_coverage',event_id=event_id,coverage=evidence_summary(chosen))
                    except Exception as exc:record('death_review_error',error=type(exc).__name__,stage='prepare',
                        reason=str(exc)[:200] if isinstance(exc,ValueError) else None)
                episode_frames.clear();episode_states.clear();episode_memory_ids.clear()
            if review_task and review_task.done():
                try:
                    completed=review_task.result();review=completed['review']
                    record('success_failure_comparison',event_id=review_job['event_id'],baselines=completed['baselines'])
                    record('death_review_result',event_id=review_job['event_id'],review=review)
                    if review.get('outcome_source')=='local_life_counter':
                        counters['deaths_confirmed_local']+=1
                        record('death_confirmed_locally',event_id=review_job['event_id'],
                            evidence=review.get('life_evidence'),model_outcome=review.get('model_outcome'))
                    trial=attempts.finish(review,review_job['candidates'],allowed=set(available_actions(controls)))
                    record('attempt_cycle_review',event_id=review_job['event_id'],result=trial)
                    attributed=attribute_review(review,review_job['candidates'],review_job['states'])
                    memory.record_outcome([a['event_id'] for a in attributed],review['outcome'],review_job['event_id'],attributions=attributed)
                    record('local_outcome_attribution',event_id=review_job['event_id'],attributions=attributed,
                        episode_consulted_ids=review_job['used_experiences'])
                    (root/'episodes'/review_job['event_id']/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
                    if review['outcome']=='death' and review['confidence']>=.7 and review.get('actionable',True):
                        ref=next(f for f in review_job['candidates'] if f['frame_hash']==review['trigger_frame_id'])
                        if ref['scene']!='gameplay':raise ValueError('Learning trigger must be playable pre-failure scene')
                        context=ref.get('context') or next((s['state'] for s in reversed(review_job['states'])
                            if 0<=ref['captured_at']-s['state']['captured_at']<=.35),{})
                        entry=memory.add(review_job['event_id'],review,ref['image'],str(review_job['stage']),context)
                        if entry:
                            counters['deaths_reviewed']+=1
                            record('experience_saved',event_id=review_job['event_id'],entry=entry)
                except Exception as exc:record('death_review_error',error=type(exc).__name__,stage='validate_or_save',
                    reason=str(exc)[:200] if isinstance(exc,ValueError) else None)
                review_task=None;review_job=None
                attempts.clear()
            if pending_effect and now-pending_effect[1]>=.25:
                before,at,key=pending_effect
                change=float(np.mean(cv2.absdiff(before,frame)))
                before_player=detect_player(before);after_player=detect_player(frame,before_player)
                dx=round(after_player[0]-before_player[0],1) if before_player and after_player else None
                dy=round(after_player[1]-before_player[1],1) if before_player and after_player else None
                last_effect=f'Mario deslocou {dx}px / {dy}px' if dx is not None and (abs(dx)>1 or abs(dy)>1) else 'Sem deslocamento do Mario confirmado'
                record('input_effect',key=key,elapsed_ms=round((now-at)*1000),pixel_change=round(change,3),
                       player_dx=dx,player_dy=dy,before_hash=exact_hash(before),after_hash=exact_hash(frame))
                cv2.imwrite(str(root/f'effect-{counters["frames"]}.png'),frame)
                pending_effect=None
            if not paused and now>=deadline:
                await send_keys(set());release_reason('Prazo de teste concluÃ­do')
            if jev_task and jev_task.done():
                try:
                    reply=jev_task.result();jev_calls=jev.calls
                    age=time.monotonic()-reply.observed_at
                    latencies.append(age*1000)
                    reasons=[]
                    if paused:reasons.append('paused')
                    if not stable or scene!='gameplay':reasons.append('not_stable_gameplay')
                    if reply.episode!=generation:reasons.append('old_generation')
                    if not 0<=age<=.5:reasons.append('expired')
                    if observed['player_box'] is None and not grace:reasons.append('player_missing')
                    guidance=next((h for h in pending_state.get('hypotheses',[]) if h.get('source')=='retry_experiment'),None)
                    latest_guidance=attempts.guidance(time.monotonic())
                    if latest_guidance and not guidance and reply.observed_at<latest_guidance['issued_at']:
                        reasons.append('pre_learning_handoff')
                    if guidance:
                        retry_rejection=attempts.validate_correction_reply(reply.action,time.monotonic(),pending_state,
                            last_control_obs.to_dict() if last_control_obs is not None else {})
                        if retry_rejection:reasons.append(retry_rejection)
                    # The provider's self-reported confidence is diagnostic, not a calibrated success probability.
                    valid=not reasons
                    record('jev_result',action=reply.action,confidence=reply.confidence,age_ms=round(age*1000),
                           accepted=valid,rejected_by=reasons,reply_generation=reply.episode,scene=scene,
                           observation_seq=reply.seq,experiences=pending_experiences)
                    if valid:
                        if guidance and attempts.accept_correction(reply.action,time.monotonic(),guidance['decision_id']):
                            decision=attempts._correction
                            record('learning_correction_decided',decision_id=guidance['decision_id'],action=reply.action,
                                proposal=guidance['proposed_plan'],action_matches_proposal=decision['matches_proposal'],
                                decision_kind='proposal' if decision['matches_proposal'] else 'jev_alternative',
                                observation_seq=reply.seq)
                        controller.accept(reply.action,time.monotonic(),generation,reply.hold_ms)
                        intent_action=reply.action
                        intent_hypotheses=[h for h in pending_state.get('hypotheses',[]) if h.get('source')=='episodic_experience']
                        counters['accepted']+=1
                        record('control_intent',source='Jev',action=reply.action,observation_seq=reply.seq)
                    else:counters['jev_rejected']+=1
                except Exception as exc:record('jev_error',error=type(exc).__name__)
                jev_task=None
            if luna_task and luna_task.done():
                try:
                    result=luna_task.result();luna_calls=luna.calls
                    record('unknown_screen_advice',result=result)
                    if not paused and scene=='unknown' and result['generation']==generation:
                        last_summary='Luna: '+result['summary']
                        # Advice never authorizes gameplay or registers a local template.
                        if result['scene'] in {'title','menu','map'} and result['confidence']>=.85:
                            keys=controls
                            key=keys.get(result['action'])
                            if key:
                                await send_keys({key},result['hold_ms']);key_deadline=now+result['hold_ms']/1000
                                last_action='Luna: '+result['action'];pending_effect=(frame.copy(),now,last_action)
                                record('keys_sent',source='Luna_unknown',keys=[key],hold_ms=result['hold_ms'])
                except Exception as exc:record('luna_error',error=type(exc).__name__)
                luna_task=None
            if native_replay_token and game.native_input:
                for executed in game.native_input.drain_executions(native_replay_token):
                    if executed.get('cancelled'):
                        attempts.cancel(executed['cancelled']);record('timed_handoff',reason=executed['cancelled']);continue
                    if last_control_obs is None:continue
                    box=last_control_obs.player.bbox
                    attempts.record(executed['action'],executed['hold_ms'],executed['sent_at'],box,source=executed['source'])
                    episode_states.append({'action':executed['action'],'hold_ms':executed['hold_ms'],
                        'executed_at':executed['sent_at'],'controller_reason':'native_timed_replay','state':last_control_obs.to_dict()})
                    counters['replay_actions']+=1
                    if executed.get('patch'):counters['retry_actions_executed']+=1
                    record('replay_action',action=executed['action'],hold_ms=executed['hold_ms'],
                        source=executed['source'],patch=executed.get('patch',False),offset_s=executed['offset_s'],
                        executed_at=executed['sent_at'],native_late_ms=executed['late_ms'])
                if not attempts.active:native_replay_token=None
            if not paused and not attempts.active and scene=='gameplay' and observed['player_box'] is None and grace and review_task is None:
                remaining=recent_visual_lease(last_control_obs,time.monotonic(),episode=generation,scene=scene,
                    stable=stable,paused=paused,identity=observed.get('identity'))
                if remaining:
                    command=controller.update(last_control_obs,time.monotonic(),scene,stable,paused,
                        allow_ground_jump=False,allow_stuck_jump=False)
                    if command:
                        hold=min(remaining,command['hold_ms']);keys=action_keys(command['action'],controls)
                        await send_keys(keys,hold);control_owned=bool(keys)
                        record('tracking_grace_action',action=command['action'],hold_ms=hold,
                            based_on_seq=last_control_obs.seq,observation_age_ms=round((time.monotonic()-last_control_obs.captured_at)*1000,1),
                            current_identity_confirmed=False)
                        state=last_control_obs.to_dict();state['hypotheses'].append({'source':'tracking_grace','status':'last_confirmed_not_current'})
                        episode_states.append({'action':command['action'],'hold_ms':hold,
                            'executed_at':time.monotonic(),'controller_reason':'tracking_grace','state':state})
                        attempts.record(command['action'],hold,time.monotonic(),last_control_obs.player.bbox,source='tracking_grace')
                        if attempts.actions:attempts.actions[-1]['position_observed_at']=last_control_obs.captured_at
            if not paused and stable and review_task is None:
                if scene=='gameplay' and observed['player_box']:
                    if pipeline is None:
                        pipeline=VisualPipeline(frame,tuple(observed['player_box']),hud_px=32,episode=generation,
                            object_detector=MarioObjectDetector() if game_profile['visual_adapter']=='mario' else None,
                            classify_motion=True)
                        record('handoff',source='local_vision',player_box=observed['player_box'],evidence=observed['evidence'])
                    obs=await asyncio.to_thread(pipeline.process,frame,now,player_bbox=tuple(observed['player_box']))
                    box=observed['player_box']
                    vx=vy=0.
                    if last_player is not None and last_player_time is not None:
                        old=center(tuple(last_player));new=center(tuple(box));dt=max(.01,now-last_player_time)
                        vx=(new[0]-old[0])/dt;vy=(new[1]-old[1])/dt
                    player_age=player_age+1 if last_player_time is not None and now-last_player_time<=.2 else 1
                    player_vx=.25*vx+.75*player_vx if player_age>1 else 0.
                    player_vy=.25*vy+.75*player_vy if player_age>1 else 0.
                    identity=observed.get('identity',{})
                    player_label='masked_player_identity' if identity.get('status')=='confirmed' else 'local_mario_palette'
                    obs.player=Track(0,tuple(box),player_vx,player_vy,.9,age=player_age,label=player_label)
                    if attempts.started_at is None:
                        # The input timeline starts when it can be scheduled,
                        # not at the older capture timestamp. Processing a
                        # frame must not make the very first key already late.
                        begun=attempts.begin(frame,box,time.monotonic(),selected_stage,root.name+'-phase-'+str(generation),
                            allow_replay=memory_mode=='influence',trusted_entry=level_entry_ready and selected_stage is not None,
                            anticipatory=True,lead_s=max(.8,float(np.median(latencies))/1000+.3) if latencies else .8,initial_keys=game.held)
                        level_entry_ready=False
                        life_frames.clear()
                        record('attempt_started',replaying=begun,mode=attempts.mode,reason=attempts.reason,
                            phase_started_at=attempts.started_at,trusted_entry=attempts.trusted_entry,
                            retry_plan=attempts.plan,stage=selected_stage)
                    raw_objects=list(obs.objects)
                    obs.objects=[o for o in raw_objects if o.bbox[1]>=35
                        and 3<=o.bbox[2]<=128 and 3<=o.bbox[3]<=128 and o.bbox[2]*o.bbox[3]<=12000
                        and (o.label.startswith('enemy_') or o.label in {'independent_motion','independent_motion_predicted'})
                        and separate_actor(o,tuple(box))]
                    counters['unverified_objects_filtered']+=len(raw_objects)-len(obs.objects)
                    obs.hazards=[h for h in obs.hazards if h.get('kind')=='gap_candidate']+[
                        h for o in obs.objects if (h:=collision_estimate(obs.player,o,require_current_evidence=True))]
                    delay=float(np.median(latencies)) if latencies else 300.
                    character_status=character.update(frame,obs,game.held,delay,controls.get('jump',''))
                    approach=character_status.get('approaching_object') or {}
                    character_status.update(jump_deadline_ms=approach.get('jump_deadline_ms'),
                        deadline_source='visual_ttc_minus_decision_latency_and_rise' if approach else None,
                        decision_latency_ms=round(delay,1),latency_source='measured' if latencies else 'initial_estimate')
                    obs.character_state=dict(character_status)
                    if attempts.active or replay.active:
                        takeover=reactive_replay_guard(obs)
                        if takeover:
                            if attempts.active:attempts.cancel(takeover)
                            replay.cancel()
                            controller.resume_reactive(time.monotonic(),generation)
                            record('reactive_replay_takeover',reason=takeover,player_box=box)
                    handoff=attempts.policy_handoff(frame,box,time.monotonic(),obs.to_dict())
                    if handoff:
                        last_jev=-1e9
                        counters['learning_handoffs']+=1
                        record('learning_handoff',guidance=handoff,player_box=box)
                    last_control_obs=obs
                    if attempts.observe(now,box,obs.camera):
                        counters['timed_prefixes_saved']+=1
                        record('timed_prefix_promoted',context=attempts.context())
                    count=character_status.get('rise_samples',0)
                    if count>saved_samples:
                        game_profile['learned_parameters'].update(jump_rise_ms=character_status['jump_rise_time_ms'],
                            jump_rise_source='observed',jump_rise_samples=base_samples+count)
                        profiles.save(game_profile);saved_samples=count
                        record('character_calibration_saved',game_id=game_profile['game_id'],parameters=game_profile['learned_parameters'])
                    last_player=box;last_player_time=now
                    if episode_frames and episode_frames[-1]['captured_at']==now:
                        episode_frames[-1]['context']=obs.to_dict()
                    for passage in encounters.update(frame,obs,list(episode_states)):
                        success_index+=1;success_id=root.name+'-passage-'+str(success_index)
                        after_filename=success_id+'-after.png'
                        cv2.imwrite(str(memory.root/after_filename),frame)
                        passage['evidence'].update(after_reference=after_filename,end_frame_hash=exact_hash(frame))
                        try:
                            entry=memory.add_success(success_id,passage['frame'],str(selected_stage),
                                passage['context'],passage['actions'],passage['evidence'])
                        except ValueError as exc:
                            entry=None;record('memory_sample_rejected',event_id=success_id,reason=str(exc)[:160])
                        if entry:
                            counters['successful_passages']+=1
                            if attempts.confirm_passage(time.monotonic(),passage['evidence']):
                                record('learned_route_extended',phase_id=attempts.phase_id,evidence=passage['evidence'])
                            event_root=root/'episodes'/success_id;event_root.mkdir(parents=True,exist_ok=True)
                            cv2.imwrite(str(event_root/'before.png'),passage['frame']);cv2.imwrite(str(event_root/'after.png'),frame)
                            (event_root/'success.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
                            record('successful_passage_saved',event_id=success_id,evidence=passage['evidence'],actions=passage['actions'])
                    prefix=prefixes.update(frame,obs,list(episode_states))
                    if prefix:
                        success_index+=1;prefix_id=root.name+'-prefix-'+str(success_index)
                        after_filename=prefix_id+'-after.png';cv2.imwrite(str(memory.root/after_filename),frame)
                        prefix['evidence'].update(after_reference=after_filename,end_frame_hash=exact_hash(frame))
                        try:
                            entry=memory.add_prefix(prefix_id,prefix['frame'],str(selected_stage),prefix['context'],prefix['actions'],prefix['evidence'])
                        except ValueError as exc:
                            entry=None;record('memory_sample_rejected',event_id=prefix_id,reason=str(exc)[:160])
                        if entry:counters['prefixes_saved']+=1;record('safe_prefix_saved',event_id=prefix_id,evidence=prefix['evidence'],actions=prefix['actions'])
                    policy_due=(not attempts.active and not replay.active and jev and jev.calls<jev.max_calls
                        and jev_task is None and now-last_jev>=.25)
                    # A known route and its explicit retry context already
                    # identify the relevant memory. Do not run an expensive
                    # global image search while replay owns the inputs.
                    explicit_retry=attempts.guidance(time.monotonic())
                    if policy_due and explicit_retry is None:
                        memory_worker.submit(frame,str(selected_stage),obs.to_dict(),now,generation)
                    # Consume a ready result only on a frame that can actually
                    # deliver it to a policy decision or start a validated replay.
                    matches=memory_worker.ready(frame,str(selected_stage),obs.to_dict(),now,generation) if policy_due and explicit_retry is None else []
                    memory_pending=(policy_due and explicit_retry is None and memory_worker.future is not None
                        and (memory_worker._active or {}).get('kind')=='revalidate'
                        and now-(memory_worker.query or {}).get('time',now)<.12)
                    if memory_mode=='influence' and not attempts.active and not replay.active and jev_task is None:
                        for candidate in matches:
                            if candidate['event_id'] not in replayed_ids and replay.start(candidate,now,box,generation):
                                replayed_ids.add(candidate['event_id']);record('replay_started',event_id=candidate['event_id']);break
                    replay_was_active=replay.active
                    timed_was_active=attempts.active
                    timed_step=attempts.next(time.monotonic(),box,native=False)
                    if timed_was_active and not attempts.active:
                        controller.resume_reactive(time.monotonic(),generation)
                        record('timed_handoff',reason=attempts.reason)
                    step=timed_step or (None if attempts.active else replay.next(now,generation,box,scene,stable,obs.hazards,obs.objects))
                    replay_intent=bool(step or attempts.active or replay.active)
                    if step:
                        controller.accept(step['action'],time.monotonic(),generation,step['hold_ms'],bounded=True,replay=True)
                        record('replay_intent',**step)
                    elif replay_was_active and not replay.active:
                        record('replay_stopped',reason='completed_or_visual_guard')
                    command=controller.update(obs,time.monotonic(),scene,stable,paused,
                        allow_ground_jump=game_profile['local_controller'].get('allow_ground_jump',False),
                        allow_stuck_jump=game_profile['local_controller'].get('allow_stuck_jump',False))
                    if command:
                        action=command['action']
                        if replay_intent and step and action!=step['action']:
                            if attempts.active:attempts.cancel('reactive_override')
                            replay.cancel();replay_intent=False
                            record('reactive_replay_takeover',reason=command['reason'],requested_action=step['action'],action=action)
                        keys=action_keys(action,controls)
                        await send_keys(keys,command['hold_ms'])
                        if replay_intent:
                            counters['replay_actions']+=1
                            record('replay_action',action=action,hold_ms=command['hold_ms'],source='reactive_proven_replay',
                                offset_s=round(time.monotonic()-attempts.started_at,4),patch=False)
                        control_owned=bool(keys)
                        last_action='Local: '+action;counters['local_commands']+=1
                        state=obs.to_dict()
                        if action==intent_action:state['hypotheses'].extend(intent_hypotheses)
                        episode_states.append({'action':action,'hold_ms':command['hold_ms'],
                            'executed_at':time.monotonic(),'controller_reason':command['reason'],'state':state})
                        correction_executed=attempts.record(action,command['hold_ms'],time.monotonic(),box,source='local_controller')
                        if correction_executed:
                            counters['retry_actions_executed']+=1
                            record('learning_correction_executed',decision=attempts._correction,
                                action_matches_proposal=attempts._correction['matches_proposal'],keys=sorted(keys),
                                character_state=character_status,player_box=box,controller_reason=command['reason'])
                        applied=[h['event_id'] for h in state.get('hypotheses',[])
                            if h.get('source')=='episodic_experience' and h.get('action_authorized') is True
                            and h.get('suggested_action')==action]
                        if applied:counters['memory_actions_executed']+=1
                        record('keys_sent',source='local_controller',keys=sorted(keys),applied_experiences=applied,**command)
                        if keys and pending_effect is None:pending_effect=(frame.copy(),now,last_action)
                    if policy_due and not attempts.active and not replay.active and not step and not memory_pending:
                        obs.hypotheses=[{'source':'local_vision','scene':scene,'goal':objective,'stage':selected_stage,
                                         'screen_evidence':observed['evidence'],'player_identity':observed.get('identity',{}),'status':'observed'}]
                        retry_guidance=attempts.guidance(time.monotonic())
                        if retry_guidance:obs.hypotheses.append(retry_guidance)
                        obs.hypotheses.append({'source':'execution_feedback','status':'executed',
                            'phase_elapsed_s':round(now-attempts.started_at,3) if attempts.started_at is not None else None,
                            'recent_actions':[{k:a[k] for k in ('action','hold_ms','offset_s','source')}
                                for a in attempts.actions[-6:]],'retry_plan':attempts.plan,
                            'retry_patch_executed':attempts.patch_executed})
                        experiences=matches if memory_mode=='influence' else []
                        if experiences:counters['memory_policy_requests']+=1
                        obs.hypotheses.extend({'source':'episodic_experience',**item} for item in experiences)
                        if matches and memory_mode=='observe':
                            record('memory_shadow_match',event_ids=[h['event_id'] for h in matches])
                        ids=tuple(item['event_id'] for item in experiences)
                        episode_memory_ids.update(ids)
                        if ids and ids!=last_memory_ids:
                            record('experience_retrieved',event_ids=ids,player_box=box,
                                matches=[{'event_id':item['event_id'],'match_kind':item.get('match_kind'),
                                    'match_evidence':item.get('match_evidence')} for item in experiences])
                        last_memory_ids=ids
                        latest_state=obs.to_dict()
                        pending_state=latest_state
                        pending_experiences=[{'event_id':item['event_id'],'suggested_action':item.get('suggested_action'),
                            'match_kind':item.get('match_kind'),'action_authorized':item.get('action_authorized'),
                            'evidence_status':item.get('evidence_status')} for item in experiences]
                        jev_task=asyncio.create_task(asyncio.to_thread(jev,obs));last_jev=now
                        cv2.imwrite(str(root/f'jev-request-{counters["frames"]}.png'),frame)
                        record('jev_request',state=obs.to_dict())
                    if jev and jev.calls>=jev.max_calls and jev_task is None:
                        await send_keys(set());release_reason('Limite Jev atingido')
                elif scene=='gameplay' and pipeline is not None:
                    # Keep known enemy tracks alive while Mario identity is
                    # missing, but never turn object-only perception into input.
                    recent_player_box = (tuple(last_player) if last_player is not None and
                                         last_player_time is not None and now-last_player_time <= .2 else None)
                    obs=await asyncio.to_thread(pipeline.process,frame,now,
                        player_bbox=recent_player_box,allow_new_objects=False)
                    obs.player=None
                    obs.hazards=[]
                    counters['object_only_frames']+=1
                elif scene in {'title','file_menu','player_menu','intro','map','hub'} and now-last_nav>1.1 and not (scene=='map' and phase_seen):
                    label=observed['evidence'].get('template','')
                    key=controls.get('right' if scene=='hub' else 'start')
                    if scene=='map':key=controls.get('left' if 'house' in label else 'confirm')
                    elif scene=='intro':key=controls.get('start')
                    if key:
                        await game.focus();await send_keys({key},200);key_deadline=time.monotonic()+.2
                        last_action='VisÃ£o local: '+key;pending_effect=(frame.copy(),now,last_action);last_nav=now
                        record('keys_sent',source='local_navigation',keys=[key],hold_ms=200,scene=scene)
                        cv2.imwrite(str(root/f'nav-{counters["frames"]}.png'),frame)
                elif scene in {'loading','death'}:
                    await send_keys(set())
            if not paused and scene=='unknown' and now-unknown_since>=2 and now-last_luna>=10 and luna_task is None:
                await send_keys(set())
                if luna and luna.calls<luna.max_calls:
                    job={'frames':list(samples),'width':256,'height':224,'generation':generation,
                         'context':{'goal':objective+'. Identify the unknown screen and next navigation action. Never erase saves.',
                                    'game_rules':rules,
                                    'known_controls':controls,'game_id':game_profile['game_id'],'mode':'unknown_only'}}
                    luna_task=asyncio.create_task(asyncio.to_thread(luna,job));last_luna=now
                    cv2.imwrite(str(root/f'luna-unknown-{counters["frames"]}.png'),frame)
                else:release_reason('Reconhecimento inconclusivo: pausado para evitar comandos na tela errada')
            owner='Replay local' if replay.active or attempts.active else ('Jev' if not paused and scene=='gameplay' else ('Luna' if luna_task or review_task else 'VisÃ£o local'))
            if not paused:
                if review_task:
                    last_summary='Luna analisando o episÃ³dio anterior antes da prÃ³xima tentativa.'
                elif scene=='gameplay' and observed['player_box']:
                    last_summary='Mario localizado; Jev escolhe os movimentos seguindo as regras do jogo.'
                    if last_memory_ids:last_summary+=' ExperiÃªncia anterior disponÃ­vel para esta situaÃ§Ã£o.'
                elif scene=='gameplay':
                    last_summary='Personagem nÃ£o localizado; teclas soltas atÃ© reconhecer novamente.'
                elif scene=='unknown':
                    last_summary='Tela ainda nÃ£o reconhecida; verificando antes de continuar.'
                elif scene=='loading':
                    last_summary='Aguardando a transiÃ§Ã£o do jogo.'
            fps=(len(frame_times)-1)/(frame_times[-1]-frame_times[0]) if len(frame_times)>1 else 0
            if now-last_metrics_sample>=1:
                record('frame_metrics',seq=packet['seq'],scene=scene,capture_fps=round(capture.fps,2),
                    processing_fps=round(fps,2),capture_ms=round((packet['completed_at']-now)*1000,2),
                    vision_ms=round((vision_finished-processing_started)*1000,2),
                    processing_ms=round((time.monotonic()-processing_started)*1000,2),
                    capture_overwritten=capture.dropped,counters=dict(counters))
                last_metrics_sample=now

            if obs is not None and now-last_object_sample>=.1:
                record('object_track_sample',seq=packet['seq'],scene=scene,
                    player_box=observed['player_box'],objects=[{'id':o.id,'bbox':list(o.bbox),
                    'missed':o.missed,'label':o.label,'confidence':o.confidence,'age':o.age} for o in obs.objects])
                last_object_sample=now
            debug_player=observed['player_box']
            display,detections=overlay_detector.update(frame,now,scene,debug_player,observation=obs,reuse=True,
                player_mask=getattr(vision,'player_mask',None))
            _,display_jpeg=cv2.imencode('.jpg',display,[cv2.IMWRITE_JPEG_QUALITY,90])
            server.update(overlay=display_jpeg.tobytes(),overlay_id=packet['seq'],overlay_time=packet['captured_at'],running=not paused,paused=paused,owner=owner,scene=scene,
                summary=last_summary or 'Perfil local reconhece as telas; Luna sÃ³ recebe situaÃ§Ãµes desconhecidas.',
                calls={'jev':jev.calls if jev else 0,'luna':(luna.calls if luna else 0)+(death_reviewer.calls if death_reviewer else 0)},fps=round(fps,1),capture_fps=round(capture.fps,1),
                processing_fps=round(fps,1),frame_age_ms=round((time.monotonic()-now)*1000,2),
                processing_ms=round((time.monotonic()-processing_started)*1000,2),capture_overwritten=capture.dropped,
                learning={'mode':memory_mode,'reviewing':review_task is not None,'deaths_reviewed':counters['deaths_reviewed'],
                    'deaths_confirmed_local':counters['deaths_confirmed_local'],
                    'retry_actions_executed':counters['retry_actions_executed'],
                    'memory_policy_requests':counters['memory_policy_requests'],
                    'memory_actions_executed':counters['memory_actions_executed'],
                    'local_controller':controller.last_reason,
                    'attempt_cycle':{'mode':attempts.mode,'reason':attempts.reason,'plan':attempts.plan,
                        'patch_executed':attempts.patch_executed,'prefixes_saved':counters['timed_prefixes_saved']},
                    'memory_search':dict(memory_worker.stats),
                    'replay_active':replay.active,'replay_actions':counters['replay_actions'],'prefixes_saved':counters['prefixes_saved'],
                    'successful_passages':counters['successful_passages']},
                capture_method=game.capture_method,capture_ms=round(game.capture_ms,2),detections=detections,
                last_action=last_action,last_effect=last_effect,output=str(root),vision=observed,counters=dict(counters))
            server.update(url=game.url,objective=objective,rules=rules,goal=goal.to_dict(),game_profile=game_profile,character_state=character_status)
            if now-last_disk>=.5:
                if game.native_input:
                    timings=list(game.native_input.timings)
                    deadlines=[t['late_ms'] for t in timings if t['reason']=='deadline']
                    server.update(input_timing={'native_worker':True,'error':game.native_input.error,
                        'deadline_releases':len(deadlines),'max_late_ms':max(deadlines,default=0.)})
                    (root/'input-timing.json').write_text(json.dumps(timings,ensure_ascii=False),encoding='utf-8')
                (root/'status.json').write_text(json.dumps(server.status,ensure_ascii=False,indent=2),encoding='utf-8');last_disk=now
            await asyncio.sleep(max(0,1/60-(time.monotonic()-tick)))
    except Exception as exc:
        if type(exc).__name__=='TargetClosedError' or (isinstance(exc,RuntimeError) and
                str(exc)=='Capture producer failed: TargetClosedError'):
            record('browser_closed',reason='game_browser_closed')
        else:
            record('runtime_error',error=type(exc).__name__,reason=str(exc)[:240] if isinstance(exc,(RuntimeError,ValueError)) else None)
            raise
    finally:
        await capture.close()
        try:await send_keys(set())
        except Exception:pass
        for task in (jev_task,luna_task,review_task):
            if task:
                try:await task
                except Exception:pass
        if jev:jev.close()
        if luna:luna.close()
        if death_reviewer:death_reviewer.close()
        await asyncio.to_thread(memory_worker.close)
        await game.close()
        for key_event in game.drain_key_events():
            attempts.record_key_event(key_event)
            record('key_event',event=key_event,phase_id=attempts.phase_id)
        server.close();log.close()
        (root/'session.json').write_text(json.dumps({'counters':dict(counters),'jev_calls':jev.calls if jev else 0,
            'luna_calls':luna.calls if luna else 0,'status':'closed'},indent=2),encoding='utf-8')
    return {'output':str(root),'status':'closed'}
