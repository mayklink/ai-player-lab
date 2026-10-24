from __future__ import annotations
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
import psutil
import cv2
import numpy as np
from .perception import VisualPipeline


def machine_report() -> dict:
    vm = psutil.virtual_memory()
    result = {'os':platform.platform(),'python':sys.version.split()[0],
              'cpu_name':platform.processor(),'physical_cores':psutil.cpu_count(logical=False),
              'logical_cores':psutil.cpu_count(),'ram_gib':round(vm.total/2**30,2),
              'available_ram_gib':round(vm.available/2**30,2),'opencv':cv2.__version__,
              'numpy':np.__version__,'git_found':bool(shutil.which('git')),
              'model_execution':'remote_api',
              'vlm_key_present':bool(os.getenv('VLM_API_KEY')),
              'vlm_url_env_present':bool(os.getenv('VLM_BASE_URL')),
              'vlm_model_env_present':bool(os.getenv('VLM_MODEL')),
              'typesafe_key_present':bool(os.getenv('TYPESAFE_API_KEY')),
              'gpu':None, 'not_a_gameplay_validation':True}
    if platform.system()=='Windows':
        command = "$cpu=Get-CimInstance Win32_Processor | Select-Object Name; $gpu=Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion; @{cpu=$cpu;gpu=$gpu} | ConvertTo-Json -Depth 4 -Compress"
        try:
            raw = subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,timeout=12)
            result['windows_hardware'] = json.loads(raw)
        except (OSError,subprocess.SubprocessError,json.JSONDecodeError):
            result['hardware_query_error']='Falha ao consultar CIM'
    return result


def fixture_frame(index: int, width: int = 320, height: int = 180, scroll: bool = False) -> tuple[np.ndarray,tuple]:
    """Fixture procedural de laboratório. Não é Mario, não simula aprendizado."""
    rng = np.random.default_rng(17)
    background = rng.integers(25,65,(height,width),dtype=np.uint8)
    for x in range(0,width,28):
        cv2.rectangle(background,(x,45),(x+12,70),95,1)
    frame = cv2.cvtColor(background,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame,(0,145),(width-1,height-1),(95,110,95),-1)
    cv2.line(frame,(0,145),(width-1,145),(220,220,220),1)
    # actor pattern has texture and does not change between frames
    px = 40 + min(index,70)
    cv2.rectangle(frame,(px,122),(px+13,143),(180,180,220),-1)
    cv2.line(frame,(px+3,125),(px+9,139),(30,30,30),2)
    ex = 265-min(index*2,190)
    cv2.rectangle(frame,(ex,127),(ex+13,143),(170,200,170),-1)
    cv2.circle(frame,(ex+4,132),2,(10,10,10),-1)
    if scroll:
        offset = -min(index,45)
        frame = cv2.warpAffine(frame,np.float32([[1,0,offset],[0,1,0]]),(width,height),borderMode=cv2.BORDER_REPLICATE)
        px += offset
    return frame,(px,122,14,22)


def draw_overlay(frame: np.ndarray, obs, action: str = 'wait') -> np.ndarray:
    out = frame.copy()
    for t in ([obs.player] if obs.player else [])+obs.objects:
        x,y,w,h = map(int,t.bbox)
        cv2.rectangle(out,(x,y),(x+w,y+h),(240,230,30) if t.id==0 else (80,220,240),1)
        cv2.putText(out,f'{t.id}: {t.vx:.0f}px/s',(max(0,x-2),max(16,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.33,(240,240,240),1)
    cv2.rectangle(out,(0,0),(out.shape[1],29),(15,15,15),-1)
    cv2.putText(out,f'FRAME {obs.seq} | {action} | objects {len(obs.objects)}',(5,12),cv2.FONT_HERSHEY_SIMPLEX,.34,(230,230,230),1)
    cv2.putText(out,'EXPERIMENTAL - not validated gameplay',(5,24),cv2.FONT_HERSHEY_SIMPLEX,.31,(230,230,230),1)
    return out


def synthetic_benchmark(output: Path, frames: int = 180, scroll: bool = False) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    first,box = fixture_frame(0,scroll=scroll)
    cv2.setNumThreads(2)
    pipeline = VisualPipeline(first,box,floor_y=145)
    samples=[]; valid=0; errors=[]; observed_tracks=set(); observations=[]
    writer=cv2.VideoWriter(str(output/'synthetic-overlay.avi'),cv2.VideoWriter_fourcc(*'MJPG'),30,(320,180))
    if not writer.isOpened(): raise RuntimeError('Codec MJPG indisponível')
    try:
        for i in range(frames):
            frame,expected = fixture_frame(i%75,scroll=scroll)
            if i%75 == 0 and i:
                pipeline = VisualPipeline(frame,expected,floor_y=145)
            start=time.perf_counter()
            obs=pipeline.process(frame,1+i/30)
            elapsed=(time.perf_counter()-start)*1000
            if i%75 >= 5: samples.append(elapsed)
            if obs.player:
                valid+=1
                errors.append(abs(obs.player.bbox[0]-expected[0]))
            observed_tracks.update(o.id for o in obs.objects)
            observations.append(obs.to_dict())
            overlay=draw_overlay(frame,obs)
            writer.write(overlay)
            if i == min(frames-1,30): cv2.imwrite(str(output/'synthetic-overlay.png'),overlay)
    finally: writer.release()
    if not samples: raise ValueError('Use pelo menos 6 frames')
    report={'kind':'synthetic_vision_only','frames':frames,'resolution':[320,180],
            'scroll':scroll,'processing_p50_ms':round(float(np.percentile(samples,50)),3),
            'processing_p95_ms':round(float(np.percentile(samples,95)),3),
            'player_detected_fraction':round(valid/frames,4),
            'player_x_mean_absolute_error_px':round(float(np.mean(errors)),3) if errors else None,
            'track_ids_created':len(observed_tracks),'machine':machine_report(),
            'excludes':['screen_capture','real_game','input_injection','Jev API','VLM inference'],
            'warning':'Not the user PC. No inference about FPS on RX 580. Synthetic actors have fixed appearances.'}
    (output/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'observations.jsonl').write_text('\n'.join(json.dumps(o) for o in observations),encoding='utf-8')
    return report
