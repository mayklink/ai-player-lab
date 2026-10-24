from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import cv2
import numpy as np

UPSTREAM_COMMIT = 'ca697eb9e681a6db657ecc395bb38731960e01f8'


def load_upstream_capture(root: Path, region: tuple[int,int,int,int]):
    """Reutiliza ScreenCapture original, sem executar main.py do Zelda."""
    file = root/'src'/'pta'/'capture'/'screen_capture.py'
    if not file.is_file():
        raise FileNotFoundError('Base ausente. Execute: python scripts/bootstrap_upstream.py')
    spec = importlib.util.spec_from_file_location('pta_upstream_capture',file)
    if spec is None or spec.loader is None: raise ImportError(str(file))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.ScreenCapture(region=region)


class UpstreamDeathDetector:
    def __init__(self, root: Path, templates: dict[str,str], threshold: float = 0.95) -> None:
        base = root/'src'/'pta'
        if not base.is_dir(): raise FileNotFoundError('Base Pixel-to-Action ausente')
        sys.path.insert(0,str(base.resolve()))
        from perception.template_match import TemplateMatchPerception
        self.detector = TemplateMatchPerception(templates=templates,threshold=threshold)
        self.last = -1e9

    def detect(self, frame: np.ndarray, now: float) -> bool:
        hits = self.detector.process(frame)['detections']
        if hits and now-self.last > 3.0:
            self.last = now; return True
        return False


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    if width < 128 or width > 1280: raise ValueError('Largura deve estar entre 128 e 1280')
    if frame.shape[1] == width: return frame.copy()
    height = max(1,round(frame.shape[0]*width/frame.shape[1]))
    return cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)


def calibrate(config_path: Path, width: int = 320) -> dict:
    import json
    from mss import mss
    with mss() as capture:
        monitor = capture.monitors[1]
        screen = np.asarray(capture.grab(monitor))[:,:,:3].copy()
    region = tuple(map(int,cv2.selectROI('1 - Selecione APENAS o jogo; Enter confirma',screen,False)))
    cv2.destroyAllWindows()
    if region[2] <= 0 or region[3] <= 0: raise ValueError('Calibração cancelada')
    x,y,w,h = region
    game = resize_frame(screen[y:y+h,x:x+w],width)
    player = tuple(map(int,cv2.selectROI('2 - Selecione o personagem completo; Enter',game,False)))
    cv2.destroyAllWindows()
    if player[2] <= 0 or player[3] <= 0: raise ValueError('Personagem não selecionado')
    result = {'capture_region':[x+monitor['left'],y+monitor['top'],w,h],
              'work_width':width,'player_box':list(player),'floor_y':None,
              'hud_px':0,'target_fps':30,'window_title':'','keys':{'left':'left','right':'right','jump':'space'},
              'upstream_path':'vendor/pixel-to-action','jev_model':'jev-1.14.0',
              'max_jev_calls':120,'jev_hz':3,'max_decision_age_ms':250,
              'vlm_base_url':'','vlm_model':'','max_vlm_calls':12,'vlm_timeout_s':30,
              'review_interval_s':10,'death_templates':{}}
    config_path.parent.mkdir(parents=True,exist_ok=True)
    config_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
