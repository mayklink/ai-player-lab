from __future__ import annotations
import importlib.util
import base64
import binascii
import sys
from pathlib import Path
import cv2
import numpy as np
from .types import BBox

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


def encode_player_template(frame: np.ndarray, box: BBox) -> dict:
    """Persist only the manually selected appearance, not a new crop on startup."""
    from .perception import SeededPlayer
    anchor = SeededPlayer(frame, box).anchor
    ok, png = cv2.imencode('.png', anchor)
    if not ok:
        raise ValueError('Falha ao guardar template da calibração')
    return {'version':1, 'frame_size':[frame.shape[1], frame.shape[0]],
            'bbox':list(map(int, box)), 'png_base64':base64.b64encode(png).decode('ascii')}


def decode_player_template(cfg: dict, frame: np.ndarray | None = None) -> np.ndarray | None:
    if cfg.get('player_box') is None:
        return None
    record = cfg.get('player_template')
    if not isinstance(record, dict):
        raise ValueError('Configuração antiga sem aparência do jogador; execute calibrate novamente')
    try:
        size, box = record['frame_size'], record['bbox']
        if (record['version'] != 1 or not isinstance(size, list) or len(size) != 2
                or any(type(x) is not int or x <= 0 for x in size)
                or box != cfg['player_box'] or len(box) != 4
                or any(type(x) is not int for x in box)):
            raise ValueError('Metadados do template inválidos; execute calibrate novamente')
        x,y,w,h = box
        if min(x,y) < 0 or min(w,h) < 4 or x+w > size[0] or y+h > size[1]:
            raise ValueError('Caixa do template fora da calibração')
        if size[0] != int(cfg.get('work_width', size[0])):
            raise ValueError('Resolução de trabalho mudou; execute calibrate novamente')
        if frame is not None and size != [frame.shape[1],frame.shape[0]]:
            raise ValueError('Tamanho do frame difere da calibração; execute calibrate novamente')
        encoded = record['png_base64']
        if not isinstance(encoded, str) or len(encoded) > 1_000_000:
            raise ValueError('Template de calibração inválido ou excessivo')
        raw = base64.b64decode(encoded, validate=True)
        if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Template da calibração precisa ser PNG')
        # Check PNG dimensions before decoding, avoiding unexpected allocations.
        if len(raw) < 24 or (int.from_bytes(raw[16:20],'big'), int.from_bytes(raw[20:24],'big')) != (w,h):
            raise ValueError('Tamanho do PNG incompatível com a calibração')
        anchor = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        if anchor is None or anchor.shape != (h,w) or float(anchor.std()) < 5:
            raise ValueError('Template inválido ou sem textura; execute calibrate novamente')
        return anchor
    except (KeyError, TypeError, binascii.Error, cv2.error) as exc:
        raise ValueError('Template corrompido; execute calibrate novamente') from exc


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
    result['player_template'] = encode_player_template(game, player)
    config_path.parent.mkdir(parents=True,exist_ok=True)
    config_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
