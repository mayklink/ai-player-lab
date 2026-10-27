
import pytest as _asset_pytest
from pathlib import Path as _AssetPath
if not (_AssetPath(__file__).parent / "fixtures" / "small-mario-stopped.png").is_file():
    _asset_pytest.skip("Optional game imagery is not distributed; see docs/ASSETS.md", allow_module_level=True)

import json
from pathlib import Path
import cv2
import httpx
from ai_player_lab.diagnostics import fixture_frame
from ai_player_lab.game_rules import MARIO_RULES
from ai_player_lab.mario_vision import LocalMarioVision
from ai_player_lab.perception import VisualPipeline
from ai_player_lab.providers import JevPolicy


def test_clear_path_is_not_an_instruction_to_wait():
    bodies=[]
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200,json={'answers':{'action':{'type':'choice','choice':'right','confidence':.7}}})
    frame,box=fixture_frame(0);obs=VisualPipeline(frame,box).process(frame,1)
    client=JevPolicy(api_key='test-only',transport=httpx.MockTransport(handle))
    try:
        assert client(obs).action=='right'
        instructions=bodies[0]['questions']['action']['instructions']
        assert 'empty objects list is NOT a reason to wait' in instructions
        assert MARIO_RULES in instructions
    finally:client.close()


def test_camera_scroll_does_not_turn_a_live_stage_into_unknown():
    root=Path(__file__).resolve().parents[1]
    vision=LocalMarioVision(root/'src/ai_player_lab/profiles/mario')
    frame=cv2.imread(str(root/'tests/fixtures/mario/phase-scrolled.png'))
    result=vision.observe(frame,1)
    assert result['scene']=='gameplay'
    assert result['evidence']['source']=='local_ocr'
    assert result['player_box'] is not None
    assert vision.observe(frame,1.2)['stable']


def test_intro_and_world_map_are_not_running_platform_stages():
    root=Path(__file__).resolve().parents[1]/'src/ai_player_lab/profiles/mario'
    for name in ['intro__welcome.png','map__house.png','map__stage1.png']:
        frame=cv2.imread(str(root/name))
        assert LocalMarioVision(root).observe(frame,1)['scene']!='gameplay'


def test_fast_canvas_capture_preserves_gameplay_recognition():
    root=Path(__file__).resolve().parents[1]
    frame=cv2.imread(str(root/'tests/fixtures/mario/phase-fast-capture.png'))
    result=LocalMarioVision(root/'src/ai_player_lab/profiles/mario').observe(frame,1)
    assert result['scene']=='gameplay' and result['player_box'] is not None


def test_compressed_hud_does_not_interrupt_a_moving_phase():
    root=Path(__file__).resolve().parents[1]
    vision=LocalMarioVision(root/'src/ai_player_lab/profiles/mario')
    # Actual failed captures from the 60-second run, alternated with a good
    # frame: scene continuity must follow current HUD evidence, not a latch.
    for index,name in enumerate(['phase-fast-capture.png','phase-hud-jpeg.png',
                                 'phase-hud-jpeg-alt.png','phase-hud-jpeg.png']):
        frame=cv2.imread(str(root/'tests/fixtures/mario'/name))
        result=vision.observe(frame,index*.05)
        assert result['scene']=='gameplay',name
        assert result['player_box'] is not None,name
        assert result['stable']==(index>0)
    # A real transition must still stop gameplay immediately.
    result=vision.observe(frame*0,.3)
    assert result['scene']=='loading' and not result['stable']
    intro=cv2.imread(str(root/'src/ai_player_lab/profiles/mario/intro__welcome.png'))
    assert vision.observe(intro,.4)['scene']=='intro'


def test_timer_label_without_digits_never_authorizes_gameplay():
    root=Path(__file__).resolve().parents[1]
    frame=cv2.imread(str(root/'tests/fixtures/mario/phase-hud-jpeg.png'))
    frame[24:30,154:168]=0
    vision=LocalMarioVision(root/'src/ai_player_lab/profiles/mario')
    assert vision.observe(frame,1)['scene']!='gameplay'
