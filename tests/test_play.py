import numpy as np
import pytest
from ai_player_lab.diagnostics import fixture_frame
from ai_player_lab.play import seed_current
from ai_player_lab.control import SafeKeyboard


def test_handoff_seed_does_not_consume_runtime_timestamp():
    frame,box=fixture_frame(0)
    pipeline=seed_current(frame,frame,list(box),3)
    result=pipeline.process(frame,1.0)
    assert result.player is not None and result.episode==3


def test_handoff_rejects_player_no_longer_present():
    frame,box=fixture_frame(0)
    with pytest.raises(ValueError):
        seed_current(frame,np.zeros_like(frame),list(box),3)


def test_disarm_tries_all_keys_even_if_one_release_fails():
    class Keyboard:
        test_backend=True
        failed=True
        released=[]
        def release(self,key):
            if key=='x' and self.failed:raise OSError('simulated')
            self.released.append(key)
    keyboard=Keyboard()
    controller=SafeKeyboard('Mario',keyboard=keyboard,focus=lambda:'Mario')
    try:
        controller.held={'x','right'}
        with pytest.raises(OSError):controller.disarm()
        assert keyboard.released==['right']
        assert controller.held=={'x'}
        keyboard.failed=False
        controller.tick()
        assert not controller.held
    finally:controller.close()
