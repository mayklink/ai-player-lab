"""Generate a fresh synthetic replay/config; never migrates real calibration silently."""
from pathlib import Path
import argparse
import json
import cv2
from ai_player_lab.diagnostics import fixture_frame
from ai_player_lab.sources import encode_player_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/synthetic-replay'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    first, box = fixture_frame(0)
    writer = cv2.VideoWriter(str(args.out/'input.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (320,180))
    if not writer.isOpened(): raise RuntimeError('Codec MJPG indisponível')
    try:
        for i in range(75): writer.write(fixture_frame(i)[0])
    finally:
        writer.release()
    config = {'game_id':'synthetic-replay', 'capture_region':[0,0,320,180],
              'work_width':320, 'player_box':list(box),
              'player_template':encode_player_template(first, box),
              'target_fps':30, 'floor_y':145, 'hud_px':0, 'window_title':'',
              'max_decision_age_ms':250}
    (args.out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(args.out.resolve())


if __name__ == '__main__': main()
