from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .diagnostics import machine_report, synthetic_benchmark


def main() -> int:
    from .configuration import load_local_environment
    load_local_environment()
    parser=argparse.ArgumentParser(description='AI Player Lab — protótipo experimental para jogos 2D')
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor'); doctor.add_argument('--out',type=Path,default=Path('runs/machine.json'))
    for name in ('demo','benchmark'):
        cmd=sub.add_parser(name); cmd.add_argument('--out',type=Path,default=Path('runs/synthetic'))
        cmd.add_argument('--frames',type=int,default=180); cmd.add_argument('--scroll',action='store_true')
    calibration=sub.add_parser('calibrate'); calibration.add_argument('--config',type=Path,default=Path('configs/local.json'))
    play=sub.add_parser('play',help='Luna navega, Jev joga; seleção apenas da área do jogo')
    play.add_argument('--config',type=Path,default=Path('configs/local.json'))
    play.add_argument('--seconds',type=float,default=180)
    play.add_argument('--enable-input',action='store_true')
    play.add_argument('--reuse-region',action='store_true')
    browser=sub.add_parser('browser-play',help='Chromium isolado, visão local, Jev e acompanhamento ao vivo')
    browser.add_argument('--url',default='https://www.smbgames.be/super-mario-for-snes.php')
    browser.add_argument('--game-id')
    browser.add_argument('--game-name')
    browser.add_argument('--port',type=int,default=8765)
    browser.add_argument('--seconds',type=float,default=120)
    browser.add_argument('--enable-input',action='store_true')
    browser.add_argument('--memory-mode',choices=['observe','influence'],default='influence')
    browser.add_argument('--memory-root',type=Path,default=Path('runs/learning'))
    browser.add_argument('--memory-baseline',type=Path)
    for name in ('observe','replay'):
        cmd=sub.add_parser(name)
        if name=='replay': cmd.add_argument('video',type=Path)
        cmd.add_argument('--config',type=Path,default=Path('configs/local.json'))
        cmd.add_argument('--seconds',type=float,default=120)
        cmd.add_argument('--jev',action='store_true'); cmd.add_argument('--vlm',action='store_true')
        cmd.add_argument('--headless',action='store_true')
        if name=='observe': cmd.add_argument('--enable-input',action='store_true')
    review=sub.add_parser('review-events')
    review.add_argument('--db',type=Path,default=Path('runs/mario-world-v1/memory.sqlite3'))
    review.add_argument('--config',type=Path,default=Path('configs/local.json'))
    review.add_argument('--limit',type=int,default=3)
    args=parser.parse_args()
    try:
        if args.command=='doctor':
            result=machine_report(); args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
        elif args.command in ('demo','benchmark'):
            if not 6 <= args.frames <= 3600: raise ValueError('frames deve ser 6..3600')
            result=synthetic_benchmark(args.out,args.frames,args.scroll)
        elif args.command=='calibrate':
            from .sources import calibrate
            result=calibrate(args.config)
        elif args.command=='play':
            from .play import run_play
            result=run_play(args.config,seconds=args.seconds,enable_input=args.enable_input,
                            reuse_region=args.reuse_region)
        elif args.command=='browser-play':
            import asyncio
            from .browser_runtime import run_browser
            result=asyncio.run(run_browser(url=args.url,port=args.port,seconds=args.seconds,enable_input=args.enable_input,
                memory_mode=args.memory_mode,memory_root=args.memory_root,memory_baseline=args.memory_baseline,game_id=args.game_id,game_name=args.game_name))
        elif args.command=='review-events':
            from .memory import MemoryStore
            from .providers import MultimodalReviewer
            if not 1<=args.limit<=20: raise ValueError('limit deve ser 1..20')
            if not args.db.exists(): raise ValueError('Banco de eventos não existe')
            cfg=json.loads(args.config.read_text(encoding='utf-8'))
            cfg['max_vlm_calls']=min(args.limit,int(cfg.get('max_vlm_calls',12)))
            model=MultimodalReviewer.from_config(cfg); store=MemoryStore(args.db)
            try:
                for job in store.pending_jobs(model.max_calls): store.attach_review(model(job))
                result=store.summary()
            finally: model.close(); store.close()
        else:
            from .runtime import run_session
            result=run_session(args.config,seconds=args.seconds,jev=args.jev,vlm=args.vlm,
                               enable_input=getattr(args,'enable_input',False),
                               video_path=getattr(args,'video',None),headless=args.headless)
        print(json.dumps(result,indent=2,ensure_ascii=False)); return 0
    except KeyboardInterrupt:
        print('Interrompido.'); return 130
    except Exception as exc:
        # Explicit errors, not silent success. No API secrets or provider bodies in stdout.
        print(f'Erro ({type(exc).__name__}): {exc if isinstance(exc,(ValueError,FileNotFoundError)) else "consulte configuração, dependências e disponibilidade dos serviços"}',file=sys.stderr)
        return 1

if __name__=='__main__': raise SystemExit(main())
