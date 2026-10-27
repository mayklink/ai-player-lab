"""Local headless acceptance check; no game or model calls."""
import asyncio,json,time
from pathlib import Path
from ai_player_lab.browser_game import BrowserGame


async def main():
    game=BrowserGame('about:blank',headed=False)
    try:
        await game.open()
        await game.page.set_content('<title>Input timing check</title><input autofocus>')
        await game.page.evaluate('''() => {window.__inputTimings=[];
            for(const type of ['keydown','keyup'])addEventListener(type,e=>window.__inputTimings.push(
                {type,key:e.key,time:performance.now(),trusted:e.isTrusted}),true);}''')
        await game.page.locator('input').focus()
        trials=[]
        for index in range(5):
            await game.page.evaluate('window.__inputTimings=[]')
            await game.set_keys({'ArrowRight'},100)
            # Block asyncio with I/O and actual Python CPU work in alternating trials.
            if index%2:
                deadline=time.perf_counter()+.3
                while time.perf_counter()<deadline:pass
            else:time.sleep(.3)
            events=await game.page.evaluate('window.__inputTimings')
            down=next(e for e in events if e['type']=='keydown')
            up=next(e for e in events if e['type']=='keyup')
            trials.append({'browser_actual_ms':round(up['time']-down['time'],3),
                'trusted':down['trusted'] and up['trusted'],'block_type':'busy_cpu' if index%2 else 'sleep'})
        report={'requested_ms':100,'blocked_main_loop_ms':300,'trials':trials,
            'min_actual_ms':min(t['browser_actual_ms'] for t in trials),'max_actual_ms':max(t['browser_actual_ms'] for t in trials),
            'trusted_native_events':all(t['trusted'] for t in trials),
            'input_released':not game.held,'worker_timings':list(game.native_input.timings)}
        Path('runs').mkdir(exist_ok=True)
        Path('runs/native-input-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
        assert report['input_released'] and report['trusted_native_events']
        assert 80<=report['min_actual_ms'] and report['max_actual_ms']<=150
    finally:await game.close()


asyncio.run(main())
