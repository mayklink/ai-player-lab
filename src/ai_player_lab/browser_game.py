"""An isolated, visible Chromium. The game interface is pixels + page key events only."""
import asyncio
import base64
from pathlib import Path
import time
import socket
from urllib.parse import urlsplit
import cv2
import numpy as np

ALLOWED_KEYS = {'Enter','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','x','z','a','s','v'}


class BrowserGame:
    def __init__(self, url: str, *, headed: bool = True) -> None:
        self.native_input=None
        self.key_events=[]
        self.url=url; self.headed=headed; self.held=set(); self.canvas=None
        self.release_task=None
        self.capture_method='element_screenshot';self.capture_ms=0.;self._pixel_failures=0
        self._last_pixel_probe=-1e9
        self.cdp=None;self._clip=None;self._clip_time=0.
        self.playwright=self.browser=self.context=self.page=None

    @property
    def held(self):
        return self.native_input.held.copy() if self.native_input else self._held

    @held.setter
    def held(self,value):self._held=value

    async def open(self):
        from playwright.async_api import async_playwright
        self.playwright=await async_playwright().start()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',0));debug_port=probe.getsockname()[1]
        launch_args=[f'--remote-debugging-port={debug_port}','--remote-debugging-address=127.0.0.1']
        try:
            self.browser=await self.playwright.chromium.launch(headless=not self.headed,chromium_sandbox=True,args=launch_args)
        except Exception as exc:
            if 'spawn UNKNOWN' not in str(exc):raise
            # Use only the installed binary, with Playwright's NEW temporary profile.
            # Never connect to an existing browser or read the user's profile.
            self.browser=await self.playwright.chromium.launch(channel='chrome',headless=not self.headed,chromium_sandbox=True,args=launch_args)
        self.context=await self.browser.new_context(viewport={'width':1100,'height':850},device_scale_factor=1)
        # Retain rendered WebGL pixels for capture in this isolated browser only.
        # This changes buffer lifetime, not the game state or available information.
        await self.context.add_init_script('''(() => {
            const getContext=HTMLCanvasElement.prototype.getContext;
            HTMLCanvasElement.prototype.getContext=function(kind,options) {
                if (['webgl','webgl2','experimental-webgl'].includes(kind))
                    options={...(options||{}),preserveDrawingBuffer:true};
                return getContext.call(this,kind,options);
            };
        })();''')
        await self.context.add_init_script('''(() => {
            window.__inputTimings=[];
            for (const type of ['keydown','keyup']) addEventListener(type,e=>{
                if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Enter','x','z','a','s','v'].includes(e.key))return;
                window.__inputTimings.push({type,key:e.key,time:performance.now(),trusted:e.isTrusted});
                if(window.__inputTimings.length>1024)window.__inputTimings.shift();
            },true);
        })();''')
        import re
        await self.context.route(re.compile(r'https?://[^/]*(?:doubleclick|googlesyndication|googleadservices|adservice)[^/]*?/'),lambda route:route.abort())
        self.page=await self.context.new_page()
        await self.page.goto(self.url,wait_until='domcontentloaded',timeout=60000)
        session=await self.context.new_cdp_session(self.page)
        target=(await session.send('Target.getTargetInfo'))['targetInfo']['targetId']
        await session.detach()
        def input_url():
            import httpx
            with httpx.Client(trust_env=False,timeout=3) as client:
                targets=client.get(f'http://127.0.0.1:{debug_port}/json/list').json()
            return next(t['webSocketDebuggerUrl'] for t in targets if t['id']==target)
        from .timed_input import CDPKeyboard,TimedInput
        ws_url=await asyncio.to_thread(input_url)
        self.native_input=TimedInput(lambda:CDPKeyboard(ws_url))
        await asyncio.wrap_future(self.native_input.ready)

    async def navigate(self,url:str):
        parsed=urlsplit(url)
        if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('URL deve ser HTTP(S), sem credenciais')
        await self.set_keys(set())
        self.url=url;self.canvas=None
        await self.page.goto(url,wait_until='domcontentloaded',timeout=60000)

    async def prepare(self):
        """Start a visible emulator shell using DOM controls, not game RAM."""
        import re
        for frame in self.page.frames:
            try:
                for candidate in await frame.get_by_text(re.compile(r'^(Start Game|Play Now|Click to Play|Play Game)$',re.I)).all():
                    if await candidate.is_visible():
                        await candidate.click(timeout=1500)
                        return True
            except Exception:continue
        return False

    async def discover(self) -> dict:
        canvases=[]; buttons=[]
        for index,frame in enumerate(self.page.frames):
            try:
                for node in await frame.locator('canvas').all():
                    if await node.is_visible():
                        box=await node.bounding_box()
                        if box:
                            in_game=await node.evaluate("n => !!n.closest('#game,.ejs_parent,.game-container,[data-game]')")
                            area=box['width']*box['height']
                            if area>=4096:
                                canvases.append((area*(5 if in_game else 1),node,{'frame':index,'in_game_container':in_game,**box}))
                buttons.extend((await frame.locator('button, [role=button]').all_text_contents())[:30])
            except Exception:pass
        if canvases:
            canvases.sort(key=lambda x:x[0],reverse=True)
            self.canvas=canvases[0][1]
        return {'url':self.page.url,'title':await self.page.title(),'canvases':[x[2] for x in canvases],
                'buttons':[x.strip()[:100] for x in buttons if x.strip()],
                'game_text':(await self.page.locator('#game').inner_text(timeout=1500))[:1000] if await self.page.locator('#game').count() else ''}

    async def capture(self) -> np.ndarray | None:
        if self.canvas is None:
            await self.discover()
            if self.canvas is None:return None
        try:
            start=time.perf_counter()
            if self._pixel_failures<2 or time.monotonic()-self._last_pixel_probe>1:
                self._last_pixel_probe=time.monotonic()
                # Pixels only: no RAM, sprite metadata, or internal game state.
                data=await self.canvas.evaluate('''(node,sync) => new Promise(resolve => {
                  const capture=() => {
                    try {
                        const output=document.createElement('canvas'); output.width=256;output.height=224;
                        const ctx=output.getContext('2d');ctx.imageSmoothingEnabled=false;
                        ctx.drawImage(node,0,0,256,224);
                        resolve(output.toDataURL('image/png').split(',')[1]);
                    } catch(e) { resolve(null); }
                  }; if(sync) requestAnimationFrame(capture); else capture();
                })''',self._pixel_failures>=2)
                frame=cv2.imdecode(np.frombuffer(base64.b64decode(data),np.uint8),cv2.IMREAD_COLOR) if data else None
                if frame is not None and np.count_nonzero(frame)>50:
                    self._pixel_failures=0
                    self.capture_method='canvas_pixels';self.capture_ms=(time.perf_counter()-start)*1000
                    return frame
                self._pixel_failures+=1
            try:
                if self.cdp is None:self.cdp=await self.context.new_cdp_session(self.page)
                if self._clip is None or time.monotonic()-self._clip_time>1:
                    self._clip=await self.canvas.bounding_box();self._clip_time=time.monotonic()
                if self._clip:
                    clip={k:self._clip[k] for k in ('x','y','width','height')}
                    clip['scale']=1
                    result=await self.cdp.send('Page.captureScreenshot',{'format':'png',
                        'clip':clip,'fromSurface':True,'optimizeForSpeed':True,'captureBeyondViewport':False})
                    frame=cv2.imdecode(np.frombuffer(base64.b64decode(result['data']),np.uint8),cv2.IMREAD_COLOR)
                    self.capture_method='cdp_png';self.capture_ms=(time.perf_counter()-start)*1000
                    return cv2.resize(frame,(256,224),interpolation=cv2.INTER_NEAREST)
            except Exception:pass
            png=await self.canvas.screenshot(type='png',timeout=3000)
            frame=cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)
            self.capture_method='element_screenshot';self.capture_ms=(time.perf_counter()-start)*1000
            return cv2.resize(frame,(256,224),interpolation=cv2.INTER_NEAREST)
        except Exception:
            self.canvas=None
            return None

    async def focus(self):
        if self.canvas is None:await self.discover()
        if self.canvas is None:raise ValueError('Canvas do jogo ainda não encontrado')
        await self.canvas.evaluate('(node) => { node.tabIndex=0; node.focus(); }')

    async def set_keys(self, keys: set[str], hold_ms: int | None = None, *, reason='superseded_by_set'):
        if not keys <= ALLOWED_KEYS:raise ValueError('Tecla fora dos controles do jogo')
        if hold_ms is not None and not 1<=hold_ms<=250:raise ValueError('Duração fora dos limites')
        if self.native_input:
            await asyncio.wrap_future(self.native_input.set(keys,hold_ms or 250,reason=reason))
            return
        if self.release_task and self.release_task is not asyncio.current_task():self.release_task.cancel()
        for key in tuple(self.held-keys):
            sent=time.monotonic();await self.page.keyboard.up(key);self.held.discard(key)
            self.key_events.append({'key':key,'down':False,'sent_at':sent,'acknowledged_at':time.monotonic(),'reason':reason})
        for key in keys-self.held:
            sent=time.monotonic();await self.page.keyboard.down(key);self.held.add(key)
            self.key_events.append({'key':key,'down':True,'sent_at':sent,'acknowledged_at':time.monotonic(),'reason':reason})
        if keys and hold_ms is not None:
            async def release():
                await asyncio.sleep(hold_ms/1000)
                await self.set_keys(set())
            self.release_task=asyncio.create_task(release())

    def drain_key_events(self):
        if self.native_input:return self.native_input.drain_key_events()
        events=self.key_events;self.key_events=[];return events

    async def pulse(self,key:str,hold_ms:int=100):
        if key not in ALLOWED_KEYS or not 20<=hold_ms<=250:raise ValueError('Comando inválido')
        await self.focus()
        await self.set_keys({key},hold_ms)
        await asyncio.sleep(hold_ms/1000)
        await self.set_keys(set())

    async def close(self):
        try:
            if self.page:await self.set_keys(set())
        finally:
            if self.native_input:await asyncio.to_thread(self.native_input.close)
            if self.context:await self.context.close()
            if self.browser:await self.browser.close()
            if self.playwright:await self.playwright.stop()
