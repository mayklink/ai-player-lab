"""Loopback live view and bounded control queue. HTTP threads never touch Playwright."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import SimpleQueue
from threading import Thread, Lock, Condition
from urllib.parse import parse_qs, urlsplit
import json
import time


class LiveServer:
    def __init__(self, port: int = 8765):
        self.lock = Lock()
        self.frames_changed = Condition(self.lock)
        self.frame = b''
        self.frame_id = 0
        self.frame_time = 0.
        self.frame_published_at = 0.
        self.overlay = b''
        self.overlay_id = None
        self.overlay_time = None
        self.video_telemetry = {
            'raw_published': 0, 'annotated_published': 0,
            'raw_requests': 0, 'raw_served': 0, 'raw_no_new_frame': 0,
            'annotated_requests': 0, 'annotated_served': 0, 'annotated_no_new_frame': 0,
        }
        self.status = {'running': False, 'paused': True, 'owner': 'Visão local', 'scene': 'starting',
                       'summary': 'Preparando navegador', 'calls': {'jev': 0, 'luna': 0}, 'fps': 0,
                       'last_action': '', 'last_effect': ''}
        self.commands = SimpleQueue()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def handle(self):
                # A browser can close a long-poll connection while the server
                # is waiting for the next frame. Treat that as a normal client
                # disconnect instead of letting the request thread traceback.
                try:
                    super().handle()
                except ConnectionAbortedError:
                    pass

            def log_message(self, *args):
                pass

            def send(self, code, body, kind='application/json', frame_id=None, frame_time=None):
                try:
                    self.send_response(code)
                    self.send_header('Content-Type', kind)
                    self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    if frame_id is not None:
                        self.send_header('X-Frame-ID', str(frame_id))
                    if frame_time is not None:
                        self.send_header('X-Frame-Time', str(frame_time))
                        age_ms = max(0., (time.monotonic() - frame_time) * 1000)
                        self.send_header('X-Frame-Age-Ms', f'{age_ms:.2f}')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    if body:
                        self.wfile.write(body)
                    return True
                except (BrokenPipeError, ConnectionResetError):
                    return False

            def do_GET(self):
                if not self.valid_host():
                    self.send(403, b'{}')
                    return
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query)
                if parsed.path == '/':
                    if self.headers.get('Host','').split(':')[0].lower()=='localhost':
                        self.send_response(302)
                        self.send_header('Location',f'http://127.0.0.1:{owner.http.server_address[1]}/')
                        self.send_header('Cache-Control','no-store')
                        self.send_header('Content-Length','0')
                        self.end_headers()
                        return
                    self.send(200, (Path(__file__).parent / 'web' / 'live.html').read_bytes(),
                              'text/html; charset=utf-8')
                elif parsed.path == '/frame.jpg':
                    try:
                        after = int(query.get('after', ['0'])[0])
                    except ValueError:
                        self.send(400, b'{"error":"invalid frame sequence"}')
                        return
                    annotated = query.get('annotated') == ['1']
                    stream_name = 'annotated' if annotated else 'raw'
                    store_name = 'overlay' if annotated else 'frame'
                    with owner.lock:
                        owner.video_telemetry[f'{stream_name}_requests'] += 1
                    # Each mode waits for a newer item in its own latest-only
                    # slot. Annotated responses are complete JPEG frames, not
                    # overlays that need to match an independently advancing raw frame.
                    with owner.frames_changed:
                        deadline = time.monotonic() + 1.5
                        while ((not getattr(owner, store_name)
                                or (annotated and owner.overlay_id is None)
                                or getattr(owner, f'{store_name}_id' if annotated else 'frame_id') <= after)
                               and time.monotonic() < deadline):
                            owner.frames_changed.wait(max(0., deadline - time.monotonic()))
                        item = getattr(owner, store_name)
                        if not item:
                            owner.video_telemetry[f'{stream_name}_no_new_frame'] += 1
                            self.send(503, b'')
                            return
                        item_id = owner.overlay_id if annotated else owner.frame_id
                        if item_id is None or item_id <= after:
                            owner.video_telemetry[f'{stream_name}_no_new_frame'] += 1
                            self.send(204, b'')
                            return
                        body = item
                        frame_time = owner.overlay_time if annotated else owner.frame_time
                    if self.send(200, body, 'image/jpeg', item_id, frame_time):
                        with owner.lock:
                            owner.video_telemetry[f'{stream_name}_served'] += 1
                elif parsed.path == '/status':
                    with owner.lock:
                        status = dict(owner.status)
                        status['video_telemetry'] = dict(owner.video_telemetry)
                    self.send(200, json.dumps(status, ensure_ascii=False).encode())
                else:
                    self.send(404, b'{}')

            def valid_host(self):
                # Reject DNS rebinding: loopback binding alone is insufficient.
                actual_port = owner.http.server_address[1]
                return self.headers.get('Host', '').lower() in {
                    f'127.0.0.1:{actual_port}', f'localhost:{actual_port}'}

            def do_POST(self):
                actual_port = owner.http.server_address[1]
                if not self.valid_host() or self.headers.get('Origin') not in (
                        None, f'http://127.0.0.1:{actual_port}',
                        f'http://localhost:{actual_port}'):
                    self.close_connection = True
                    self.send(403, b'{}')
                    return
                try:
                    n = int(self.headers.get('Content-Length', 0))
                    if not 0 < n <= 4096:
                        raise ValueError()
                    data = json.loads(self.rfile.read(n))
                    if not isinstance(data, dict):
                        raise ValueError('Expected a JSON object')
                    if self.path == '/control' and data.get('action') in {
                            'start', 'pause', 'stop', 'reset', 'shutdown', 'inspect'}:
                        owner.commands.put(data)
                        self.send(200, b'{"ok":true}')
                    elif self.path == '/key' and data.get('key') in {
                            'Enter', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'x', 'z', 'a', 's', 'v'} \
                            and 20 <= data.get('hold_ms', 100) <= 250:
                        owner.commands.put({'action': 'key', **data})
                        self.send(200, b'{"ok":true}')
                    else:
                        self.send(400, b'{"error":"Comando invalido"}')
                except (ValueError, TypeError):
                    self.close_connection = True
                    self.send(400, b'{}')

        self.http = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = Thread(target=self.http.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def update(self, *, frame=None, frame_id=None, frame_time=None, overlay=None,
               overlay_id=None, overlay_time=None, **status):
        """Publish raw video and processed overlay independently.

        ``frame_time``/``overlay_time`` are producer monotonic timestamps. They
        are returned as server timestamps and must only be compared against
        other values from that same clock. A separate overlay never advances or
        changes the raw frame sequence.
        """
        with self.frames_changed:
            if frame is not None:
                self.frame = frame
                self.frame_id = int(frame_id) if frame_id is not None else self.frame_id + 1
                self.frame_time = float(frame_time) if frame_time is not None else time.monotonic()
                self.frame_published_at = time.monotonic()
                self.video_telemetry['raw_published'] += 1
                self.frames_changed.notify_all()
            if overlay is not None:
                self.overlay = overlay
                self.overlay_id = int(overlay_id) if overlay_id is not None else self.frame_id
                self.overlay_time = float(overlay_time) if overlay_time is not None else time.monotonic()
                self.video_telemetry['annotated_published'] += 1
                self.frames_changed.notify_all()
            self.status.update(status)

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=2)
