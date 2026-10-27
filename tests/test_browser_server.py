import json
import math
import socket
import time
from threading import Thread
from urllib.request import urlopen
from http.server import BaseHTTPRequestHandler
from http.client import HTTPConnection

from ai_player_lab.browser_server import LiveServer


def test_localhost_panel_redirects_once_to_direct_loopback():
    server=LiveServer(port=0);server.start()
    port=server.http.server_address[1]
    client=HTTPConnection('127.0.0.1',port,timeout=2)
    try:
        client.request('GET','/',headers={'Host':f'localhost:{port}'})
        response=client.getresponse()
        assert response.status==302
        assert response.headers['Location']==f'http://127.0.0.1:{port}/'
        response.read()
        client.request('GET','/')
        response=client.getresponse()
        assert response.status==200 and b'gameFrame' in response.read()
    finally:
        client.close();server.close()


def test_raw_and_annotated_streams_advance_independently_and_report_counts():
    server = LiveServer(port=0)
    server.start()
    base = f'http://127.0.0.1:{server.http.server_address[1]}'
    captured_at = time.monotonic() - .05
    try:
        # The latest raw image can be several captures ahead of the last
        # processed annotation. Annotated mode must still serve its own newest
        # complete JPEG rather than require equality with raw's sequence.
        server.update(frame=b'raw-20', frame_id=20, frame_time=captured_at)
        server.update(overlay=b'annotated-12', overlay_id=12, overlay_time=captured_at)

        with urlopen(f'{base}/frame.jpg?after=0') as response:
            assert response.read() == b'raw-20'
            assert response.headers['X-Frame-ID'] == '20'
            assert float(response.headers['X-Frame-Age-Ms']) >= 40

        with urlopen(f'{base}/frame.jpg?annotated=1&after=10') as response:
            assert response.read() == b'annotated-12'
            assert response.headers['X-Frame-ID'] == '12'
            assert float(response.headers['X-Frame-Age-Ms']) >= 40

        # The long-poll waits for a newer overlay even if no newer raw frame
        # was published. Only the annotated stream is being advanced here.
        def publish_next_overlay():
            time.sleep(.02)
            server.update(overlay=b'annotated-15', overlay_id=15,
                          overlay_time=time.monotonic() - .01)

        publisher = Thread(target=publish_next_overlay)
        publisher.start()
        with urlopen(f'{base}/frame.jpg?annotated=1&after=12', timeout=2) as response:
            assert response.read() == b'annotated-15'
            assert response.headers['X-Frame-ID'] == '15'
            assert math.isfinite(float(response.headers['X-Frame-Age-Ms']))
        publisher.join(timeout=1)

        with urlopen(f'{base}/status') as response:
            telemetry = json.load(response)['video_telemetry']
        assert telemetry == {
            'raw_published': 1, 'annotated_published': 2,
            'raw_requests': 1, 'raw_served': 1, 'raw_no_new_frame': 0,
            'annotated_requests': 2, 'annotated_served': 2, 'annotated_no_new_frame': 0,
        }
    finally:
        server.close()


def test_handler_treats_aborted_keepalive_connection_as_client_disconnect(monkeypatch):
    server = LiveServer(port=0)
    server.start()
    client, request = socket.socketpair()

    def aborted_handle(_handler):
        raise ConnectionAbortedError('client closed the connection')

    try:
        monkeypatch.setattr(BaseHTTPRequestHandler, 'handle', aborted_handle)
        # Constructing the request handler exercises its outer handle() guard;
        # the simulated disconnect must not escape into the server thread.
        server.http.RequestHandlerClass(request, ('127.0.0.1', 1), server.http)
    finally:
        client.close()
        server.close()
