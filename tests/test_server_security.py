from http.client import HTTPConnection

from ai_player_lab.browser_server import LiveServer


def test_rebinding_hosts_and_cross_origin_commands_are_rejected():
    server = LiveServer(port=0)
    server.start()
    port = server.http.server_address[1]
    client = HTTPConnection('127.0.0.1', port, timeout=2)
    try:
        client.request('GET', '/status', headers={'Host': 'attacker.example'})
        response = client.getresponse()
        assert response.status == 403
        response.read()
        client.request('POST', '/control', '{"action":"start"}',
                       {'Origin': 'https://attacker.example'})
        response = client.getresponse()
        assert response.status == 403
        response.read()
        assert server.commands.empty()
        client.close()
        client = HTTPConnection('127.0.0.1', port, timeout=2)
        client.request('POST', '/control', '[]',
                       {'Origin': f'http://127.0.0.1:{port}'})
        response = client.getresponse()
        assert response.status == 400
        response.read()
        client.close()
        client = HTTPConnection('127.0.0.1', port, timeout=2)
        client.request('POST', '/control', '{"action":"pause"}',
                       {'Origin': f'http://127.0.0.1:{port}'})
        response = client.getresponse()
        assert response.status == 200
        response.read()
        assert server.commands.get()['action'] == 'pause'
    finally:
        client.close()
        server.close()
