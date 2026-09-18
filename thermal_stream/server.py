import asyncio
import json
import websockets
from websockets.asyncio.server import serve
from websockets.http11 import Response
from websockets.datastructures import Headers

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Thermal Stream</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; font-family:monospace; display:flex; flex-direction:column; align-items:center; min-height:100vh; }
  h1 { color:#888; font-size:14px; font-weight:normal; padding:10px; }
  img { width:100%; max-width:768px; image-rendering:pixelated; display:block; }
  #s { color:#4a4; padding:8px; font-size:12px; }
  #bt { color:#ddd; font-size:20px; padding:2px 8px 6px; }
  #t { color:#666; font-size:11px; padding-bottom:10px; }
  .e { color:#a44; }
</style>
</head>
<body>
<h1>Thermal Stream</h1>
<img id="i" src="" alt="thermal" />
<div id="s">connecting...</div>
<div id="bt">body: -- °C</div>
<div id="t"></div>
<script>
const elS=document.getElementById('s'),elT=document.getElementById('t'),elBT=document.getElementById('bt'),img=document.getElementById('i');
let B=0,C=0,T=Date.now(),blobURL='';
function U(){const s=(Date.now()-T)/1000;elT.textContent=C+' frames | '+(B/1024).toFixed(0)+' KB | '+(s>0?(B*8/1e3/s).toFixed(0):0)+' kbps | '+s.toFixed(0)+'s'}
function c(){const w=new WebSocket('ws://'+location.host);w.binaryType='arraybuffer';
w.onmessage=e=>{C++;B+=e.data.byteLength;
  const old=blobURL;
  blobURL=URL.createObjectURL(new Blob([e.data],{type:'image/jpeg'}));
  img.src=blobURL;
  if(old)URL.revokeObjectURL(old)};
w.onopen=()=>elS.textContent='streaming';
w.onerror=()=>{elS.textContent='WS error';elS.className='e'};
w.onclose=()=>{elS.textContent='disconnected - retry...';elS.className='e';setTimeout(c,2000)}
setInterval(U,2000)}
function b(){const w=new WebSocket('ws://'+location.host+'/body-temperature');
w.onmessage=e=>{const d=JSON.parse(e.data);elBT.textContent=d.valid?'body: '+d.body_temp_c.toFixed(2)+' °C':'body: -- °C'};
w.onclose=()=>setTimeout(b,2000)}
c();
b();
</script>
</body>
</html>"""


def _json_response(payload, status=200):
    headers = Headers()
    headers['Content-Type'] = 'application/json; charset=utf-8'
    headers['Access-Control-Allow-Origin'] = '*'
    return Response(status, 'OK', headers, json.dumps(payload).encode())


async def http_handler(connection, request, latest_body_temp=None):
    if request.headers.get('Upgrade', '').lower() == 'websocket':
        return None
    if request.path == '/':
        headers = Headers()
        headers['Content-Type'] = 'text/html; charset=utf-8'
        return Response(200, 'OK', headers, HTML_PAGE.encode())
    if request.path == '/body-temperature':
        payload = latest_body_temp or {"type": "body_temperature", "valid": False}
        return _json_response(payload)
    return connection.respond(404, "Not Found")


async def broadcast_loop(queue: asyncio.Queue, clients: set):
    while True:
        data = await queue.get()
        stale = []
        for ws in list(clients):
            try:
                await ws.send(data)
            except websockets.exceptions.ConnectionClosed:
                stale.append(ws)
        for ws in stale:
            clients.discard(ws)


async def body_temperature_loop(queue: asyncio.Queue, clients: set, latest: dict):
    while True:
        payload = await queue.get()
        latest.clear()
        latest.update(payload)
        data = json.dumps(payload)
        stale = []
        for ws in list(clients):
            try:
                await ws.send(data)
            except websockets.exceptions.ConnectionClosed:
                stale.append(ws)
        for ws in stale:
            clients.discard(ws)


def _ws_path(ws):
    request = getattr(ws, 'request', None)
    if request is not None:
        return getattr(request, 'path', '/')
    return getattr(ws, 'path', '/')


async def ws_handler(ws, frame_clients: set, body_temp_clients: set, latest_body_temp: dict):
    path = _ws_path(ws)
    if path == '/body-temperature':
        clients = body_temp_clients
        if latest_body_temp:
            await ws.send(json.dumps(latest_body_temp))
    else:
        clients = frame_clients

    clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)


async def start_server(queue: asyncio.Queue, host='0.0.0.0', port=7864, body_temp_queue=None):
    frame_clients = set()
    body_temp_clients = set()
    latest_body_temp = {}
    async with serve(
        lambda ws: ws_handler(ws, frame_clients, body_temp_clients, latest_body_temp),
        host, port,
        process_request=lambda conn, req: http_handler(conn, req, latest_body_temp),
        max_size=2**24,
    ):
        print(f"Server listening: http://{host}:{port}")
        broadcast_task = asyncio.create_task(broadcast_loop(queue, frame_clients))
        body_temp_task = None
        if body_temp_queue is not None:
            body_temp_task = asyncio.create_task(
                body_temperature_loop(body_temp_queue, body_temp_clients, latest_body_temp)
            )
        try:
            await asyncio.get_running_loop().create_future()
        finally:
            broadcast_task.cancel()
            if body_temp_task is not None:
                body_temp_task.cancel()
