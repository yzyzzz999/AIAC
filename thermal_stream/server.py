import asyncio
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
  #t { color:#666; font-size:11px; }
  .e { color:#a44; }
</style>
</head>
<body>
<h1>Thermal Stream</h1>
<img id="i" src="" alt="thermal" />
<div id="s">connecting...</div>
<div id="t"></div>
<script>
const elS=document.getElementById('s'),elT=document.getElementById('t'),img=document.getElementById('i');
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
c();
</script>
</body>
</html>"""


async def http_handler(connection, request):
    if request.headers.get('Upgrade', '').lower() == 'websocket':
        return None
    if request.path == '/':
        headers = Headers()
        headers['Content-Type'] = 'text/html; charset=utf-8'
        return Response(200, 'OK', headers, HTML_PAGE.encode())
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


async def ws_handler(ws, clients: set):
    clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)


async def start_server(queue: asyncio.Queue, host='0.0.0.0', port=7864):
    clients = set()
    async with serve(
        lambda ws: ws_handler(ws, clients),
        host, port,
        process_request=http_handler,
        max_size=2**24,
    ):
        print(f"Server listening: http://{host}:{port}")
        broadcast_task = asyncio.create_task(broadcast_loop(queue, clients))
        try:
            await asyncio.get_running_loop().create_future()
        finally:
            broadcast_task.cancel()
