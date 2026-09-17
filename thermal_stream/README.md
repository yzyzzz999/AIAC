# Thermal Stream

256×192 红外热成像实时彩色视频流，Ironbow 伪彩，JPEG 编码，WebSocket 推送。

## 启动服务

```bash
cd /path/to/thermal_stream
conda activate AIAC
./start.sh            # 启动
./stop.sh             # 停止
./restart.sh          # 重启
```

服务监听 `0.0.0.0:7864`。

## 取流方式

### 方式 1：浏览器（最简单）

打开 `http://<服务器IP>:7864/`，页面内置播放器，自动连接 WebSocket 并实时刷新 JPEG 画面。

### 方式 2：Python 取流

```python
import asyncio
import websockets
from PIL import Image
from io import BytesIO

async def stream():
    async with websockets.connect("ws://192.168.0.122:7864") as ws:
        while True:
            jpeg_bytes = await ws.recv()
            img = Image.open(BytesIO(jpeg_bytes))  # 256×192 RGB
            # do something with img ...

asyncio.run(stream())
```

### 方式 3：命令行取单帧

```bash
python3 -c "
import asyncio, websockets
async def main():
    async with websockets.connect('ws://192.168.0.122:7864') as ws:
        data = await ws.recv()
        with open('frame.jpg', 'wb') as f: f.write(data)
asyncio.run(main())
"
```

### 方式 4：OpenCV 取流

```python
import asyncio
import websockets
import cv2
import numpy as np

async def stream():
    async with websockets.connect("ws://192.168.0.122:7864") as ws:
        while True:
            jpeg_bytes = await ws.recv()
            arr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
            cv2.imshow("Thermal", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

asyncio.run(stream())
```

## 协议

| 项 | 值 |
|---|---|
| 传输 | WebSocket (binary) |
| 格式 | JPEG |
| 分辨率 | 256×192 |
| 帧率 | ~25fps (由 .env 中 FPS 配置) |
| 色彩映射 | Ironbow (黑→蓝→紫→红→橙→黄→白) |
| 每次消息 | 一帧完整 JPEG 图像 |

## 配置

编辑 `.env`：

```
PORT=7864      # HTTP/WS 端口
FPS=25         # 帧率
WIDTH=256      # 图像宽度
HEIGHT=192     # 图像高度
```
