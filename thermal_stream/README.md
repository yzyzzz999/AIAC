# Thermal Stream

256×192 红外热成像实时彩色视频流，Ironbow 伪彩，JPEG 编码，WebSocket 推送。
同时提供体温估算接口：从温度矩阵筛选 35~38°C 温度点，求平均并推送给前端。

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

### 方式 2.1：Python 订阅体温

```python
import asyncio
import json
import websockets

async def stream_body_temperature():
    async with websockets.connect("ws://192.168.0.122:7864/body-temperature") as ws:
        while True:
            data = json.loads(await ws.recv())
            # {"type":"body_temperature","body_temp_c":36.42,"valid":true,...}
            print(data)

asyncio.run(stream_body_temperature())
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

## 体温接口

| 项 | 值 |
|---|---|
| WebSocket | `ws://<服务器IP>:7864/body-temperature` |
| HTTP 当前值 | `http://<服务器IP>:7864/body-temperature` |
| 格式 | JSON |
| 默认计算频率 | 5Hz |
| 默认推送频率 | 2Hz |
| 计算方式 | 过滤温度矩阵中 35~38°C 的点，数量达到门槛后取均值并做 EMA 平滑 |

示例：

```json
{
  "type": "body_temperature",
  "body_temp_c": 36.42,
  "valid": true,
  "sample_count": 184,
  "range_c": [22.1, 39.6],
  "ts": 12345.678
}
```

说明：热成像视频可保持较高帧率保证画面流畅，但体温数值没有必要每帧计算、每帧推送。默认 5Hz 计算足够跟踪人体温变化，2Hz 推送可降低前端渲染和 WebSocket 广播开销。如果设备 CPU 余量充足，可提高 `BODY_TEMP_CALC_HZ`；如果前端只展示数字，通常不建议高于 5Hz 推送。

## 配置

编辑 `.env`：

```
PORT=7864      # HTTP/WS 端口
FPS=25         # 帧率
WIDTH=256      # 图像宽度
HEIGHT=192     # 图像高度
BODY_TEMP_CALC_HZ=5       # 体温计算频率
BODY_TEMP_PUSH_HZ=2       # 体温推送频率
BODY_TEMP_MIN_C=35        # 体温候选点下限
BODY_TEMP_MAX_C=38        # 体温候选点上限
BODY_TEMP_MIN_PIXELS=25   # 候选点少于该值时认为无效
BODY_TEMP_EMA_ALPHA=0.35  # 平滑系数，越大越灵敏
```
