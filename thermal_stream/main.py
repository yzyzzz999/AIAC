#!/usr/bin/env python3
"""Thermal camera colorized video streaming server.

Reads config from thermal_stream/.env, captures frames from USB thermal camera,
applies Ironbow colormap, encodes H.264 via NVENC, and streams via WebSocket
for browser MSE playback.
"""

import os
import sys
import signal
import asyncio
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from thermal_driver import ThermalCamera
from thermal_stream.colorize import apply_colormap
from thermal_stream.streamer import ThermalStreamer
from thermal_stream.server import start_server


def load_config():
    """Load config from .env file, return dict with defaults."""
    config = {'PORT': 7864, 'FPS': 25, 'WIDTH': 256, 'HEIGHT': 192}
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip()
                    if k in config:
                        config[k] = type(config[k])(v)
    return config


async def async_main(port, fps, width, height):
    print("Initializing thermal camera...")
    try:
        cam = ThermalCamera()
    except Exception as exc:
        print(f"Failed to initialize thermal camera: {exc}", file=sys.stderr)
        return

    queue = asyncio.Queue(maxsize=10)
    streamer = ThermalStreamer(width=width, height=height, fps=fps)
    streamer.start(queue)
    print("GStreamer pipeline started.")

    # Push priming frames to stabilize encoder
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(10):
        streamer.push_frame(dummy)
        await asyncio.sleep(0.01)

    running = True
    last_ts = 0.0
    count = 0

    loop = asyncio.get_event_loop()

    def shutdown():
        nonlocal running
        running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass

    async def capture_loop():
        nonlocal last_ts, count, running
        while running:
            now = time.monotonic()
            elapsed = now - last_ts
            if elapsed < 1.0 / fps:
                await asyncio.sleep(0.001)
                continue

            ir_gray, temp = cam.get_frame()
            if temp is None:
                await asyncio.sleep(0.001)
                continue

            rgb = apply_colormap(temp)
            streamer.push_frame(rgb)

            last_ts = now
            count += 1
            if count % 100 == 0:
                print(f"[{count}] temp: {temp.min():.1f}~{temp.max():.1f}°C")

    capture = asyncio.create_task(capture_loop())
    await start_server(queue, host='0.0.0.0', port=port)
    await capture
    cam.close()
    streamer.stop()


def main():
    cfg = load_config()
    port = cfg['PORT']
    fps = cfg['FPS']
    width = cfg['WIDTH']
    height = cfg['HEIGHT']

    print(f"Config: {width}x{height} @ {fps}fps, port {port}")

    try:
        asyncio.run(async_main(port, fps, width, height))
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopped.")


if __name__ == '__main__':
    main()
