#!/usr/bin/env python3
"""Thermal camera colorized video streaming server.

Reads config from thermal_stream/.env, captures frames from USB thermal camera,
applies Ironbow colormap, encodes JPEG frames, and streams via WebSocket.
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
from thermal_stream.body_temperature import BodyTemperatureEstimator
from thermal_stream.colorize import apply_colormap
from thermal_stream.streamer import ThermalStreamer
from thermal_stream.server import start_server


def load_config():
    """Load config from .env file, return dict with defaults."""
    config = {
        'PORT': 7864,
        'FPS': 25,
        'WIDTH': 256,
        'HEIGHT': 192,
        'BODY_TEMP_CALC_HZ': 5.0,
        'BODY_TEMP_PUSH_HZ': 2.0,
        'BODY_TEMP_MIN_C': 35.0,
        'BODY_TEMP_MAX_C': 38.0,
        'BODY_TEMP_MIN_PIXELS': 25,
        'BODY_TEMP_EMA_ALPHA': 0.35,
    }
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


def _put_latest(queue: asyncio.Queue, payload):
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(payload)


async def async_main(
    port,
    fps,
    width,
    height,
    body_temp_calc_hz,
    body_temp_push_hz,
    body_temp_min_c,
    body_temp_max_c,
    body_temp_min_pixels,
    body_temp_ema_alpha,
):
    print("Initializing thermal camera...")
    try:
        cam = ThermalCamera()
    except Exception as exc:
        print(f"Failed to initialize thermal camera: {exc}", file=sys.stderr)
        return

    queue = asyncio.Queue(maxsize=10)
    body_temp_queue = asyncio.Queue(maxsize=1)
    body_temp_estimator = BodyTemperatureEstimator(
        min_c=body_temp_min_c,
        max_c=body_temp_max_c,
        min_pixels=body_temp_min_pixels,
        ema_alpha=body_temp_ema_alpha,
    )
    streamer = ThermalStreamer(width=width, height=height, fps=fps)
    streamer.start(queue)
    print("GStreamer pipeline started.")
    print(
        "Body temperature: "
        f"{body_temp_min_c:.1f}~{body_temp_max_c:.1f}°C, "
        f"calc {body_temp_calc_hz:g}Hz, push {body_temp_push_hz:g}Hz"
    )

    # Push priming frames to stabilize encoder
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(10):
        streamer.push_frame(dummy)
        await asyncio.sleep(0.01)

    running = True
    last_ts = 0.0
    last_body_calc_ts = 0.0
    last_body_push_ts = 0.0
    latest_body_reading = None
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
        nonlocal last_ts, last_body_calc_ts, last_body_push_ts
        nonlocal latest_body_reading, count, running
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

            if body_temp_calc_hz > 0 and now - last_body_calc_ts >= 1.0 / body_temp_calc_hz:
                latest_body_reading = body_temp_estimator.estimate(temp, now)
                last_body_calc_ts = now

            if (
                latest_body_reading is not None
                and body_temp_push_hz > 0
                and now - last_body_push_ts >= 1.0 / body_temp_push_hz
            ):
                _put_latest(body_temp_queue, latest_body_reading.to_payload())
                last_body_push_ts = now

            last_ts = now
            count += 1
            if count % 100 == 0:
                body = "--"
                if latest_body_reading and latest_body_reading.valid:
                    body = f"{latest_body_reading.body_temp_c:.2f}°C"
                print(f"[{count}] temp: {temp.min():.1f}~{temp.max():.1f}°C body={body}")

    capture = asyncio.create_task(capture_loop())
    await start_server(queue, host='0.0.0.0', port=port, body_temp_queue=body_temp_queue)
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
        asyncio.run(async_main(
            port,
            fps,
            width,
            height,
            cfg['BODY_TEMP_CALC_HZ'],
            cfg['BODY_TEMP_PUSH_HZ'],
            cfg['BODY_TEMP_MIN_C'],
            cfg['BODY_TEMP_MAX_C'],
            cfg['BODY_TEMP_MIN_PIXELS'],
            cfg['BODY_TEMP_EMA_ALPHA'],
        ))
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopped.")


if __name__ == '__main__':
    main()
