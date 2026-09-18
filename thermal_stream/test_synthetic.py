#!/usr/bin/env python3
"""Synthetic test: stream a moving thermal gradient without real camera."""

import os, sys, asyncio, signal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thermal_stream.colorize import apply_colormap
from thermal_stream.streamer import ThermalStreamer
from thermal_stream.server import start_server
from thermal_stream.body_temperature import BodyTemperatureEstimator


async def main():
    queue = asyncio.Queue(maxsize=10)
    body_temp_queue = asyncio.Queue(maxsize=1)
    body_temp_estimator = BodyTemperatureEstimator()
    streamer = ThermalStreamer(width=256, height=192, fps=25)
    streamer.start(queue)
    print("GStreamer pipeline started (synthetic mode).")

    running = True
    loop = asyncio.get_event_loop()

    def shutdown():
        nonlocal running
        running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass

    frame_idx = 0
    last_body_push_ts = 0.0
    server_task = asyncio.create_task(start_server(queue, port=7864, body_temp_queue=body_temp_queue))

    while running:
        now = asyncio.get_running_loop().time()
        phase = frame_idx * 0.02
        x = np.linspace(20 + 50 * np.sin(phase), 80 + 50 * np.cos(phase * 0.7), 256)
        y = np.linspace(30, 120, 192)
        temp = ((x[np.newaxis, :] + y[:, np.newaxis]) / 2).astype(np.float32)
        rgb = apply_colormap(temp)
        streamer.push_frame(rgb)
        if now - last_body_push_ts >= 0.5:
            reading = body_temp_estimator.estimate(temp, now)
            if body_temp_queue.full():
                try:
                    body_temp_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            body_temp_queue.put_nowait(reading.to_payload())
            last_body_push_ts = now
        frame_idx += 1
        await asyncio.sleep(0.04)

    streamer.stop()
    server_task.cancel()


if __name__ == '__main__':
    asyncio.run(main())
