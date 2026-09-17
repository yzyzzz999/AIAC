"""GStreamer pipeline: appsrc → jpegenc → appsink (MJPEG)."""

import sys
import asyncio
import threading
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)


class ThermalStreamer:
    def __init__(self, width=256, height=192, fps=25):
        self.width = width
        self.height = height
        self.fps = fps
        self._queue = None
        self._loop = None
        self._glib_thread = None

        pipeline_str = (
            f"appsrc name=src is-live=true format=time "
            f"caps=video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 "
            f"! videoconvert "
            f"! jpegenc quality=85 "
            f"! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name('src')
        self.appsink = self.pipeline.get_by_name('sink')

    def _on_new_sample(self, appsink):
        sample = appsink.emit('pull-sample')
        if sample and self._queue is not None:
            buf = sample.get_buffer()
            ok, map_info = buf.map(Gst.MapFlags.READ)
            if ok:
                try:
                    self._queue.put_nowait(bytes(map_info.data))
                except asyncio.QueueFull:
                    pass
                finally:
                    buf.unmap(map_info)
        return Gst.FlowReturn.OK

    def _run_glib_loop(self):
        self._loop = GLib.MainLoop()
        self._loop.run()

    def start(self, queue: asyncio.Queue):
        self._queue = queue
        self.appsink.connect('new-sample', self._on_new_sample)

        self._glib_thread = threading.Thread(
            target=self._run_glib_loop, name='glib-loop', daemon=True
        )
        self._glib_thread.start()

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")

    def push_frame(self, rgb_frame: np.ndarray):
        if not rgb_frame.flags['C_CONTIGUOUS']:
            rgb_frame = np.ascontiguousarray(rgb_frame)
        buf = Gst.Buffer.new_wrapped(rgb_frame.tobytes())
        self.appsrc.emit('push-buffer', buf)

    def stop(self):
        self._queue = None
        self.pipeline.set_state(Gst.State.NULL)
        if self._loop:
            self._loop.quit()
