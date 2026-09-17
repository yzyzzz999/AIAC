import ctypes
import os
import sys
import numpy as np
from struct_defs import *  # 确保 struct_defs.py 在同一目录下

class ThermalCamera:
    def __init__(self, lib_path_base='./stream_sample_python/sample/linux/'):
        self.width = 256
        self.height = 192
        self.raw_byte_size = self.width * self.height * 2 * 2 + 256 * 2 * 2 # Image + Temp + Info lines
        
        # 加载库
        self.lib_ircmd = self._load_lib(os.path.join(lib_path_base, 'libircmd.so'))
        self.lib_ircam = self._load_lib(os.path.join(lib_path_base, 'libircam.so'))
        self.lib_iruvc = self._load_lib(os.path.join(lib_path_base, 'libiruvc.so'))
        
        if not (self.lib_ircmd and self.lib_ircam and self.lib_iruvc):
            raise Exception("无法加载红外库文件，请检查路径")

        self.iruvc_handle = None
        self.raw_frame_data_ptr = None
        self._init_device()

    def _load_lib(self, path):
        try:
            return ctypes.CDLL(path)
        except OSError:
            return None

    def _init_device(self):
        # 1. 设置日志
        self.lib_iruvc.iruvc_log_register(2, None, None) # 2=不打印日志，避免刷屏

        # 2. 创建句柄
        ir_video_handle = ctypes.POINTER(IrVideoHandle_t)()
        self.lib_ircam.ir_video_handle_create(ctypes.pointer(ir_video_handle))
        
        self.lib_iruvc.iruvc_camera_handle_create.restype = ctypes.POINTER(IruvcHandle_t)
        self.iruvc_handle = self.lib_iruvc.iruvc_camera_handle_create(ir_video_handle)
        
        param = IruvcDevParam_t()
        param.pid, param.vid, param.same_idx = 0x4321, 0x3474, 0
        
        # 3. 打开与初始化
        if self.lib_iruvc.iruvc_camera_open(self.iruvc_handle, ctypes.byref(param)) != 0:
            raise Exception("红外相机打开失败")
        self.lib_iruvc.iruvc_camera_init(self.iruvc_handle, ctypes.byref(param))
        
        # 4. 启动流
        video_params = IruvcCamStreamParams_t()
        video_params.camera_param.width = self.width
        video_params.camera_param.height = self.height + 192 + 2 # Image + Temp + Info
        video_params.camera_param.frame_size = self.raw_byte_size
        video_params.camera_param.fps = 25
        video_params.camera_param.format = b"YUYV"
        
        self.lib_iruvc.iruvc_camera_start_stream(self.iruvc_handle, ctypes.byref(video_params))
        
        # 准备数据 buffer
        self.raw_buffer = ctypes.create_string_buffer(self.raw_byte_size)
        self.raw_frame_data_ptr = ctypes.cast(self.raw_buffer, ctypes.POINTER(ctypes.c_uint8))

    def get_frame(self):
        """
        返回: (image_gray_8bit, temp_celsius_matrix)
        """
        ret = self.lib_iruvc.iruvc_camera_frame_get(self.iruvc_handle, None, self.raw_frame_data_ptr, self.raw_byte_size)
        if ret != 0:
            return None, None

        frame_buffer = np.frombuffer(ctypes.string_at(self.raw_frame_data_ptr, self.raw_byte_size), dtype=np.uint8)
        
        # --- 解析图像 ---
        image_byte_size = self.width * self.height * 2
        image_raw = frame_buffer[0 : image_byte_size]
        
        # 提取 Y 通道 (YUYV -> Y)，这是最清晰的灰度图，适合标定
        # Y 在索引 0, 2, 4...
        image_y = image_raw[0::2]
        image_y = image_y[::-1].reshape((self.height, self.width))
        
        # --- 解析温度 (如果需要) ---
        # 略过信息行
        info_size = self.width * 2 * 2
        temp_start = image_byte_size + info_size
        temp_end = temp_start + (self.width * self.height * 2)
        temp_raw = frame_buffer[temp_start : temp_end].view(dtype=np.uint16)
        
        # 简单转换，请根据实际情况修改
        temp_celsius = (temp_raw.reshape((self.height, self.width)) / 64.0) - 273.15
        
        return image_y, temp_celsius

    def close(self):
        if self.iruvc_handle:
            stop_params = IruvcStreamStopParams_t()
            self.lib_iruvc.iruvc_camera_stop_stream(self.iruvc_handle, ctypes.byref(stop_params))
            self.lib_iruvc.iruvc_camera_close(self.iruvc_handle)