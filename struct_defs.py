from ctypes import (
    CDLL, Structure, c_void_p, c_int, c_uint8, c_uint32, c_char_p,
    POINTER, CFUNCTYPE, cast , c_uint
)

class IrControlHandle_t(Structure):
    pass

class IrVideoHandle_t(Structure):
    pass

class IruvcDevParam_t(Structure):
    _fields_ = [
        ("pid", c_uint),
        ("vid", c_uint),
        ("same_idx", c_uint),
    ]


class IruvcHandle_t(Structure):
    pass

class IrcmdHandle_t(Structure):
    pass

class VideoOutputInfo_t(Structure):
    _fields_ = [
        ("video_output_status", c_int),
        ("video_output_format", c_int),
        ("video_output_fps", c_int),
        ("video_output_num", c_int),
        ("video_output_mode", c_int),
        ("video_output_info_sw", c_int),
    ]

class CameraParam_t(Structure):
    _fields_ = [
        ("format", c_char_p),
        ("width", c_uint),
        ("height", c_uint),
        ("frame_size", c_uint),
        ("fps", c_uint),
        ("timeout_ms_delay", c_uint)
    ]

class UserCallback_t(Structure):
    _fields_ = [
        ("iruvc_handle", POINTER(IruvcHandle_t)),
        ("usr_func", c_void_p),
        ("usr_param", c_void_p)
    ]

class IruvcCamStreamParams_t(Structure):
    _fields_ = [
        ("camera_param", CameraParam_t),
        ("usr_callback", UserCallback_t),
        ("reserved", (c_uint8 * 48))
    ]

class cam_side_preview_ctl(c_int):
    CLOSE_CAM_SIDE_PREVIEW = 0
    KEEP_CAM_SIDE_PREVIEW  = 1

class IruvcStreamStopParams_t(Structure):
    _fields_ = [
        ("stop_mode", cam_side_preview_ctl),
        ("reserved", (c_uint8 * 60))
    ]
class IruvcFrameGetParams_t(Structure):
    _fields_ = [
        ("reserved", (c_int * 32))
    ]
