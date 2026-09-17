# service/__init__.py
# 延迟导入，避免 IPC 等轻量模块被重依赖拖慢

def __getattr__(name):
    if name == "FaceDetector" or name == "DetectedFace":
        from .detector import FaceDetector, DetectedFace
        return FaceDetector if name == "FaceDetector" else DetectedFace
    if name == "VehicleRecognizer":
        from .recognizer import VehicleRecognizer
        return VehicleRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["FaceDetector", "DetectedFace", "VehicleRecognizer"]
