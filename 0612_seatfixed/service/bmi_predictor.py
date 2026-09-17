import sys, os, gc, warnings, time, logging
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import cv2, threading, time
import torch
from PIL import Image
warnings.filterwarnings("ignore")

log = logging.getLogger("bmi")

BMI_MODELS_DIR = Path(__file__).parent.parent / "models" / "bmi"

from service.bmi_backend import CachedYOLOv8FLIPModel, RealtimeFeatureExtractor

CLASS_NAMES = ["C0-偏瘦", "C1-正常", "C2-超重", "C3-肥胖"]
MODEL_PATH = str(BMI_MODELS_DIR / "best_yolov8_flip_bmi_cached2.pth")

class BMIPredictor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ext = None
        self._model = None
        self._ready = False
        self._load_time = None
        self._latest_result = None
        self._result_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._camera_running = False
        self._camera_thread = None
        self._stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    def ensure_loaded(self):
        if self._ready:
            return
        with self._load_lock:
            if self._ready:
                return
            t0 = time.time()
            log.info("Loading models...")
            gc.collect()
            yolo_path = str(BMI_MODELS_DIR / "yolov8n.pt")
            flip_path = str(BMI_MODELS_DIR / "FLIP-base-16")
            self._ext = RealtimeFeatureExtractor(yolo_path, flip_path, self.device)
            gc.collect()
            log.info("Loading BMI model...")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"BMI model not found: {MODEL_PATH}")
            self._model = CachedYOLOv8FLIPModel()
            ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict):
                for k in ["model_state_dict", "state_dict"]:
                    if k in ckpt:
                        self._model.load_state_dict(ckpt[k], strict=False)
                        break
                else:
                    self._model.load_state_dict(ckpt, strict=False)
            else:
                self._model = ckpt
            self._model.eval().to(self.device)
            self._ready = True  # 核心模型就绪，人脸检测器按需加载
            self._load_time = time.time() - t0
            log.info("Ready in %.1fs", self._load_time)
    
    @torch.no_grad()
    def predict(self, image_bgr, passenger_face=None, driver_face=None):
        """image_bgr: numpy array (H,W,3) in BGR format. Returns dict with passenger/driver predictions.

        passenger_face/driver_face: optional PIL.Image, 预裁剪的人脸图片。
        传入则跳过该侧的人脸检测，复用主流程检测结果。
        """
        self.ensure_loaded()
        if image_bgr.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_bgr
        pil_img = Image.fromarray(image_rgb)
        w, h = pil_img.size
        left_pil = pil_img.crop((0, 0, w // 2, h))
        right_pil = pil_img.crop((w // 2, 0, w, h))
        pre_faces = {"passenger": passenger_face, "driver": driver_face}

        def process_side(half_pil, half_name):
            try:
                face_img = pre_faces.get(half_name)
                # 无预裁剪人脸 → 该侧无人，直接返回（不再加载后备检测器）
                if face_img is None:
                    return {"face_detected": False, "class_id": -1, "class_name": "no_face", "confidence": 0.0}
                body = half_pil.resize((640, 640), Image.BILINEAR)
                body_t = torch.from_numpy(np.array(body).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
                face_inputs = self._ext.flip_processor(images=face_img, return_tensors="pt")
                face_t = face_inputs["pixel_values"]
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        yf = self._ext.extract_yolo(body_t)
                        ff = self._ext.extract_flip(face_t).to(self.device)
                        bmi_v, cls_v, _ = self._model(yf[0].to(self.device), yf[1].to(self.device),
                                                       yf[2].to(self.device), ff)
                else:
                    yf = self._ext.extract_yolo(body_t)
                    ff = self._ext.extract_flip(face_t).to(self.device)
                    bmi_v, cls_v, _ = self._model(yf[0].to(self.device), yf[1].to(self.device),
                                                   yf[2].to(self.device), ff)
                cls_id = int(cls_v[0].argmax().item())
                cls_conf = float(torch.softmax(cls_v[0], dim=-1).max().item())
                return {"face_detected": True, "class_id": cls_id,
                        "class_name": CLASS_NAMES[cls_id] if 0 <= cls_id < 4 else "",
                        "confidence": cls_conf, "bmi": float(bmi_v[0].item())}
            except Exception as e:
                log.exception("BMI predict side=%s failed", half_name)
                return {"face_detected": False, "class_id": -1, "class_name": "error", "confidence": 0.0, "error": str(e)}
        
        passenger = process_side(left_pil, "passenger")
        driver = process_side(right_pil, "driver")
        return {"status": "ok", "passenger": passenger, "driver": driver}
    
    @property
    def ready(self):
        return self._ready



    def start_camera(self, camera_id=0, interval=5.0):
        if self._camera_running:
            return {"status": "already_running"}
        self._camera_running = True
        self._camera_thread = threading.Thread(
            target=self._camera_loop, args=(camera_id, interval), daemon=True)
        self._camera_thread.start()
        return {"status": "started", "camera_id": camera_id, "interval": interval}

    def stop_camera(self):
        self._camera_running = False
        if self._camera_thread:
            self._camera_thread.join(timeout=10)
            self._camera_thread = None
        return {"status": "stopped"}

    def get_latest(self):
        with self._result_lock:
            if not self._ready:
                return {"status": "loading"}
            if self._latest_result is None:
                return {"status": "no_result"}
            return self._latest_result

    def _camera_loop(self, camera_id, interval):
        try:
            self.ensure_loaded()
        except Exception as e:
            with self._result_lock:
                self._latest_result = {"error": "Model load failed: " + str(e)}
            self._camera_running = False
            return
        cap = cv2.VideoCapture(camera_id)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                with self._result_lock:
                    self._latest_result = {"error": "Cannot open camera " + str(camera_id)}
                self._camera_running = False
                return
            while self._camera_running:
                ret, frame = cap.read()
                if ret:
                    try:
                        result = self.predict(frame)
                        with self._result_lock:
                            self._latest_result = result
                    except Exception as e:
                        with self._result_lock:
                            self._latest_result = {"error": str(e)}
                time.sleep(interval)
        finally:
            cap.release()
