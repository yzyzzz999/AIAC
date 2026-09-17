"""身高估计模块：纯 NumPy + ONNX + ultralytics YOLO。"""
from __future__ import annotations
import math, os, sys, logging, numpy as np, cv2
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from threading import Lock

log = logging.getLogger("height")

TARGET_SIZE = 320
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
NOSE_IDX = 0; LEFT_EYE_IDX = 1; RIGHT_EYE_IDX = 2; LEFT_EAR_IDX = 3; RIGHT_EAR_IDX = 4


def _calc_head_size(keypoints, img_w, img_h):
    """基于面部关键点估算头部大小（归一化），用于后排过滤。"""
    face_ids = [NOSE_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX, LEFT_EAR_IDX, RIGHT_EAR_IDX]
    pts = []
    for fid in face_ids:
        if keypoints[fid][2] > 0.3:
            pts.append((keypoints[fid][0], keypoints[fid][1]))
    if len(pts) >= 2:
        max_dist = 0.0
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                dx = pts[i][0] - pts[j][0]; dy = pts[i][1] - pts[j][1]
                max_dist = max(max_dist, np.sqrt(dx*dx + dy*dy))
        diag = np.sqrt(img_w*img_w + img_h*img_h)
        return max_dist / diag if diag > 0 else 0.05
    # fallback: 用肩宽估算
    ls = keypoints[LEFT_SHOULDER_IDX]; rs = keypoints[RIGHT_SHOULDER_IDX]
    if ls[2] > 0.3 and rs[2] > 0.3:
        sw = abs(ls[0] - rs[0])
        diag = np.sqrt(img_w*img_w + img_h*img_h)
        return sw / diag * 0.5 if diag > 0 else 0.05
    return 0.05


def _ensure_torch_dll_path():
    for sp in sys.path:
        cand = os.path.join(sp, "torch", "lib")
        if os.path.isdir(cand):
            try: os.add_dll_directory(cand)
            except: pass
            break


# ════════════ 几何高度 ════════════
class NumPyHeightEstimator:
    def __init__(self, geo: Dict[str, float]):
        self.offset = geo["offset"]; self.slope = geo["slope"]
        self.C = geo["C"]; self.D = geo["D"]; self.E = geo["E"]
        self.delta = geo["delta"]
        self.sc_x_d = geo["seat_center_x_driver"]; self.sc_x_p = geo["seat_center_x_passenger"]
        self.sc_m = abs(geo["scale_male"]); self.sc_f = abs(geo["scale_female"]); self.sc_u = abs(geo["scale_unknown"])
        self.fx = geo["fx"]; self.fy = geo["fy"]; self.cx = geo["cx"]; self.cy = geo["cy"]
        self.SEAT_HALF = geo["SEAT_HALF_LEN"]; self.SEAT_CY = geo["SEAT_CENTER_Y_FIXED"]
        self.alpha_seat = geo.get("alpha_seat", 0.0); self.beta_seat = geo.get("beta_seat", 0.0)

    def _Z_shoulder(self, x, y, d):
        rad = (self.D * y + self.E * (1 - y)) * (math.pi / 180)
        return self.offset + self.slope * x - self.C * math.cos(rad) - d

    def _Z_seat(self, x): return self.offset + self.slope * x

    def _to_camera(self, u, v, Z):
        return np.array([(u - self.cx) * Z / self.fx, (v - self.cy) * Z / self.fy, Z], dtype=np.float32)

    def _pt_to_seg(self, pt, Z_seat, sc_x):
        A = np.array([sc_x - self.SEAT_HALF, self.SEAT_CY, Z_seat], dtype=np.float32)
        B = np.array([sc_x + self.SEAT_HALF, self.SEAT_CY, Z_seat], dtype=np.float32)
        AB = B - A; AP = pt - A
        t = np.clip(np.dot(AP, AB) / max(np.dot(AB, AB), 1e-8), 0, 1)
        return float(np.linalg.norm(pt - (A + t * AB)))

    def _scale(self, gender):
        if gender < -0.5:  # unknown sentinel
            return self.sc_u
        g = float(np.clip(gender, 0.0, 1.0))
        return self.sc_f + g * (self.sc_m - self.sc_f)

    def predict(self, u, v, sx, sy, is_drv, gender, du=0, dv=0, dd=0):
        Zs = self._Z_shoulder(sx, sy, self.delta + dd)
        pt = self._to_camera(u + du, v + dv, Zs)
        dist = self._pt_to_seg(pt, self._Z_seat(sx), self.sc_x_d if is_drv else self.sc_x_p)
        sm = 1.0 + self.alpha_seat * (sx - 0.5) + self.beta_seat * (sy - 0.5)
        return dist * self._scale(gender) * max(0.8, min(1.2, sm))


# ════════════ EMA 累加器 ════════════
def _iqr_filter(values, mul=1.5):
    if len(values) < 4: return values
    a = np.array(values); q1, q3 = np.percentile(a, 25), np.percentile(a, 75)
    lo, hi = q1 - mul * (q3 - q1), q3 + mul * (q3 - q1)
    return a[(a >= lo) & (a <= hi)].tolist()


class HeightAccumulator:
    def __init__(self, max_window=100, ema_alpha=0.2, stability_window=30, stability_thresh=0.5):
        self._w = max_window; self._a = 0.05
        self._d_raw = deque(maxlen=max_window); self._p_raw = deque(maxlen=max_window)
        self._d_filt = deque(maxlen=max_window); self._p_filt = deque(maxlen=max_window)
        self._d_ema = None; self._p_ema = None
        self._d_h = deque(maxlen=stability_window); self._p_h = deque(maxlen=stability_window)
        self._thresh = stability_thresh

    def _push(self, raw_dq, filt_dq, ema, hist, h):
        raw_dq.append(h); f = _iqr_filter(list(raw_dq))
        filt_dq = deque(f, maxlen=self._w)
        med = float(np.median(f)) if f else h
        ema = med if ema is None else self._a * med + (1 - self._a) * ema
        hist.append(ema)
        return ema, filt_dq

    def add_driver(self, h): self._d_ema, self._d_filt = self._push(self._d_raw, self._d_filt, self._d_ema, self._d_h, h)
    def add_passenger(self, h): self._p_ema, self._p_filt = self._push(self._p_raw, self._p_filt, self._p_ema, self._p_h, h)

    def _stats(self, filt, ema, hist):
        if not filt: return None
        a = np.array(filt, dtype=np.float64)
        ema_hist = list(hist)
        if len(ema_hist) >= hist.maxlen:
            ema_std = float(np.std(np.array(ema_hist, dtype=np.float64)))
            median_std = float(np.std(a)) if len(a) >= 30 else float("inf")
            stable = (ema_std < self._thresh and len(a) >= 30 and median_std < self._thresh * 4)
        else:
            ema_std = None; stable = False
        return {"ema": ema, "mean": float(np.mean(a)), "median": float(np.median(a)),
                "std": float(np.std(a)), "min": float(np.min(a)), "max": float(np.max(a)),
                "n": len(a), "ema_std": ema_std, "stable": stable, "fill_pct": len(a)/self._w*100}

    def driver_stats(self): return self._stats(list(self._d_filt), self._d_ema, self._d_h)
    def passenger_stats(self): return self._stats(list(self._p_filt), self._p_ema, self._p_h)

    def reset(self):
        self._d_raw.clear(); self._p_raw.clear()
        self._d_ema = None; self._p_ema = None
        self._d_h.clear(); self._p_h.clear()


# ════════════ HeightEstimator ════════════
class HeightEstimator:
    def __init__(self, models_dir: str = "height"):
        onnx_yolo = os.path.join(models_dir, "yolov8n-pose.onnx")
        from service.height_model_68abs import HeightModel68Abs
        self._model = HeightModel68Abs(models_dir)

        _ensure_torch_dll_path()
        from ultralytics import YOLO
        self._yolo = YOLO(onnx_yolo, task="pose")

        self._acc = HeightAccumulator()
        self._lock = Lock()
        # 默认座椅参数 = 数据集均值占位 (driver xs0.62/ys0.26, passenger xs0.41/ys0.59);
        # CAN 到达后 set_seat_params 覆盖为真实值
        self._seat = {"dx": 0.62, "dy": 0.26, "px": 0.41, "py": 0.59}
        self._gender = {"d": 1, "p": 1}
        self._stable_mode = "reduce"   # "continue" | "reduce" | "stop"
        self._reduce_interval = 5      # reduce时每隔N个predict调用实际跑一次YOLO
        self._stable_skip_count = 0
        self._ready = True
        self._can_received = False
        log.info("ready | model=68abs_huber")

    def set_seat_params(self, dx, dy, px, py, d_gender=1.0, p_gender=1.0):
        with self._lock:
            self._seat = {"dx": dx, "dy": dy, "px": px, "py": py}
            self._gender = {"d": d_gender, "p": p_gender}
            self._acc.reset()
        log.info("seat: d=(%s,%s) g=%s | p=(%s,%s) g=%s", dx, dy, d_gender, px, py, p_gender)

    def set_gender(self, driver_gender: float = 1.0, passenger_gender: float = 1.0):
        """从人脸属性识别更新性别（1.0=男, 0.0=女, -1=未知, 中间值连续插值）。"""
        with self._lock:
            self._gender = {"d": driver_gender, "p": passenger_gender}

    def set_stable_mode(self, mode: str):
        """mode: "continue" | "reduce" | "stop" """
        if mode not in ("continue", "reduce", "stop"):
            raise ValueError(f"Invalid stable_mode: {mode}")
        self._stable_mode = mode
        self._stable_skip_count = 0
        log.info("stable_mode=%s", mode)

    def mark_can_received(self):
        self._can_received = True
        log.info("CAN signal received, height estimation now valid")

    def is_stable(self) -> bool:
        with self._lock:
            ds = self._acc.driver_stats()
            ps = self._acc.passenger_stats()
            return (ds is not None and ps is not None and ds.get("stable") and ps.get("stable"))

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        # 稳定后根据mode决定是否跳过YOLO重型检测（仅CAN到达后生效）
        if self._can_received and self._stable_mode in ("reduce", "stop"):
            self._stable_skip_count += 1
            if self._stable_mode == "stop" and self.is_stable():
                return self.get_status()
            if self._stable_mode == "reduce" and self.is_stable():
                if self._stable_skip_count % self._reduce_interval != 0:
                    return self.get_status()

        h_img, w_img = frame.shape[:2]
        results = self._yolo(frame, conf=0.5, iou=0.7, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None or r.keypoints is None: continue
            boxes = r.boxes.data.cpu().numpy(); kpts = r.keypoints.data.cpu().numpy()
            for box, kp in zip(boxes, kpts):
                x1,y1,x2,y2,conf,_ = box
                ls=kp[LEFT_SHOULDER_IDX]; rs=kp[RIGHT_SHOULDER_IDX]
                su=sv=None
                if ls[2]>0.3 and rs[2]>0.3:
                    su=(ls[0]+rs[0])/2; sv=(ls[1]+rs[1])/2
                head_size = _calc_head_size(kp, w_img, h_img)
                dets.append({"bbox":[x1,y1,x2,y2],"conf":conf,"su":su,"sv":sv,"head_size":head_size,"kp":kp})

        # 后排过滤
        if len(dets) > 1:
            max_head = max(d["head_size"] for d in dets)
            dets = [d for d in dets if d["head_size"] >= max_head * 0.6]

        # 按照 infer_stream 的逻辑确定角色: 左=乘客, 右=驾驶员
        # 多人时按bbox位置分配，不依赖单个检测的 center_x 判断
        if len(dets) >= 2:
            dets_sorted = sorted(dets, key=lambda d: d["bbox"][0])  # 按x1排序
            dets = []
            # 最左 = 乘客
            p = dets_sorted[0]; p["role"] = "passenger"
            dets.append(p)
            # 最右 = 驾驶员（且不能是同一个bbox）
            d = dets_sorted[-1]
            if abs(d["bbox"][0] - p["bbox"][0]) > 10:
                d["role"] = "driver"; dets.append(d)
        elif len(dets) == 1:
            d = dets[0]
            d["role"] = "driver" if (d["bbox"][0]+d["bbox"][2])/2 > w_img/2 else "passenger"

        with self._lock:
            for d in dets:
                u,v=d["su"],d["sv"]
                if u is None: continue
                kp = d.get("kp")
                if kp is None: continue
                role=d["role"]; is_drv=(role=="driver")
                sx=self._seat["dx"] if is_drv else self._seat["px"]
                sy=self._seat["dy"] if is_drv else self._seat["py"]
                gender=self._gender["d"] if is_drv else self._gender["p"]
                # 生产 YOLO 输出全图像素 17 点 (x,y,conf) — 与研究缓存像素化后一致
                kp_px = np.asarray(kp, dtype=np.float64)
                h = self._model.predict(kp_px, sx, sy, gender, is_drv, shoulder_u=u, shoulder_v=v)
                if 120 <= h <= 220:
                    if is_drv: self._acc.add_driver(h)
                    else: self._acc.add_passenger(h)
            return {"driver":self._acc.driver_stats(),"passenger":self._acc.passenger_stats()}

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            ds = self._acc.driver_stats()
            ps = self._acc.passenger_stats()
            stable = bool(ds is not None and ps is not None and ds.get("stable") and ps.get("stable"))
            # 把座椅参数和性别合并到每个角色的统计数据里
            driver_info = dict(ds) if ds else None
            passenger_info = dict(ps) if ps else None
            if driver_info is not None:
                driver_info["seat_x"] = self._seat["dx"]
                driver_info["seat_y"] = self._seat["dy"]
                driver_info["gender"] = self._gender["d"]
            if passenger_info is not None:
                passenger_info["seat_x"] = self._seat["px"]
                passenger_info["seat_y"] = self._seat["py"]
                passenger_info["gender"] = self._gender["p"]
            return {"driver": driver_info, "passenger": passenger_info,
                    "stable_mode": self._stable_mode, "stable": stable,
                    "can_received": self._can_received}

    @property
    def ready(self) -> bool: return self._ready
