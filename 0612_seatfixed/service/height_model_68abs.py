"""68+abs (107维) Huber 身高回归 — 生产纯 NumPy 推理。
替代旧 t24 CNN+几何方案。权重由 AIAC_Standalone/regression/export_tef_huber.py 导出。

特征管线复刻 final_tef_model.py:30-90，与训练时完全一致：
  - 输入关键点须为 17 点全图像素坐标 (COCO 顺序), conf 为第3列
  - 座椅外参来自 models/height/seat_params_bilevel_auto.json
  - 输出: pred = ((X - mean) / std) @ coef + intercept  (cm)
"""
from __future__ import annotations
import json, os, math
import numpy as np

# ── 座椅几何常量 (bilevel_autograd.py) ──
ALPHA_SEAT = -math.radians(15)
COSA, SINA = math.cos(ALPHA_SEAT), math.sin(ALPHA_SEAT)
R_ALIGN = np.diag([1.0, -1.0, -1.0])
J0r = np.array([-0.00179506, -0.22333345, 0.02821913])
PELV_CTR = 0.15


def rod_np(r):
    """Rodrigues 旋转矩阵 (bilevel_autograd.rod_torch 的 NumPy 版)。"""
    th = float(np.linalg.norm(r)) + 1e-8
    k = r / th
    k0, k1, k2 = k
    Kmat = np.array([[0.0, -k2, k1], [k2, 0.0, -k0], [-k1, k0, 0.0]])
    return np.eye(3) + math.sin(th) * Kmat + (1 - math.cos(th)) * (Kmat @ Kmat)


def seat_geom_np(theta, role, xs, ys):
    """骨盆原点 Os 与旋转/平移 (bilevel_autograd.seat_geom_torch 的 NumPy 版)。
    theta (18,): [driver rvec(3), tx,ty,tz, b,d,e,  passenger rvec(3), tx,tz, d,e, c, h_pel]
    """
    if role == 'driver':
        rvec = theta[0:3]; tx, ty, tz = theta[3], theta[4], theta[5]
        b, d, e = theta[6], theta[7], theta[8]; c = theta[16]
    else:
        rvec = theta[9:12]; tx, ty, tz = theta[12], theta[4], theta[13]
        b, d, e = theta[6], theta[14], theta[15]; c = theta[16]
    h_pel = theta[17] if theta.size > 17 else PELV_CTR
    R = rod_np(rvec)
    t = np.array([tx, ty, tz])
    th = d * ys + e
    s, cc = math.sin(th), math.cos(th)
    E = np.array([0.0, -c * s, b * xs - c * cc])
    Ph = np.array([0.0, 0.0, b * xs])
    dY = Ph[1] - E[1]; dZ = Ph[2] - E[2]
    Pv = np.array([0.0, E[1] + COSA * dY - SINA * dZ, E[2] + SINA * dY + COSA * dZ])
    Os = Pv + np.array([0.0, -h_pel, 0.0]) - R_ALIGN @ J0r
    return R, t, E, Pv, Os


class HeightModel68Abs:
    """107维特征 + Huber 线性回归推理。"""

    def __init__(self, models_dir: str):
        npz_path = os.path.join(models_dir, "huber_68abs_tef.npz")
        seat_path = os.path.join(models_dir, "seat_params_bilevel_auto.json")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"缺少权重: {npz_path}")
        if not os.path.exists(seat_path):
            raise FileNotFoundError(f"缺少座椅外参: {seat_path}")

        d = np.load(npz_path)
        self.coef = d["coef"].astype(np.float64)
        self.intercept = float(d["intercept"][0])
        self.mean = d["feature_mean"].astype(np.float64)
        self.std = d["feature_std"].astype(np.float64)
        self.fx, self.fy = float(d["fx"]), float(d["fy"])
        self.cx, self.cy = float(d["cx"]), float(d["cy"])
        self.A, self.B_, self.Cg = float(d["seat_A"]), float(d["seat_B"]), float(d["seat_Cg"])
        self.Ep, self.Dp = float(d["seat_Ep"]), float(d["seat_Dp"])
        self.Db = d["seat_Db"].astype(np.float64)
        self.Pb = d["seat_Pb"].astype(np.float64)

        prm = json.load(open(seat_path))
        dp, pp = prm["driver"], prm["passenger"]
        self.theta = np.array([
            *dp["rvec"], dp["tvec"][0], dp["tvec"][1], dp["tvec"][2],
            dp["b"], dp["d"], dp["e"],
            *pp["rvec"], pp["tvec"][0], pp["tvec"][2], pp["d"], pp["e"],
            pp["c"], prm["h_pel"],
        ], dtype=np.float64)

    def _TZ(self, xs, ys, is_drv):
        R, t, _, _, Os = seat_geom_np(self.theta, "driver" if is_drv else "passenger", xs, ys)
        return float((Os @ R.T + t)[2])

    def build_features(self, kp, xs, ys, gender, is_drv, shoulder_u=None, shoulder_v=None):
        """kp: (17,3) 全图像素 (x,y,conf)。返回 (107,)。
        研究约定: pts = 肩中点(全图像素), kpts 归一化→×2560/1440。
        生产 YOLO 直接给全图像素, 故 KX/KY 直接用 kp[:,0], kp[:,1]。
        is_drv: bool, 角色决定座椅外参 (driver/passenger) 与靠背投影表 Db/Pb。
        """
        N = 1
        KX = kp[:, 0].reshape(1, 17).astype(np.float64)
        KY = kp[:, 1].reshape(1, 17).astype(np.float64)
        CF = kp[:, 2].reshape(1, 17).astype(np.float64)
        # 肩中点 (全图像素), 与研究 pts 一致
        if shoulder_u is None or shoulder_v is None:
            su = (KX[:, 5] + KX[:, 6]) / 2.0
            sv = (KY[:, 5] + KY[:, 6]) / 2.0
        else:
            su = np.array([[shoulder_u]], np.float64)
            sv = np.array([[shoulder_v]], np.float64)
        PU = su.copy(); PV = sv.copy()
        XS = np.array([[xs]], np.float64)
        YS = np.array([[ys]], np.float64)
        G = np.array([[gender]], np.float64)
        D = np.array([[1.0 if is_drv else 0.0]])

        # 骨盆深度 (按角色选外参)
        TZ = np.array([[self._TZ(xs, ys, is_drv)]])

        F = np.zeros((N, 68))
        sw = np.hypot(KX[:, 5] - KX[:, 6], KY[:, 5] - KY[:, 6])
        sx = (KX[:, 5] + KX[:, 6]) / 2; sy = (KY[:, 5] + KY[:, 6]) / 2
        hh = np.hypot(KX[:, 0] - sx, KY[:, 0] - sy)
        hx = (KX[:, 11] + KX[:, 12]) / 2; hy = (KY[:, 11] + KY[:, 12]) / 2
        to = np.hypot(sx - hx, sy - hy)
        F[:, 0] = PV; F[:, 1] = PU; F[:, 2] = XS; F[:, 3] = YS
        F[:, 4] = sw; F[:, 5] = hh; F[:, 6] = to
        F[:, 7] = sw / (hh + 1e-6); F[:, 8] = sw / (to + 1e-6)
        F[:, 9] = CF.mean(1); F[:, 10] = self.fy / (PV - self.cy + 1e-8)
        XbYbA1A2 = self.Db[None, :] if D[0, 0] == 1 else self.Pb[None, :]
        Xb_, Yb_, A1_, A2_ = XbYbA1A2[:, 0], XbYbA1A2[:, 1], XbYbA1A2[:, 2], XbYbA1A2[:, 3]
        th_a = self.Ep * YS + self.Dp; Z = self.A * XS + self.B_ + self.Cg * np.cos(th_a)
        F[:, 11] = self.fx * (Xb_ + A1_ * np.sin(th_a)) / Z + self.cx
        F[:, 12] = self.fy * (Yb_ + A2_ * XS) / Z + self.cy
        F[:, 13] = PU - F[:, 11]; F[:, 14] = PV - F[:, 12]; F[:, 15] = Z; F[:, 16] = G
        for c in range(17):
            F[:, 17 + c * 3] = KX[:, c]; F[:, 17 + c * 3 + 1] = KY[:, c]; F[:, 17 + c * 3 + 2] = CF[:, c]

        Abs = np.zeros((N, 11, 3))
        for c in range(11):
            Abs[:, c, 0] = (KY[:, c] - self.cy) / self.fy * TZ
            Abs[:, c, 1] = (KX[:, c] - self.cx) / self.fx * TZ
        Ex = np.zeros((N, 22))
        for c in range(11):
            Ex[:, c] = Abs[:, c, 0]; Ex[:, 11 + c] = Abs[:, c, 1]
        Rel = np.zeros((N, 11))
        for c in range(11):
            Rel[:, c] = Abs[:, c, 0] - Abs[:, 5, 0]
        AL = np.zeros((N, 6))
        AL[:, 0] = sw * TZ / self.fy; AL[:, 1] = Abs[:, 5, 0]; AL[:, 2] = Abs[:, 0, 0]
        AL[:, 3] = Abs[:, 0, 0] - Abs[:, 5, 0]; AL[:, 4] = TZ; AL[:, 5] = XS
        X = np.hstack([F, np.hstack([Ex, Rel, AL])])
        return X[0]

    def predict(self, kp, xs, ys, gender, is_drv, shoulder_u=None, shoulder_v=None):
        X = self.build_features(kp, xs, ys, gender, is_drv, shoulder_u, shoulder_v)
        return float(((X - self.mean) / self.std) @ self.coef + self.intercept)
