#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运动检测可行性测试。
读取摄像头，在左右座椅 ROI 内计算帧差能量，叠加显示。
不依赖任何模型，纯 cv2 + numpy。
Ctrl+C 退出。
"""

import cv2
import numpy as np
import os
import time

# ── 与 body_detector 相同的 ROI ──
BODY_ROIS = {
    "left":  (0.00, 0.05, 0.40, 0.80),
    "right": (0.52, 0.05, 1.00, 0.80),
}

CAMERA_ID = int(os.getenv("CAMERA_ID", "6"))
PREVIEW_W, PREVIEW_H = 1280, 720

def draw_text(img, text, x, y, color=(255, 255, 255)):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"无法打开摄像头 {CAMERA_ID}")
        return

    prev_gray = None
    motion_history = {"left": [], "right": []}
    frame_count = 0

    print("运动能量测试开始。观察每个 ROI 的 Motion 值。")
    print("空座椅期望: ~0.5-2  有人期望: ~3-30   Ctrl+C 退出")
    print()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        scale_x, scale_y = PREVIEW_W / W, PREVIEW_H / H

        # ── 运动能量计算 ──
        motion = {}
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            for slot_name, roi in BODY_ROIS.items():
                rx1 = int(roi[0] * W)
                ry1 = int(roi[1] * H)
                rx2 = int(roi[2] * W)
                ry2 = int(roi[3] * H)
                roi_diff = diff[ry1:ry2, rx1:rx2]
                energy = float(np.mean(roi_diff))
                motion[slot_name] = round(energy, 2)
                motion_history[slot_name].append(energy)
                if len(motion_history[slot_name]) > 50:
                    motion_history[slot_name].pop(0)
        else:
            motion = {"left": 0.0, "right": 0.0}

        prev_gray = gray

        # ── 绘制 ──
        roi_colors = {"left": (255, 0, 0), "right": (0, 255, 0)}
        for slot_name, roi in BODY_ROIS.items():
            rx1 = int(roi[0] * PREVIEW_W)
            ry1 = int(roi[1] * PREVIEW_H)
            rx2 = int(roi[2] * PREVIEW_W)
            ry2 = int(roi[3] * PREVIEW_H)
            cv2.rectangle(preview, (rx1, ry1), (rx2, ry2), roi_colors[slot_name], 2)

            e = motion.get(slot_name, 0.0)
            hist = motion_history[slot_name]
            avg_10 = round(float(np.mean(hist[-10:])), 2) if len(hist) >= 10 else e

            # 颜色：能量高=暖色，低=冷色
            if e < 3:
                e_color = (200, 200, 200)  # 灰：空
            elif e < 8:
                e_color = (0, 255, 255)     # 黄：呼吸级微动
            elif e < 20:
                e_color = (0, 165, 255)     # 橙：活动
            else:
                e_color = (0, 0, 255)       # 红：大幅运动

            draw_text(preview, f"{slot_name} motion={e:.2f} avg10={avg_10:.2f}",
                      rx1 + 6, ry1 + 22, e_color)

        # 全局信息
        draw_text(preview, f"Frame: {frame_count}", 10, PREVIEW_H - 10, (200, 200, 200))

        cv2.imshow("Motion Test - Ctrl+C or Q to quit", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("测试结束")


if __name__ == "__main__":
    main()
