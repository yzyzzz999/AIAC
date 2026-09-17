#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运动检测 + YOLO body bbox 位置快速采样。"""
import cv2, numpy as np, os, time, sys

BODY_ROIS = {"left": (0.00, 0.05, 0.40, 0.80), "right": (0.52, 0.05, 1.00, 0.80)}
CAMERA_ID = int(os.getenv("CAMERA_ID", "6"))
DURATION = 10  # 秒

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"FAIL: camera {CAMERA_ID}")
    exit(1)

# 尝试加载 YOLO body 模型
yolo_model = None
try:
    from ultralytics import YOLO
    model_path = os.getenv("BODY_DET_MODEL_PATH", "./models/body/yolov8n-pose.pt")
    yolo_model = YOLO(model_path)
    import torch
    if torch.cuda.is_available():
        yolo_model.to("cuda")
    print("YOLO body model loaded OK")
except Exception as e:
    print(f"YOLO not available: {e}")

prev_gray = None
samples = {"left": [], "right": []}
body_bottoms = {"left": [], "right": []}  # body bbox bottom / H
frames = 0
start = time.time()

while time.time() - start < DURATION:
    ok, frame = cap.read()
    if not ok:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    frames += 1

    # ── 运动能量 ──
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        for slot, roi in BODY_ROIS.items():
            rx1, ry1 = int(roi[0]*W), int(roi[1]*H)
            rx2, ry2 = int(roi[2]*W), int(roi[3]*H)
            roi_diff = diff[ry1:ry2, rx1:rx2]
            samples[slot].append(float(np.mean(roi_diff)))
    prev_gray = gray

    # ── YOLO body bbox bottom ──
    if yolo_model is not None and frames % 10 == 0:
        results = yolo_model(frame, verbose=False)
        if results and results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                xc = (x1 + x2) / 2
                slot = "right" if xc >= W / 2 else "left"
                bottom_ratio = y2 / H
                body_bottoms[slot].append(bottom_ratio)

    # 保存首帧
    if frames == 1:
        cv2.imwrite("/tmp/motion_test_first.jpg", frame)

cap.release()

print(f"采样完成: {frames} 帧, {DURATION} 秒")
print()
for slot in ("left", "right"):
    if samples[slot]:
        arr = np.array(samples[slot])
        print(f"  [{slot}] 运动能量: min={arr.min():.2f}  max={arr.max():.2f}  "
              f"mean={arr.mean():.2f}  median={np.median(arr):.2f}  std={arr.std():.2f}")
        below = sum(1 for v in samples[slot] if v < 3)
        print(f"          <3.0: {below}帧 ({100*below/len(arr):.0f}%)  "
              f">=3.0: {len(arr)-below}帧 ({100*(len(arr)-below)/len(arr):.0f}%)")
        if body_bottoms[slot]:
            bb = body_bottoms[slot]
            print(f"          YOLO body bottom/y: min={min(bb):.2f}  max={max(bb):.2f}  "
                  f"mean={np.mean(bb):.2f}  median={np.median(bb):.2f}")
            front = sum(1 for b in bb if b > 0.55)
            rear = sum(1 for b in bb if b <= 0.55)
            print(f"          bottom>0.55(前排): {front}次  bottom<=0.55(后排/高): {rear}次")
    else:
        print(f"  [{slot}]: 无数据")
    print()
