"""
camera_preview.py
================
实时摄像头预览 + 人脸识别可视化

用法:
    python camera_preview.py
    python camera_preview.py --camera 0
    python camera_preview.py --camera 0 --no-window   # 无窗口模式，仅通过 /preview 接口查看
"""

import argparse
import cv2
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.gallery import Gallery, DEFAULT_SIM_THRESHOLD
from service.detector import FaceDetector
from service.gender_age_recognizer import GenderAgeRecognizer


def normalize_emb(emb):
    return emb / (np.linalg.norm(emb) + 1e-8)

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="摄像头实时人脸识别预览")
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备ID")
    parser.add_argument("--gallery", type=str, default="gallery.pkl", help="Gallery 文件路径")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIM_THRESHOLD, help="识别阈值")
    parser.add_argument("--no-window", action="store_true", help="不显示窗口，仅输出日志")
    args = parser.parse_args()

    print(f"初始化人脸检测器...")
    detector = FaceDetector()
    detector.ensure()

    print(f"加载 Gallery: {args.gallery}")
    gallery = Gallery(args.gallery, sim_threshold=args.threshold)
    print(f"Gallery 已加载: {gallery.count()} 个身份")

    print("初始化年龄/性别识别器...")
    gender_age = GenderAgeRecognizer()

    print(f"打开摄像头 {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {args.camera}")
        return

    frame_count = 0
    fps = 0
    fps_timer = time.time()
    fps_count = 0

    print("\n开始实时预览 (按 ESC 或 Q 退出)")
    print(f"{'帧号':>6} | {'检测':>4} | {'识别':>4} | {'身份':<30} | FPS")
    print("-" * 70)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("警告: 无法读取帧")
            time.sleep(0.1)
            continue

        frame_count += 1

        # FPS 计算
        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_timer = time.time()

        # 人脸检测 (每帧都检测)
        detected = detector.detect_frame(frame)
        H, W = frame.shape[:2]

        identities = []
        total_known = 0

        for face in detected:
            # 识别
            match = gallery.recognize(face.emb, args.threshold)
            is_known = match["is_known"]
            identity_id = match["identity_id"] or ""
            name = match.get("name") or identity_id
            similarity = match["similarity"]
            attr = gender_age.predict_face(frame, face.bbox)
            attr_label = gender_age.format_attr(attr)

            if is_known:
                total_known += 1

            identities.append((face, match))

            # 绘制
            x1, y1, x2, y2 = map(int, face.bbox)
            color = (0, 255, 0) if is_known else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            label = f"ID: {name or 'unknown'}  SIM: {similarity:.2f}"
            if attr_label:
                label = f"{label}  {attr_label}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.85
            thickness = 2
            padding = 6
            (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_y = max(y1 - 10, lh + baseline + padding * 2)
            bg_top = text_y - lh - baseline - padding
            bg_bottom = text_y + baseline + padding
            bg_right = min(frame.shape[1] - 1, x1 + lw + padding * 2)
            cv2.rectangle(frame, (x1, bg_top), (bg_right, bg_bottom), color, -1)
            cv2.putText(frame, label, (x1 + padding, text_y), font, font_scale, (0, 0, 0), thickness)

        # 信息叠加
        info = f"Faces: {len(detected)}  Known: {total_known}  Gallery: {gallery.count()}  FPS: {fps}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # 控制台输出 (每秒一次)
        if fps_count == 0:
            face_ids = ", ".join([m[1].get("name") or m[1]["identity_id"] or "?" for m in identities]) or "none"
            print(f"{frame_count:>6} | {len(detected):>4} | {total_known:>4} | {face_ids:<30} | {fps}")

        if not args.no_window:
            cv2.imshow("Vehicle Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("\n退出预览")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
