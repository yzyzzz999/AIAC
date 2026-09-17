"""
main.py — 离线评估 + 摄像头模式入口。

模块拆分:
  pipeline/data_loader.py  — GT 标注解析、data/ 图片扫描
  pipeline/fusion.py       — 多帧性别/年龄融合、RGB/NIR 跨模态融合
  pipeline/sequence.py     — 单 RGB/NIR 序列完整处理
  pipeline/evaluate.py     — 统计与评估
  pipeline/visualize.py    — fused 可视化 + RUN_ONLY_FUSED 模式
"""

import os
import sys
import time
import math
import atexit
import argparse
from datetime import datetime, timedelta

import cv2
import numpy as np

from service.detector import FaceDetector
from service.gender_age_backends import create_gender_age_backend
from utils import safe_float, draw_results, select_front_row_faces

from pipeline.data_loader import (ensure_dir, norm_str, natural_key,
                                  find_person_summary_file, build_gt_from_person_summary_csv,
                                  collect_all_folders_from_data, build_sequence_dict_from_folders,
                                  save_csv)
from pipeline.fusion import (is_valid_prediction, compute_frame_weight,
                             fuse_rgb_nir_summary_row, get_row_value)
from pipeline.sequence import process_one_sequence, assign_faces_to_slots, predict_face_attr
from pipeline.evaluate import save_statistics, evaluate_predictions
from pipeline.visualize import (save_fused_visualization,
                                run_fused_only_from_existing_results)

# ---- 常量 ----
DEFAULT_SOURCE = "folder"
DEFAULT_CAMERA_ID = 0
DEFAULT_CAMERA_FRAME_INTERVAL = 3
DEFAULT_CAMERA_CTX_ID = -1
DEFAULT_ATTR_DEVICE = "cpu"


# ---- 工具 ----

def safe_int(x, default=0):
    try:
        return int(float(norm_str(x)))
    except Exception:
        return default


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def setup_run_log(save_root):
    ensure_dir(save_root)
    log_path = os.path.join(save_root, "run.log")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(old_stdout, log_file)
    sys.stderr = TeeStream(old_stderr, log_file)
    def close_log():
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_file.close()
    atexit.register(close_log)
    print("=" * 60)
    print(f"[LOG] run log: {log_path}")
    print(f"[LOG] start time: {datetime.now().isoformat(timespec='seconds')}")
    return log_path


# ---- CLI ----

def parse_args():
    parser = argparse.ArgumentParser(description="AI 空调项目：文件夹批处理 / 实时摄像头识别")
    parser.add_argument("--source", choices=["folder", "camera"], default=DEFAULT_SOURCE)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--camera-id", default=str(DEFAULT_CAMERA_ID))
    parser.add_argument("--camera-frame-interval", type=int, default=DEFAULT_CAMERA_FRAME_INTERVAL)
    parser.add_argument("--ctx-id", type=int, default=DEFAULT_CAMERA_CTX_ID)
    parser.add_argument("--attr-device", choices=["auto", "cpu", "cuda"], default=DEFAULT_ATTR_DEVICE)
    parser.add_argument("--attr-backend", choices=["onnx", "tensorrt"],
                        default=os.getenv("GENDER_AGE_BACKEND", "onnx"))
    parser.add_argument("--gender-thr", type=float, default=0.45)
    parser.add_argument("--age-thr", type=float, default=0.30)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def parse_camera_id(camera_id):
    camera_id = str(camera_id).strip()
    if camera_id.isdigit():
        return int(camera_id)
    return camera_id


# ---- 摄像头模式 ----

def predict_frame_attrs(image_bgr, detector, predictor, gender_thr=0.45, age_thr=0.30):
    h, w = image_bgr.shape[:2]
    raw_faces = detector.detect_faces_dicts(image_bgr)
    selected_faces = select_front_row_faces(raw_faces, w, h)
    assigned = assign_faces_to_slots(selected_faces, {"face1": None, "face2": None})
    current_predictions = {}
    for face_name in ["face1", "face2"]:
        assigned_face = assigned.get(face_name)
        if assigned_face is None:
            continue
        try:
            pred = predict_face_attr(image_bgr, assigned_face, predictor)
            if is_valid_prediction(pred, gender_thr=gender_thr, age_thr=age_thr):
                current_predictions[face_name] = pred
        except Exception as e:
            print(f"[WARN] camera {face_name} prediction failed: {e}")
    return current_predictions


def run_camera_mode(args):
    camera_id = parse_camera_id(args.camera_id)
    frame_interval = max(1, int(args.camera_frame_interval))

    print("=" * 60)
    print("[INFO] Source: camera")
    print(f"[INFO] camera_id={camera_id}, frame_interval={frame_interval}")
    print("[INFO] 按 ESC 或 q 退出摄像头模式")

    detector = FaceDetector()
    attr_device = None if args.attr_device == "auto" else args.attr_device
    predictor = create_gender_age_backend(backend=args.attr_backend, model_dir="./FLIP-base-32", device=attr_device)
    print(f"[INFO] attr_backend={args.attr_backend}, attr_device={getattr(predictor, 'device', attr_device or 'auto')}")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头/视频流: {camera_id}")

    frame_count = 0
    latest_predictions = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 摄像头读取失败")
                break
            frame_count += 1
            if frame_count % frame_interval == 0:
                latest_predictions = predict_frame_attrs(frame, detector, predictor)
                face1 = latest_predictions.get("face1", {})
                face2 = latest_predictions.get("face2", {})
                print(f"[FRAME {frame_count}] 副驾(face1): {face1.get('gender', 'unknown')} {face1.get('age_group', 'unknown')} | "
                      f"主驾(face2): {face2.get('gender', 'unknown')} {face2.get('age_group', 'unknown')}")
            if not args.no_show:
                vis = draw_results(frame, latest_predictions) if latest_predictions else frame
                cv2.imshow("AI AC Camera Recognition", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()
    print("[INFO] Camera mode stopped.")


# ---- 主函数 ----

def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    setup_run_log(os.path.join(base_dir, "outputs_sequence"))

    if args.source == "camera":
        run_camera_mode(args)
        return

    # ---- 路径 ----
    person_summary_path = find_person_summary_file(base_dir)
    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(base_dir, data_root)

    save_root = os.path.join(base_dir, "outputs_sequence")
    save_vis_dir = os.path.join(save_root, "vis")
    save_json_dir = os.path.join(save_root, "json")
    save_fused_vis_dir = os.path.join(save_vis_dir, "fused")
    save_summary_csv = os.path.join(save_root, "result_summary_detailed.csv")
    save_debug_csv = os.path.join(save_root, "result_debug.csv")
    save_stats_json = os.path.join(save_root, "result_stats.json")
    save_with_result_csv = os.path.join(save_root, "sequences_with_result.csv")
    save_without_result_csv = os.path.join(save_root, "sequences_without_result.csv")
    save_gt_csv = os.path.join(save_root, "gt_from_second_file.csv")
    save_one_folder_csv = os.path.join(save_root, "person_one_folder_one_sample.csv")
    save_clean_rows_csv = os.path.join(save_root, "person_second_file_clean_rows.csv")
    save_person_stats_json = os.path.join(save_root, "person_second_file_stats.json")
    save_fused_summary_csv = os.path.join(save_root, "result_summary_rgb_nir_fused.csv")
    save_eval_merged_csv = os.path.join(save_root, "eval_merged_fused.csv")
    save_eval_json = os.path.join(save_root, "eval_metrics_fused.json")

    for d in [save_root, save_vis_dir, save_json_dir, save_fused_vis_dir]:
        ensure_dir(d)

    print(f"[DEBUG] cwd: {os.getcwd()}")
    print(f"[DEBUG] base_dir: {base_dir}")
    print(f"[DEBUG] data_root: {data_root}")
    print(f"[DEBUG] person_summary_path: {person_summary_path}")

    # ---- RUN_ONLY_FUSED 开关 ----
    RUN_ONLY_FUSED = False

    if RUN_ONLY_FUSED:
        print("=" * 60)
        print("[INFO] RUN_ONLY_FUSED = True")
        build_gt_from_person_summary_csv(
            person_summary_path=person_summary_path, save_gt_csv=save_gt_csv,
            save_one_folder_csv=save_one_folder_csv, save_clean_rows_csv=save_clean_rows_csv,
            save_person_stats_json=save_person_stats_json,
        )
        run_fused_only_from_existing_results(
            save_summary_csv=save_summary_csv, save_fused_summary_csv=save_fused_summary_csv,
            save_fused_vis_dir=save_fused_vis_dir, save_json_dir=save_json_dir,
        )
        evaluate_predictions(gt_path=save_gt_csv, pred_csv_path=save_fused_summary_csv,
                             save_merged_csv=save_eval_merged_csv, save_eval_json=save_eval_json)
        print("=" * 60)
        print("[INFO] Fused-only Done.")
        return

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    # ---- Step 1: 读取标注 ----
    _, _, gt_folders = build_gt_from_person_summary_csv(
        person_summary_path=person_summary_path, save_gt_csv=save_gt_csv,
        save_one_folder_csv=save_one_folder_csv, save_clean_rows_csv=save_clean_rows_csv,
        save_person_stats_json=save_person_stats_json,
    )

    # ---- Step 2: 扫描 data ----
    sample_folders = collect_all_folders_from_data(data_root)
    seq_dict, _ = build_sequence_dict_from_folders(data_root=data_root, sample_folders=sample_folders)
    print(f"[INFO] folders with GT: {len(gt_folders)}, folders to predict: {len(sample_folders)}")

    sequence_ids = sorted(seq_dict.keys(), key=natural_key)
    if not sequence_ids:
        raise FileNotFoundError(f"data 目录中的 folder 都没有找到图片: {data_root}")

    # ---- Step 3: 参数 ----
    frame_stride = 1
    max_frames_per_sequence = 12
    save_frame_vis = False
    gender_thr = args.gender_thr
    age_thr = args.age_thr
    bbox_history_min = 4
    bbox_center_x_scale = 1.0; bbox_center_y_scale = 1.0
    bbox_area_tol_low = 0.6; bbox_area_tol_high = 1.8
    bbox_wh_tol_low = 0.65; bbox_wh_tol_high = 1.5
    bbox_weight_lambda = 0.25

    total_images = sum(len(v.get("rgb", [])) + len(v.get("nir", [])) for v in seq_dict.values())
    print("=" * 60)
    print("[INFO] Start Prediction")
    print(f"[INFO] total folders: {len(sample_folders)}, with images: {len(sequence_ids)}, total images: {total_images}")

    run_start_dt = datetime.now()
    run_start_ts = time.monotonic()
    print(f"[TIME] run_start: {run_start_dt.isoformat(timespec='seconds')}")

    # ---- Step 4: 加载模型 ----
    detector = FaceDetector()
    attr_device = None if args.attr_device == "auto" else args.attr_device
    predictor = create_gender_age_backend(backend=args.attr_backend, model_dir="./FLIP-base-32", device=attr_device)
    print(f"[INFO] attr_backend={args.attr_backend}, attr_device={getattr(predictor, 'device', attr_device or 'auto')}")

    # ---- Step 5: 逐个 folder 处理 ----
    summary_rows, debug_rows, fused_rows = [], [], []
    total_sequences = len(sequence_ids)

    for idx, seq_id in enumerate(sequence_ids, start=1):
        folder_start_ts = time.monotonic()
        now_dt = datetime.now()
        elapsed = folder_start_ts - run_start_ts
        if idx > 1:
            avg = elapsed / float(idx - 1)
            eta_seconds = avg * float(total_sequences - idx + 1)
            eta_finish = now_dt + timedelta(seconds=eta_seconds)
            print(f"\n[TIME] progress={idx - 1}/{total_sequences}, elapsed={format_duration(elapsed)}, "
                  f"avg={format_duration(avg)}, eta={format_duration(eta_seconds)}, "
                  f"eta_finish={eta_finish.isoformat(timespec='seconds')}")
        print(f"[INFO] Processing {idx}/{total_sequences}: {seq_id}")

        rgb_paths = seq_dict[seq_id].get("rgb", [])
        nir_paths = seq_dict[seq_id].get("nir", [])
        rgb_summary_row = nir_summary_row = None

        for modality, modal_paths in [("rgb", rgb_paths), ("nir", nir_paths)]:
            if not modal_paths:
                print(f"[WARN] {seq_id} 没有 {modality.upper()} 图片")
                continue
            summary_row, debug_row = process_one_sequence(
                seq_id=f"{seq_id}_{modality}", image_paths=modal_paths,
                detector=detector, predictor=predictor,
                save_vis_dir=save_vis_dir, save_json_dir=save_json_dir,
                frame_stride=frame_stride, max_frames_per_sequence=max_frames_per_sequence,
                save_frame_vis=save_frame_vis, gender_thr=gender_thr, age_thr=age_thr,
                bbox_history_min=bbox_history_min, bbox_center_x_scale=bbox_center_x_scale,
                bbox_center_y_scale=bbox_center_y_scale, bbox_area_tol_low=bbox_area_tol_low,
                bbox_area_tol_high=bbox_area_tol_high, bbox_wh_tol_low=bbox_wh_tol_low,
                bbox_wh_tol_high=bbox_wh_tol_high, bbox_weight_lambda=bbox_weight_lambda,
            )
            summary_row["modality"] = modality
            debug_row["modality"] = modality
            summary_rows.append(summary_row)
            debug_rows.append(debug_row)
            if modality == "rgb":
                rgb_summary_row = summary_row
            else:
                nir_summary_row = summary_row

        if rgb_summary_row is not None or nir_summary_row is not None:
            fused_row = fuse_rgb_nir_summary_row(seq_id, rgb_summary_row, nir_summary_row)
            fused_vis_path = save_fused_visualization(
                seq_id=seq_id, fused_row=fused_row,
                rgb_row=rgb_summary_row or {}, nir_row=nir_summary_row or {},
                save_fused_vis_dir=save_fused_vis_dir, save_json_dir=save_json_dir,
            )
            fused_row["fused_vis_path"] = fused_vis_path
            fused_rows.append(fused_row)
            print(f"[INFO] 融合结果: 主驾 {fused_row['pred_gender0']}({fused_row['pred_age0']}), "
                  f"副驾 {fused_row['pred_gender1']}({fused_row['pred_age1']})")

        folder_elapsed = time.monotonic() - folder_start_ts
        elapsed = time.monotonic() - run_start_ts
        avg = elapsed / float(idx)
        eta_seconds = avg * float(total_sequences - idx)
        eta_finish = datetime.now() + timedelta(seconds=eta_seconds)
        print(f"[TIME] folder_done: {datetime.now().isoformat(timespec='seconds')}, "
              f"folder_elapsed={format_duration(folder_elapsed)}, progress={idx}/{total_sequences}, "
              f"elapsed={format_duration(elapsed)}, avg={format_duration(avg)}, "
              f"eta={format_duration(eta_seconds)}, eta_finish={eta_finish.isoformat(timespec='seconds')}")

    # ---- Step 6: 保存 ----
    summary_fieldnames = [
        "sequence_id", "video", "modality", "num_frames", "processed_frames", "sampled_frames", "frame_stride",
        "frames_with_any_face", "frames_with_two_faces",
        "face1_attr_reject_count", "face2_attr_reject_count",
        "face1_valid_frame_count", "face2_valid_frame_count", "face1_found", "face2_found", "remark",
        "face1_bbox", "face1_gender", "face1_gender_score", "face1_age_group", "face1_age_score", "face1_frame_weight_mean",
        "face2_bbox", "face2_gender", "face2_gender_score", "face2_age_group", "face2_age_score", "face2_frame_weight_mean",
        "pred_person_forward_number", "pred_gender0", "pred_age0", "pred_gender1", "pred_age1",
        "rep_frame_path", "result_vis_path",
    ]
    debug_fieldnames = [
        "sequence_id", "video", "modality", "num_frames", "processed_frames", "sampled_frames",
        "frames_with_any_face", "frames_with_two_faces",
        "face1_attr_reject_count", "face2_attr_reject_count",
        "face1_valid_frame_count", "face2_valid_frame_count", "face1_found", "face2_found", "remark",
    ]
    fused_fieldnames = [
        "sequence_id", "video",
        "rgb_pred_person_forward_number", "rgb_pred_gender0", "rgb_pred_age0", "rgb_pred_gender1", "rgb_pred_age1",
        "nir_pred_person_forward_number", "nir_pred_gender0", "nir_pred_age0", "nir_pred_gender1", "nir_pred_age1",
        "pred_person_forward_number", "pred_gender0", "pred_age0", "pred_gender1", "pred_age1",
        "rgb_result_vis_path", "nir_result_vis_path", "fused_vis_path",
        "rgb_rep_frame_path", "nir_rep_frame_path",
    ]
    save_csv(summary_rows, save_summary_csv, summary_fieldnames)
    save_csv(debug_rows, save_debug_csv, debug_fieldnames)
    save_csv(fused_rows, save_fused_summary_csv, fused_fieldnames)

    # ---- Step 7: 评估 ----
    save_statistics(debug_rows=debug_rows, save_stats_json=save_stats_json,
                    save_with_result_csv=save_with_result_csv, save_without_result_csv=save_without_result_csv)
    evaluate_predictions(gt_path=save_gt_csv, pred_csv_path=save_fused_summary_csv,
                         save_merged_csv=save_eval_merged_csv, save_eval_json=save_eval_json)

    print("=" * 60)
    total_elapsed = time.monotonic() - run_start_ts
    print("[INFO] Done.")
    print(f"[TIME] run_end: {datetime.now().isoformat(timespec='seconds')}")
    print(f"[TIME] total_elapsed: {format_duration(total_elapsed)}")
    print(f"[INFO] fused summary: {save_fused_summary_csv}")
    print(f"[INFO] eval metrics: {save_eval_json}")


if __name__ == "__main__":
    main()
