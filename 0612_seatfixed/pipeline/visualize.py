"""
pipeline/visualize.py
=====================
Fused 可视化图生成 + RUN_ONLY_FUSED 模式。
"""

import os
import ast
import json

import cv2
import numpy as np

from pipeline.data_loader import norm_str, ensure_dir, natural_key, save_csv
from pipeline.fusion import fuse_rgb_nir_summary_row, get_row_value


def parse_bbox_value(v):
    if isinstance(v, list):
        if len(v) == 4:
            try:
                return [float(x) for x in v]
            except Exception:
                return []
        return []
    s = norm_str(v)
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)) and len(obj) == 4:
            return [float(x) for x in obj]
    except Exception:
        pass
    return []


def draw_fused_results_no_score(image_bgr, results):
    vis = image_bgr.copy()
    role_styles = {
        "face1": {"color": (255, 255, 0), "name": "FACE1"},
        "face2": {"color": (255, 255, 0), "name": "FACE2"},
    }

    def draw_text_with_outline(img, text, org, font, font_scale, text_color, outline_color, thickness):
        x, y = org
        cv2.putText(img, text, (x, y), font, font_scale, outline_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    placed_label_boxes = []

    def overlaps(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    for role, info in results.items():
        bbox = info.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        style = role_styles.get(role, {"color": (255, 255, 0), "name": role.upper()})
        color = style["color"]
        role_name = style["name"]
        gender = str(info.get("gender", "") or "")
        age_group = str(info.get("age_group", "") or "")

        label_1 = f"{role_name}"
        label_2 = f"Gender: {gender}"
        label_3 = f"Age: {age_group}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 4)

        font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.72; thickness = 2
        pad_x = 10; pad_y = 8; line_gap = 10

        (w1, h1), _ = cv2.getTextSize(label_1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(label_2, font, font_scale, thickness)
        (w3, h3), _ = cv2.getTextSize(label_3, font, font_scale, thickness)

        text_w = max(w1, w2, w3)
        text_h = h1 + h2 + h3 + line_gap * 2 + pad_y * 2
        img_h, img_w = vis.shape[:2]

        box_x1 = x1; box_y2 = y1 - 8; box_y1 = box_y2 - text_h
        box_x2 = box_x1 + text_w + pad_x * 2

        if box_y1 < 0:
            box_y1 = y2 + 8; box_y2 = box_y1 + text_h
        if box_x2 > img_w:
            box_x1 = max(0, img_w - (text_w + pad_x * 2)); box_x2 = box_x1 + text_w + pad_x * 2
        if box_y2 > img_h:
            box_y2 = img_h - 2; box_y1 = max(0, box_y2 - text_h)

        candidate_box = [box_x1, box_y1, box_x2, box_y2]
        shift_step = text_h + 8
        max_try = 6; try_count = 0

        while any(overlaps(candidate_box, old_box) for old_box in placed_label_boxes) and try_count < max_try:
            new_y1 = candidate_box[1] - shift_step; new_y2 = candidate_box[3] - shift_step
            if new_y1 >= 0:
                candidate_box = [candidate_box[0], new_y1, candidate_box[2], new_y2]
            else:
                new_y1 = candidate_box[1] + shift_step; new_y2 = candidate_box[3] + shift_step
                if new_y2 <= img_h:
                    candidate_box = [candidate_box[0], new_y1, candidate_box[2], new_y2]
                else:
                    break
            try_count += 1

        box_x1, box_y1, box_x2, box_y2 = candidate_box
        placed_label_boxes.append(candidate_box)
        cv2.rectangle(vis, (box_x1, box_y1), (box_x2, box_y2), color, -1)
        cv2.rectangle(vis, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)

        text_x = box_x1 + pad_x; y_text = box_y1 + pad_y + h1
        draw_text_with_outline(vis, label_1, (text_x, y_text), font, font_scale, (255, 255, 255), (0, 0, 0), thickness)
        y_text += h2 + line_gap
        draw_text_with_outline(vis, label_2, (text_x, y_text), font, font_scale, (255, 255, 255), (0, 0, 0), thickness)
        y_text += h3 + line_gap
        draw_text_with_outline(vis, label_3, (text_x, y_text), font, font_scale, (255, 255, 255), (0, 0, 0), thickness)

    return vis


def load_rep_frame_bbox_from_json(json_path, rep_frame_path, slot_name):
    if not json_path or not os.path.isfile(json_path):
        return []
    rep_frame_path = norm_str(rep_frame_path)
    if not rep_frame_path:
        return []
    key = "all_frames_face1" if slot_name == "face1" else "all_frames_face2"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    frames = data.get(key, [])
    for item in frames:
        if not isinstance(item, dict):
            continue
        if norm_str(item.get("frame_path", "")) == rep_frame_path:
            bbox = parse_bbox_value(item.get("bbox", []))
            if bbox:
                return bbox
    rep_base = os.path.basename(rep_frame_path)
    for item in frames:
        if not isinstance(item, dict):
            continue
        if os.path.basename(norm_str(item.get("frame_path", ""))) == rep_base:
            bbox = parse_bbox_value(item.get("bbox", []))
            if bbox:
                return bbox
    return []


def pick_rgb_bbox_for_fused(seq_id, rgb_row, nir_row, save_json_dir, slot_name):
    rgb_rep_frame = norm_str(get_row_value(rgb_row, "rep_frame_path", ""))
    rgb_json_path = os.path.join(save_json_dir, f"{seq_id}_rgb.json") if save_json_dir else ""
    bbox = load_rep_frame_bbox_from_json(rgb_json_path, rgb_rep_frame, slot_name)
    if bbox:
        return bbox
    bbox = parse_bbox_value(get_row_value(rgb_row, f"{slot_name}_bbox", []))
    if bbox:
        return bbox
    nir_rep_frame = norm_str(get_row_value(nir_row, "rep_frame_path", ""))
    nir_json_path = os.path.join(save_json_dir, f"{seq_id}_nir.json") if save_json_dir else ""
    bbox = load_rep_frame_bbox_from_json(nir_json_path, nir_rep_frame, slot_name)
    if bbox:
        return bbox
    bbox = parse_bbox_value(get_row_value(nir_row, f"{slot_name}_bbox", []))
    if bbox:
        return bbox
    return []


def save_fused_visualization(seq_id, fused_row, rgb_row, nir_row, save_fused_vis_dir, save_json_dir=None):
    ensure_dir(save_fused_vis_dir)
    bg_path = norm_str(get_row_value(rgb_row, "rep_frame_path", ""))
    if not bg_path or not os.path.isfile(bg_path):
        bg_path = norm_str(get_row_value(nir_row, "rep_frame_path", ""))

    if bg_path and os.path.isfile(bg_path):
        image_bgr = cv2.imread(bg_path)
    else:
        image_bgr = None

    if image_bgr is None:
        image_bgr = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(image_bgr, "No background frame", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    face1_bbox = pick_rgb_bbox_for_fused(seq_id, rgb_row, nir_row, save_json_dir, "face1")
    face2_bbox = pick_rgb_bbox_for_fused(seq_id, rgb_row, nir_row, save_json_dir, "face2")

    predictions = {}
    if face1_bbox:
        predictions["face1"] = {
            "bbox": face1_bbox,
            "gender": get_row_value(fused_row, "pred_gender1", "0"),
            "age_group": get_row_value(fused_row, "pred_age1", "0"),
        }
    if face2_bbox:
        predictions["face2"] = {
            "bbox": face2_bbox,
            "gender": get_row_value(fused_row, "pred_gender0", "0"),
            "age_group": get_row_value(fused_row, "pred_age0", "0"),
        }

    if predictions:
        vis = draw_fused_results_no_score(image_bgr, predictions)
    else:
        vis = image_bgr.copy()
        cv2.putText(vis, "No fused bbox", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    text = f"FUSED  Driver:{fused_row['pred_gender0']} {fused_row['pred_age0']}  Passenger:{fused_row['pred_gender1']} {fused_row['pred_age1']}"
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    save_path = os.path.join(save_fused_vis_dir, f"{seq_id}_fused_result.jpg")
    cv2.imwrite(save_path, vis)
    return save_path


def split_base_and_modality(video_name):
    video_name = norm_str(video_name)
    if video_name.endswith("_rgb"):
        return video_name[:-4], "rgb"
    if video_name.endswith("_nir"):
        return video_name[:-4], "nir"
    return video_name, ""


def run_fused_only_from_existing_results(save_summary_csv, save_fused_summary_csv, save_fused_vis_dir, save_json_dir):
    if not os.path.isfile(save_summary_csv):
        raise FileNotFoundError(f"找不到已有预测结果文件: {save_summary_csv}")

    with open(save_summary_csv, "r", encoding="utf-8-sig", newline="") as f:
        import csv
        rows = [{norm_str(k): norm_str(v) for k, v in row.items()} for row in csv.DictReader(f)]

    grouped = {}
    for row in rows:
        video = norm_str(row.get("video", ""))
        base_seq, modality = split_base_and_modality(video)
        if modality not in ("rgb", "nir"):
            continue
        grouped.setdefault(base_seq, {})[modality] = row

    fused_rows = []
    for seq_id in sorted(grouped.keys(), key=natural_key):
        rgb_row = grouped[seq_id].get("rgb", {})
        nir_row = grouped[seq_id].get("nir", {})
        fused_row = fuse_rgb_nir_summary_row(seq_id, rgb_row, nir_row)
        fused_vis_path = save_fused_visualization(
            seq_id=seq_id, fused_row=fused_row, rgb_row=rgb_row, nir_row=nir_row,
            save_fused_vis_dir=save_fused_vis_dir, save_json_dir=save_json_dir,
        )
        fused_row["fused_vis_path"] = fused_vis_path
        fused_rows.append(fused_row)
        print(f"[INFO] Fused {seq_id}: 主驾 {fused_row['pred_gender0']}({fused_row['pred_age0']}), 副驾 {fused_row['pred_gender1']}({fused_row['pred_age1']})")

    fused_fieldnames = [
        "sequence_id", "video",
        "rgb_pred_person_forward_number", "rgb_pred_gender0", "rgb_pred_age0", "rgb_pred_gender1", "rgb_pred_age1",
        "nir_pred_person_forward_number", "nir_pred_gender0", "nir_pred_age0", "nir_pred_gender1", "nir_pred_age1",
        "pred_person_forward_number", "pred_gender0", "pred_age0", "pred_gender1", "pred_age1",
        "rgb_result_vis_path", "nir_result_vis_path", "fused_vis_path",
        "rgb_rep_frame_path", "nir_rep_frame_path",
    ]
    save_csv(fused_rows, save_fused_summary_csv, fused_fieldnames)
    return fused_rows
