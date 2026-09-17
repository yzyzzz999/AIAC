"""
pipeline/sequence.py
====================
处理一个 RGB 或 NIR 序列的完整流程。
"""

import os
import json

import cv2
import numpy as np

from utils import (safe_float, normalize_bbox, bbox_center, bbox_area,
                   crop_face_pil, draw_results, select_front_row_faces)
from pipeline.data_loader import norm_str, ensure_dir, uniform_sample_paths
from pipeline.fusion import (fuse_by_count_and_confidence, is_valid_prediction,
                             compute_frame_weight, compute_bbox_soft_score)


# ---- 工具 ----

def iou_xyxy(box1, box2):
    from utils import bbox_area
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    ix1 = max(x11, x21); iy1 = max(y11, y21)
    ix2 = min(x12, x22); iy2 = min(y12, y22)
    inter_w = max(0.0, ix2 - ix1); inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    union = bbox_area(box1) + bbox_area(box2) - inter + 1e-6
    return inter / union


def to_jsonable_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.float32, np.float64)):
            out[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            out[k] = int(v)
        elif isinstance(v, dict):
            out[k] = to_jsonable_dict(v)
        elif isinstance(v, list):
            new_list = []
            for x in v:
                if isinstance(x, (np.float32, np.float64)):
                    new_list.append(float(x))
                elif isinstance(x, (np.int32, np.int64)):
                    new_list.append(int(x))
                else:
                    new_list.append(x)
            out[k] = new_list
        else:
            out[k] = v
    return out


def safe_get(d, *keys, default=""):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def bbox_wh(bbox):
    x1, y1, x2, y2 = bbox
    return max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))


def is_good_representative_frame(frame_predictions, image_shape, min_face_area_ratio=0.02, min_bbox_soft_score=0.50):
    if len(frame_predictions) < 2:
        return False
    image_h, image_w = image_shape[:2]
    image_area = max(float(image_w * image_h), 1.0)
    for pred in frame_predictions.values():
        bbox = normalize_bbox(pred.get("bbox"))
        if len(bbox) != 4:
            return False
        area_ratio = bbox_area(bbox) / image_area
        bbox_soft_score = safe_float(pred.get("bbox_soft_score"), default=0.0)
        if area_ratio < min_face_area_ratio:
            return False
        if bbox_soft_score < min_bbox_soft_score:
            return False
    return True


# ---- faces 分配 ----

def assign_faces_to_slots(current_faces, prev_slots):
    slots = {"face1": None, "face2": None}
    current_faces = list(current_faces)
    if not current_faces:
        return slots

    SLOT_TO_FACE = {"left": "face1", "right": "face2"}
    marked_faces = [f for f in current_faces if f.get("_front_row_slot") in SLOT_TO_FACE]
    if marked_faces:
        for slot_name in ["face1", "face2"]:
            slot_candidates = [f for f in marked_faces if SLOT_TO_FACE.get(f.get("_front_row_slot")) == slot_name]
            if not slot_candidates:
                continue
            slots[slot_name] = max(slot_candidates, key=lambda f: safe_float(f.get("_front_row_score"), default=0.0))
        return slots

    if prev_slots.get("face1") is None and prev_slots.get("face2") is None:
        current_faces = sorted(current_faces, key=lambda x: bbox_center(normalize_bbox(x["bbox"]))[0])
        slots["face1"] = current_faces[0] if len(current_faces) >= 1 else None
        slots["face2"] = current_faces[1] if len(current_faces) >= 2 else None
        return slots

    used = set()
    for slot_name in ["face1", "face2"]:
        prev_face = prev_slots.get(slot_name)
        if prev_face is None:
            continue
        best_idx = -1; best_iou = -1.0
        for i, f in enumerate(current_faces):
            if i in used:
                continue
            cur_iou = iou_xyxy(normalize_bbox(prev_face["bbox"]), normalize_bbox(f["bbox"]))
            if cur_iou > best_iou:
                best_iou = cur_iou; best_idx = i
        if best_idx >= 0:
            slots[slot_name] = current_faces[best_idx]
            used.add(best_idx)

    remaining = [f for i, f in enumerate(current_faces) if i not in used]
    remaining = sorted(remaining, key=lambda x: bbox_center(normalize_bbox(x["bbox"]))[0])
    if slots["face1"] is None and remaining:
        slots["face1"] = remaining.pop(0)
    if slots["face2"] is None and remaining:
        slots["face2"] = remaining.pop(0)
    return slots


# ---- 属性预测 ----

def predict_face_attr(image_bgr, face_info, predictor):
    bbox = normalize_bbox(face_info["bbox"])
    face_pil = crop_face_pil(image_bgr, bbox)
    if face_pil is None:
        raise ValueError("crop_face_pil returned None")
    attr_result = predictor.predict_all(face_pil)
    result = {
        "bbox": bbox,
        "gender": norm_str(safe_get(attr_result, "gender", "label", default="")),
        "gender_score": safe_float(safe_get(attr_result, "gender", "score", default=0.0)),
        "gender_probs": safe_get(attr_result, "gender", "probs", default={}),
        "age_group": norm_str(safe_get(attr_result, "age", "label", default="")),
        "age_score": safe_float(safe_get(attr_result, "age", "score", default=0.0)),
        "age_probs": safe_get(attr_result, "age", "probs", default={}),
    }
    return to_jsonable_dict(result)


# ---- 主流程 ----

def process_one_sequence(
    seq_id, image_paths, detector, predictor,
    save_vis_dir, save_json_dir,
    frame_stride=1, max_frames_per_sequence=12, save_frame_vis=False,
    gender_thr=0.45, age_thr=0.30,
    bbox_history_min=4, bbox_center_x_scale=1.0, bbox_center_y_scale=1.0,
    bbox_area_tol_low=0.6, bbox_area_tol_high=1.8,
    bbox_wh_tol_low=0.65, bbox_wh_tol_high=1.5,
    bbox_weight_lambda=0.25,
):
    original_num_frames = len(image_paths)
    image_paths = uniform_sample_paths(image_paths, max_frames=max_frames_per_sequence)

    print("=" * 60)
    print(f"[INFO] folder / sequence_id: {seq_id}")
    print(f"[INFO] total_frames_in_folder: {original_num_frames}")
    print(f"[INFO] frames_selected_for_inference: {len(image_paths)}")

    save_vis_frames_dir = os.path.join(save_vis_dir, f"{seq_id}_frames")
    if save_frame_vis:
        ensure_dir(save_vis_frames_dir)

    prev_slots = {"face1": None, "face2": None}
    face1_all_frames = []
    face2_all_frames = []

    processed_frames = 0; sampled_frames = 0
    frames_with_any_face = 0; frames_with_two_faces = 0
    face1_attr_reject_count = 0; face2_attr_reject_count = 0

    image_width = None
    rep_frame_bgr = None; rep_draw_results = None; rep_frame_path = ""; rep_score = -1.0
    fallback_rep_frame_bgr = None; fallback_rep_draw_results = None
    fallback_rep_frame_path = ""; fallback_rep_score = -1.0

    for frame_idx, image_path in enumerate(image_paths):
        processed_frames += 1
        if frame_idx % frame_stride != 0:
            continue
        sampled_frames += 1

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[WARN] Cannot read image: {image_path}")
            face1_all_frames.append(None); face2_all_frames.append(None)
            continue

        if image_width is None:
            image_width = image_bgr.shape[1]

        raw_faces = detector.detect_faces_dicts(image_bgr)
        h, w = image_bgr.shape[:2]
        selected_faces = select_front_row_faces(raw_faces, w, h)

        if len(raw_faces) > 0:
            frames_with_any_face += 1
        if len(selected_faces) >= 2:
            frames_with_two_faces += 1

        assigned = assign_faces_to_slots(selected_faces, prev_slots)
        prev_slots = assigned

        current_frame_predictions = {}

        for face_name in ["face1", "face2"]:
            frame_data = None
            assigned_face = assigned.get(face_name)
            if assigned_face is not None:
                history_frames = face1_all_frames if face_name == "face1" else face2_all_frames
                history_boxes = [f["bbox"] for f in history_frames if f is not None]
                bbox_soft_score = compute_bbox_soft_score(
                    normalize_bbox(assigned_face["bbox"]), history_boxes,
                    min_history=bbox_history_min, center_x_scale=bbox_center_x_scale,
                    center_y_scale=bbox_center_y_scale, area_tol_low=bbox_area_tol_low,
                    area_tol_high=bbox_area_tol_high, wh_tol_low=bbox_wh_tol_low,
                    wh_tol_high=bbox_wh_tol_high,
                )
                try:
                    pred = predict_face_attr(image_bgr, assigned_face, predictor)
                    if is_valid_prediction(pred, gender_thr=gender_thr, age_thr=age_thr):
                        frame_weight, attr_score = compute_frame_weight(
                            pred, bbox_soft_score, bbox_weight_lambda=bbox_weight_lambda)
                        pred["bbox_soft_score"] = round(float(bbox_soft_score), 4)
                        pred["base_attr_score"] = round(float(attr_score), 4)
                        pred["frame_weight"] = round(float(frame_weight), 4)
                        pred["frame_path"] = image_path
                        pred["frame_index"] = frame_idx
                        frame_data = pred
                        current_frame_predictions[face_name] = pred
                    else:
                        if face_name == "face1":
                            face1_attr_reject_count += 1
                        else:
                            face2_attr_reject_count += 1
                except Exception as e:
                    print(f"[WARN] {face_name} prediction failed: {e}")

            if face_name == "face1":
                face1_all_frames.append(frame_data)
            else:
                face2_all_frames.append(frame_data)

        if save_frame_vis and current_frame_predictions:
            vis_frame = draw_results(image_bgr, current_frame_predictions)
            frame_vis_path = os.path.join(save_vis_frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_vis_path, vis_frame)

        cur_rep_score = sum(safe_float(p.get("frame_weight")) for p in current_frame_predictions.values())
        if current_frame_predictions:
            if cur_rep_score > fallback_rep_score:
                fallback_rep_score = cur_rep_score
                fallback_rep_frame_bgr = image_bgr.copy()
                fallback_rep_draw_results = current_frame_predictions
                fallback_rep_frame_path = image_path
            if (is_good_representative_frame(current_frame_predictions, image_bgr.shape)
                    and cur_rep_score > rep_score):
                rep_score = cur_rep_score
                rep_frame_bgr = image_bgr.copy()
                rep_draw_results = current_frame_predictions
                rep_frame_path = image_path

    if rep_frame_bgr is None and fallback_rep_frame_bgr is not None:
        rep_frame_bgr = fallback_rep_frame_bgr
        rep_draw_results = fallback_rep_draw_results
        rep_frame_path = fallback_rep_frame_path
        print(f"[WARN] {seq_id} 没有满足代表图门槛的帧，回退到最高 frame_weight 帧: {rep_frame_path}")

    valid_face1 = [f for f in face1_all_frames if f is not None]
    valid_face2 = [f for f in face2_all_frames if f is not None]

    final_face1 = fuse_by_count_and_confidence(valid_face1)
    final_face2 = fuse_by_count_and_confidence(valid_face2)

    face1_found = bool(final_face1["gender"] and final_face1["age_group"])
    face2_found = bool(final_face2["gender"] and final_face2["age_group"])

    if sampled_frames == 0:
        remark = "no_sampled_frames"
    elif frames_with_any_face == 0:
        remark = "no_face_detected_in_sequence"
    elif face1_found and face2_found:
        remark = "two_faces_fused_ok"
    elif face1_found or face2_found:
        remark = "only_one_face_fused"
    else:
        remark = "faces_detected_but_no_valid_fusion"

    pred_person_forward_number = int(face1_found) + int(face2_found)
    pred_gender0 = pred_age0 = "0"
    pred_gender1 = pred_age1 = "0"

    valid_faces = []
    if face1_found:
        valid_faces.append(final_face1)
    if face2_found:
        valid_faces.append(final_face2)

    if len(valid_faces) >= 2:
        valid_faces = sorted(valid_faces, key=lambda x: bbox_center(x["bbox"])[0], reverse=True)
        pred_gender0 = valid_faces[0]["gender"]; pred_age0 = valid_faces[0]["age_group"]
        pred_gender1 = valid_faces[1]["gender"]; pred_age1 = valid_faces[1]["age_group"]
    elif len(valid_faces) == 1:
        one_face = valid_faces[0]
        cx, _ = bbox_center(one_face["bbox"])
        if image_width is None or cx >= image_width / 2.0:
            pred_gender0 = one_face["gender"]; pred_age0 = one_face["age_group"]
        else:
            pred_gender1 = one_face["gender"]; pred_age1 = one_face["age_group"]

    detail_result = {
        "sequence_id": seq_id, "video": seq_id,
        "num_frames": original_num_frames, "processed_frames": processed_frames,
        "sampled_frames": sampled_frames, "frame_stride": frame_stride,
        "all_frames_face1": face1_all_frames, "all_frames_face2": face2_all_frames,
        "final_fused_face1": final_face1, "final_fused_face2": final_face2,
        "status": {"face1_found": face1_found, "face2_found": face2_found, "remark": remark},
        "representative_frame_path": rep_frame_path,
    }

    json_path = os.path.join(save_json_dir, seq_id + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detail_result, f, ensure_ascii=False, indent=2)

    vis_path = os.path.join(save_vis_dir, seq_id + "_result.jpg")
    if rep_frame_bgr is not None and rep_draw_results:
        vis = draw_results(rep_frame_bgr, rep_draw_results)
    else:
        vis = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(vis, "No valid result", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(vis_path, vis)

    print(f"[INFO] 有效帧 face1: {len(valid_face1)}, face2: {len(valid_face2)}")
    print(f"[INFO] 融合结果 主驾: {pred_gender0}({pred_age0}), 副驾: {pred_gender1}({pred_age1})")
    if save_frame_vis:
        print(f"[INFO] 每帧可视化目录: {save_vis_frames_dir}")
    else:
        print("[INFO] 每帧可视化: disabled，仅保存序列代表帧和 fused 图")

    summary_row = {
        "sequence_id": seq_id, "video": seq_id,
        "num_frames": original_num_frames, "processed_frames": processed_frames,
        "sampled_frames": sampled_frames, "frame_stride": frame_stride,
        "frames_with_any_face": frames_with_any_face, "frames_with_two_faces": frames_with_two_faces,
        "face1_attr_reject_count": face1_attr_reject_count, "face2_attr_reject_count": face2_attr_reject_count,
        "face1_valid_frame_count": len(valid_face1), "face2_valid_frame_count": len(valid_face2),
        "face1_found": face1_found, "face2_found": face2_found, "remark": remark,
        "face1_bbox": final_face1["bbox"], "face1_gender": final_face1["gender"],
        "face1_gender_score": final_face1["gender_score"], "face1_age_group": final_face1["age_group"],
        "face1_age_score": final_face1["age_score"], "face1_frame_weight_mean": final_face1["frame_weight_mean"],
        "face2_bbox": final_face2["bbox"], "face2_gender": final_face2["gender"],
        "face2_gender_score": final_face2["gender_score"], "face2_age_group": final_face2["age_group"],
        "face2_age_score": final_face2["age_score"], "face2_frame_weight_mean": final_face2["frame_weight_mean"],
        "pred_person_forward_number": pred_person_forward_number,
        "pred_gender0": pred_gender0, "pred_age0": pred_age0,
        "pred_gender1": pred_gender1, "pred_age1": pred_age1,
        "rep_frame_path": rep_frame_path, "result_vis_path": vis_path,
    }

    debug_row = {
        "sequence_id": seq_id, "video": seq_id,
        "num_frames": original_num_frames, "processed_frames": processed_frames,
        "sampled_frames": sampled_frames, "frames_with_any_face": frames_with_any_face,
        "frames_with_two_faces": frames_with_two_faces,
        "face1_attr_reject_count": face1_attr_reject_count, "face2_attr_reject_count": face2_attr_reject_count,
        "face1_valid_frame_count": len(valid_face1), "face2_valid_frame_count": len(valid_face2),
        "face1_found": face1_found, "face2_found": face2_found, "remark": remark,
    }

    return summary_row, debug_row
