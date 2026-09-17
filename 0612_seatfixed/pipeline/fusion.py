"""
pipeline/fusion.py
==================
多帧性别/年龄融合 + 保守后处理 + RGB/NIR 跨模态融合。
"""

import math

from utils import safe_float, normalize_bbox

AGE_GROUPS = ["12-17", "18-40", "41-59", "60-74", "75+"]

# 年龄融合保守策略参数
AGE_4159_MIN_RATIO = 0.65
AGE_4159_NEED_OVER_1840 = 1.20
AGE_1217_MIN_SOFT_RATIO = 0.38
AGE_1217_REL_TO_1840 = 0.90
AGE_1217_HARD_MIN_COUNT = 2
AGE_1217_HARD_QUALITY_REL = 0.85
AGE_1217_FALSE_POSITIVE_BACKOFF = 1.10
AGE_MODAL_4159_NEED_OVER_1840 = 1.35
AGE_MODAL_1217_NEED_OVER_1840 = 1.20


# ---- 通用 ----

def norm_str(x):
    if x is None:
        return ""
    return str(x).strip()


def safe_int(x, default=0):
    try:
        return int(float(norm_str(x)))
    except Exception:
        return default


# ---- bbox 稳定性 ----

def ratio_score(ratio, low, high):
    ratio = max(float(ratio), 1e-6)
    if low <= ratio <= high:
        return 1.0
    if ratio < low:
        return max(0.0, min(1.0, ratio / low))
    return max(0.0, min(1.0, high / ratio))


def compute_bbox_soft_score(bbox, history_boxes, min_history=4,
                            center_x_scale=1.0, center_y_scale=1.0,
                            area_tol_low=0.6, area_tol_high=1.8,
                            wh_tol_low=0.65, wh_tol_high=1.5):
    from statistics import median
    history_boxes = [b for b in history_boxes if b]
    if len(history_boxes) < min_history:
        return 1.0
    from utils import bbox_center, bbox_wh, bbox_area
    ref_cx = median([bbox_center(b)[0] for b in history_boxes])
    ref_cy = median([bbox_center(b)[1] for b in history_boxes])
    ref_w = median([bbox_wh(b)[0] for b in history_boxes])
    ref_h = median([bbox_wh(b)[1] for b in history_boxes])
    ref_area = median([bbox_area(b) for b in history_boxes])
    cx, cy = bbox_center(bbox)
    w, h = bbox_wh(bbox)
    area = bbox_area(bbox)
    sx = max(center_x_scale * ref_w, 1e-6)
    sy = max(center_y_scale * ref_h, 1e-6)
    score_x = math.exp(-0.5 * ((cx - ref_cx) / sx) ** 2)
    score_y = math.exp(-0.5 * ((cy - ref_cy) / sy) ** 2)
    center_score = math.sqrt(score_x * score_y)
    area_score = ratio_score(area / max(ref_area, 1e-6), area_tol_low, area_tol_high)
    w_score = ratio_score(w / max(ref_w, 1e-6), wh_tol_low, wh_tol_high)
    h_score = ratio_score(h / max(ref_h, 1e-6), wh_tol_low, wh_tol_high)
    wh_score = math.sqrt(w_score * h_score)
    return max(0.0, min(1.0, 0.5 * center_score + 0.25 * area_score + 0.25 * wh_score))


def weighted_average_bbox(items, weight_key="frame_weight"):
    valid = [x for x in items if x and x.get("bbox")]
    if not valid:
        return []
    total_w = 0.0
    sx1 = sy1 = sx2 = sy2 = 0.0
    for item in valid:
        b = item["bbox"]
        w = max(1e-6, safe_float(item.get(weight_key)))
        sx1 += b[0] * w; sy1 += b[1] * w; sx2 += b[2] * w; sy2 += b[3] * w
        total_w += w
    return [sx1 / total_w, sy1 / total_w, sx2 / total_w, sy2 / total_w]


# ---- 帧过滤与权重 ----

def is_valid_prediction(pred, gender_thr=0.45, age_thr=0.30):
    if not pred:
        return False
    if safe_float(pred.get("gender_score")) < gender_thr:
        return False
    if safe_float(pred.get("age_score")) < age_thr:
        return False
    if norm_str(pred.get("gender")) == "":
        return False
    if norm_str(pred.get("age_group")) == "":
        return False
    return True


def compute_frame_weight(pred, bbox_soft_score, bbox_weight_lambda=0.25):
    g = safe_float(pred.get("gender_score"))
    a = safe_float(pred.get("age_score"))
    attr_score = math.sqrt(max(g, 0.0) * max(a, 0.0))
    frame_weight = (1.0 - bbox_weight_lambda) * attr_score + bbox_weight_lambda * bbox_soft_score
    return max(0.0, min(1.0, frame_weight)), attr_score


# ---- 年龄融合 ----

def get_age_hard_stat(age_hard_stats, age_name, field, default=0.0):
    if not isinstance(age_hard_stats, dict):
        return default
    item = age_hard_stats.get(age_name, {})
    if not isinstance(item, dict):
        return default
    return safe_float(item.get(field, default), default=default)


def normalize_age_prob_stats(age_prob_stats, total_quality):
    if total_quality <= 1e-6:
        return {age: 0.0 for age in AGE_GROUPS}
    return {age: safe_float(age_prob_stats.get(age, 0.0), default=0.0) / max(total_quality, 1e-6)
            for age in AGE_GROUPS}


def conservative_age_postprocess(age_prob_stats, age_hard_stats, total_quality, raw_best_age):
    norm_probs = normalize_age_prob_stats(age_prob_stats, total_quality)
    p12 = norm_probs.get("12-17", 0.0)
    p18 = norm_probs.get("18-40", 0.0)
    p41 = norm_probs.get("41-59", 0.0)
    c12 = get_age_hard_stat(age_hard_stats, "12-17", "count", 0.0)
    c18 = get_age_hard_stat(age_hard_stats, "18-40", "count", 0.0)
    c41 = get_age_hard_stat(age_hard_stats, "41-59", "count", 0.0)
    q12 = get_age_hard_stat(age_hard_stats, "12-17", "quality_sum", 0.0)
    q18 = get_age_hard_stat(age_hard_stats, "18-40", "quality_sum", 0.0)

    final_age = raw_best_age
    reason = "raw_probability_best"

    if raw_best_age == "18-40":
        soft_support_12 = (p12 >= AGE_1217_MIN_SOFT_RATIO and p12 >= p18 * AGE_1217_REL_TO_1840)
        hard_support_12 = (c12 >= AGE_1217_HARD_MIN_COUNT and q12 >= q18 * AGE_1217_HARD_QUALITY_REL
                           and p12 >= AGE_1217_MIN_SOFT_RATIO * 0.75)
        if soft_support_12 or hard_support_12:
            final_age = "12-17"
            reason = "strong_protect_12_17_from_18_40"

    if final_age == "12-17":
        soft_18_stronger = p18 >= p12 * AGE_1217_FALSE_POSITIVE_BACKOFF
        hard_18_stronger = (c18 > c12 and q18 >= q12 * AGE_1217_FALSE_POSITIVE_BACKOFF)
        very_strong_12 = (p12 >= 0.55 and c12 >= max(AGE_1217_HARD_MIN_COUNT, c18) and q12 >= q18)
        if (soft_18_stronger or hard_18_stronger) and not very_strong_12:
            final_age = "18-40"
            reason = "backoff_false_12_17_to_18_40"

    if raw_best_age == "41-59":
        hard_vote_prefers_18 = (c18 > 0 and c18 >= c41)
        soft_41_not_enough = (p41 < AGE_4159_MIN_RATIO)
        soft_41_not_much_better = (p41 < p18 * AGE_4159_NEED_OVER_1840)
        if hard_vote_prefers_18 or soft_41_not_enough or soft_41_not_much_better:
            final_age = "18-40"
            reason = "protect_18_40_from_41_59"

    final_score = norm_probs.get(final_age, 0.0)
    debug_info = {
        "normalized_probs": {k: round(v, 4) for k, v in norm_probs.items()},
        "raw_best_age": raw_best_age, "final_age": final_age,
        "postprocess_reason": reason,
    }
    return final_age, final_score, debug_info


def fuse_age_by_probs(frames):
    frames = [f for f in frames if f is not None]
    if not frames:
        return "", 0.0, {}

    age_prob_stats = {age: 0.0 for age in AGE_GROUPS}
    age_hard_stats = {}
    total_quality = 0.0

    for f in frames:
        age_label = norm_str(f.get("age_group", ""))
        age_score = safe_float(f.get("age_score"), default=0.0)
        bbox_score = safe_float(f.get("bbox_soft_score"), default=1.0)
        frame_weight = safe_float(f.get("frame_weight"), default=1.0)
        quality = 0.50 * frame_weight + 0.30 * age_score + 0.20 * bbox_score
        quality = max(0.0, min(1.0, quality))
        if age_label:
            age_hard_stats.setdefault(age_label, {"count": 0, "quality_sum": 0.0, "score_sum": 0.0})
            age_hard_stats[age_label]["count"] += 1
            age_hard_stats[age_label]["quality_sum"] += quality
            age_hard_stats[age_label]["score_sum"] += age_score
        age_probs = f.get("age_probs", {})
        used_probs = False
        if isinstance(age_probs, dict) and age_probs:
            for age in AGE_GROUPS:
                p = safe_float(age_probs.get(age), default=0.0)
                if p > 0:
                    used_probs = True
                age_prob_stats[age] += quality * p
        if not used_probs and age_label in AGE_GROUPS:
            age_prob_stats[age_label] += quality * max(age_score, 1e-6)
        total_quality += quality

    if total_quality <= 1e-6:
        return "", 0.0, {"prob_stats": age_prob_stats, "hard_stats": age_hard_stats, "total_quality": 0.0}

    raw_best_age = max(age_prob_stats.items(), key=lambda x: x[1])[0]
    raw_best_score = age_prob_stats[raw_best_age] / total_quality
    best_age, age_score, post_debug = conservative_age_postprocess(
        age_prob_stats=age_prob_stats, age_hard_stats=age_hard_stats,
        total_quality=total_quality, raw_best_age=raw_best_age,
    )
    return best_age, round(age_score, 4), {
        "prob_stats": age_prob_stats, "hard_stats": age_hard_stats,
        "total_quality": round(total_quality, 4),
        "raw_best_age": raw_best_age, "raw_best_score": round(raw_best_score, 4),
        "final_age": best_age, "final_age_score": round(age_score, 4),
        "postprocess": post_debug,
    }


# ---- 性别融合 ----

def fuse_by_count_and_confidence(frames):
    frames = [f for f in frames if f is not None]
    if not frames:
        return {"bbox": [], "gender": "", "gender_score": 0.0, "age_group": "", "age_score": 0.0,
                "gender_stats": {}, "age_stats": {}, "total_valid_frames": 0,
                "frame_weight_mean": 0.0, "bbox_score_mean": 0.0}

    gender_stats = {}
    for f in frames:
        g = f["gender"]
        w = safe_float(f.get("frame_weight"), default=1.0)
        g_score = safe_float(f.get("gender_score"), default=0.0)
        gender_stats.setdefault(g, {"count": 0, "weighted_count": 0.0, "total_score": 0.0, "weighted_score": 0.0})
        gender_stats[g]["count"] += 1
        gender_stats[g]["weighted_count"] += w
        gender_stats[g]["total_score"] += g_score
        gender_stats[g]["weighted_score"] += w * g_score

    best_gender = max(gender_stats.items(), key=lambda x: (x[1]["weighted_score"], x[1]["weighted_count"], x[1]["total_score"]))[0]
    gender_weight = max(gender_stats[best_gender]["weighted_count"], 1e-6)
    gender_score = gender_stats[best_gender]["weighted_score"] / gender_weight
    best_age, age_score, age_stats = fuse_age_by_probs(frames)
    frame_weights = [safe_float(f.get("frame_weight")) for f in frames]
    bbox_scores = [safe_float(f.get("bbox_soft_score")) for f in frames]

    return {
        "bbox": weighted_average_bbox(frames, "frame_weight"),
        "gender": best_gender, "gender_score": round(gender_score, 4),
        "age_group": best_age, "age_score": round(age_score, 4),
        "gender_stats": gender_stats, "age_stats": age_stats,
        "total_valid_frames": len(frames),
        "frame_weight_mean": round(sum(frame_weights) / max(1, len(frame_weights)), 4),
        "bbox_score_mean": round(sum(bbox_scores) / max(1, len(bbox_scores)), 4),
    }


# ---- RGB/NIR 跨模态融合 ----

def valid_label(x):
    x = norm_str(x)
    return x not in ("", "0", "None", "none")


def choose_value(primary, backup, default="0"):
    primary = norm_str(primary)
    backup = norm_str(backup)
    if valid_label(primary):
        return primary
    if valid_label(backup):
        return backup
    return default


def get_row_value(row, key, default="0"):
    if not isinstance(row, dict):
        return default
    return row.get(key, default)


def age_quality(row, slot_name):
    if not isinstance(row, dict) or not row:
        return 0.0
    valid_frames = safe_int(row.get(f"{slot_name}_valid_frame_count", 0))
    age_score = safe_float(row.get(f"{slot_name}_age_score", 0.0))
    frame_weight = safe_float(row.get(f"{slot_name}_frame_weight_mean", 0.0))
    if valid_frames <= 0:
        return 0.0
    return math.log1p(valid_frames) * age_score * max(frame_weight, 1e-6)


def gender_quality(row, slot_name):
    if not isinstance(row, dict) or not row:
        return 0.0
    valid_frames = safe_int(row.get(f"{slot_name}_valid_frame_count", 0))
    gender_score = safe_float(row.get(f"{slot_name}_gender_score", 0.0))
    frame_weight = safe_float(row.get(f"{slot_name}_frame_weight_mean", 0.0))
    if valid_frames <= 0:
        return 0.0
    return math.log1p(valid_frames) * gender_score * max(frame_weight, 1e-6)


def choose_gender_by_quality(rgb_row, nir_row, pred_key, slot_name):
    rgb_gender = get_row_value(rgb_row, pred_key, "0")
    nir_gender = get_row_value(nir_row, pred_key, "0")
    if not valid_label(rgb_gender) and not valid_label(nir_gender):
        return "0"
    if valid_label(rgb_gender) and not valid_label(nir_gender):
        return rgb_gender
    if valid_label(nir_gender) and not valid_label(rgb_gender):
        return nir_gender
    if rgb_gender == nir_gender:
        return rgb_gender
    rgb_q = gender_quality(rgb_row, slot_name)
    nir_q = gender_quality(nir_row, slot_name)
    if rgb_q >= nir_q * 0.95:
        return rgb_gender
    return nir_gender


def choose_age_by_quality(rgb_row, nir_row, pred_key, slot_name):
    rgb_age = get_row_value(rgb_row, pred_key, "0")
    nir_age = get_row_value(nir_row, pred_key, "0")
    if not valid_label(rgb_age) and not valid_label(nir_age):
        return "0"
    if valid_label(rgb_age) and not valid_label(nir_age):
        return rgb_age
    if valid_label(nir_age) and not valid_label(rgb_age):
        return nir_age
    if rgb_age == nir_age:
        return rgb_age
    rgb_q = age_quality(rgb_row, slot_name)
    nir_q = age_quality(nir_row, slot_name)

    if set([rgb_age, nir_age]) == set(["18-40", "41-59"]):
        if rgb_age == "18-40":
            q18, q41 = rgb_q, nir_q
        else:
            q18, q41 = nir_q, rgb_q
        if q41 >= q18 * AGE_MODAL_4159_NEED_OVER_1840:
            return "41-59"
        return "18-40"

    if set([rgb_age, nir_age]) == set(["12-17", "18-40"]):
        if rgb_age == "12-17":
            q12, q18 = rgb_q, nir_q
        else:
            q12, q18 = nir_q, rgb_q
        if q12 >= q18 * AGE_MODAL_1217_NEED_OVER_1840:
            return "12-17"
        return "18-40"

    if rgb_q >= nir_q:
        return rgb_age
    return nir_age


def fuse_rgb_nir_summary_row(seq_id, rgb_row=None, nir_row=None):
    rgb_row = rgb_row or {}
    nir_row = nir_row or {}
    pred_person = max(
        safe_int(get_row_value(rgb_row, "pred_person_forward_number", "0")),
        safe_int(get_row_value(nir_row, "pred_person_forward_number", "0")),
    )
    pred_gender0 = choose_gender_by_quality(rgb_row, nir_row, "pred_gender0", "face2")
    pred_gender1 = choose_gender_by_quality(rgb_row, nir_row, "pred_gender1", "face1")
    pred_age0 = choose_age_by_quality(rgb_row, nir_row, "pred_age0", "face2")
    pred_age1 = choose_age_by_quality(rgb_row, nir_row, "pred_age1", "face1")

    return {
        "sequence_id": seq_id, "video": seq_id,
        "rgb_pred_person_forward_number": get_row_value(rgb_row, "pred_person_forward_number", "0"),
        "rgb_pred_gender0": get_row_value(rgb_row, "pred_gender0", "0"),
        "rgb_pred_age0": get_row_value(rgb_row, "pred_age0", "0"),
        "rgb_pred_gender1": get_row_value(rgb_row, "pred_gender1", "0"),
        "rgb_pred_age1": get_row_value(rgb_row, "pred_age1", "0"),
        "nir_pred_person_forward_number": get_row_value(nir_row, "pred_person_forward_number", "0"),
        "nir_pred_gender0": get_row_value(nir_row, "pred_gender0", "0"),
        "nir_pred_age0": get_row_value(nir_row, "pred_age0", "0"),
        "nir_pred_gender1": get_row_value(nir_row, "pred_gender1", "0"),
        "nir_pred_age1": get_row_value(nir_row, "pred_age1", "0"),
        "pred_person_forward_number": str(pred_person),
        "pred_gender0": pred_gender0, "pred_age0": pred_age0,
        "pred_gender1": pred_gender1, "pred_age1": pred_age1,
        "rgb_result_vis_path": get_row_value(rgb_row, "result_vis_path", ""),
        "nir_result_vis_path": get_row_value(nir_row, "result_vis_path", ""),
        "rgb_rep_frame_path": get_row_value(rgb_row, "rep_frame_path", ""),
        "nir_rep_frame_path": get_row_value(nir_row, "rep_frame_path", ""),
    }
