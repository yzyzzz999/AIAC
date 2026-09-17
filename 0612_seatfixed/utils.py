# utils.py
# -*- coding: utf-8 -*-

from PIL import Image
import cv2


# ============================================================
# 1. 基础工具函数
# ============================================================

def safe_float(x, default=0.0):
    """
    安全转 float。
    """
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def normalize_bbox(bbox):
    """
    把 bbox 转成普通 float list。
    bbox 格式：[x1, y1, x2, y2]
    """
    if bbox is None:
        return []
    return [float(x) for x in bbox]


def bbox_center(bbox):
    """
    计算 bbox 中心点。
    """
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def bbox_area(bbox):
    """
    计算 bbox 面积。
    """
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


# ============================================================
# 2. 前排人脸 ROI 选择
# ============================================================

# 基于 data/ 中前排主副驾 bbox 分布标定。
# right 保留中间余量，兼容主驾侧脸向中间探身。
FRONT_ROW_ROIS = {
    "left": (0.00, 0.28, 0.34, 0.92),
    "right": (0.56, 0.28, 1.00, 0.92),
}
FRONT_ROW_MIN_FACE_AREA_RATIO = 0.015
FRONT_ROW_MIN_BOTTOM_RATIO = 0.52
FRONT_ROW_MIN_ROI_OVERLAP_RATIO = 0.60

# ── 前排人体检测 ROI（基于 camera 实际标定）──
FRONT_ROW_BODY_ROIS = {
    "left":  (0.00, 0.30, 0.42, 0.95),
    "right": (0.55, 0.30, 1.00, 0.95),
}
FRONT_ROW_BODY_MIN_AREA_RATIO = 0.04


def _kps_in_roi(kps_xy, kps_conf, roi_rect, min_conf=0.25):
    """统计关键点在 ROI 内的数量和比例。"""
    total = 0
    inside = 0
    shoulder_inside = {"left_shoulder": False, "right_shoulder": False}
    for idx in range(len(kps_xy)):
        if kps_conf[idx] < min_conf:
            continue
        total += 1
        x, y = kps_xy[idx]
        rx1, ry1, rx2, ry2 = roi_rect
        if rx1 <= x <= rx2 and ry1 <= y <= ry2:
            inside += 1
            if idx == 5:    # left_shoulder
                shoulder_inside["left_shoulder"] = True
            elif idx == 6:  # right_shoulder
                shoulder_inside["right_shoulder"] = True
    ratio = inside / max(total, 1)
    return inside, ratio, shoulder_inside


def select_front_row_bodies(bodies, image_w, image_h):
    """
    从人体检测结果中选择前排左右的人体。
    匹配条件（满足任一即可）：
      - 双肩关键点都在 ROI 内
      - 大部分关键点（>50%）在 ROI 内

    bodies: list[dict], 每个 dict 需包含:
        "bbox": [x1, y1, x2, y2]
        "score": float
        "xc" / "yc": float (肩部中点)
        "kps_xy": [[x,y]*17] | None  (17点坐标)
        "kps_conf": [conf*17] | None (17点置信度)

    返回: {
        "left": bool, "right": bool,
        "_left_body": dict|None, "_right_body": dict|None,
    }
    """
    if not bodies or image_w <= 0 or image_h <= 0:
        return {"left": False, "right": False, "_left_body": None, "_right_body": None}

    result = {"left": False, "right": False, "_left_body": None, "_right_body": None}

    import logging as _logging
    _log = _logging.getLogger("utils")

    for b in bodies:
        bbox = normalize_bbox(b["bbox"])
        if len(bbox) != 4:
            continue

        kps_xy = b.get("kps_xy")
        kps_conf = b.get("kps_conf")

        # 没关键点 → 回退到 bbox 重叠方式
        if kps_xy is None or kps_conf is None:
            body_cx = b.get("xc", (bbox[0] + bbox[2]) / 2.0)
            best_slot = None
            best_overlap = 0.0
            for slot_name, roi in FRONT_ROW_BODY_ROIS.items():
                if result[slot_name]:
                    continue
                if slot_name == "left" and body_cx >= image_w / 2:
                    continue
                if slot_name == "right" and body_cx < image_w / 2:
                    continue
                roi_rect = _scale_roi(roi, image_w, image_h)
                area = bbox_area(bbox)
                overlap = _intersection_area(bbox, roi_rect) / max(area, 1e-6)
                if overlap >= 0.50 and overlap > best_overlap:
                    best_slot = slot_name
                    best_overlap = overlap
            if best_slot:
                result[best_slot] = True
                result[f"_{best_slot}_body"] = {"xc": body_cx, "yc": b.get("yc", 0), "bbox": bbox}
            continue

        # ── 关键点匹配 ──
        best_slot = None
        best_score = 0.0  # 综合得分

        for slot_name, roi in FRONT_ROW_BODY_ROIS.items():
            if result[slot_name]:
                continue
            roi_rect = _scale_roi(roi, image_w, image_h)
            inside, ratio, shoulders = _kps_in_roi(kps_xy, kps_conf, roi_rect)

            # 条件1：双肩都在 ROI 内
            both_shoulders = shoulders["left_shoulder"] and shoulders["right_shoulder"]
            # 条件2：大部分关键点在 ROI 内
            majority_kps = ratio > 0.5

            if not both_shoulders and not majority_kps:
                _log.info("[BodySelect] kp-match=NONE slot=%s kps_in=%d ratio=%.2f "
                          "shoulders=(L=%s R=%s)",
                          slot_name, inside, ratio,
                          shoulders["left_shoulder"], shoulders["right_shoulder"])
                continue

            # 得分：双肩优先
            score = (0.7 if both_shoulders else 0.0) + 0.3 * ratio
            if score > best_score:
                best_slot = slot_name
                best_score = score

            _log.info("[BodySelect] kp-match=%s slot=%s kps_in=%d/%d ratio=%.2f "
                      "shoulders=(L=%s R=%s) score=%.2f",
                      "BOTH" if both_shoulders else "MAJORITY",
                      slot_name, inside, len(kps_xy), ratio,
                      shoulders["left_shoulder"], shoulders["right_shoulder"], score)

        # ── 关键点匹配失败 → bbox 重叠兜底（人靠方向盘、身体出画等）──
        if best_slot is None:
            _log.info("[BodySelect] kp-match=NONE-ALL → 回退bbox法 bbox=%s",
                      [round(v, 2) for v in bbox])
            body_cx = b.get("xc", (bbox[0] + bbox[2]) / 2.0)
            best_overlap = 0.0
            for slot_name, roi in FRONT_ROW_BODY_ROIS.items():
                if result[slot_name]:
                    continue
                if slot_name == "left" and body_cx >= image_w / 2:
                    continue
                if slot_name == "right" and body_cx < image_w / 2:
                    continue
                roi_rect = _scale_roi(roi, image_w, image_h)
                area = bbox_area(bbox)
                overlap = _intersection_area(bbox, roi_rect) / max(area, 1e-6)
                if overlap >= 0.35 and overlap > best_overlap:
                    best_slot = slot_name
                    best_overlap = overlap
            if best_slot:
                _log.info("[BodySelect] bbox-fallback match=%s overlap=%.2f", best_slot, best_overlap)

        if best_slot:
            result[best_slot] = True
            body_cx = b.get("xc", (bbox[0] + bbox[2]) / 2.0)
            result[f"_{best_slot}_body"] = {"xc": body_cx, "yc": b.get("yc", 0), "bbox": bbox}

    return result


def get_body_rois() -> dict:
    """获取当前 body ROI 配置。"""
    return {
        "left": list(FRONT_ROW_BODY_ROIS["left"]),
        "right": list(FRONT_ROW_BODY_ROIS["right"]),
    }


def set_body_rois(left: list, right: list):
    """动态更新 body ROI 配置。"""
    global FRONT_ROW_BODY_ROIS
    FRONT_ROW_BODY_ROIS = {
        "left": tuple(float(v) for v in left[:4]),
        "right": tuple(float(v) for v in right[:4]),
    }


def _scale_roi(roi, image_w, image_h):
    x1, y1, x2, y2 = roi
    return [x1 * image_w, y1 * image_h, x2 * image_w, y2 * image_h]


def _point_in_rect(x, y, rect):
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= x <= rx2 and ry1 <= y <= ry2


def _intersection_area(box, rect):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = rect
    ix1 = max(x1, rx1)
    iy1 = max(y1, ry1)
    ix2 = min(x2, rx2)
    iy2 = min(y2, ry2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def select_front_row_faces(faces, image_w, image_h):
    """
    从检测到的人脸中选择前排两张脸（主驾+副驾）。

    faces: list[dict], 每个 dict 需包含:
        "bbox": [x1, y1, x2, y2]
        "score": float (检测置信度)

    返回: list[dict], 选中的脸, 附带:
        "_front_row_slot": "left" | "right"
        "_front_row_score": float

    为什么不用简单 top2？
    车内多乘员场景下，后排乘客可能更正脸、更清晰，
    简单按检测分数或面积选 top2 时，可能会选到后排。
    """
    if not faces or image_w <= 0 or image_h <= 0:
        return []

    img_area = float(image_w * image_h)
    candidates = {"left": [], "right": []}

    import logging as _logging
    _log = _logging.getLogger("utils")
    _log_filter_count = 0
    for f in faces:
        bbox = normalize_bbox(f["bbox"])
        if len(bbox) != 4:
            continue

        cx, cy = bbox_center(bbox)
        x1, y1, x2, y2 = bbox
        area = bbox_area(bbox)

        if area < img_area * FRONT_ROW_MIN_FACE_AREA_RATIO:
            _log_filter_count += 1
            if _log_filter_count <= 3:
                _log.info("[FaceFilter] 面积太小: area=%.4f 需要=%.4f bbox=%s", area/img_area, FRONT_ROW_MIN_FACE_AREA_RATIO, bbox)
            continue
        if y2 < image_h * FRONT_ROW_MIN_BOTTOM_RATIO:
            _log_filter_count += 1
            if _log_filter_count <= 3:
                _log.info("[FaceFilter] 位置太高: y2=%.2f 需要>=%.2f bbox=%s", y2/image_h, FRONT_ROW_MIN_BOTTOM_RATIO, bbox)
            continue

        det_score = safe_float(f.get("det_score", f.get("score", 1.0)), 1.0)
        area_score = area / img_area
        bottom_score = y2 / float(image_h)

        matched_any = False
        for slot_name, roi in FRONT_ROW_ROIS.items():
            roi_rect = _scale_roi(roi, image_w, image_h)
            center_inside = _point_in_rect(cx, cy, roi_rect)
            overlap_ratio = _intersection_area(bbox, roi_rect) / max(area, 1e-6)

            if not center_inside and overlap_ratio < FRONT_ROW_MIN_ROI_OVERLAP_RATIO:
                continue
            matched_any = True

        if not matched_any:
            _log_filter_count += 1
            if _log_filter_count <= 3:
                _log.info("[FaceFilter] ROI不匹配: cx=%.2f cy=%.2f bbox=%s", cx/image_w, cy/image_h, bbox)

        for slot_name, roi in FRONT_ROW_ROIS.items():
            roi_rect = _scale_roi(roi, image_w, image_h)
            center_inside = _point_in_rect(cx, cy, roi_rect)
            overlap_ratio = _intersection_area(bbox, roi_rect) / max(area, 1e-6)

            if not center_inside and overlap_ratio < FRONT_ROW_MIN_ROI_OVERLAP_RATIO:
                continue

            score = (
                0.35 * overlap_ratio
                + 0.30 * area_score
                + 0.20 * bottom_score
                + 0.15 * det_score
            )

            candidates[slot_name].append({
                "raw": f,
                "score": score,
                "slot": slot_name,
            })

    for slot_name in candidates:
        candidates[slot_name] = sorted(candidates[slot_name], key=lambda x: x["score"], reverse=True)

    selected = []
    for slot_name in ("left", "right"):
        if not candidates[slot_name]:
            continue
        face = candidates[slot_name][0]["raw"]
        face["_front_row_slot"] = slot_name
        face["_front_row_score"] = round(float(candidates[slot_name][0]["score"]), 6)
        selected.append(face)

    selected = sorted(selected, key=lambda f: bbox_center(normalize_bbox(f["bbox"]))[0])
    return selected[:2]


# ============================================================
# 4. 裁剪人脸
# ============================================================

def crop_face_pil(image_bgr, bbox, expand_ratio=0.15):
    """
    从原图裁剪人脸，并稍微扩一点边界。
    输入是 OpenCV 的 BGR 图像，输出是 PIL RGB 图像。
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        raise ValueError(f"Invalid bbox: {bbox}")

    ex = int(bw * expand_ratio)
    ey = int(bh * expand_ratio)

    x1 = max(0, x1 - ex)
    y1 = max(0, y1 - ey)
    x2 = min(w, x2 + ex)
    y2 = min(h, y2 + ey)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Expanded bbox out of bounds: {[x1, y1, x2, y2]}")

    crop_bgr = image_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        raise ValueError(f"Empty crop for bbox: {bbox}")

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(crop_rgb)
    return face_pil


# ============================================================
# 5. 可视化绘制
# ============================================================

def draw_results(image_bgr, results):
    """
    results 格式:
    {
        "face1": {...},   # 图像左侧人脸，通常是副驾
        "face2": {...},   # 图像右侧人脸，通常是主驾
    }
    """
    vis = image_bgr.copy()

    role_styles = {
        "face1": {
            "color": (255, 255, 0),   # 青色 BGR
            "name": "FACE1"
        },
        "face2": {
            "color": (255, 255, 0),   # 青色 BGR
            "name": "FACE2"
        }
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
        gender_score = safe_float(info.get("gender_score", 0.0), 0.0)
        age_group = str(info.get("age_group", "") or "")
        age_score = safe_float(info.get("age_score", 0.0), 0.0)

        label_1 = f"{role_name}"
        label_2 = f"Gender: {gender} ({gender_score:.2f})"
        label_3 = f"Age: {age_group} ({age_score:.2f})"

        # 人脸框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.72
        thickness = 2
        pad_x = 10
        pad_y = 8
        line_gap = 10

        (w1, h1), _ = cv2.getTextSize(label_1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(label_2, font, font_scale, thickness)
        (w3, h3), _ = cv2.getTextSize(label_3, font, font_scale, thickness)

        text_w = max(w1, w2, w3)
        text_h = h1 + h2 + h3 + line_gap * 2 + pad_y * 2

        img_h, img_w = vis.shape[:2]

        # 默认放在人脸框上方
        box_x1 = x1
        box_y2 = y1 - 8
        box_y1 = box_y2 - text_h
        box_x2 = box_x1 + text_w + pad_x * 2

        # 上方放不下就放下方
        if box_y1 < 0:
            box_y1 = y2 + 8
            box_y2 = box_y1 + text_h

        # 右边越界修正
        if box_x2 > img_w:
            box_x1 = max(0, img_w - (text_w + pad_x * 2))
            box_x2 = box_x1 + text_w + pad_x * 2

        # 下方越界修正
        if box_y2 > img_h:
            box_y2 = img_h - 2
            box_y1 = max(0, box_y2 - text_h)

        # 标签避让
        candidate_box = [box_x1, box_y1, box_x2, box_y2]
        shift_step = text_h + 8
        max_try = 6
        try_count = 0

        while any(overlaps(candidate_box, old_box) for old_box in placed_label_boxes) and try_count < max_try:
            new_y1 = candidate_box[1] - shift_step
            new_y2 = candidate_box[3] - shift_step

            if new_y1 >= 0:
                candidate_box = [candidate_box[0], new_y1, candidate_box[2], new_y2]
            else:
                new_y1 = candidate_box[1] + shift_step
                new_y2 = candidate_box[3] + shift_step
                if new_y2 <= img_h:
                    candidate_box = [candidate_box[0], new_y1, candidate_box[2], new_y2]
                else:
                    break

            try_count += 1

        box_x1, box_y1, box_x2, box_y2 = candidate_box
        placed_label_boxes.append(candidate_box)

        # 标签底框
        cv2.rectangle(vis, (box_x1, box_y1), (box_x2, box_y2), color, -1)
        cv2.rectangle(vis, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)

        text_x = box_x1 + pad_x
        y_text = box_y1 + pad_y + h1

        draw_text_with_outline(
            vis, label_1, (text_x, y_text),
            font, font_scale, (255, 255, 255), (0, 0, 0), thickness
        )

        y_text += h2 + line_gap
        draw_text_with_outline(
            vis, label_2, (text_x, y_text),
            font, font_scale, (255, 255, 255), (0, 0, 0), thickness
        )

        y_text += h3 + line_gap
        draw_text_with_outline(
            vis, label_3, (text_x, y_text),
            font, font_scale, (255, 255, 255), (0, 0, 0), thickness
        )

    return vis