"""
pipeline/data_loader.py
=======================
GT 标注解析 + data/ 目录图片扫描。
"""

import os
import re
import csv
import json
from collections import Counter

import numpy as np

import numpy as np

from utils import safe_float

# ---- 常量 ----

PERSON_FILE_CANDIDATES = [
    "person_information_summary.csv",
    "person_information_summary.txt",
    "粘贴的文本 (2).txt",
    os.path.join("data", "person_information_summary.csv"),
    os.path.join("data", "person_information_summary.txt"),
    os.path.join("AIACdataset", "person_information_summary.csv"),
    os.path.join("AIACdataset", "person_information_summary.txt"),
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AGE_GROUPS = ["12-17", "18-40", "41-59", "60-74", "75+"]


# ---- 通用工具 ----

def norm_str(x):
    if x is None:
        return ""
    return str(x).strip()


def str_to_bool(x):
    return norm_str(x).lower() in ("true", "1", "yes", "y", "t")


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTS)


def uniform_sample_paths(image_paths, max_frames=12):
    image_paths = list(image_paths or [])
    if max_frames is None or max_frames <= 0:
        return image_paths
    n = len(image_paths)
    if n <= max_frames:
        return image_paths
    idxs = np.linspace(0, n - 1, int(max_frames)).round().astype(int).tolist()
    idxs = sorted(set(idxs))
    return [image_paths[i] for i in idxs]


def natural_key(s):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(s))]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def save_csv(rows, save_path, fieldnames):
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] saved csv: {save_path}")


# ---- CSV 读取 ----

def read_csv_rows_safely(csv_path):
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_error = None
    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return [{norm_str(k): norm_str(v) for k, v in row.items()} for row in reader]
        except UnicodeDecodeError as e:
            last_error = e
    raise UnicodeDecodeError("unknown", b"", 0, 1,
                             f"无法读取文件编码: {csv_path}, last_error={last_error}")


def find_person_summary_file(base_dir):
    for rel_path in PERSON_FILE_CANDIDATES:
        p = os.path.join(base_dir, rel_path)
        if os.path.isfile(p):
            return p
    for root, _, files in os.walk(base_dir):
        for name in files:
            if not name.lower().endswith((".csv", ".txt")):
                continue
            p = os.path.join(root, name)
            try:
                with open(p, "r", encoding="utf-8-sig", errors="ignore") as f:
                    first_line = f.readline().strip()
                if first_line.startswith("folder,txt_path,position") or first_line.startswith("folder,position"):
                    return p
            except Exception:
                pass
    raise FileNotFoundError(
        "没有找到第二个文件。请把它保存为 person_information_summary.csv，"
        "并放在 main.py 同级目录，或者 data 目录下。"
    )


# ---- GT 构建 ----

def cn_gender_to_en(gender):
    gender = norm_str(gender).lower()
    mapping = {
        "男": "male", "男性": "male", "m": "male", "male": "male",
        "女": "female", "女性": "female", "f": "female", "female": "female",
    }
    return mapping.get(gender, "0")


def age_to_group(age):
    age = norm_str(age)
    if age in AGE_GROUPS:
        return age
    try:
        age_value = int(float(age))
    except Exception:
        return "0"
    if age_value <= 17:
        return "12-17"
    if age_value <= 40:
        return "18-40"
    if age_value <= 59:
        return "41-59"
    if age_value <= 74:
        return "60-74"
    return "75+"


def build_gt_from_person_summary_csv(
    person_summary_path, save_gt_csv, save_one_folder_csv,
    save_clean_rows_csv, save_person_stats_json,
):
    rows = read_csv_rows_safely(person_summary_path)
    if not rows:
        raise ValueError(f"第二个文件为空: {person_summary_path}")

    required_cols = ["folder", "position", "年龄", "性别"]
    for col in required_cols:
        if col not in rows[0]:
            raise ValueError(f"第二个文件缺少字段: {col}，当前字段为: {list(rows[0].keys())}")

    clean_rows = []
    folder_map = {}

    for r in rows:
        folder = norm_str(r.get("folder", ""))
        position = norm_str(r.get("position", ""))
        if folder == "":
            continue

        cur = {
            "folder": folder,
            "txt_path": norm_str(r.get("txt_path", "")),
            "position": position,
            "id": norm_str(r.get("id", "")),
            "年龄": norm_str(r.get("年龄", "")),
            "性别": norm_str(r.get("性别", "")),
            "身高(cm)": norm_str(r.get("身高(cm)", "")),
            "体重": norm_str(r.get("体重", "")),
            "BMI": norm_str(r.get("BMI", "")),
            "BMI分类": norm_str(r.get("BMI分类", "")),
            "衣着": norm_str(r.get("衣着", "")),
            "gender_en": cn_gender_to_en(r.get("性别", "")),
            "age_group": age_to_group(r.get("年龄", "")),
        }
        clean_rows.append(cur)

        if folder not in folder_map:
            folder_map[folder] = {"主驾": None, "副驾": None, "all_rows": []}
        folder_map[folder]["all_rows"].append(cur)
        if position == "主驾":
            folder_map[folder]["主驾"] = cur
        elif position == "副驾":
            folder_map[folder]["副驾"] = cur

    gt_rows = []
    one_folder_rows = []

    for folder in sorted(folder_map.keys(), key=natural_key):
        driver = folder_map[folder]["主驾"]
        passenger = folder_map[folder]["副驾"]
        driver_exists = driver is not None
        passenger_exists = passenger is not None
        person_forward_number = int(driver_exists) + int(passenger_exists)

        gt_rows.append({
            "video": folder,
            "person_forward_number": str(person_forward_number),
            "gender0": driver["gender_en"] if driver_exists else "0",
            "age0": driver["age_group"] if driver_exists else "0",
            "gender1": passenger["gender_en"] if passenger_exists else "0",
            "age1": passenger["age_group"] if passenger_exists else "0",
        })
        one_folder_rows.append({
            "folder": folder,
            "person_forward_number": str(person_forward_number),
            "driver_id": driver["id"] if driver_exists else "",
            "driver_gender_raw": driver["性别"] if driver_exists else "",
            "driver_gender": driver["gender_en"] if driver_exists else "0",
            "driver_age_raw": driver["年龄"] if driver_exists else "",
            "driver_age_group": driver["age_group"] if driver_exists else "0",
            "passenger_id": passenger["id"] if passenger_exists else "",
            "passenger_gender_raw": passenger["性别"] if passenger_exists else "",
            "passenger_gender": passenger["gender_en"] if passenger_exists else "0",
            "passenger_age_raw": passenger["年龄"] if passenger_exists else "",
            "passenger_age_group": passenger["age_group"] if passenger_exists else "0",
        })

    gt_fieldnames = ["video", "person_forward_number", "gender0", "age0", "gender1", "age1"]
    clean_fieldnames = [
        "folder", "txt_path", "position", "id", "年龄", "性别", "身高(cm)", "体重",
        "BMI", "BMI分类", "衣着", "gender_en", "age_group"
    ]
    one_folder_fieldnames = [
        "folder", "person_forward_number",
        "driver_id", "driver_gender_raw", "driver_gender", "driver_age_raw", "driver_age_group",
        "passenger_id", "passenger_gender_raw", "passenger_gender", "passenger_age_raw", "passenger_age_group",
    ]

    save_csv(gt_rows, save_gt_csv, gt_fieldnames)
    save_csv(clean_rows, save_clean_rows_csv, clean_fieldnames)
    save_csv(one_folder_rows, save_one_folder_csv, one_folder_fieldnames)

    stats = {
        "source_file": person_summary_path,
        "total_raw_rows": len(rows),
        "total_clean_rows": len(clean_rows),
        "total_folders": len(folder_map),
        "total_gt_samples": len(gt_rows),
        "position_count": dict(Counter(r["position"] for r in clean_rows)),
        "gender_count_raw": dict(Counter(r["性别"] for r in clean_rows)),
        "gender_count_en": dict(Counter(r["gender_en"] for r in clean_rows)),
        "age_group_count": dict(Counter(r["age_group"] for r in clean_rows)),
    }
    with open(save_person_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("[INFO] 第二个文件读取完成")
    print(f"[INFO] source file: {person_summary_path}")
    print(f"[INFO] raw rows: {len(rows)}")
    print(f"[INFO] clean rows: {len(clean_rows)}")
    print(f"[INFO] folders / samples: {len(folder_map)}")

    sample_folders = [r["video"] for r in gt_rows]
    return gt_rows, one_folder_rows, sample_folders


# ---- data/ 图片扫描 ----

def collect_images_for_one_folder_modalities(data_root, folder_name):
    folder_dir = os.path.join(data_root, folder_name)
    modality_dirs = {
        "rgb": [os.path.join(folder_dir, "rgb"), os.path.join(folder_dir, "RGB")],
        "nir": [os.path.join(folder_dir, "nir"), os.path.join(folder_dir, "NIR")],
    }
    result = {"rgb": [], "nir": []}
    for modality, dirs in modality_dirs.items():
        image_paths = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for name in files:
                    if is_image_file(name):
                        image_paths.append(os.path.join(root, name))
        result[modality] = sorted(image_paths, key=natural_key)
    return result


def build_sequence_dict_from_folders(data_root, sample_folders):
    seq_dict = {}
    missing_folders = []
    for folder in sample_folders:
        modality_images = collect_images_for_one_folder_modalities(data_root, folder)
        if modality_images["rgb"] or modality_images["nir"]:
            seq_dict[folder] = modality_images
        else:
            missing_folders.append(folder)

    print("=" * 60)
    print("[INFO] 根据指定 folder 列表收集 RGB / NIR 图片完成")
    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] expected folders: {len(sample_folders)}")
    print(f"[INFO] folders with images: {len(seq_dict)}")
    print(f"[INFO] folders without images: {len(missing_folders)}")
    if missing_folders:
        print("[WARN] 以下 folder 没找到图片：")
        for x in missing_folders[:50]:
            print(f"  - {x}")
        if len(missing_folders) > 50:
            print(f"  ... 还有 {len(missing_folders) - 50} 个")
    return seq_dict, missing_folders


def collect_all_folders_from_data(data_root):
    folders = []
    if not os.path.isdir(data_root):
        return folders
    for name in os.listdir(data_root):
        p = os.path.join(data_root, name)
        if os.path.isdir(p):
            folders.append(name)
    folders = sorted(folders, key=natural_key)
    print("=" * 60)
    print("[INFO] 扫描 data 目录完成")
    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] folders found in data: {len(folders)}")
    return folders
