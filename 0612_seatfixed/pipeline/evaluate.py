"""
pipeline/evaluate.py
====================
统计与评估指标。
"""

import os
import csv
import json

from pipeline.data_loader import norm_str, str_to_bool, save_csv


def safe_acc(correct_count, total_count):
    if total_count <= 0:
        return 0.0
    return float(correct_count) / float(total_count)


def save_statistics(debug_rows, save_stats_json, save_with_result_csv, save_without_result_csv):
    total_sequences = len(debug_rows)
    any_result_rows = [r for r in debug_rows if str_to_bool(r.get("face1_found")) or str_to_bool(r.get("face2_found"))]
    no_result_rows = [r for r in debug_rows if not (str_to_bool(r.get("face1_found")) or str_to_bool(r.get("face2_found")))]
    both_found_rows = [r for r in debug_rows if str_to_bool(r.get("face1_found")) and str_to_bool(r.get("face2_found"))]

    stats = {
        "total_sequences": total_sequences,
        "sequences_with_any_result": len(any_result_rows),
        "sequences_without_any_result": len(no_result_rows),
        "sequences_with_both_faces": len(both_found_rows),
        "sequences_no_face_detected": len([r for r in debug_rows if r.get("remark") == "no_face_detected_in_sequence"]),
        "sequences_only_one_face_fused": len([r for r in debug_rows if r.get("remark") == "only_one_face_fused"]),
        "sequences_two_faces_fused_ok": len([r for r in debug_rows if r.get("remark") == "two_faces_fused_ok"]),
        "sequences_faces_detected_but_no_valid_fusion": len([r for r in debug_rows if r.get("remark") == "faces_detected_but_no_valid_fusion"]),
    }
    with open(save_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    debug_fieldnames = [
        "sequence_id", "video", "modality", "num_frames", "processed_frames", "sampled_frames",
        "frames_with_any_face", "frames_with_two_faces",
        "face1_attr_reject_count", "face2_attr_reject_count",
        "face1_valid_frame_count", "face2_valid_frame_count",
        "face1_found", "face2_found", "remark",
    ]
    save_csv(any_result_rows, save_with_result_csv, debug_fieldnames)
    save_csv(no_result_rows, save_without_result_csv, debug_fieldnames)

    print("=" * 60)
    print("[INFO] Statistics Summary")
    for k, v in stats.items():
        print(f"[INFO] {k}: {v}")


def evaluate_predictions(gt_path, pred_csv_path, save_merged_csv, save_eval_json):
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")
    if not os.path.isfile(pred_csv_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_csv_path}")

    with open(gt_path, "r", encoding="utf-8-sig", newline="") as f:
        gt_rows = [{norm_str(k): norm_str(v) for k, v in row.items()} for row in csv.DictReader(f)]
    with open(pred_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        pred_rows = [{norm_str(k): norm_str(v) for k, v in row.items()} for row in csv.DictReader(f)]

    pred_by_video = {}
    for row in pred_rows:
        video = norm_str(row.get("video", ""))
        if video:
            pred_by_video[video] = row

    merged_rows = []
    person_correct = person_total = 0
    driver_gender_correct = driver_gender_total = 0
    driver_age_correct = driver_age_total = 0
    passenger_gender_correct = passenger_gender_total = 0
    passenger_age_correct = passenger_age_total = 0
    all_correct_count = all_total = 0

    for gt in gt_rows:
        video = norm_str(gt.get("video", ""))
        pred = pred_by_video.get(video, {})

        gt_person = norm_str(gt.get("person_forward_number", ""))
        gt_gender0 = norm_str(gt.get("gender0", "")); gt_age0 = norm_str(gt.get("age0", ""))
        gt_gender1 = norm_str(gt.get("gender1", "")); gt_age1 = norm_str(gt.get("age1", ""))

        pred_person = norm_str(pred.get("pred_person_forward_number", "0"))
        pred_gender0 = norm_str(pred.get("pred_gender0", "0")); pred_age0 = norm_str(pred.get("pred_age0", "0"))
        pred_gender1 = norm_str(pred.get("pred_gender1", "0")); pred_age1 = norm_str(pred.get("pred_age1", "0"))

        person_ok = gt_person == pred_person
        person_correct += int(person_ok); person_total += 1

        driver_gender_ok = driver_age_ok = passenger_gender_ok = passenger_age_ok = False

        if gt_gender0 != "0":
            driver_gender_ok = gt_gender0 == pred_gender0
            driver_gender_correct += int(driver_gender_ok); driver_gender_total += 1
        if gt_age0 != "0":
            driver_age_ok = gt_age0 == pred_age0
            driver_age_correct += int(driver_age_ok); driver_age_total += 1
        if gt_gender1 != "0":
            passenger_gender_ok = gt_gender1 == pred_gender1
            passenger_gender_correct += int(passenger_gender_ok); passenger_gender_total += 1
        if gt_age1 != "0":
            passenger_age_ok = gt_age1 == pred_age1
            passenger_age_correct += int(passenger_age_ok); passenger_age_total += 1

        if gt_person == "1":
            all_ok = person_ok and driver_gender_ok and driver_age_ok
        else:
            all_ok = person_ok and driver_gender_ok and driver_age_ok and passenger_gender_ok and passenger_age_ok
        all_correct_count += int(all_ok); all_total += 1

        merged_rows.append({
            "video": video, "gt_person_forward_number": gt_person,
            "gt_gender0_driver": gt_gender0, "gt_age0_driver": gt_age0,
            "gt_gender1_passenger": gt_gender1, "gt_age1_passenger": gt_age1,
            "pred_person_forward_number": pred_person,
            "pred_gender0_driver": pred_gender0, "pred_age0_driver": pred_age0,
            "pred_gender1_passenger": pred_gender1, "pred_age1_passenger": pred_age1,
            "person_correct": int(person_ok), "driver_gender_correct": int(driver_gender_ok),
            "driver_age_correct": int(driver_age_ok), "passenger_gender_correct": int(passenger_gender_ok),
            "passenger_age_correct": int(passenger_age_ok), "all_correct": int(all_ok),
        })

    merged_fieldnames = [
        "video", "gt_person_forward_number", "gt_gender0_driver", "gt_age0_driver",
        "gt_gender1_passenger", "gt_age1_passenger",
        "pred_person_forward_number", "pred_gender0_driver", "pred_age0_driver",
        "pred_gender1_passenger", "pred_age1_passenger",
        "person_correct", "driver_gender_correct", "driver_age_correct",
        "passenger_gender_correct", "passenger_age_correct", "all_correct",
    ]
    save_csv(merged_rows, save_merged_csv, merged_fieldnames)

    metrics = {
        "total_samples": all_total,
        "person_count_acc": safe_acc(person_correct, person_total),
        "person_count_correct": person_correct, "person_count_total": person_total,
        "driver_gender_acc": safe_acc(driver_gender_correct, driver_gender_total),
        "driver_gender_correct": driver_gender_correct, "driver_gender_total": driver_gender_total,
        "driver_age_acc": safe_acc(driver_age_correct, driver_age_total),
        "driver_age_correct": driver_age_correct, "driver_age_total": driver_age_total,
        "passenger_gender_acc": safe_acc(passenger_gender_correct, passenger_gender_total),
        "passenger_gender_correct": passenger_gender_correct, "passenger_gender_total": passenger_gender_total,
        "passenger_age_acc": safe_acc(passenger_age_correct, passenger_age_total),
        "passenger_age_correct": passenger_age_correct, "passenger_age_total": passenger_age_total,
        "all_correct_acc": safe_acc(all_correct_count, all_total),
        "all_correct_count": all_correct_count, "all_total": all_total,
    }
    with open(save_eval_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Person count acc: {metrics['person_count_acc']:.4f} ({person_correct}/{person_total})")
    print(f"Driver gender acc: {metrics['driver_gender_acc']:.4f} ({driver_gender_correct}/{driver_gender_total})")
    print(f"Driver age acc: {metrics['driver_age_acc']:.4f} ({driver_age_correct}/{driver_age_total})")
    print(f"Passenger gender acc: {metrics['passenger_gender_acc']:.4f} ({passenger_gender_correct}/{passenger_gender_total})")
    print(f"Passenger age acc: {metrics['passenger_age_acc']:.4f} ({passenger_age_correct}/{passenger_age_total})")
    print(f"All-correct acc: {metrics['all_correct_acc']:.4f} ({all_correct_count}/{all_total})")

    return metrics
