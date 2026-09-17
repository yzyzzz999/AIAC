#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""偏好层端到端快速测试脚本。

模拟完整的"RF推荐 → 人工接管 → 数据收集 → 车端训练 → 联合推理"流程。
将20分钟缩短为120秒，方便快速验证。

用法:
    cd /home/data/AIAC/sls_recommendation_weather26
    python test_preference_layer.py
"""

from __future__ import annotations

import os
import sys
import time
import threading
import warnings

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.core.models.base_model_loader import RandomForestModelLoader
from src.preference_learning.preference_layer_manager import (
    CollectedSample,
    PreferenceLayerManager,
    PreferenceState,
)

# ============================================================
# 测试配置（缩短版）
# ============================================================
TEST_COLLECTION_SECONDS = 120      # 2分钟收集（正式版是1200）
TEST_MIN_SAMPLES = 3               # 最少3条（正式版是30，因为采样间隔可能不同）
TEST_SAMPLE_INTERVAL = 10.0        # 10秒一条（正式版是5秒）
TEST_TAKEOVER_CONFIRM = 2.0        # 2秒接管确认（正式版是5秒）


def create_synthetic_features(feature_columns: list, seed: int = 42) -> dict:
    """生成一帧合法的26维特征。"""
    rng = np.random.default_rng(seed)
    feat = {}
    for col in feature_columns:
        if col in ("主驾PMV", "副驾PMV"):
            feat[col] = float(rng.uniform(-3, 3))
        elif col == "天气状况":
            feat[col] = float(rng.choice([1, 2, 3, 4]))
        elif col in ("主驾性别", "副驾性别"):
            feat[col] = float(rng.choice([0, 1]))
        elif "年龄" in col:
            feat[col] = float(rng.integers(0, 5))
        elif "衣着" in col:
            feat[col] = float(rng.integers(0, 10))
        elif "BMI" in col:
            feat[col] = float(rng.integers(0, 4))
        elif "身高" in col:
            feat[col] = float(rng.uniform(150, 190))
        elif col == "风向":
            feat[col] = float(rng.integers(0, 8))
        elif col == "实时车速":
            feat[col] = float(rng.uniform(0, 120))
        elif col == "阳光传感器":
            feat[col] = float(rng.uniform(0, 1000))
        elif col == "气压(hPa)":
            feat[col] = float(rng.uniform(990, 1030))
        elif "湿度" in col:
            feat[col] = float(rng.uniform(30, 80))
        else:
            feat[col] = float(rng.uniform(18, 32))
    return feat


def format_result(d: dict | None, label: str = "") -> str:
    if d is None:
        return "None"
    return (f"driver={d.get('driver_temp', '?')}°C, "
            f"passenger={d.get('passenger_temp', '?')}°C, "
            f"wind={d.get('wind_speed', '?')}, "
            f"mode={d.get('air_mode', '?')}")


def main():
    print("=" * 70)
    print("  偏好层端到端测试")
    print(f"  采集时长: {TEST_COLLECTION_SECONDS}秒  |  接管确认: {TEST_TAKEOVER_CONFIRM}秒")
    print("=" * 70)

    # ---- 1. 加载26维RF ----
    print("\n[1/6] 加载weather26随机森林...")
    loader = RandomForestModelLoader(
        model_dir="models/weather26",
        feature_config="models/weather26/manifest.json",
        package_mode="weather26",
    )
    if not loader.load_all():
        print("FAIL: RF模型加载失败")
        return 1
    features = loader.get_feature_columns()
    print(f"  加载成功: {len(features)}维特征")
    print(f"  前5个特征: {features[:5]}")

    # ---- 2. 创建偏好层管理器 (缩短时间) ----
    print("\n[2/6] 创建偏好层管理器...")
    output_dir = "models/preference_layer"
    os.makedirs(output_dir, exist_ok=True)

    mgr = PreferenceLayerManager(
        model_loader=loader,
        output_dir=output_dir,
    )
    # 覆盖时间参数为测试版
    mgr.COLLECTION_DURATION = TEST_COLLECTION_SECONDS
    mgr.MIN_TRAINING_SAMPLES = TEST_MIN_SAMPLES
    mgr.SAMPLE_INTERVAL = TEST_SAMPLE_INTERVAL
    mgr.TAKEOVER_CONFIRM_SECONDS = TEST_TAKEOVER_CONFIRM

    status = mgr.get_status()
    print(f"  初始状态: {status.state_name}")

    # ---- 3. 运行RF获取基准推荐 ----
    print("\n[3/6] 运行RF获取基准推荐...")
    feat0 = create_synthetic_features(features, seed=0)
    feat_df = pd.DataFrame([feat0], columns=features)
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    base_result = loader.predict(feat_df)
    rf_dict = {
        "driver_temp": base_result.driver_temp,
        "passenger_temp": base_result.passenger_temp,
        "wind_speed": float(base_result.wind_speed),
        "air_mode": float(base_result.air_mode),
    }
    print(f"  RF基准推荐: {format_result(rf_dict)}")
    # 实车流程只在命令真实发送并被CAN回读确认后开放差异接管检测。
    mgr.note_command_sent(rf_dict)
    mgr.update(feat0, rf_dict, rf_dict)

    # ---- 4. 模拟人工接管 ----
    print(f"\n[4/6] 模拟人工接管 (5秒高温偏好: 主驾26→28°C, 副驾28→26°C, 风量3→5)...")
    human_actions = {
        "driver_temp": 28.0,
        "passenger_temp": 26.0,
        "wind_speed": 5.0,
        "air_mode": 1.0,
    }
    print(f"  人工设定: {format_result(human_actions)}")

    # 调用update直到确认接管
    takeover_confirmed = False
    for i in range(20):
        resp = mgr.update(feat0, rf_dict, human_actions)
        if resp.get("state") == "collecting":
            takeover_confirmed = True
            print(f"  ✓ 人工接管已确认! (第{i+1}次调用)")
            break
        time.sleep(0.5)

    if not takeover_confirmed:
        print("  FAIL: 接管未确认")
        return 1

    # ---- 5. 收集训练样本 ----
    print(f"\n[5/6] 收集训练样本 (每{TEST_SAMPLE_INTERVAL}秒一条, 共{TEST_COLLECTION_SECONDS}秒)...")

    # 预设多种偏好场景的样本（模拟真实用户操作变化）
    scenarios = [
        {"driver_temp": 28.0, "passenger_temp": 26.0, "wind_speed": 5.0, "air_mode": 1.0},
        {"driver_temp": 27.5, "passenger_temp": 26.5, "wind_speed": 4.0, "air_mode": 1.0},
        {"driver_temp": 28.0, "passenger_temp": 27.0, "wind_speed": 5.0, "air_mode": 2.0},
        {"driver_temp": 27.0, "passenger_temp": 25.5, "wind_speed": 4.0, "air_mode": 1.0},
    ]

    collection_start = time.time()
    sample_count = 0
    last_status_print = 0

    training_triggered = False
    while time.time() - collection_start < TEST_COLLECTION_SECONDS:
        now = time.time()
        # 每10秒轮换偏好场景，模拟真实用户变化
        scenario_idx = int((now - collection_start) / 30) % len(scenarios)
        scenario = scenarios[scenario_idx]

        # 刷新特征（模拟车辆环境变化）
        feat = create_synthetic_features(features, seed=int(now))

        resp = mgr.update(feat, rf_dict, scenario)

        # 检查训练触发
        if resp.get("trigger_training"):
            training_triggered = True
            print(f"\n  >>> 训练触发! (在 {now - collection_start:.0f}秒)")
            break

        # 打印进度
        if now - last_status_print >= 10:
            status = mgr.get_status()
            elapsed = now - collection_start
            print(f"  进度: {elapsed:.0f}/{TEST_COLLECTION_SECONDS}s, "
                  f"样本={status.sample_count}, 状态={status.state_name}")
            last_status_print = now

        time.sleep(1)

    # 如果循环结束时还没触发，补一次update触发状态迁移
    if not training_triggered:
        print("  等待最后一帧触发训练...")
        for _ in range(5):
            resp = mgr.update(
                create_synthetic_features(features, seed=9999),
                rf_dict,
                scenarios[-1],
            )
            if resp.get("trigger_training"):
                training_triggered = True
                print("  >>> 训练触发!")
                break
            time.sleep(1.0)

    status = mgr.get_status()
    print(f"\n  收集完成: {status.sample_count}条样本, 状态={status.state_name}")

    if status.state_name == "training":
        # ---- 6. 执行训练 ----
        print("\n[6/6] 执行车端MLP训练...")
        try:
            samples_df = mgr.get_training_dataframe()
            print(f"  训练数据: {len(samples_df)}行 x {len(samples_df.columns)}列")

            from preference_learning.online_trainer_v26 import Weather26OnlineTrainer

            trainer = Weather26OnlineTrainer(
                model_loader=loader,
                output_dir=output_dir,
                min_samples=TEST_MIN_SAMPLES,
                validation_ratio=0.2,
            )
            report = trainer.train(samples_df)
            mgr.on_training_done(report)

            print(f"\n{'='*70}")
            print(f"  训练结果")
            print(f"  {'='*70}")
            print(f"  success:      {report.get('success')}")
            print(f"  accepted:     {report.get('accepted')}")
            print(f"  rows:         {report.get('rows')}")
            print(f"  base_score:   {report.get('base_score', 0):.4f}")
            print(f"  updated_score:{report.get('updated_score', 0):.4f}")
            print(f"  training_time:{report.get('training_time_s', 0):.1f}s")

            if report.get("success"):
                # 验证MLP联合推理
                print(f"\n  验证联合推理 (RF+MLP)...")
                test_feat = create_synthetic_features(features, seed=999)
                joint = mgr.apply_mlp_residual(test_feat, rf_dict)
                if joint:
                    print(f"  RF基准:  {format_result(rf_dict)}")
                    print(f"  MLP修正后: {format_result(joint)}")
                    print(f"  MLP残差Δ: driver={joint.get('driver_delta', 0):+.1f}°C, "
                          f"passenger={joint.get('passenger_delta', 0):+.1f}°C")

                print(f"\n  模型文件已保存: {output_dir}/mlp_model.joblib")
                print(f"  训练报告: {output_dir}/training_report.json")

                # 最终状态
                final_status = mgr.get_status()
                print(f"\n  最终状态: {final_status.state_name}")
                print(f"  MLP已加载: {final_status.mlp_loaded}")

            print(f"\n{'='*70}")
            if report.get("success") and report.get("accepted"):
                print("  ✓ 偏好层测试通过! RF+MLP联合推理已启用")
            else:
                print(f"  ⚠ 训练完成但未接受: {report.get('reason', '?')}")
            print(f"{'='*70}")

        except Exception as e:
            print(f"\n  FAIL: 训练异常: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print(f"\n  WARNING: 未达到训练触发条件 (状态={status.state_name}, 样本={status.sample_count})")
        print(f"  可能原因: 样本数不足 {TEST_MIN_SAMPLES} 条 或 收集时长不足 {TEST_COLLECTION_SECONDS}秒")

    return 0


if __name__ == "__main__":
    sys.exit(main())
