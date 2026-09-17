#!/usr/bin/env python3
"""偏好层完整流程演示：RF推荐 → 人工接管 → 收集 → 训练 → 联合推理

模拟场景:
- RF推荐 22°C, 用户觉得太冷, 手动调到 26°C
- 系统检测到接管, 收集5分钟数据, 训练MLP
- 训练完成后 MLP学会用户偏好, 联合推理输出接近26°C

用法: python test_preference_flow.py
"""

import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.core.models.base_model_loader import RandomForestModelLoader
from src.preference_learning.preference_layer_manager import PreferenceLayerManager
from src.preference_learning.online_trainer_v26 import Weather26OnlineTrainer

# ===== 测试参数 =====
COLLECTION_SEC = 120   # 2分钟收集
MIN_SAMPLES = 3
TAKEOVER_CONFIRM = 3.0  # 3秒接管确认
SAMPLE_INTERVAL = 10.0  # 10秒一条

# ===== 模拟场景: 用户喜欢26°C高温, RF推荐22°C太冷 =====
RF_PREDICTS = {"driver_temp": 22.0, "passenger_temp": 22.5, "wind_speed": 4.0, "air_mode": 2.0}
HUMAN_PREFERS = {"driver_temp": 26.0, "passenger_temp": 26.0, "wind_speed": 5.0, "air_mode": 1.0}


def make_features(feature_cols, seed=0):
    rng = np.random.default_rng(seed)
    f = {}
    for col in feature_cols:
        if col in ("主驾PMV", "副驾PMV"): f[col] = float(rng.uniform(-3, 3))
        elif col == "天气状况": f[col] = float(rng.choice([1,2,3,4]))
        elif col in ("主驾性别","副驾性别"): f[col] = float(rng.choice([0,1]))
        elif "年龄" in col: f[col] = float(rng.integers(0,5))
        elif "衣着" in col: f[col] = float(rng.integers(0,10))
        elif "BMI" in col: f[col] = float(rng.integers(0,4))
        elif "身高" in col: f[col] = float(rng.uniform(150,190))
        elif col == "风向": f[col] = float(rng.integers(0,8))
        elif col == "实时车速": f[col] = float(rng.uniform(0,120))
        elif col == "阳光传感器": f[col] = float(rng.uniform(0,1000))
        elif col == "气压(hPa)": f[col] = float(rng.uniform(990,1030))
        elif "湿度" in col: f[col] = float(rng.uniform(30,80))
        else: f[col] = float(rng.uniform(18,32))
    return f


def main():
    print("=" * 60)
    print("  偏好层完整流程演示")
    print("=" * 60)

    # ---- 1. 加载RF ----
    print("\n[1] 加载26维随机森林...")
    loader = RandomForestModelLoader(
        model_dir="models/weather26",
        feature_config="models/weather26/manifest.json",
        package_mode="weather26",
    )
    loader.load_all()
    features = loader.get_feature_columns()
    print(f"    RF加载成功: {len(features)}维, 900棵树")

    # ---- 2. 创建偏好层管理器 ----
    os.makedirs("models/preference_layer", exist_ok=True)
    mgr = PreferenceLayerManager(
        model_loader=loader,
        output_dir="models/preference_layer",
    )
    mgr.set_feature_columns(features)
    mgr.COLLECTION_DURATION = COLLECTION_SEC
    mgr.MIN_TRAINING_SAMPLES = MIN_SAMPLES
    mgr.SAMPLE_INTERVAL = SAMPLE_INTERVAL
    mgr.TAKEOVER_CONFIRM_SECONDS = TAKEOVER_CONFIRM

    # ---- 3. RF推荐阶段 (模拟正常行驶) ----
    print(f"\n[2] === RF推荐阶段 ===")
    feat = make_features(features, seed=0)
    feat_df = pd.DataFrame([feat], columns=features).apply(pd.to_numeric, errors="coerce")
    base = loader.predict(feat_df)
    rf_result = {
        "driver_temp": base.driver_temp,
        "passenger_temp": base.passenger_temp,
        "wind_speed": float(base.wind_speed),
        "air_mode": float(base.air_mode),
    }
    print(f"    RF基础推荐: driver={rf_result['driver_temp']}°C, "
          f"passenger={rf_result['passenger_temp']}°C, "
          f"wind={int(rf_result['wind_speed'])}, mode={int(rf_result['air_mode'])}")
    # 模拟命令已成功发送且被CAN状态回读确认。
    mgr.note_command_sent(rf_result)
    mgr.update(feat, rf_result, rf_result)

    # 正常RF推荐持续10秒
    print("    正常RF推荐中... (模拟行驶10秒)")
    for i in range(10):
        resp = mgr.update(make_features(features, seed=i), rf_result, rf_result)  # CAN=RF, 无接管
        assert resp["state"] == "rf_only", f"意外状态: {resp['state']}"
        time.sleep(0.5)
    print("    RF推荐阶段完成, 状态正常")

    # ---- 4. 模拟人工接管 ----
    print(f"\n[3] === 人工接管 ===")
    print(f"    用户觉得 {rf_result['driver_temp']}°C 太冷, 手动调到 {HUMAN_PREFERS['driver_temp']}°C")
    print(f"    等待 {TAKEOVER_CONFIRM} 秒确认接管...")

    takeover_ok = False
    for i in range(20):
        resp = mgr.update(make_features(features, seed=100+i), rf_result, HUMAN_PREFERS)
        if resp["state"] == "collecting":
            takeover_ok = True
            print(f"    接管已确认! 进入数据收集")
            break
        time.sleep(0.5)

    if not takeover_ok:
        print("    FAIL: 接管未确认")
        return 1

    # ---- 5. 数据收集 ----
    print(f"\n[4] === 数据收集 ({COLLECTION_SEC}秒) ===")
    start = time.time()
    last_print = 0
    trained = False
    report = None

    while time.time() - start < COLLECTION_SEC + 10:
        now = time.time()
        seed = int(now * 100) % 10000

        # 定期轮换偏好场景
        scenario_idx = int((now - start) / 30) % 3
        if scenario_idx == 0:
            can = HUMAN_PREFERS
        elif scenario_idx == 1:
            can = {"driver_temp": 26.5, "passenger_temp": 25.5, "wind_speed": 5.0, "air_mode": 1.0}
        else:
            can = {"driver_temp": 26.0, "passenger_temp": 27.0, "wind_speed": 6.0, "air_mode": 2.0}

        resp = mgr.update(make_features(features, seed=seed), rf_result, can)

        if resp.get("trigger_training") and not trained:
            trained = True
            print(f"\n    收集完成, 触发训练...")

            samples_df = mgr.get_training_dataframe()
            print(f"    样本: {len(samples_df)}条, {len(samples_df.columns)}列")

            trainer = Weather26OnlineTrainer(
                model_loader=loader,
                output_dir="models/preference_layer",
                min_samples=MIN_SAMPLES,
                validation_ratio=0.2,
            )
            report = trainer.train(samples_df)
            mgr.on_training_done(report)

            if report.get("success"):
                print(f"    训练成功! base_score={report['base_score']:.2f} → "
                      f"updated_score={report['updated_score']:.2f}")
            else:
                print(f"    训练未通过: {report.get('reason','?')}")
            break

        if now - last_print >= 15:
            status = mgr.get_status()
            elapsed = now - start
            print(f"    收集进度: {elapsed:.0f}/{COLLECTION_SEC}s, "
                  f"样本={status.sample_count}, 状态={status.state_name}")
            last_print = now

        time.sleep(0.5)

    # ---- 6. 验证联合推理 ----
    print(f"\n[5] === 联合推理验证 ===")
    status = mgr.get_status()
    print(f"    当前状态: {status.state_name}")

    test_feat = make_features(features, seed=9999)
    joint = mgr.apply_mlp_residual(test_feat, rf_result)

    print(f"    RF基础推荐:     driver={rf_result['driver_temp']}°C, "
          f"passenger={rf_result['passenger_temp']}°C, "
          f"wind={int(rf_result['wind_speed'])}, mode={int(rf_result['air_mode'])}")

    if joint:
        d_delta = joint.get("driver_delta", 0)
        p_delta = joint.get("passenger_delta", 0)
        print(f"    MLP残差修正:     Δ driver={d_delta:+.1f}°C, Δ passenger={p_delta:+.1f}°C")
        print(f"    RF+MLP最终推荐:  driver={joint['driver_temp']}°C, "
              f"passenger={joint['passenger_temp']}°C, "
              f"wind={int(joint['wind_speed'])}, mode={int(joint['air_mode'])}")
    else:
        print(f"    MLP未加载或推理失败")
        print(f"    文件: models/preference_layer/mlp_model.joblib")

    print(f"\n    人工偏好目标:    driver={HUMAN_PREFERS['driver_temp']}°C, "
          f"passenger={HUMAN_PREFERS['passenger_temp']}°C, "
          f"wind={int(HUMAN_PREFERS['wind_speed'])}, mode={int(HUMAN_PREFERS['air_mode'])}")

    print(f"\n{'='*60}")
    if report and report.get("success"):
        print(f"  ✓ 偏好层流程完成!")
        print(f"  RF({rf_result['driver_temp']}°C) → 接管 → 训练 → "
              f"RF+MLP({joint['driver_temp']}°C) ≈ 人工偏好({HUMAN_PREFERS['driver_temp']}°C)")
    else:
        print(f"  流程完成 (训练{'成功' if report and report.get('success') else '未通过'})")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
