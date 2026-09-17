#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""车端在线训练管道：用26维RF基础推荐 + 人工偏好数据训练30维残差MLP。

训练流程:
1. 接收人工偏好样本 (26维特征 + 4个人工动作)
2. 使用26维固定RF重新生成4维基础推荐
3. 构造残差标签 (人工 - RF基础)
4. 按时间顺序80/20切分
5. 多组MLP候选配置搜索
6. 仅当验证分数优于纯RF时接受更新
7. 原子文件替换
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.preference_learning.online_ordinal_mlp import (
    MODE_CLASSES,
    OrdinalMLPConfig,
    OnlineOrdinalMultiHeadClassifier,
    TEMPERATURE_DELTAS,
    WIND_DELTAS,
)

BASE_COLUMNS = [
    "base_driver_temperature",
    "base_passenger_temperature",
    "base_wind",
    "base_mode",
]
LABEL_COLUMNS = [
    "driver_temperature_class",
    "passenger_temperature_class",
    "wind_class",
    "mode_class",
]
ACTION_COLUMNS = [
    "主驾温度显示",
    "副驾温度显示",
    "UI界面_风速",
    "空调出风模式请求",
]


class Weather26OnlineTrainer:
    """26维RF + 30维MLP 车端在线训练器。"""

    MLP_VERSION = "v17_online_supervised_residual_wind_v3_v1"

    def __init__(
        self,
        model_loader,
        output_dir: Path | str,
        min_samples: int = 50,
        validation_ratio: float = 0.2,
    ):
        self._model_loader = model_loader
        self._output_dir = Path(output_dir)
        self._min_samples = min_samples
        self._validation_ratio = validation_ratio

    # ---------------------------------------------------------------
    # 公开入口
    # ---------------------------------------------------------------
    def train(
        self,
        samples: pd.DataFrame,
        user_id: str = "",
    ) -> dict:
        """训练并返回报告。samples 必须含26维特征 + 4个人工动作列。"""
        t0 = time.time()
        feature_columns = self._model_loader.get_feature_columns()
        missing = [c for c in (*feature_columns, *ACTION_COLUMNS) if c not in samples.columns]
        if missing:
            raise ValueError(f"训练样本缺少字段: {missing}")
        if len(samples) < self._min_samples:
            return {
                "success": False,
                "reason": f"样本不足: {len(samples)} < {self._min_samples}",
                "rows": len(samples),
            }

        frame = self._add_base_predictions(samples, feature_columns)
        frame, overflow = self._add_labels(frame)
        feature_order = [*feature_columns, *BASE_COLUMNS]

        split = int(round(len(frame) * (1.0 - self._validation_ratio)))
        split = min(max(split, 1), len(frame) - 1)
        train_df = frame.iloc[:split]
        val_df = frame.iloc[split:]

        candidates = self._candidate_configs()
        best_model, best_metrics, best_score, best_idx = self._select_best(
            train_df, val_df, feature_order, candidates
        )
        if best_model is None:
            return {"success": False, "reason": "没有任何候选配置训练成功", "rows": len(frame)}

        base_score = self._score(best_metrics, "base")
        updated_score = self._score(best_metrics, "final")
        accepted = updated_score < base_score
        preference_profile = self._build_preference_profile(samples)

        report = {
            "version": "v17_wind_v3_online_update_report_v1",
            "base_rf_package": "weather26",
            "rows": int(len(frame)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "selected_candidate": best_idx,
            "base_score": base_score,
            "updated_score": updated_score,
            "accepted": accepted,
            "overflow_rates": overflow,
            "validation_metrics": best_metrics,
            "training_time_s": round(time.time() - t0, 2),
            "user_id": user_id,
            "preference_profile": preference_profile,
        }

        report_path = self._output_dir / "training_report.json"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        if not accepted:
            return {**report, "success": False, "reason": "验证分数未优于纯RF基座"}

        # 保存模型
        package = {
            "version": self.MLP_VERSION,
            "base_random_forest_package": "weather26",
            "profile": "vehicle_online_updated_user",
            "description": "基于weather26 RF的在线偏好残差MLP (30维输入)",
            "feature_order": feature_order,
            "decision_interval_seconds": 5,
            "temperature_delta_values": TEMPERATURE_DELTAS.tolist(),
            "wind_delta_values": WIND_DELTAS.tolist(),
            "mode_class_meaning": {
                0: "保持随机森林模式",
                **{v: f"切换到模式{v}" for v in range(1, 8)},
            },
            "preference_profile": preference_profile,
            "model": best_model,
        }
        tmp_path = self._output_dir / "mlp_model.joblib.tmp"
        final_path = self._output_dir / "mlp_model.joblib"
        joblib.dump(package, tmp_path)
        os.replace(tmp_path, final_path)

        profile_path = self._output_dir / "preference_profile.json"
        profile_tmp_path = self._output_dir / "preference_profile.json.tmp"
        profile_tmp_path.write_text(
            json.dumps(preference_profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(profile_tmp_path, profile_path)

        print(json.dumps(report, ensure_ascii=False, indent=2))
        return {**report, "success": True, "model_path": str(final_path)}

    @staticmethod
    def _build_preference_profile(samples: pd.DataFrame) -> dict:
        """提取高置信度的离散偏好，作为小样本MLP外推时的安全约束。"""
        profile = {"sample_count": int(len(samples))}

        for column, prefix in (
            ("主驾温度显示", "driver_temperature"),
            ("副驾温度显示", "passenger_temperature"),
        ):
            if column not in samples:
                continue
            values = pd.to_numeric(samples[column], errors="coerce").dropna()
            values = (values * 2.0).round().div(2.0).clip(16.0, 31.0)
            if values.empty:
                continue
            preferred = float(values.mode().iloc[-1])
            confidence = float((values == preferred).mean())
            profile.update({
                f"preferred_{prefix}": preferred,
                f"{prefix}_confidence": confidence,
                f"{prefix}_locked": confidence >= 0.8,
            })

        if "UI界面_风速" in samples:
            wind = pd.to_numeric(samples["UI界面_风速"], errors="coerce").dropna()
            wind = wind.round().clip(1, 9).astype(int)
            if not wind.empty:
                preferred_wind = int(wind.mode().iloc[-1])
                confidence = float((wind == preferred_wind).mean())
                profile.update({
                    "preferred_wind": preferred_wind,
                    "wind_confidence": confidence,
                    "wind_locked": confidence >= 0.8,
                })
        return profile

    # ---------------------------------------------------------------
    # 内部方法
    # ---------------------------------------------------------------
    def _add_base_predictions(self, frame: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """用主模型加载器重新生成26维RF基础推荐。"""
        features_df = frame[feature_columns].copy()
        features_df = features_df.apply(pd.to_numeric, errors="coerce")
        out = frame.copy()
        # BaseModelLoader 的公开接口只返回输入DataFrame第一行的结果。在线训练
        # 必须逐行调用，否则第一条RF推荐会被广播到整个五分钟窗口，导致残差
        # 标签与随后实时推理的RF基座不一致。
        base_driver = []
        base_passenger = []
        base_wind = []
        base_mode = []
        for row_index in range(len(features_df)):
            result = self._model_loader.predict(features_df.iloc[[row_index]])
            base_driver.append(float(result.driver_temp))
            base_passenger.append(float(result.passenger_temp))
            base_wind.append(int(result.wind_speed))
            base_mode.append(int(result.air_mode))
        out["base_driver_temperature"] = base_driver
        out["base_passenger_temperature"] = base_passenger
        out["base_wind"] = base_wind
        out["base_mode"] = base_mode
        return out

    def _add_labels(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """构造4个残差类别标签。"""
        out = frame.copy()
        overflow: dict[str, float] = {}
        for prefix, act, base in (
            ("driver", "主驾温度显示", "base_driver_temperature"),
            ("passenger", "副驾温度显示", "base_passenger_temperature"),
        ):
            raw = out[act].to_numpy(float) - out[base].to_numpy(float)
            overflow[prefix] = float(
                ((raw < TEMPERATURE_DELTAS.min()) | (raw > TEMPERATURE_DELTAS.max())).mean()
            )
            clipped = np.clip(raw, TEMPERATURE_DELTAS.min(), TEMPERATURE_DELTAS.max())
            quantized = np.round(clipped * 2.0) / 2.0
            out[f"{prefix}_temperature_delta"] = quantized
            out[f"{prefix}_temperature_class"] = np.rint(
                (quantized - TEMPERATURE_DELTAS.min()) * 2.0
            ).astype(int)

        raw_wind = out["UI界面_风速"].to_numpy(int) - out["base_wind"].to_numpy(int)
        overflow["wind"] = float(
            ((raw_wind < WIND_DELTAS.min()) | (raw_wind > WIND_DELTAS.max())).mean()
        )
        wind_delta = np.clip(raw_wind, WIND_DELTAS.min(), WIND_DELTAS.max()).astype(int)
        out["wind_delta"] = wind_delta
        out["wind_class"] = (wind_delta - int(WIND_DELTAS.min())).astype(int)
        out["mode_class"] = np.where(
            out["空调出风模式请求"].astype(int) == out["base_mode"].astype(int),
            0,
            out["空调出风模式请求"].astype(int),
        )
        return out, overflow

    @staticmethod
    def _candidate_configs() -> list[OrdinalMLPConfig]:
        return [
            OrdinalMLPConfig(
                hidden=(32, 16), ordinal_strength=0.5, direction_strength=0.5,
                alpha=0.001, epochs=400, batch_size=128, patience=40,
            ),
            OrdinalMLPConfig(
                hidden=(64, 32), ordinal_strength=1.0, direction_strength=0.5,
                alpha=0.001, epochs=400, batch_size=128, patience=40,
            ),
            OrdinalMLPConfig(
                hidden=(64, 32), ordinal_strength=1.0, direction_strength=1.0,
                alpha=0.001, epochs=400, batch_size=128, patience=40,
            ),
            OrdinalMLPConfig(
                hidden=(32,), ordinal_strength=1.0, direction_strength=1.0,
                alpha=0.003, epochs=400, batch_size=128, patience=40,
            ),
        ]

    def _select_best(self, train_df, val_df, feature_order, candidates):
        best_model, best_metrics, best_score, best_idx = None, None, float("inf"), -1
        for idx, cfg in enumerate(candidates):
            try:
                model = OnlineOrdinalMultiHeadClassifier(cfg).fit(
                    train_df[feature_order].to_numpy(float),
                    train_df[LABEL_COLUMNS].to_numpy(int),
                    val_df[feature_order].to_numpy(float),
                    val_df[LABEL_COLUMNS].to_numpy(int),
                )
                preds = self._decode(val_df, model, feature_order)
                metrics = self._metrics(val_df, preds)
                score = self._score(metrics, "final")
                if score < best_score:
                    best_model, best_metrics, best_score, best_idx = model, metrics, score, idx
                print(f"  候选{idx}: hidden={cfg.hidden} ord={cfg.ordinal_strength} "
                      f"dir={cfg.direction_strength} score={score:.4f}")
            except Exception as e:
                print(f"  候选{idx}: 训练失败 - {e}")
        return best_model, best_metrics, best_score, best_idx

    @staticmethod
    def _decode(frame, model, feature_order):
        probs = model.predict_proba(frame[feature_order].to_numpy(float))
        classes = []
        for hi, deltas, base_col, lo, hi_val in (
            (0, TEMPERATURE_DELTAS, "base_driver_temperature", 16.0, 31.0),
            (1, TEMPERATURE_DELTAS, "base_passenger_temperature", 16.0, 31.0),
            (2, WIND_DELTAS, "base_wind", 1.0, 9.0),
        ):
            p = probs[hi].copy()
            b = frame[base_col].to_numpy(float)
            legal = (b[:, None] + deltas[None, :] >= lo) & (b[:, None] + deltas[None, :] <= hi_val)
            raw_classes = p.argmax(axis=1)
            p[~legal] = 0.0
            total = p.sum(axis=1, keepdims=True)
            p /= np.where(total > 0, total, 1.0)
            expected = np.sum(p * deltas[None, :], axis=1)
            dist = np.abs(expected[:, None] - deltas[None, :])
            dist[~legal] = np.inf
            decoded = dist.argmin(axis=1)

            # 若模型最可能的类别越过16/31℃边界，应投影到最近合法残差。
            # 不能直接删掉该类别后对剩余小概率重新归一化，否则低温偏好会
            # 从-4℃反向跳成+1℃。
            for row_index, raw_class in enumerate(raw_classes):
                if legal[row_index, raw_class]:
                    continue
                legal_classes = np.flatnonzero(legal[row_index])
                decoded[row_index] = legal_classes[
                    np.abs(deltas[legal_classes] - deltas[raw_class]).argmin()
                ]
            classes.append(decoded)
        classes.append(probs[3].argmax(axis=1))

        pred = pd.DataFrame({
            "pred_driver_temp_class": classes[0],
            "pred_passenger_temp_class": classes[1],
            "pred_wind_class": classes[2],
            "pred_mode_class": classes[3],
        })
        pred["pred_driver_delta"] = TEMPERATURE_DELTAS[classes[0]]
        pred["pred_passenger_delta"] = TEMPERATURE_DELTAS[classes[1]]
        pred["pred_wind_delta"] = WIND_DELTAS[classes[2]].astype(int)
        pred["final_driver_temperature"] = np.clip(
            frame["base_driver_temperature"].to_numpy(float) + pred["pred_driver_delta"], 16.0, 31.0
        )
        pred["final_passenger_temperature"] = np.clip(
            frame["base_passenger_temperature"].to_numpy(float) + pred["pred_passenger_delta"], 16.0, 31.0
        )
        pred["final_wind"] = np.clip(
            frame["base_wind"].to_numpy(int) + pred["pred_wind_delta"], 1, 9
        ).astype(int)
        pred["final_mode"] = np.where(
            pred["pred_mode_class"] == 0,
            frame["base_mode"].to_numpy(int),
            pred["pred_mode_class"],
        ).astype(int)
        return pred

    @staticmethod
    def _metrics(frame, pred):
        m = {}
        for name, act, base, final in (
            ("driver_temperature", "主驾温度显示", "base_driver_temperature", "final_driver_temperature"),
            ("passenger_temperature", "副驾温度显示", "base_passenger_temperature", "final_passenger_temperature"),
            ("wind", "UI界面_风速", "base_wind", "final_wind"),
        ):
            truth = frame[act].to_numpy(float)
            b = frame[base].to_numpy(float)
            f = pred[final].to_numpy(float)
            m[name] = {
                "base_mae": float(np.abs(truth - b).mean()),
                "final_mae": float(np.abs(truth - f).mean()),
                "original_mean": float(truth.mean()),
                "base_mean": float(b.mean()),
                "final_mean": float(f.mean()),
            }
        mt = frame["空调出风模式请求"].to_numpy(int)
        mb = frame["base_mode"].to_numpy(int)
        mf = pred["final_mode"].to_numpy(int)
        m["mode"] = {
            "base_accuracy": float((mt == mb).mean()),
            "final_accuracy": float((mt == mf).mean()),
        }
        return m

    @staticmethod
    def _score(metrics, prefix):
        temp = 0.5 * (metrics["driver_temperature"][f"{prefix}_mae"] +
                       metrics["passenger_temperature"][f"{prefix}_mae"])
        mode_err = 1.0 - metrics["mode"][f"{prefix}_accuracy"]
        return temp + 0.4 * metrics["wind"][f"{prefix}_mae"] + 0.5 * mode_err
