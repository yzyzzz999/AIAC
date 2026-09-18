"""
冒烟测试: 车载 PMV Runtime 包
验证目标：
  1. 读取 sample_input.json，运行 CHTD + PMV 一步
  2. 输出结构符合 config/schemas/runtime_output_schema.json
  3. forbidden_input_detected = False
  4. status in {ok, degraded}（允许 degraded 但不允许 blocked/error）
  5. use_next_state = True 且 x_next 字段存在（或 None — 表示模型未实现状态传递）
"""
import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_INPUT = Path(__file__).parent / "sample_input.json"
EXPECTED_OUTPUT_KEYS = Path(__file__).parent / "expected_output_keys.json"
RESULT_OUTPUT = Path(tempfile.gettempdir()) / "vehicle_pmv_smoke_test_output.json"

EXPECTED = json.loads(EXPECTED_OUTPUT_KEYS.read_text(encoding="utf-8"))
REQUIRED_TOP_LEVEL = EXPECTED["required_top_level"]
REQUIRED_MODEL_STATE = EXPECTED["required_model_state"]
REQUIRED_PMV = EXPECTED["required_pmv"]
REQUIRED_DIAG = EXPECTED["required_diagnostics"]
FORBIDDEN_INPUT_KEYS = {
    "TA_FdHeadTempLe", "TA_FpHeadTempLe", "TA_FdFloorTemp1",
    "TA_FdFloorTemp2", "TA_CarbinFrntTempLe",
    "measured_head_temperature", "measured_feet_temperature",
}

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check(condition, name, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def run_smoke_test():
    print("=" * 60)
    print("车载 PMV Runtime 冒烟测试")
    print("=" * 60)
    results = []

    # ── T1: 加载 sample_input ────────────────────────────────────
    print("\nT1: 加载 sample_input.json")
    raw = json.loads(SAMPLE_INPUT.read_text(encoding="utf-8"))
    results.append(check(isinstance(raw, dict), "sample_input.json 可读且为字典"))

    # ── T2: 无禁止输入键 ─────────────────────────────────────────
    print("\nT2: 检查禁止输入")
    found_forbidden = [k for k in raw if k in FORBIDDEN_INPUT_KEYS]
    results.append(check(
        len(found_forbidden) == 0,
        "sample_input 不含禁止键",
        f"发现: {found_forbidden}" if found_forbidden else "OK"
    ))

    # ── T3: 调用 run_vehicle_pmv.run() ──────────────────────────
    print("\nT3: 调用 run_vehicle_pmv.run()")
    try:
        from run_vehicle_pmv import run as pmv_run
        out = pmv_run(dict(raw))
        results.append(check(True, "run() 无异常退出"))
    except Exception as e:
        results.append(check(False, "run() 无异常退出", str(e)))
        out = {}

    # ── T4: 输出结构检查 ─────────────────────────────────────────
    print("\nT4: 输出结构")
    for key in REQUIRED_TOP_LEVEL:
        results.append(check(key in out, f"顶层字段 '{key}' 存在"))

    ms = out.get("model_state", {}) or {}
    for key in REQUIRED_MODEL_STATE:
        results.append(check(key in ms, f"model_state.{key} 存在"))

    pmv = out.get("pmv", {}) or {}
    for key in REQUIRED_PMV:
        results.append(check(key in pmv, f"pmv.{key} 存在"))

    diag = out.get("diagnostics", {}) or {}
    for key in REQUIRED_DIAG:
        results.append(check(key in diag, f"diagnostics.{key} 存在"))

    # ── T5: forbidden_input_detected = False ─────────────────────
    print("\nT5: forbidden_input_detected")
    results.append(check(
        diag.get("forbidden_input_detected") is False,
        "forbidden_input_detected == False",
        f"actual: {diag.get('forbidden_input_detected')}"
    ))

    # ── T6: status ok/degraded (not error/blocked) ───────────────
    print("\nT6: status")
    status_val = out.get("status", "missing")
    results.append(check(
        status_val in {"ok", "degraded"},
        f"status 为 ok 或 degraded",
        f"actual: {status_val}"
    ))

    # ── T7: provenance 字段 ──────────────────────────────────────
    print("\nT7: provenance")
    prov = out.get("provenance", {}) or {}
    results.append(check(
        prov.get("calibration_status")
        == "spring_summer_v8_l1_candidate_not_vehicle_calibration",
        "calibration_status 正确"
    ))
    results.append(check(
        prov.get("chtd_param_mode") == "default",
        "chtd_param_mode = default"
    ))

    # ── T8: use_next_state 语义 ──────────────────────────────────
    print("\nT8: use_next_state 语义")
    has_x_next = "x_next" in out
    results.append(check(
        raw.get("use_next_state") is True,
        "sample_input 设置了 use_next_state=True",
    ))
    if has_x_next:
        results.append(check(True, "x_next 字段存在于输出"))
    else:
        print(f"  [{WARN}] x_next 字段不存在 — 模型可能未实现状态传递（允许 WARN）")

    # ── 结果汇总 ─────────────────────────────────────────────────
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    RESULT_OUTPUT.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )
    print("\n" + "=" * 60)
    print(f"结果: {n_pass}/{len(results)} PASS, {n_fail} FAIL")
    verdict = PASS if n_fail == 0 else FAIL
    print(f"SMOKE_TEST_VERDICT: {verdict}")
    print("输出已写入:", RESULT_OUTPUT)
    print("=" * 60)
    return n_fail == 0


if __name__ == "__main__":
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)


def test_runtime_smoke():
    assert run_smoke_test()
