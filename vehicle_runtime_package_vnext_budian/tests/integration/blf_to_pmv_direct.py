#!/usr/bin/env python3
"""直接读 BLF → DBC 解析 → PMV 运行（无需 vcan/socket，纯进程内验证）。"""
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import can
import cantools

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_vehicle_pmv import run as run_pmv

# DBC 信号 → PMV 输入字段（直接用 DBC 信号名，不走 RTE）
DBC_TO_PMV = {
    "VIU_AmbT":              "amb_t_c",
    "RSM_RelHum":            "rh_percent",
    "IPB_VehicleSpeed":      "vehicle_speed_kph",
    "RSM_LeSolarInten":      "solar_driver_w_m2",
    "RSM_RiSolarInten":      "solar_passenger_w_m2",
    "AC_FEvapCurrentTemp":   "eva_t_c",
    "AC_FrntInCarT":         "ict_c",
    "AC_DrvrFaceVentActT":   "driver_face_tma",
    "AC_PassFaceVentActT":   "passenger_face_tma",
    "AC_Forward_BlwPwmOut":  "eva_t_c",  # 备用
}

# 关注的 CAN ID（与生产者订阅一致）
TARGET_IDS = {0x18F, 0x338, 0x33E, 0x33F, 0x35B, 0x3B9}


def build_pmv_input(signal_cache: dict) -> dict:
    pmv = {"chtd_param_mode": "safe_preview", "use_next_state": True}
    tma = {}
    for dbc_name, value in signal_cache.items():
        pmv_field = DBC_TO_PMV.get(dbc_name)
        if pmv_field is None:
            continue
        if pmv_field == "driver_face_tma":
            tma["FrntFdvFlow"] = value
        elif pmv_field == "passenger_face_tma":
            tma["FrntFpvFlow"] = value
        else:
            pmv[pmv_field] = value
    if tma:
        pmv["tma"] = tma
    return pmv


def main():
    blf_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not blf_path:
        runs_dir = Path("/home/data/data_collection/outputs/records_no_hw/runs/run_0001")
        blf_files = list(runs_dir.glob("*.blf"))
        blf_path = str(blf_files[0]) if blf_files else None
    if not blf_path:
        print("ERROR: no BLF file found")
        sys.exit(1)

    dbc_path = "/home/data/data_collection/CAN_utf8.dbc"
    db = cantools.database.load_file(dbc_path)

    print(f"BLF: {blf_path}")
    print(f"DBC: {dbc_path}")

    reader = can.BLFReader(blf_path)
    messages = [m for m in reader if m.arbitration_id in TARGET_IDS]
    reader.stop()
    print(f"Messages: {len(messages)} (filtered from {len(TARGET_IDS)} CAN IDs)")

    # 按 1 秒窗口聚合 + 运行 PMV
    signal_cache = {}
    x_next = None
    t0 = messages[0].timestamp
    window_start = t0
    results = []
    decode_errors = 0

    for msg in messages:
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data, decode_choices=False)
        except Exception:
            decode_errors += 1
            continue

        # 提取数值型信号
        for sig_name, sig_value in decoded.items():
            if isinstance(sig_value, (int, float)):
                signal_cache[sig_name] = float(sig_value)

        # 每秒运行一次 PMV
        if msg.timestamp - window_start >= 1.0:
            if "VIU_AmbT" in signal_cache:
                pmv_input = build_pmv_input(signal_cache)
                if x_next is not None:
                    pmv_input["x_init"] = x_next
                try:
                    output = run_pmv(pmv_input)
                    x_next = output.get("x_next")
                    results.append({
                        "t": round(msg.timestamp - t0, 1),
                        "amb": signal_cache.get("VIU_AmbT"),
                        "speed": signal_cache.get("IPB_VehicleSpeed"),
                        "pmv_drv": output["pmv"]["pmv_driver"],
                        "pmv_pass": output["pmv"]["pmv_passenger"],
                        "drv_head": output["model_state"]["driver_head_temp_c"],
                        "pass_head": output["model_state"]["passenger_head_temp_c"],
                        "status": output["status"],
                    })
                except Exception as e:
                    results.append({"t": round(msg.timestamp - t0, 1), "error": str(e)})

            window_start = msg.timestamp

    print(f"Decode errors: {decode_errors}")
    print(f"PMV runs: {len(results)}")

    if results:
        # 输出摘要
        ok = [r for r in results if r.get("status") == "ok"]
        err = [r for r in results if r.get("status") == "error"]
        print(f"  OK: {len(ok)}, Error: {len(err)}")

        # 前 10 条结果
        print("\nFirst 10 results:")
        print(f"{'Time':>6s} {'Amb':>6s} {'Speed':>6s} {'PMV_Drv':>8s} {'PMV_Pass':>8s} {'DrvHd':>7s} {'PassHd':>7s} {'Status':>6s}")
        for r in results[:10]:
            err = r.get("error", "")
            if err:
                print(f"{r['t']:>6.1f} ERROR: {err[:60]}")
            else:
                print(f"{r['t']:>6.1f} {r['amb']:>6.1f} {r['speed']:>6.0f} "
                      f"{r['pmv_drv']:>8.3f} {r['pmv_pass']:>8.3f} "
                      f"{r['drv_head']:>7.2f} {r['pass_head']:>7.2f} {r['status']:>6s}")

        # 保存完整结果
        out_file = "/tmp/pmv_blf_test_results.json"
        Path(out_file).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    main()
