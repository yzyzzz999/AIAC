"""Helper: generate all config JSON files for the runtime package."""
import json
from pathlib import Path

CFG = Path(r'F:\TempWork\SeriesPMV\001_APP\delivery\vehicle_runtime_package_vnext\config')
PKG = Path(r'F:\TempWork\SeriesPMV\001_APP\delivery\vehicle_runtime_package_vnext')
TESTS = PKG / 'tests'

# ── vehicle_runtime_config.json ─────────────────────────────────────────────
vehicle_cfg = {
    "schema": "vehicle_runtime_config_v1",
    "package": "vehicle_runtime_package_vnext",
    "calibration_status": "midterm_preview_not_vehicle_calibration",
    "airflow_mode": "actuator_first_m8_anchor_preview",
    "chtd_param_mode": "safe_preview",
    "feature_flags": {
        "enable_ir_fusion": False,
        "enable_image_occupancy": True,
        "enable_runtime_conditioning": False,
        "enable_post_glass_defrost": False,
        "enable_corrected_chtd": False,
        "enable_vehicle_adapted_params": False,
        "use_next_state": True
    },
    "forbidden_inputs": [
        "TA_FdHeadTempLe", "TA_FpHeadTempLe",
        "TA_FdFloorTemp1", "TA_FdFloorTemp2",
        "TA_FpFloorTemp1", "TA_FpFloorTemp2",
        "TA_CarbinFrntTempLe", "TA_CarbinFrntTempRi",
        "measured_head_temperature", "measured_feet_temperature",
        "measured_cabin_temperature"
    ],
    "can_signal_exceptions": True,
    "can_exception_list": "can_signal_exception_list.json"
}
(CFG / 'vehicle_runtime_config.json').write_text(
    json.dumps(vehicle_cfg, indent=2, ensure_ascii=False), encoding='utf-8')

# ── calibration_params_preview.json ─────────────────────────────────────────
calib = {
    "schema": "calibration_params_preview_v1",
    "calibration_status": "midterm_preview_not_vehicle_calibration",
    "chtd_param_mode": "safe_preview",
    "param_source": "hvac_sim.chtd.safe_preview_params",
    "param_module": "hvac_sim/chtd/safe_preview_params.py",
    "provenance": {
        "based_on": "physical_baseline_v2 + safe_preview adjustments",
        "from_invalid_pso": False,
        "from_stage2_v2_invalid": False,
        "from_cache_v5_pso": False,
        "cache_v5_validated": False,
        "production_ready": False
    },
    "known_risk": "calibrated_before_airflow_v5_fix",
    "notes": "model_runs_but_accuracy_not_final. Use for functional integration only."
}
(CFG / 'calibration_params_preview.json').write_text(
    json.dumps(calib, indent=2, ensure_ascii=False), encoding='utf-8')

# ── signal_mapping_sig_fte.json ──────────────────────────────────────────────
sig_map = {
    "schema": "signal_mapping_sig_fte_v1",
    "note": "Maps SIG/FTE engineering quantities to RuntimeSignalInput fields",
    "mappings": [
        {"runtime_field": "amb_t_c",            "sig_fte_name": "SIG.AmbientTemperature_C",      "can_fallback": "VIU_AmbT",               "unit": "degC",  "required": True},
        {"runtime_field": "rh_percent",          "sig_fte_name": "SIG.RelativeHumidity_pct",      "can_fallback": "RH_pct",                 "unit": "%",     "required": False, "default": 50.0},
        {"runtime_field": "vehicle_speed_kph",   "sig_fte_name": "SIG.VehicleSpeed_kph",          "can_fallback": "IPB_VehicleSpeed",        "unit": "km/h",  "required": False, "default": 0.0},
        {"runtime_field": "solar_driver_w_m2",   "sig_fte_name": "SIG.SolarIrradiance_Driver",    "can_fallback": "RSM_LeSolarInten",        "unit": "W/m2",  "required": False, "default": 0.0},
        {"runtime_field": "solar_passenger_w_m2","sig_fte_name": "SIG.SolarIrradiance_Passenger", "can_fallback": "RSM_RiSolarInten",        "unit": "W/m2",  "required": False, "default": 0.0},
        {"runtime_field": "eva_t_c",             "sig_fte_name": "FTE.EvapOutletTemp_C",          "can_fallback": None,                      "unit": "degC",  "required": False},
        {"runtime_field": "driver_face_flow",    "sig_fte_name": "FTE.FrntDrvFaceFlow_m3h",       "can_fallback": "AC_DrvrFaceVentActT",     "unit": "m3/h",  "required": False},
        {"runtime_field": "passenger_face_flow", "sig_fte_name": "FTE.FrntPsgFaceFlow_m3h",       "can_fallback": "AC_PassFaceVentActT",     "unit": "m3/h",  "required": False},
        {"runtime_field": "driver_floor_flow",   "sig_fte_name": "FTE.FrntDrvFloorFlow_m3h",      "can_fallback": "AC_DrvrFootVentActT",     "unit": "m3/h",  "required": False},
        {"runtime_field": "passenger_floor_flow","sig_fte_name": "FTE.FrntPsgFloorFlow_m3h",      "can_fallback": "AC_PassFootVentActT",     "unit": "m3/h",  "required": False},
    ]
}
(CFG / 'signal_mapping_sig_fte.json').write_text(
    json.dumps(sig_map, indent=2, ensure_ascii=False), encoding='utf-8')

# ── can_signal_exception_list.json ───────────────────────────────────────────
can_exc = {
    "schema": "can_signal_exception_list_v1",
    "generated": "2026-06-15",
    "note": "CAN signals still used directly; SIG/FTE equivalents not yet available. Replace before production.",
    "exceptions": [
        {"can_signal": "VIU_AmbT",               "runtime_field": "amb_t_c",              "unit": "degC",  "sig_fte_replacement": "SIG.AmbientTemperature_C",        "deployment_impact": "Required"},
        {"can_signal": "RH_pct",                  "runtime_field": "rh_percent",           "unit": "%",     "sig_fte_replacement": "SIG.RelativeHumidity_pct",        "deployment_impact": "Defaults to 50% if missing"},
        {"can_signal": "IPB_VehicleSpeed",         "runtime_field": "vehicle_speed_kph",    "unit": "km/h",  "sig_fte_replacement": "SIG.VehicleSpeed_kph",           "deployment_impact": "Defaults to 0 if missing"},
        {"can_signal": "RSM_LeSolarInten",         "runtime_field": "solar_driver_w_m2",    "unit": "W/m2",  "sig_fte_replacement": "SIG.SolarIrradiance_Driver",     "deployment_impact": "Defaults to 0 if missing"},
        {"can_signal": "RSM_RiSolarInten",         "runtime_field": "solar_passenger_w_m2", "unit": "W/m2",  "sig_fte_replacement": "SIG.SolarIrradiance_Passenger",  "deployment_impact": "Defaults to 0 if missing"},
        {"can_signal": "AC_Forward_AirFlowTarget", "runtime_field": "flows.total_m3h",      "unit": "m3/h",  "sig_fte_replacement": "FTE.TotalAirflowTarget_m3h",     "deployment_impact": "Required for airflow"},
        {"can_signal": "AC_DrvrFaceVentActT",      "runtime_field": "driver_face_flow",     "unit": "m3/h",  "sig_fte_replacement": "FTE.FrntDrvFaceFlow_m3h",        "deployment_impact": "Required for head zone"},
        {"can_signal": "AC_PassFaceVentActT",      "runtime_field": "passenger_face_flow",  "unit": "m3/h",  "sig_fte_replacement": "FTE.FrntPsgFaceFlow_m3h",        "deployment_impact": "Required for head zone"},
        {"can_signal": "AC_DrvrFootVentActT",      "runtime_field": "driver_floor_flow",    "unit": "m3/h",  "sig_fte_replacement": "FTE.FrntDrvFloorFlow_m3h",       "deployment_impact": "Required for feet zone"},
        {"can_signal": "AC_PassFootVentActT",      "runtime_field": "passenger_floor_flow", "unit": "m3/h",  "sig_fte_replacement": "FTE.FrntPsgFloorFlow_m3h",       "deployment_impact": "Required for feet zone"},
        {"can_signal": "AC_BLOW_FaceVentilaPosn",  "runtime_field": "tma.face_posn_v",      "unit": "V",     "sig_fte_replacement": "FTE.FaceVentPosition_V",         "deployment_impact": "Mode detection"},
        {"can_signal": "AC_FrantFootVentPosn",     "runtime_field": "tma.foot_posn_v",      "unit": "V",     "sig_fte_replacement": "FTE.FootVentPosition_V",         "deployment_impact": "Mode detection"},
        {"can_signal": "AC_DefrostVentilaPosn",    "runtime_field": "tma.defrost_posn_v",   "unit": "V",     "sig_fte_replacement": "FTE.DefrostVentPosition_V",      "deployment_impact": "Mode detection"},
        {"can_signal": "AC_FrntInCarT",            "runtime_field": "ict_c",                "unit": "degC",  "sig_fte_replacement": "SIG.InCarTemp_C",                "deployment_impact": "Optional — estimated internally"},
    ]
}
(CFG / 'can_signal_exception_list.json').write_text(
    json.dumps(can_exc, indent=2, ensure_ascii=False), encoding='utf-8')

# ── runtime_input_schema.json ────────────────────────────────────────────────
input_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "RuntimeSignalInput",
    "description": "Vehicle PMV runtime input. Only SIG/FTE fields allowed. Thermocouple temperatures forbidden.",
    "type": "object",
    "required": ["amb_t_c"],
    "properties": {
        "amb_t_c":            {"type": "number", "description": "Ambient temperature (degC). SIG source."},
        "rh_percent":         {"type": "number", "default": 50.0, "description": "Relative humidity (%)."},
        "vehicle_speed_kph":  {"type": "number", "default": 0.0,  "description": "Vehicle speed (km/h)."},
        "solar_driver_w_m2":  {"type": "number", "default": 0.0,  "description": "Driver-side solar (W/m2)."},
        "solar_passenger_w_m2":{"type": "number","default": 0.0,  "description": "Passenger-side solar (W/m2)."},
        "eva_t_c":            {"type": ["number","null"],          "description": "Evap outlet temperature (degC). FTE source."},
        "hct_c":              {"type": ["number","null"],          "description": "Heater core outlet temperature (degC). FTE."},
        "ict_c":              {"type": ["number","null"],          "description": "In-car target temperature setpoint (degC). CAN exception: AC_FrntInCarT."},
        "driver_face_flow":   {"type": ["number","null"],          "description": "Driver face outlet flow (m3/h). FTE."},
        "passenger_face_flow":{"type": ["number","null"],          "description": "Passenger face outlet flow (m3/h). FTE."},
        "driver_floor_flow":  {"type": ["number","null"],          "description": "Driver floor outlet flow (m3/h). FTE."},
        "passenger_floor_flow":{"type": ["number","null"],         "description": "Passenger floor outlet flow (m3/h). FTE."},
        "driver_defrost_flow":{"type": ["number","null"],          "description": "Driver defrost flow (m3/h). FTE."},
        "passenger_defrost_flow":{"type": ["number","null"],       "description": "Passenger defrost flow (m3/h). FTE."},
        "tma": {
            "type": "object",
            "description": "Vent air temperature map (degC per outlet key).",
            "properties": {
                "FrntFdvFlow": {"type": "number"},
                "FrntFpvFlow": {"type": "number"},
                "FrntFdfFlow": {"type": "number"},
                "FrntFpfFlow": {"type": "number"},
            }
        },
        "flows": {
            "type": "object",
            "description": "Additional flow overrides (m3/h per outlet key)."
        },
        "image_inputs": {
            "type": ["object","null"],
            "description": "Optional image-module occupant data (IR surface temp, occupancy)."
        },
        "use_next_state":     {"type": "boolean", "default": True},
        "chtd_param_mode":    {"type": "string",  "enum": ["default","safe_preview","phase3_engineering_preview"]},
        "x_init":             {"type": ["array","null"], "description": "Optional 28-element CHTD state vector for warm start."},
    },
    "forbidden_keys": [
        "TA_FdHeadTempLe","TA_FpHeadTempLe","TA_FdFloorTemp1","TA_FdFloorTemp2",
        "TA_FpFloorTemp1","TA_FpFloorTemp2","TA_CarbinFrntTempLe","TA_CarbinFrntTempRi",
        "measured_head_temperature","measured_feet_temperature","measured_cabin_temperature"
    ]
}
(PKG / 'runtime_input_schema.json').write_text(
    json.dumps(input_schema, indent=2, ensure_ascii=False), encoding='utf-8')

# ── runtime_output_schema.json ───────────────────────────────────────────────
output_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "RuntimePMVOutput",
    "type": "object",
    "properties": {
        "timestamp":     {"type": "string"},
        "status":        {"type": "string", "enum": ["ok","degraded","error"]},
        "schema_version":{"type": "string"},
        "model_state": {
            "type": "object",
            "properties": {
                "driver_head_temp_c":                  {"type": "number"},
                "passenger_head_temp_c":               {"type": "number"},
                "driver_feet_temp_c":                  {"type": "number"},
                "passenger_feet_temp_c":               {"type": "number"},
                "second_row_driver_head_temp_c":       {"type": "number"},
                "second_row_passenger_head_temp_c":    {"type": "number"},
                "second_row_driver_feet_temp_c":       {"type": "number"},
                "second_row_passenger_feet_temp_c":    {"type": "number"},
                "front_cabin_temp_c":                  {"type": "number"},
                "passenger_cabin_temp_c":              {"type": "number"},
            }
        },
        "pmv": {
            "type": "object",
            "properties": {
                "pmv_driver":              {"type": "number"},
                "pmv_passenger":           {"type": "number"},
                "pmv_second_row_driver":   {"type": "number"},
                "pmv_second_row_passenger":{"type": "number"},
            }
        },
        "comfort_flags": {
            "type": "object",
            "properties": {
                "driver_comfortable":    {"type": "boolean"},
                "passenger_comfortable": {"type": "boolean"},
            }
        },
        "diagnostics": {
            "type": "object",
            "properties": {
                "airflow_distribution_source": {"type": "string"},
                "airflow_total_source":        {"type": "string"},
                "mode_key":                    {"type": "string"},
                "face_fraction":               {"type": "number"},
                "foot_fraction":               {"type": "number"},
                "defrost_fraction":            {"type": "number"},
                "lr_split_source":             {"type": "string"},
                "rh_source":                   {"type": "string"},
                "forbidden_input_detected":    {"type": "boolean"},
                "fallback_used":               {"type": "boolean"},
                "missing_signal_list":         {"type": "array", "items": {"type": "string"}},
            }
        },
        "provenance": {
            "type": "object",
            "properties": {
                "calibration_status": {"type": "string"},
                "airflow_mode":       {"type": "string"},
                "chtd_param_mode":    {"type": "string"},
                "package_version":    {"type": "string"},
            }
        }
    }
}
(PKG / 'runtime_output_schema.json').write_text(
    json.dumps(output_schema, indent=2, ensure_ascii=False), encoding='utf-8')

# ── runtime_input_example_minimal.json ──────────────────────────────────────
ex_min = {
    "_comment": "Minimal runtime input — only amb_t_c required. All others default.",
    "amb_t_c": 35.0,
    "chtd_param_mode": "safe_preview",
    "use_next_state": True
}
(PKG / 'runtime_input_example_minimal.json').write_text(
    json.dumps(ex_min, indent=2, ensure_ascii=False), encoding='utf-8')

# ── runtime_input_example_full.json ─────────────────────────────────────────
ex_full = {
    "_comment": "Full runtime input example — summer face-mode cooling scenario. No thermocouple inputs.",
    "amb_t_c": 38.0,
    "rh_percent": 45.0,
    "vehicle_speed_kph": 60.0,
    "solar_driver_w_m2": 600.0,
    "solar_passenger_w_m2": 400.0,
    "eva_t_c": 8.0,
    "ict_c": 24.0,
    "driver_face_flow": 45.0,
    "passenger_face_flow": 45.0,
    "driver_floor_flow": 0.0,
    "passenger_floor_flow": 0.0,
    "driver_defrost_flow": 0.0,
    "passenger_defrost_flow": 0.0,
    "tma": {
        "FrntFdvFlow": 14.0,
        "FrntFpvFlow": 14.5,
        "FrntFdfFlow": 30.0,
        "FrntFpfFlow": 30.5
    },
    "image_inputs": {
        "driver":    {"occupied": True,  "clothing_clo": 0.5, "activity_met": 1.1},
        "passenger": {"occupied": False, "clothing_clo": 0.5, "activity_met": 1.0}
    },
    "chtd_param_mode": "safe_preview",
    "use_next_state": True
}
(PKG / 'runtime_input_example_full.json').write_text(
    json.dumps(ex_full, indent=2, ensure_ascii=False), encoding='utf-8')

# ── tests/sample_input.json ──────────────────────────────────────────────────
(TESTS / 'sample_input.json').write_text(
    json.dumps(ex_full, indent=2, ensure_ascii=False), encoding='utf-8')

# ── tests/expected_output_keys.json ─────────────────────────────────────────
expected_keys = {
    "required_top_level": ["status", "model_state", "pmv", "diagnostics", "provenance"],
    "required_model_state": ["driver_head_temp_c", "passenger_head_temp_c",
                             "driver_feet_temp_c", "passenger_feet_temp_c"],
    "required_pmv": ["pmv_driver", "pmv_passenger"],
    "required_diagnostics": ["forbidden_input_detected", "fallback_used"],
    "forbidden_input_must_be_false": True
}
(TESTS / 'expected_output_keys.json').write_text(
    json.dumps(expected_keys, indent=2, ensure_ascii=False), encoding='utf-8')

print("All JSON files written OK")
