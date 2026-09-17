"""Build a unified parameter master table from Excel exports, dataclasses, and defaults."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.afe.fan import FanParams, FanSet
from hvac_sim.afe.params import AFEParams
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.config.param_loader import ParamTableError, load_initial_params
from hvac_sim.config.param_mapping import (
    optional_param_table_paths,
    resolve_python_param_name,
)
from hvac_sim.glass_temp import WindshieldGlassParams
from hvac_sim.pipeline import PipelineInputs

try:
    from hvac_sim.calibration.de_calibration import _AFE_CALIB_FIELDS as AFE_PSO_FIELDS
except ImportError:  # pragma: no cover
    AFE_PSO_FIELDS = []

INVENTORY_FIELD_NAMES: Tuple[str, ...] = (
    "parameter_name",
    "module",
    "category",
    "current_value",
    "value_shape",
    "source_file",
    "source_status",
    "is_lut",
    "is_placeholder_like",
    "used_in_python",
    "used_in_model_location",
    "needs_vehicle_calibration",
    "pso_candidate",
    "priority",
    "chinese_description",
)

_AFE_MISSING_IN_EXCEL = frozenset({"OsaFlapRessCo", "RecFlapRessCo"})
_AFE_PYTHON_ONLY = frozenset({"eps"})

_CHINESE_RULES: Tuple[Tuple[str, str], ...] = (
    ("RoofMassAtb", "车顶质量"),
    ("MassAtb", "质量"),
    ("AreaAtb", "面积"),
    ("SolarRadCo", "太阳辐射吸收/增益系数"),
    ("LeakageCo", "漏气换热系数"),
    ("ConvCo", "对流换热系数"),
    ("RadCo", "辐射换热系数"),
    ("AirVAtb", "空气体积"),
    ("CpAtb", "比热"),
    ("RessCo", "流阻系数"),
    ("AmbT_X", "环境温度断点轴"),
    ("QgainCo", "热增益全局系数"),
    ("QlossCo", "热损失全局系数"),
)

_FAN_PSO_SUFFIXES = ("n0", "p0", "a0")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _params_pkg() -> Path:
    return _repo_root() / "simulink_conversion_package" / "params"


def _geometry_defaults_path() -> Path:
    return Path(__file__).resolve().parent / "geometry_defaults.json"


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return float(value)
    if isinstance(value, (int, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    return value


def _value_shape(value: Any) -> str:
    if isinstance(value, np.ndarray):
        rows, cols = (1, value.size) if value.ndim == 1 else value.shape
        if value.ndim == 1:
            return f"({value.size},)"
        return str(tuple(value.shape))
    if isinstance(value, list):
        return f"({len(value)},)"
    if isinstance(value, dict):
        return "dict"
    return "scalar"


def _is_lut_value(value: Any, name: str = "") -> bool:
    if name.endswith("_M") or name.endswith("_X"):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size > 1
    if isinstance(value, list):
        return len(value) > 1
    return False


def _numeric_elements(value: Any) -> List[float]:
    if isinstance(value, np.ndarray):
        flat = value.astype(float).ravel()
        return [float(x) for x in flat.tolist()]
    if isinstance(value, list):
        out: List[float] = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                return []
        return out
    return []


def _is_placeholder_like(value: Any, name: str = "") -> bool:
    nums = _numeric_elements(value)
    if not nums:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            v = float(value)
            return v == 1.0 or v == 0.0
        return False
    if len(nums) == 1:
        return nums[0] in (0.0, 1.0)
    return all(x == 1.0 for x in nums) or all(x == 0.0 for x in nums)


def _chinese_description(name: str) -> str:
    for token, label in _CHINESE_RULES:
        if token in name:
            return label
    return "待工程师补充中文含义"


def _infer_module(name: str, module_hint: str = "") -> str:
    if module_hint:
        return module_hint
    upper = name.upper()
    if upper.startswith("CHTD_") or name in {f.name for f in fields(CHTDParams)}:
        return "CHTD"
    if name in {f.name for f in fields(AFEParams)} or upper.startswith("AFE_"):
        return "AFE"
    if name.startswith(("F1_", "F2_", "S_")) or name.endswith(_FAN_PSO_SUFFIXES):
        return "AFE"
    if name.startswith(("ir_", "ekf_")):
        return "observer"
    if name.startswith("glass_") or name.startswith("defrost_"):
        return "observer"
    if "geometry" in name or "outlet" in name or "attenuation" in name:
        return "geometry"
    return "workspace"


def _infer_category(name: str) -> str:
    if name.endswith("_M"):
        return "lut_map"
    if name.endswith("_X"):
        return "lut_axis"
    if name.endswith("_P"):
        return "scalar_param"
    if name.endswith("RessCo") or "Flap" in name:
        return "resistance"
    if name.endswith(_FAN_PSO_SUFFIXES) or name.startswith(("F1_", "F2_", "S_")):
        return "fan_curve"
    if name.startswith(("ir_", "ekf_")):
        return "ir_ekf"
    if name.startswith("glass_"):
        return "glass_observer"
    if name.startswith("defrost_"):
        return "defrost_observer"
    if "geometry" in name or "outlet" in name:
        return "air_speed_geometry"
    if name.endswith("UAAtb"):
        return "deprecated_legacy"
    return "general"


def _model_location(module: str, category: str) -> str:
    mapping = {
        "CHTD": "hvac_sim/chtd/thermal.py",
        "AFE": "hvac_sim/afe/",
        "geometry": "hvac_sim/air_speed.py + config/geometry_defaults.json",
        "observer": "hvac_sim/pipeline.py",
        "workspace": "Simulink workspace / param.xlsx",
    }
    if category == "glass_observer":
        return "hvac_sim/glass_temp.py"
    if category == "defrost_observer":
        return "hvac_sim/defrost.py"
    if category == "ir_ekf":
        return "hvac_sim/pipeline.py + hvac_sim/ekf/head_temp_filter.py"
    return mapping.get(module, "unknown")


def _priority_for(
    *,
    name: str,
    source_status: str,
    is_placeholder_like: bool,
    is_lut: bool,
    module: str,
    category: str,
) -> str:
    if name == "CHTD_AirCpAtb_P" or name == "CHTD_AmbT_X" or name == "eps":
        return "P3"
    if source_status == "runtime_observer_default":
        return "P2"
    if source_status == "geometry_placeholder":
        return "P0"
    if name in _AFE_MISSING_IN_EXCEL:
        return "P0"
    if module == "AFE" and category == "fan_curve" and is_placeholder_like:
        return "P0"
    if module == "CHTD" and is_lut and is_placeholder_like:
        return "P0"
    if module == "CHTD" and is_placeholder_like and category == "scalar_param":
        if any(token in name for token in ("MassAtb", "CpAtb", "AreaAtb", "AirVAtb")):
            return "P0"
    if source_status == "missing_in_excel":
        return "P0"
    if is_placeholder_like:
        return "P1"
    if module in ("AFE", "CHTD"):
        return "P1"
    return "P2"


def _needs_calibration(
    *,
    source_status: str,
    is_placeholder_like: bool,
    module: str,
    measured: Optional[bool] = None,
) -> bool:
    if source_status in ("geometry_placeholder", "missing_in_excel", "python_default"):
        if module == "geometry":
            return True
    if measured is False:
        return True
    if is_placeholder_like and module in ("CHTD", "AFE", "geometry"):
        return True
    if source_status == "missing_in_excel":
        return True
    return False


def _is_pso_candidate(name: str, module: str) -> bool:
    if name in AFE_PSO_FIELDS:
        return True
    if module == "AFE" and name.endswith(_FAN_PSO_SUFFIXES):
        return True
    if name.startswith(("F1_", "F2_", "S_")) and name.split("_")[-1] in _FAN_PSO_SUFFIXES:
        return True
    return False


def _blank_row(name: str) -> Dict[str, Any]:
    return {key: None for key in INVENTORY_FIELD_NAMES} | {"parameter_name": name}


def _finalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    name = str(row["parameter_name"])
    value = row.get("_raw_value")
    row["current_value"] = _serialize_value(value)
    row["value_shape"] = _value_shape(value)
    row["is_lut"] = _is_lut_value(value, name)
    row["is_placeholder_like"] = _is_placeholder_like(value, name)
    row["chinese_description"] = _chinese_description(name)
    if row.get("module") is None:
        row["module"] = _infer_module(name)
    if row.get("category") is None:
        row["category"] = _infer_category(name)
    if row.get("used_in_model_location") is None:
        row["used_in_model_location"] = _model_location(row["module"], row["category"])
    if row.get("priority") is None:
        row["priority"] = _priority_for(
            name=name,
            source_status=str(row.get("source_status") or ""),
            is_placeholder_like=bool(row["is_placeholder_like"]),
            is_lut=bool(row["is_lut"]),
            module=str(row["module"]),
            category=str(row["category"]),
        )
    if row.get("needs_vehicle_calibration") is None:
        row["needs_vehicle_calibration"] = _needs_calibration(
            source_status=str(row.get("source_status") or ""),
            is_placeholder_like=bool(row["is_placeholder_like"]),
            module=str(row["module"]),
            measured=row.get("_measured"),
        )
    if row.get("pso_candidate") is None:
        row["pso_candidate"] = _is_pso_candidate(name, str(row["module"]))
    row.pop("_raw_value", None)
    row.pop("_measured", None)
    return {k: row[k] for k in INVENTORY_FIELD_NAMES}


def _canonical_param_name(name: str) -> Tuple[str, bool]:
    """Return (canonical_name, used_in_python)."""
    resolved, _ = resolve_python_param_name(name, target_type=_infer_target_type(name))
    if resolved:
        return resolved, True
    return name, False


def _infer_target_type(name: str) -> str:
    from hvac_sim.config.param_loader import infer_target_type

    return infer_target_type(name)


def _load_workspace_params_fallback() -> Tuple[List[Dict[str, Any]], str]:
    path = _params_pkg() / "workspace_params.json"
    if not path.is_file():
        return [], ""
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for item in doc.get("vars", []):
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("name", "")).strip()
        if not raw_name:
            continue
        name, used = _canonical_param_name(raw_name)
        rows.append(
            {
                "parameter_name": name,
                "module": _infer_module(name),
                "category": _infer_category(name),
                "_raw_value": item.get("value"),
                "source_file": str(path.relative_to(_repo_root())),
                "source_status": "excel_param",
                "used_in_python": used,
            }
        )
    return rows, str(path)


def _load_excel_records() -> Tuple[List[Dict[str, Any]], List[str]]:
    paths = optional_param_table_paths()
    if not paths:
        return [], []
    try:
        doc = load_initial_params(paths[0])
    except (ParamTableError, OSError):
        return [], [str(p) for p in paths]
    rows: List[Dict[str, Any]] = []
    for rec in doc.parameters:
        name = rec.name.strip()
        resolved, _ = resolve_python_param_name(
            name, target_type=rec.target_type or _infer_target_type(name)
        )
        py_name = resolved or name
        used = resolved is not None
        rows.append(
            {
                "parameter_name": py_name if used else name,
                "module": _infer_module(py_name or name),
                "category": _infer_category(py_name or name),
                "_raw_value": rec.value,
                "source_file": str(paths[0].relative_to(_repo_root())),
                "source_status": "excel_param",
                "used_in_python": used,
            }
        )
    return rows, [str(p) for p in paths]


def _dataclass_rows(
    instance: Any,
    *,
    module: str,
    source_file: str,
    source_status: str = "python_default",
    name_prefix: str = "",
) -> List[Dict[str, Any]]:
    if not is_dataclass(instance):
        return []
    rows: List[Dict[str, Any]] = []
    for f in fields(instance):
        value = getattr(instance, f.name)
        if isinstance(value, FanParams):
            for sub in fields(FanParams):
                pname = f"{name_prefix}{f.name}_{sub.name}" if name_prefix else f"{f.name}_{sub.name}"
                rows.append(
                    {
                        "parameter_name": pname,
                        "module": module,
                        "category": "fan_curve",
                        "_raw_value": getattr(value, sub.name),
                        "source_file": source_file,
                        "source_status": source_status,
                        "used_in_python": True,
                    }
                )
            continue
        pname = f"{name_prefix}{f.name}" if name_prefix else f.name
        rows.append(
            {
                "parameter_name": pname,
                "module": module,
                "category": _infer_category(pname),
                "_raw_value": value,
                "source_file": source_file,
                "source_status": source_status,
                "used_in_python": True,
            }
        )
    return rows


def _observer_default_rows() -> List[Dict[str, Any]]:
    src = "hvac_sim/pipeline.py"
    defaults = {
        "ir_process_var": PipelineInputs.ir_process_var,
        "ir_measurement_var": PipelineInputs.ir_measurement_var,
        "ir_surface_to_air_offset_c": PipelineInputs.ir_surface_to_air_offset_c,
        "ir_min_confidence": PipelineInputs.ir_min_confidence,
        "ir_initial_variance": PipelineInputs.ir_initial_variance,
    }
    rows = [
        {
            "parameter_name": name,
            "module": "observer",
            "category": "ir_ekf",
            "_raw_value": val,
            "source_file": src,
            "source_status": "runtime_observer_default",
            "used_in_python": True,
        }
        for name, val in defaults.items()
    ]
    glass = WindshieldGlassParams()
    for f in fields(glass):
        rows.append(
            {
                "parameter_name": f"glass_{f.name}",
                "module": "observer",
                "category": "glass_observer",
                "_raw_value": getattr(glass, f.name),
                "source_file": "hvac_sim/glass_temp.py",
                "source_status": "runtime_observer_default",
                "used_in_python": True,
            }
        )
    defrost_defaults = {
        "defrost_posn_fdh_angle_min_deg": 0.0,
        "defrost_posn_fdh_angle_max_deg": 90.0,
        "defrost_posn_fdh_method": "linear",
        "defrost_tmadef_fallback": "eva",
    }
    for name, val in defrost_defaults.items():
        rows.append(
            {
                "parameter_name": name,
                "module": "observer",
                "category": "defrost_observer",
                "_raw_value": val,
                "source_file": "hvac_sim/defrost.py",
                "source_status": "runtime_observer_default",
                "used_in_python": True,
            }
        )
    return rows


def _geometry_rows() -> List[Dict[str, Any]]:
    path = _geometry_defaults_path()
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    rel = str(path.relative_to(_repo_root()))
    rows: List[Dict[str, Any]] = []
    for outlet_name, cfg in doc.get("outlets", {}).items():
        if not isinstance(cfg, dict):
            continue
        for key, val in cfg.items():
            if key in ("measured", "note"):
                continue
            pname = f"geometry_{outlet_name}_{key}"
            rows.append(
                {
                    "parameter_name": pname,
                    "module": "geometry",
                    "category": "air_speed_geometry",
                    "_raw_value": val,
                    "source_file": rel,
                    "source_status": "geometry_placeholder",
                    "used_in_python": True,
                    "_measured": cfg.get("measured"),
                }
            )
    global_cfg = doc.get("global", {})
    for key, val in global_cfg.items():
        if isinstance(val, dict) and "value" in val:
            rows.append(
                {
                    "parameter_name": f"geometry_global_{key}",
                    "module": "geometry",
                    "category": "air_speed_geometry",
                    "_raw_value": val["value"],
                    "source_file": rel,
                    "source_status": "geometry_placeholder",
                    "used_in_python": True,
                    "_measured": val.get("measured"),
                }
            )
    return rows


def _merge_rows(primary: Dict[str, Dict[str, Any]], incoming: Iterable[Dict[str, Any]]) -> None:
    for row in incoming:
        name = str(row["parameter_name"])
        existing = primary.get(name)
        if existing is None:
            primary[name] = dict(row)
            continue
        # Prefer excel_param over python_default; keep used_in_python=True if either source uses it.
        status_rank = {
            "excel_param": 4,
            "python_default": 3,
            "geometry_placeholder": 2,
            "runtime_observer_default": 2,
            "missing_in_excel": 1,
        }
        new_rank = status_rank.get(str(row.get("source_status")), 0)
        old_rank = status_rank.get(str(existing.get("source_status")), 0)
        if new_rank >= old_rank:
            merged = dict(existing)
            merged.update(row)
            if existing.get("used_in_python") or row.get("used_in_python"):
                merged["used_in_python"] = True
            primary[name] = merged
        else:
            if row.get("used_in_python"):
                existing["used_in_python"] = True


def _mark_missing_in_excel(rows: Dict[str, Dict[str, Any]], excel_names: set[str]) -> None:
    afe_fields = {f.name for f in fields(AFEParams)}
    for name, row in rows.items():
        if name not in afe_fields:
            continue
        sim_name = f"AFE_FH{name}_P"
        in_excel = name in excel_names or sim_name in excel_names
        if name in _AFE_PYTHON_ONLY:
            continue
        if name in _AFE_MISSING_IN_EXCEL or not in_excel:
            if row.get("source_status") == "python_default":
                row["source_status"] = "missing_in_excel"
                row["needs_vehicle_calibration"] = True
                row["priority"] = "P0"


def build_parameter_inventory() -> Dict[str, Any]:
    """Collect parameters from all configured sources into one inventory document."""
    merged: Dict[str, Dict[str, Any]] = {}

    chtd = CHTDParams()
    afe = AFEParams()
    fans = FanSet()

    _merge_rows(
        merged,
        _dataclass_rows(
            chtd,
            module="CHTD",
            source_file="python_impl/hvac_sim/chtd/params.py",
        ),
    )
    _merge_rows(
        merged,
        _dataclass_rows(
            afe,
            module="AFE",
            source_file="python_impl/hvac_sim/afe/params.py",
        ),
    )
    _merge_rows(
        merged,
        _dataclass_rows(
            fans,
            module="AFE",
            source_file="python_impl/hvac_sim/afe/fan.py",
        ),
    )
    _merge_rows(merged, _geometry_rows())
    _merge_rows(merged, _observer_default_rows())

    excel_rows, excel_paths = _load_excel_records()
    excel_names = {str(r["parameter_name"]) for r in excel_rows}
    if excel_rows:
        _merge_rows(merged, excel_rows)
    else:
        ws_rows, _ = _load_workspace_params_fallback()
        excel_names = {str(r["parameter_name"]) for r in ws_rows}
        _merge_rows(merged, ws_rows)

    _mark_missing_in_excel(merged, excel_names)

    parameters = [_finalize_row(row) for row in sorted(merged.values(), key=lambda r: r["parameter_name"])]

    lut_count = sum(1 for p in parameters if p["is_lut"])
    placeholder_like_count = sum(1 for p in parameters if p["is_placeholder_like"])
    p0_count = sum(1 for p in parameters if p["priority"] == "P0")
    pso_candidate_count = sum(1 for p in parameters if p["pso_candidate"])
    missing_in_excel_count = sum(1 for p in parameters if p["source_status"] == "missing_in_excel")

    return {
        "schema": "parameter_master_table_v1",
        "description": "Unified parameter inventory for engineer review (Excel + Python defaults + geometry + observers)",
        "source_files": {
            "excel_candidates": excel_paths,
            "python_chtd_params": "python_impl/hvac_sim/chtd/params.py",
            "python_afe_params": "python_impl/hvac_sim/afe/params.py",
            "python_fan_params": "python_impl/hvac_sim/afe/fan.py",
            "geometry_defaults": str(_geometry_defaults_path().relative_to(_repo_root())),
            "workspace_fallback": str((_params_pkg() / "workspace_params.json").relative_to(_repo_root())),
        },
        "summary": {
            "total_parameters": len(parameters),
            "lut_count": lut_count,
            "placeholder_like_count": placeholder_like_count,
            "p0_count": p0_count,
            "pso_candidate_count": pso_candidate_count,
            "missing_in_excel_count": missing_in_excel_count,
        },
        "parameters": parameters,
    }


def write_parameter_master_table(
    output_dir: Path,
    *,
    pretty: bool = True,
) -> Dict[str, Path]:
    """Write ``parameter_master_table.json`` and ``parameter_master_table.csv``."""
    doc = build_parameter_inventory()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "parameter_master_table.json"
    csv_path = output_dir / "parameter_master_table.csv"

    json_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INVENTORY_FIELD_NAMES))
        writer.writeheader()
        for row in doc["parameters"]:
            out = dict(row)
            if isinstance(out.get("current_value"), (list, dict)):
                out["current_value"] = json.dumps(out["current_value"], ensure_ascii=False)
            writer.writerow(out)

    return {"json": json_path, "csv": csv_path}


def filter_parameter_inventory(
    inventory: Optional[Dict[str, Any]] = None,
    *,
    priority: Optional[str] = None,
    module: Optional[str] = None,
    pso_candidate: Optional[bool] = None,
    placeholder_only: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Return parameter rows matching optional filters."""
    doc = inventory if inventory is not None else build_parameter_inventory()
    rows = list(doc.get("parameters", []))
    if priority is not None:
        rows = [r for r in rows if r.get("priority") == priority]
    if module is not None:
        rows = [r for r in rows if r.get("module") == module]
    if pso_candidate is not None:
        rows = [r for r in rows if bool(r.get("pso_candidate")) == pso_candidate]
    if placeholder_only is not None:
        rows = [r for r in rows if bool(r.get("is_placeholder_like")) == placeholder_only]
    return rows


def _write_inventory_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INVENTORY_FIELD_NAMES))
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("current_value"), (list, dict)):
                out["current_value"] = json.dumps(out["current_value"], ensure_ascii=False)
            writer.writerow(out)


def export_parameter_review_slices(output_dir: Path) -> Dict[str, Path]:
    """Write focused CSV review slices (includes ``chinese_description``)."""
    doc = build_parameter_inventory()
    output_dir.mkdir(parents=True, exist_ok=True)

    slices = {
        "parameter_p0_review.csv": filter_parameter_inventory(doc, priority="P0"),
        "parameter_pso_candidates.csv": filter_parameter_inventory(doc, pso_candidate=True),
        "parameter_placeholder_luts.csv": [
            r
            for r in doc["parameters"]
            if r.get("is_lut") and r.get("is_placeholder_like")
        ],
        "parameter_geometry_measurement_todo.csv": [
            r
            for r in doc["parameters"]
            if r.get("module") == "geometry" and r.get("needs_vehicle_calibration")
        ],
    }

    paths: Dict[str, Path] = {}
    for filename, rows in slices.items():
        path = output_dir / filename
        _write_inventory_csv(path, rows)
        paths[filename] = path
    return paths
