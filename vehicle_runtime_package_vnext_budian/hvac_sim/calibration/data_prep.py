"""Wind-speed CSV → validation / airflow-calibration dataset conversion.

CSV columns are probe **wind speed**, not volumetric flow. Outlet volume flow is
computed only when ``effective_area_m2`` is marked ``measured=true`` in geometry
or a user-supplied area table:

    outlet_flow_m3h = wind_speed_m_s * effective_area_m2 * 3600

Without any measured areas, only the local-air-speed validation dataset is emitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from hvac_sim.config.geometry import load_geometry_config

from .wind_speed import (
    FACE_SHARE,
    SIDE_DEFROST_SHARE,
    summarize_wind_csv,
    sample_to_dict,
    parse_condition_file,
    _csv_condition_index,
)


VALIDATION_SCHEMA = "prepared_local_air_speed_validation_v1"
AIRFLOW_SCHEMA = "prepared_airflow_calibration_v1"

VALIDATION_FILENAME = "prepared_local_air_speed_validation_dataset.json"
AIRFLOW_FILENAME = "prepared_airflow_calibration_dataset.json"

FLOW_FORMULA = "outlet_flow_m3h = wind_speed_m_s * effective_area_m2 * 3600"

# Outlet key → wind-speed field on summarized sample (m/s).
_OUTLET_SPEED_FIELD: Dict[str, str] = {
    "driver_face": "measured_driver_air_speed_m_s",
    "passenger_face": "measured_passenger_air_speed_m_s",
    "driver_floor": "measured_driver_floor_speed_m_s",
    "passenger_floor": "measured_passenger_floor_speed_m_s",
    "driver_defrost": "estimated_driver_side_defrost_speed_m_s",
    "passenger_defrost": "estimated_passenger_side_defrost_speed_m_s",
}

_SIDE_DEFROST_OUTLETS = frozenset({"driver_defrost", "passenger_defrost"})


class CalibrationDataPrepError(ValueError):
    """Invalid calibration data-prep input."""


@dataclass
class OutletAreaInfo:
    effective_area_m2: float
    measured: bool
    source: str = ""
    note: str = ""


@dataclass
class PrepareCalibrationDatasetsResult:
    validation_path: Path
    airflow_path: Optional[Path]
    validation_document: Dict[str, Any]
    airflow_document: Optional[Dict[str, Any]]
    any_measured_area: bool


def _flow_from_speed(speed_m_s: float, area_m2: float) -> float:
    return max(0.0, float(speed_m_s)) * float(area_m2) * 3600.0


def load_area_table(path: Union[str, Path]) -> Dict[str, Any]:
    """Load geometry or user area-table JSON."""
    src = Path(path)
    if not src.exists():
        raise CalibrationDataPrepError(f"area table not found: {src}")
    doc = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise CalibrationDataPrepError("area table root must be an object")
    return doc


def resolve_outlet_areas(
    *,
    geometry_path: Optional[Union[str, Path]] = None,
    area_table_path: Optional[Union[str, Path]] = None,
) -> Dict[str, OutletAreaInfo]:
    """Merge geometry defaults with optional user area-table overrides."""
    if area_table_path is not None:
        base_doc = load_area_table(area_table_path)
        source_label = Path(area_table_path).name
    elif geometry_path is not None:
        base_doc = load_geometry_config(geometry_path)
        source_label = Path(geometry_path).name
    else:
        base_doc = load_geometry_config()
        source_label = "geometry_defaults.json"

    outlets = base_doc.get("outlets", {})
    if not isinstance(outlets, dict):
        raise CalibrationDataPrepError("area table missing 'outlets' object")

    resolved: Dict[str, OutletAreaInfo] = {}
    for outlet_name, payload in outlets.items():
        if not isinstance(payload, dict):
            continue
        area_raw = payload.get("effective_area_m2")
        if area_raw is None:
            continue
        resolved[outlet_name] = OutletAreaInfo(
            effective_area_m2=float(area_raw),
            measured=bool(payload.get("measured", False)),
            source=source_label,
            note=str(payload.get("note", "")),
        )
    return resolved


def any_measured_outlet_area(outlet_areas: Mapping[str, OutletAreaInfo]) -> bool:
    return any(info.measured for info in outlet_areas.values())


def _area_measurement_status(outlet_areas: Mapping[str, OutletAreaInfo]) -> Dict[str, Any]:
    measured = sorted(name for name, info in outlet_areas.items() if info.measured)
    placeholder = sorted(name for name, info in outlet_areas.items() if not info.measured)
    return {
        "any_measured": bool(measured),
        "measured_outlets": measured,
        "placeholder_outlets": placeholder,
    }


def _rear_foot_block(sample: Mapping[str, Any]) -> Dict[str, Any]:
    sensor_means = sample.get("sensor_speed_mean_m_s", {})
    rear_probe_ids = ("2", "3", "11", "13")
    present = [sensor_means[k] for k in rear_probe_ids if k in sensor_means]
    return {
        "rear_driver_foot_speed_m_s": None,
        "rear_passenger_foot_speed_m_s": None,
        "rear_foot_outlet_flow_m3h": None,
        "status": "missing_no_rear_foot_probe_mapping",
        "provenance": {
            "note": (
                "Rear foot outlet reserved; experiment CSV has rear face/center probes "
                "but no dedicated rear-foot speed channel"
            ),
            "rear_probe_speeds_present": bool(present),
            "rear_probe_ids": list(rear_probe_ids),
        },
    }


def _validation_sample_from_wind(
    sample: Mapping[str, Any],
    *,
    outlet_areas: Mapping[str, OutletAreaInfo],
) -> Dict[str, Any]:
    measured_speeds = {
        "driver_face": float(sample["measured_driver_air_speed_m_s"]),
        "passenger_face": float(sample["measured_passenger_air_speed_m_s"]),
        "driver_floor": float(sample["measured_driver_floor_speed_m_s"]),
        "passenger_floor": float(sample["measured_passenger_floor_speed_m_s"]),
        "front_windshield_defrost": float(sample["measured_front_defrost_speed_m_s"]),
        "driver_side_defrost_estimated": float(
            sample["estimated_driver_side_defrost_speed_m_s"]
        ),
        "passenger_side_defrost_estimated": float(
            sample["estimated_passenger_side_defrost_speed_m_s"]
        ),
    }
    return {
        "source_file": sample.get("source_file"),
        "condition": sample.get("condition"),
        "n_rows": sample.get("n_rows"),
        "time_start": sample.get("time_start"),
        "time_end": sample.get("time_end"),
        "sensor_speed_mean_m_s": sample.get("sensor_speed_mean_m_s", {}),
        "sensor_speed_std_m_s": sample.get("sensor_speed_std_m_s", {}),
        "sensor_temp_mean_c": sample.get("sensor_temp_mean_c", {}),
        "measured_wind_speeds_m_s": measured_speeds,
        "rear_foot": _rear_foot_block(sample),
        "area_measurement_status": _area_measurement_status(outlet_areas),
        "provenance": {
            "classification": "validation_not_formal_calibration",
            "no_outlet_flow_from_placeholder_area": True,
            "side_defrost_rule": (
                "total_side_defrost = (driver_face + passenger_face) * 0.07 / 0.93; "
                "split equally to driver/passenger side defrost"
            ),
        },
        "notes": list(sample.get("notes", [])),
    }


def _outlet_flow_record(
    outlet_name: str,
    speed_m_s: float,
    area_info: OutletAreaInfo,
    *,
    derived_from: Optional[str] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "outlet_flow_m3h": _flow_from_speed(speed_m_s, area_info.effective_area_m2),
        "wind_speed_m_s": float(speed_m_s),
        "effective_area_m2": area_info.effective_area_m2,
        "measured": area_info.measured,
        "area_source": area_info.source,
    }
    if derived_from:
        record["derived_from"] = derived_from
    if area_info.note:
        record["note"] = area_info.note
    return record


def _airflow_sample_from_wind(
    sample: Mapping[str, Any],
    *,
    outlet_areas: Mapping[str, OutletAreaInfo],
) -> Dict[str, Any]:
    outlet_flows: Dict[str, Any] = {}
    skipped_unmeasured: List[str] = []

    for outlet_name, speed_field in _OUTLET_SPEED_FIELD.items():
        area_info = outlet_areas.get(outlet_name)
        speed_m_s = float(sample[speed_field])
        if area_info is None:
            outlet_flows[outlet_name] = {
                "outlet_flow_m3h": None,
                "status": "missing_area_definition",
                "wind_speed_m_s": speed_m_s,
            }
            continue
        if not area_info.measured:
            skipped_unmeasured.append(outlet_name)
            outlet_flows[outlet_name] = {
                "outlet_flow_m3h": None,
                "status": "skipped_unmeasured_area",
                "wind_speed_m_s": speed_m_s,
                "effective_area_m2": area_info.effective_area_m2,
                "measured": False,
                "provenance": "effective area not measured; flow not computed",
            }
            continue
        derived = None
        if outlet_name in _SIDE_DEFROST_OUTLETS:
            derived = "93_7_side_defrost_rule"
        outlet_flows[outlet_name] = _outlet_flow_record(
            outlet_name,
            speed_m_s,
            area_info,
            derived_from=derived,
        )

    rear_area = outlet_areas.get("rear_foot")
    if rear_area is None or not rear_area.measured:
        outlet_flows["rear_foot"] = {
            "outlet_flow_m3h": None,
            "status": "missing_rear_foot_data",
            "wind_speed_m_s": None,
            "measured": False,
            "provenance": _rear_foot_block(sample)["provenance"],
        }
    else:
        outlet_flows["rear_foot"] = {
            "outlet_flow_m3h": 0.0,
            "wind_speed_m_s": 0.0,
            "effective_area_m2": rear_area.effective_area_m2,
            "measured": True,
            "status": "zero_placeholder_no_rear_foot_probe",
            "provenance": _rear_foot_block(sample)["provenance"],
        }

    return {
        "source_file": sample.get("source_file"),
        "condition": sample.get("condition"),
        "outlet_flows_m3h": outlet_flows,
        "skipped_unmeasured_outlets": skipped_unmeasured,
        "provenance": {
            "formula": FLOW_FORMULA,
            "side_defrost_rule": (
                f"face_share={FACE_SHARE}, side_share={SIDE_DEFROST_SHARE}; "
                "side speeds estimated then multiplied by measured side-defrost area"
            ),
        },
    }


def build_validation_dataset(
    wind_samples: List[Mapping[str, Any]],
    *,
    source_dir: Union[str, Path],
    condition_file: Optional[str],
    outlet_areas: Mapping[str, OutletAreaInfo],
) -> Dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "classification": "validation_not_formal_calibration",
        "purpose": "local_air_speed_prediction_validation",
        "not_formal_airflow_calibration": True,
        "source_dir": str(source_dir),
        "condition_file": condition_file,
        "area_measurement_status": _area_measurement_status(outlet_areas),
        "distribution_policy": {
            "front_face_share": FACE_SHARE,
            "front_side_defrost_total_share": SIDE_DEFROST_SHARE,
            "side_defrost_rule": (
                "total_side_defrost = (driver_face + passenger_face) * 0.07 / 0.93; "
                "split equally"
            ),
            "rear_foot_flow": "reserved_null_until_rear_foot_probe_available",
        },
        "notes": [
            "CSV wind speed validates local air-speed prediction only",
            "Outlet volume flow is NOT derived from placeholder effective areas",
            "Formal airflow calibration requires measured effective_area_m2",
        ],
        "samples": [
            _validation_sample_from_wind(s, outlet_areas=outlet_areas) for s in wind_samples
        ],
    }


def build_airflow_calibration_dataset(
    wind_samples: List[Mapping[str, Any]],
    *,
    source_dir: Union[str, Path],
    outlet_areas: Mapping[str, OutletAreaInfo],
) -> Dict[str, Any]:
    measured = _area_measurement_status(outlet_areas)
    if not measured["any_measured"]:
        raise CalibrationDataPrepError(
            "airflow calibration dataset requires at least one measured=true effective area"
        )
    return {
        "schema": AIRFLOW_SCHEMA,
        "classification": "airflow_calibration_candidate",
        "requires_measured_effective_area": True,
        "not_production_calibration_until_reviewed": True,
        "source_dir": str(source_dir),
        "formula": FLOW_FORMULA,
        "area_measurement_status": measured,
        "area_sources": {
            name: {
                "effective_area_m2": info.effective_area_m2,
                "measured": info.measured,
                "source": info.source,
            }
            for name, info in sorted(outlet_areas.items())
        },
        "distribution_policy": {
            "front_face_share": FACE_SHARE,
            "front_side_defrost_total_share": SIDE_DEFROST_SHARE,
            "side_defrost_rule": (
                "driver/passenger side defrost speed from 93/7 face split; "
                "flows use measured side-defrost effective areas"
            ),
        },
        "samples": [
            _airflow_sample_from_wind(s, outlet_areas=outlet_areas) for s in wind_samples
        ],
    }


def _collect_wind_samples(
    csv_dir: Union[str, Path],
    *,
    condition_file: Optional[Union[str, Path]] = None,
) -> tuple[List[Dict[str, Any]], Path, Optional[str]]:
    base = Path(csv_dir)
    if condition_file is None:
        condition_path = base / "多站点记录.txt"
    else:
        condition_path = Path(condition_file)
    conditions = parse_condition_file(condition_path) if condition_path.exists() else []
    condition_by_index = {c.sequence_index: c for c in conditions}

    samples: List[Dict[str, Any]] = []
    for csv_path in sorted(base.glob("*.csv"), key=_csv_condition_index):
        idx = _csv_condition_index(csv_path)
        wind_sample = summarize_wind_csv(
            csv_path,
            condition=condition_by_index.get(idx),
        )
        samples.append(sample_to_dict(wind_sample))

    if not samples:
        raise CalibrationDataPrepError(f"no CSV samples found under {base}")

    cond_str = str(condition_path) if condition_path.exists() else None
    return samples, base, cond_str


def prepare_calibration_datasets(
    csv_dir: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    geometry_path: Optional[Union[str, Path]] = None,
    area_table_path: Optional[Union[str, Path]] = None,
    condition_file: Optional[Union[str, Path]] = None,
    pretty: bool = True,
) -> PrepareCalibrationDatasetsResult:
    """Convert wind-speed CSV directory into validation and optional airflow JSON."""
    wind_samples, base, cond_str = _collect_wind_samples(
        csv_dir,
        condition_file=condition_file,
    )
    outlet_areas = resolve_outlet_areas(
        geometry_path=geometry_path,
        area_table_path=area_table_path,
    )
    has_measured = any_measured_outlet_area(outlet_areas)

    if output_dir is None:
        out_base = Path(__file__).resolve().parents[3] / "simulink_conversion_package" / "python_targets"
    else:
        out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    validation_doc = build_validation_dataset(
        wind_samples,
        source_dir=base,
        condition_file=cond_str,
        outlet_areas=outlet_areas,
    )
    validation_path = out_base / VALIDATION_FILENAME
    validation_path.write_text(
        json.dumps(validation_doc, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )

    airflow_doc: Optional[Dict[str, Any]] = None
    airflow_path: Optional[Path] = None
    if has_measured:
        airflow_doc = build_airflow_calibration_dataset(
            wind_samples,
            source_dir=base,
            outlet_areas=outlet_areas,
        )
        airflow_path = out_base / AIRFLOW_FILENAME
        airflow_path.write_text(
            json.dumps(airflow_doc, ensure_ascii=False, indent=2 if pretty else None) + "\n",
            encoding="utf-8",
        )

    return PrepareCalibrationDatasetsResult(
        validation_path=validation_path,
        airflow_path=airflow_path,
        validation_document=validation_doc,
        airflow_document=airflow_doc,
        any_measured_area=has_measured,
    )


def prepare_from_wind_payload(
    wind_payload: Mapping[str, Any],
    *,
    output_dir: Union[str, Path],
    geometry_path: Optional[Union[str, Path]] = None,
    area_table_path: Optional[Union[str, Path]] = None,
    pretty: bool = True,
) -> PrepareCalibrationDatasetsResult:
    """Build prepared datasets from an existing ``convert_wind_experiment_dir`` payload."""
    samples = wind_payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise CalibrationDataPrepError("wind payload must contain non-empty 'samples' list")

    outlet_areas = resolve_outlet_areas(
        geometry_path=geometry_path,
        area_table_path=area_table_path,
    )
    has_measured = any_measured_outlet_area(outlet_areas)
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    validation_doc = build_validation_dataset(
        samples,
        source_dir=wind_payload.get("source_dir", out_base),
        condition_file=wind_payload.get("condition_file"),
        outlet_areas=outlet_areas,
    )
    validation_path = out_base / VALIDATION_FILENAME
    validation_path.write_text(
        json.dumps(validation_doc, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )

    airflow_doc = None
    airflow_path = None
    if has_measured:
        airflow_doc = build_airflow_calibration_dataset(
            samples,
            source_dir=wind_payload.get("source_dir", out_base),
            outlet_areas=outlet_areas,
        )
        airflow_path = out_base / AIRFLOW_FILENAME
        airflow_path.write_text(
            json.dumps(airflow_doc, ensure_ascii=False, indent=2 if pretty else None) + "\n",
            encoding="utf-8",
        )

    return PrepareCalibrationDatasetsResult(
        validation_path=validation_path,
        airflow_path=airflow_path,
        validation_document=validation_doc,
        airflow_document=airflow_doc,
        any_measured_area=has_measured,
    )
