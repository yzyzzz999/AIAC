"""Formal prepared calibration dataset schemas and validation (T10-21).

Filenames (under ``python_targets/`` or user output dir):

- ``prepared_airflow_calibration_dataset.json``
- ``prepared_chtd_calibration_dataset.json``
- ``prepared_ekf_calibration_dataset.json``

Legacy wind-prep airflow uses ``prepared_airflow_calibration_v1``; formal bench
schema is ``prepared_airflow_calibration_v2``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hvac_sim.chtd.bus_index import N_U, N_X_STATES, X_INDEX

from .data_prep import AIRFLOW_SCHEMA as AIRFLOW_V1_SCHEMA
from .de_calibration import AFEMeasurement
from .schema import CalibrationSchemaError

# Formal schema ids
AIRFLOW_FORMAL_SCHEMA = "prepared_airflow_calibration_v2"
CHTD_FORMAL_SCHEMA = "prepared_chtd_calibration_v2"
EKF_FORMAL_SCHEMA = "prepared_ekf_calibration_v2"

AIRFLOW_FILENAME = "prepared_airflow_calibration_dataset.json"
CHTD_FILENAME = "prepared_chtd_calibration_dataset.json"
EKF_FILENAME = "prepared_ekf_calibration_dataset.json"

_DEMO_AIRFLOW_SCHEMA = "afe_calibration_demo_v1"
_DEMO_CHTD_SCHEMA = "chtd_calibration_demo_v1"
_DEMO_EKF_SCHEMA = "ekf_calibration_demo_v1"

_EKF_SAMPLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "model_head_temp_c": ("model_head_temp", "model_temp_c"),
    "ir_surface_temp_c": ("ir_surface_temp",),
    "measured_head_air_temp_c": (
        "measured_head_air_temp",
        "target_head_air_temp_c",
        "target_fused_temp_c",
    ),
    "air_speed_m_s": ("air_speed",),
    "mrt_c": ("mrt", "mean_radiant_temp_c"),
    "confidence": (),
}

_EKF_REQUIRED_FIELDS = (
    "model_head_temp_c",
    "ir_surface_temp_c",
    "measured_head_air_temp_c",
)


class CalibrationDatasetSchemaError(CalibrationSchemaError):
    """Prepared calibration JSON failed schema validation."""


def _path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _require_mapping(
    value: Any,
    *,
    path: str,
    optional: bool = False,
) -> Mapping[str, Any]:
    if value is None:
        if optional:
            return {}
        raise CalibrationDatasetSchemaError(f"{path}: required object missing")
    if not isinstance(value, Mapping):
        raise CalibrationDatasetSchemaError(f"{path}: must be an object")
    return value


def _require_list(
    value: Any,
    *,
    path: str,
    min_len: int = 1,
) -> List[Any]:
    if not isinstance(value, list):
        raise CalibrationDatasetSchemaError(f"{path}: must be a list")
    if len(value) < min_len:
        raise CalibrationDatasetSchemaError(
            f"{path}: must contain at least {min_len} element(s), got {len(value)}"
        )
    return value


def _require_keys(
    data: Mapping[str, Any],
    keys: Sequence[str],
    *,
    path: str,
) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise CalibrationDatasetSchemaError(
            f"{path}: missing required field(s): {', '.join(missing)}"
        )


def _require_quality_flags(sample: Mapping[str, Any], *, path: str) -> None:
    flags = sample.get("quality_flags")
    if flags is None:
        raise CalibrationDatasetSchemaError(f"{path}: missing required 'quality_flags' object")
    _require_mapping(flags, path=_path(path, "quality_flags"))


def _numeric_flow_dict(
    raw: Any,
    *,
    path: str,
) -> Dict[str, float]:
    block = _require_mapping(raw, path=path)
    flows: Dict[str, float] = {}
    for key, val in block.items():
        if val is None:
            continue
        try:
            flows[str(key)] = float(val)
        except (TypeError, ValueError) as exc:
            raise CalibrationDatasetSchemaError(
                f"{path}.{key}: measured_flow_m3h values must be numeric or null"
            ) from exc
    if not flows:
        raise CalibrationDatasetSchemaError(
            f"{path}: measured_flow_m3h must include at least one numeric outlet flow"
        )
    return flows


def _float_vector(
    value: Any,
    length: int,
    *,
    path: str,
) -> np.ndarray:
    if not isinstance(value, list):
        raise CalibrationDatasetSchemaError(f"{path}: must be a list of length {length}")
    if len(value) != length:
        raise CalibrationDatasetSchemaError(
            f"{path}: expected length {length}, got {len(value)}"
        )
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CalibrationDatasetSchemaError(f"{path}: non-numeric element") from exc


def validate_airflow_calibration_dataset_v1(doc: Mapping[str, Any]) -> None:
    """Wind-prep / bench outlet_flows_m3h format (``prepared_airflow_calibration_v1``)."""
    if str(doc.get("schema", "")) != AIRFLOW_V1_SCHEMA:
        raise CalibrationDatasetSchemaError(
            f"schema must be {AIRFLOW_V1_SCHEMA!r}, got {doc.get('schema')!r}"
        )
    samples = _require_list(doc.get("samples"), path="samples", min_len=1)
    for idx, sample in enumerate(samples):
        sp = _path("samples", f"[{idx}]")
        s = _require_mapping(sample, path=sp)
        flows = s.get("outlet_flows_m3h")
        if flows is None:
            raise CalibrationDatasetSchemaError(
                f"{sp}: missing 'outlet_flows_m3h' (v1 wind-prep layout)"
            )
        _require_mapping(flows, path=_path(sp, "outlet_flows_m3h"))


def validate_airflow_calibration_dataset_v2(doc: Mapping[str, Any]) -> None:
    """Formal AFE bench schema (operating point + measured outlet flows)."""
    if str(doc.get("schema", "")) != AIRFLOW_FORMAL_SCHEMA:
        raise CalibrationDatasetSchemaError(
            f"schema must be {AIRFLOW_FORMAL_SCHEMA!r}, got {doc.get('schema')!r}"
        )
    if not str(doc.get("source", "")).strip():
        raise CalibrationDatasetSchemaError("root: missing required 'source' string")
    samples = _require_list(doc.get("samples"), path="samples", min_len=1)
    for idx, sample in enumerate(samples):
        sp = _path("samples", f"[{idx}]")
        s = _require_mapping(sample, path=sp)
        _require_keys(
            s,
            ("operating_condition", "flaps", "fan_set", "measured_flow_m3h", "quality_flags"),
            path=sp,
        )
        _require_mapping(s["operating_condition"], path=_path(sp, "operating_condition"))
        _require_mapping(s["flaps"], path=_path(sp, "flaps"))
        _require_mapping(s["fan_set"], path=_path(sp, "fan_set"))
        _numeric_flow_dict(s["measured_flow_m3h"], path=_path(sp, "measured_flow_m3h"))
        _require_quality_flags(s, path=sp)
        if "hvac_mode" in s and not isinstance(s["hvac_mode"], str):
            raise CalibrationDatasetSchemaError(f"{sp}.hvac_mode: must be a string")


def validate_chtd_calibration_dataset_v2(doc: Mapping[str, Any]) -> None:
    """Formal CHTD time-series / one-step calibration schema."""
    if str(doc.get("schema", "")) != CHTD_FORMAL_SCHEMA:
        raise CalibrationDatasetSchemaError(
            f"schema must be {CHTD_FORMAL_SCHEMA!r}, got {doc.get('schema')!r}"
        )
    if not str(doc.get("source", "")).strip():
        raise CalibrationDatasetSchemaError("root: missing required 'source' string")
    meta = doc.get("metadata")
    if meta is not None:
        _require_mapping(meta, path="metadata")
    samples = _require_list(doc.get("samples"), path="samples", min_len=1)
    root_dt = None
    if isinstance(meta, Mapping) and "dt_s" in meta:
        root_dt = float(meta["dt_s"])

    for idx, sample in enumerate(samples):
        sp = _path("samples", f"[{idx}]")
        s = _require_mapping(sample, path=sp)
        _require_quality_flags(s, path=sp)
        has_series = "time_series" in s
        has_x0 = "x0" in s
        if not has_series and not has_x0:
            raise CalibrationDatasetSchemaError(
                f"{sp}: require 'x0' or 'time_series' object"
            )
        if "measured_zone_temps" not in s:
            raise CalibrationDatasetSchemaError(f"{sp}: missing 'measured_zone_temps'")
        temps = _require_mapping(
            s["measured_zone_temps"],
            path=_path(sp, "measured_zone_temps"),
        )
        if not temps:
            raise CalibrationDatasetSchemaError(
                f"{sp}.measured_zone_temps: at least one zone required"
            )
        for zname, zval in temps.items():
            if isinstance(zval, list):
                if not zval:
                    raise CalibrationDatasetSchemaError(
                        f"{sp}.measured_zone_temps.{zname}: empty series"
                    )
            else:
                try:
                    float(zval)
                except (TypeError, ValueError) as exc:
                    raise CalibrationDatasetSchemaError(
                        f"{sp}.measured_zone_temps.{zname}: must be numeric or list"
                    ) from exc

        if has_x0:
            _float_vector(s["x0"], N_X_STATES, path=_path(sp, "x0"))
            if "u" not in s:
                raise CalibrationDatasetSchemaError(f"{sp}: missing 'u' with 'x0'")
            _float_vector(s["u"], N_U, path=_path(sp, "u"))

        if has_series:
            ts = _require_mapping(s["time_series"], path=_path(sp, "time_series"))
            if "dt_s" not in ts and root_dt is None and "dt_s" not in s:
                raise CalibrationDatasetSchemaError(
                    f"{sp}: missing dt_s in time_series, sample, or metadata"
                )
            if "u" not in ts and "u" not in s:
                raise CalibrationDatasetSchemaError(
                    f"{sp}.time_series: missing 'u' (vector or list of vectors)"
                )
            u_raw = ts.get("u", s.get("u"))
            if isinstance(u_raw, list) and u_raw and isinstance(u_raw[0], list):
                for j, row in enumerate(u_raw):
                    _float_vector(row, N_U, path=_path(sp, f"time_series.u[{j}]"))
            else:
                _float_vector(u_raw, N_U, path=_path(sp, "time_series.u"))
            if "x0" in ts:
                _float_vector(ts["x0"], N_X_STATES, path=_path(sp, "time_series.x0"))
            elif "x0" in s:
                _float_vector(s["x0"], N_X_STATES, path=_path(sp, "x0"))
            else:
                raise CalibrationDatasetSchemaError(
                    f"{sp}.time_series: missing 'x0' (in time_series or sample)"
                )


def validate_ekf_calibration_dataset_v2(doc: Mapping[str, Any]) -> None:
    """Formal EKF / IR head-temperature calibration schema."""
    if str(doc.get("schema", "")) != EKF_FORMAL_SCHEMA:
        raise CalibrationDatasetSchemaError(
            f"schema must be {EKF_FORMAL_SCHEMA!r}, got {doc.get('schema')!r}"
        )
    if not str(doc.get("source", "")).strip():
        raise CalibrationDatasetSchemaError("root: missing required 'source' string")
    samples = _require_list(doc.get("samples"), path="samples", min_len=1)
    for idx, sample in enumerate(samples):
        sp = _path("samples", f"[{idx}]")
        s = _require_mapping(sample, path=sp)
        _require_quality_flags(s, path=sp)
        for canonical in _EKF_REQUIRED_FIELDS:
            aliases = _EKF_SAMPLE_ALIASES.get(canonical, ())
            if canonical in s or any(a in s for a in aliases):
                continue
            raise CalibrationDatasetSchemaError(
                f"{sp}: missing required field {canonical!r} "
                f"(aliases: {', '.join(aliases) or 'none'})"
            )
        try:
            for canonical in _EKF_REQUIRED_FIELDS:
                float(resolve_ekf_sample_field(s, canonical))
        except (TypeError, ValueError, KeyError) as exc:
            raise CalibrationDatasetSchemaError(
                f"{sp}: EKF numeric fields invalid: {exc}"
            ) from exc


def resolve_ekf_sample_field(sample: Mapping[str, Any], canonical: str) -> Any:
    """Read EKF sample field accepting documented aliases."""
    if canonical in sample:
        return sample[canonical]
    for alias in _EKF_SAMPLE_ALIASES.get(canonical, ()):
        if alias in sample:
            return sample[alias]
    raise KeyError(canonical)


def normalize_ekf_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """Map formal EKF sample to HeadTempEKFCalibrationAdapter keys."""
    ir_val: Optional[float] = None
    if _has_ekf_field(sample, "ir_surface_temp_c"):
        ir_val = _optional_float(resolve_ekf_sample_field(sample, "ir_surface_temp_c"))
    confidence = (
        float(resolve_ekf_sample_field(sample, "confidence"))
        if _has_ekf_field(sample, "confidence")
        else 1.0
    )
    return {
        "model_temp_c": float(resolve_ekf_sample_field(sample, "model_head_temp_c")),
        "ir_surface_temp_c": ir_val,
        "confidence": confidence,
        "measured_head_air_temp_c": float(
            resolve_ekf_sample_field(sample, "measured_head_air_temp_c")
        ),
        "air_speed_m_s": _optional_float(
            sample.get("air_speed_m_s", sample.get("air_speed"))
        ),
        "mrt_c": _optional_float(sample.get("mrt_c", sample.get("mrt"))),
        "quality_flags": dict(sample.get("quality_flags") or {}),
    }


def _has_ekf_field(sample: Mapping[str, Any], canonical: str) -> bool:
    if canonical in sample:
        return sample[canonical] is not None
    return any(sample.get(a) is not None for a in _EKF_SAMPLE_ALIASES.get(canonical, ()))


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def validate_prepared_dataset(
    doc: Mapping[str, Any],
    *,
    expected_schema: str,
) -> None:
    """Dispatch validation by schema id."""
    schema = str(doc.get("schema", ""))
    if schema != expected_schema:
        raise CalibrationDatasetSchemaError(
            f"schema must be {expected_schema!r}, got {schema!r}"
        )
    if expected_schema == AIRFLOW_V1_SCHEMA:
        validate_airflow_calibration_dataset_v1(doc)
    elif expected_schema == AIRFLOW_FORMAL_SCHEMA:
        validate_airflow_calibration_dataset_v2(doc)
    elif expected_schema == CHTD_FORMAL_SCHEMA:
        validate_chtd_calibration_dataset_v2(doc)
    elif expected_schema == EKF_FORMAL_SCHEMA:
        validate_ekf_calibration_dataset_v2(doc)
    else:
        raise CalibrationDatasetSchemaError(f"unsupported schema: {expected_schema!r}")


def normalize_dataset_classification(raw: str) -> str:
    """Map dataset document classification to PSO result classification."""
    key = str(raw).strip()
    if key in ("demo_only", "calibration_candidate"):
        return key
    if key.endswith("_calibration_candidate") or key == "airflow_calibration_candidate":
        return "calibration_candidate"
    return key


def validate_dataset_for_pso_target(
    doc: Mapping[str, Any],
    *,
    target: str,
) -> str:
    """Validate a user-supplied dataset for ``run_calibration_pso`` target.

    Returns classification string (``calibration_candidate``).

    Demo schemas skip strict formal validation.
    """
    schema = str(doc.get("schema", ""))
    target_key = target.strip().lower()

    if target_key == "airflow":
        if schema in (_DEMO_AIRFLOW_SCHEMA,):
            return "demo_only"
        if schema == AIRFLOW_V1_SCHEMA:
            validate_airflow_calibration_dataset_v1(doc)
            return normalize_dataset_classification(
                str(doc.get("classification", "calibration_candidate"))
            )
        if schema == AIRFLOW_FORMAL_SCHEMA:
            validate_airflow_calibration_dataset_v2(doc)
            return normalize_dataset_classification(
                str(doc.get("classification", "calibration_candidate"))
            )
        raise CalibrationDatasetSchemaError(
            "airflow dataset schema must be one of: "
            f"{AIRFLOW_V1_SCHEMA!r}, {AIRFLOW_FORMAL_SCHEMA!r}, "
            f"or demo {_DEMO_AIRFLOW_SCHEMA!r}; got {schema!r}"
        )

    if target_key == "chtd":
        if schema in (_DEMO_CHTD_SCHEMA, "chtd_golden_wrapper_v1"):
            return "demo_only"
        if schema == CHTD_FORMAL_SCHEMA:
            validate_chtd_calibration_dataset_v2(doc)
            return normalize_dataset_classification(
                str(doc.get("classification", "calibration_candidate"))
            )
        if "samples" in doc:
            # Legacy / golden-style — minimal check
            _require_list(doc.get("samples"), path="samples", min_len=1)
            return normalize_dataset_classification(
                str(doc.get("classification", "calibration_candidate"))
            )
        if "cases" in doc:
            return "demo_only"
        raise CalibrationDatasetSchemaError(
            "CHTD dataset must use schema "
            f"{CHTD_FORMAL_SCHEMA!r}, include non-empty 'samples', or golden 'cases'; "
            f"got schema={schema!r}"
        )

    if target_key == "ekf":
        if schema in (_DEMO_EKF_SCHEMA,):
            return "demo_only"
        if schema == EKF_FORMAL_SCHEMA:
            validate_ekf_calibration_dataset_v2(doc)
            return normalize_dataset_classification(
                str(doc.get("classification", "calibration_candidate"))
            )
        raise CalibrationDatasetSchemaError(
            "EKF dataset schema must be one of: "
            f"{EKF_FORMAL_SCHEMA!r} or demo {_DEMO_EKF_SCHEMA!r}; got {schema!r}"
        )

    raise CalibrationDatasetSchemaError(
        f"validate_dataset_for_pso_target: unsupported target {target!r}"
    )


def airflow_v2_sample_to_afe_measurement(sample: Mapping[str, Any]) -> AFEMeasurement:
    """Map formal v2 bench sample to ``AFEMeasurement`` for AFE PSO."""
    from hvac_sim.afe.fan import FanSet
    from hvac_sim.afe.resistance import FlapSet

    flaps_raw = sample.get("flaps", {})
    fans_raw = sample.get("fan_set", {})
    op = sample.get("operating_condition", {})
    flows = _numeric_flow_dict(
        sample["measured_flow_m3h"],
        path="measured_flow_m3h",
    )

    def _flow(name: str) -> Optional[float]:
        return flows.get(name)

    n_f1 = float(fans_raw.get("n_F1", op.get("n_F1", 0.55)))
    n_f2 = float(fans_raw.get("n_F2", op.get("n_F2", 0.50)))
    n_s = float(fans_raw.get("n_S", op.get("n_S", 0.35)))
    temp_c = float(op.get("temp_C", sample.get("temp_C", 25.0)))
    dp1 = float(op.get("dp1", sample.get("dp1", 0.0)))

    q1p = _flow("driver_face")
    q2p = _flow("passenger_face")
    qm = _flow("Qm") or _flow("total_mass_flow")

    return AFEMeasurement(
        flaps=FlapSet(),
        fan_set=FanSet(),
        n_F1=n_f1,
        n_F2=n_f2,
        n_S=n_s,
        temp_C=temp_c,
        dp1=dp1,
        Qm_measured=qm,
        Q1p_measured=q1p,
        Q2p_measured=q2p,
        w_Qm=1.0 if qm is not None else 0.0,
        w_Q1p=1.0 if q1p is not None else 0.0,
        w_Q2p=1.0 if q2p is not None else 0.0,
    )


def chtd_v2_sample_to_one_step(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """Build one-step smoke sample ``{x0, u, t_new}`` from formal CHTD v2 row."""
    if "x0" in sample:
        x0 = _float_vector(sample["x0"], N_X_STATES, path="x0")
    else:
        ts = _require_mapping(sample["time_series"], path="time_series")
        x0 = _float_vector(
            ts.get("x0", sample.get("x0")),
            N_X_STATES,
            path="time_series.x0",
        )

    if "u" in sample:
        u = _float_vector(sample["u"], N_U, path="u")
    else:
        ts = _require_mapping(sample["time_series"], path="time_series")
        u_raw = ts["u"]
        if isinstance(u_raw, list) and u_raw and isinstance(u_raw[0], list):
            u = _float_vector(u_raw[0], N_U, path="time_series.u[0]")
        else:
            u = _float_vector(u_raw, N_U, path="time_series.u")

    t_new = np.array(x0, dtype=float, copy=True)
    zone_temps = sample["measured_zone_temps"]
    for zname, zval in zone_temps.items():
        if zname not in X_INDEX:
            continue
        idx = int(X_INDEX[zname])
        if isinstance(zval, list):
            if not zval:
                continue
            t_new[idx] = float(zval[0])
        else:
            t_new[idx] = float(zval)

    return {
        "x0": x0.tolist(),
        "u": u.tolist(),
        "t_new": t_new.tolist(),
        "name": sample.get("name"),
        "quality_flags": dict(sample.get("quality_flags") or {}),
    }


def _airflow_v1_to_measurements(
    doc: Mapping[str, Any],
) -> Tuple[List[AFEMeasurement], List[str]]:
    """Map prepared_airflow_calibration_v1 outlet flows to AFE flow targets."""
    from hvac_sim.afe.fan import FanSet
    from hvac_sim.afe.resistance import FlapSet

    warnings: List[str] = []
    measurements: List[AFEMeasurement] = []
    for idx, sample in enumerate(doc.get("samples", [])):
        if not isinstance(sample, dict):
            warnings.append(f"samples[{idx}] skipped: not an object")
            continue
        flows = sample.get("outlet_flows_m3h", {})
        if not isinstance(flows, dict):
            warnings.append(f"samples[{idx}] skipped: missing outlet_flows_m3h")
            continue

        def _flow_m3h(outlet: str) -> Optional[float]:
            block = flows.get(outlet)
            if not isinstance(block, dict):
                return None
            raw = block.get("outlet_flow_m3h")
            if raw is None:
                return None
            return float(raw)

        q1p = _flow_m3h("driver_face")
        q2p = _flow_m3h("passenger_face")
        if q1p is None and q2p is None:
            warnings.append(f"samples[{idx}] skipped: no measured outlet flows")
            continue
        measurements.append(
            AFEMeasurement(
                flaps=FlapSet(),
                fan_set=FanSet(),
                n_F1=0.55,
                n_F2=0.50,
                n_S=0.35,
                Q1p_measured=q1p,
                Q2p_measured=q2p,
                w_Q1p=1.0 if q1p is not None else 0.0,
                w_Q2p=1.0 if q2p is not None else 0.0,
            )
        )
    if not measurements:
        raise CalibrationDatasetSchemaError(
            "prepared airflow v1 dataset produced no usable AFE measurements"
        )
    warnings.append(
        "prepared v1 outlet flows mapped to AFE Q1_prime/Q2_prime with default fan/flap"
    )
    return measurements, warnings


def airflow_measurements_from_dataset(
    doc: Mapping[str, Any],
) -> Tuple[List[AFEMeasurement], List[str]]:
    """Parse airflow prepared JSON (v1 or v2) into AFE measurements."""
    schema = str(doc.get("schema", ""))
    warnings: List[str] = []
    measurements: List[AFEMeasurement] = []

    if schema == AIRFLOW_FORMAL_SCHEMA:
        for idx, sample in enumerate(doc.get("samples", [])):
            if not isinstance(sample, dict):
                warnings.append(f"samples[{idx}] skipped: not an object")
                continue
            measurements.append(airflow_v2_sample_to_afe_measurement(sample))
        if not measurements:
            raise CalibrationDatasetSchemaError(
                "airflow v2 dataset produced no usable samples"
            )
        warnings.append(
            "formal v2: flaps from sample used for metadata only; "
            "AFE FlapSet defaults applied in smoke PSO"
        )
        return measurements, warnings

    if schema == AIRFLOW_V1_SCHEMA:
        return _airflow_v1_to_measurements(doc)

    raise CalibrationDatasetSchemaError(
        f"airflow_measurements_from_dataset: unsupported schema {schema!r}"
    )


def chtd_one_step_samples_from_dataset(
    doc: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize CHTD dataset samples to one-step list for smoke PSO."""
    schema = str(doc.get("schema", ""))
    warnings: List[str] = []

    if schema == CHTD_FORMAL_SCHEMA:
        out: List[Dict[str, Any]] = []
        for idx, sample in enumerate(doc.get("samples", [])):
            if not isinstance(sample, dict):
                warnings.append(f"samples[{idx}] skipped: not an object")
                continue
            out.append(chtd_v2_sample_to_one_step(sample))
        if not out:
            raise CalibrationDatasetSchemaError("CHTD v2 dataset has no usable samples")
        warnings.append(
            "CHTD v2 formal dataset: using first timestep / scalar zone temps for smoke one-step PSO"
        )
        return out, warnings

    samples = list(doc.get("samples", []))
    if not samples:
        raise CalibrationDatasetSchemaError("CHTD dataset has no samples")
    return samples, warnings


def ekf_samples_from_dataset(
    doc: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize EKF prepared JSON (v2 or demo) for PSO fusion objective."""
    schema = str(doc.get("schema", ""))
    warnings: List[str] = []
    out: List[Dict[str, Any]] = []

    if schema not in (EKF_FORMAL_SCHEMA, _DEMO_EKF_SCHEMA):
        raise CalibrationDatasetSchemaError(
            f"ekf_samples_from_dataset: unsupported schema {schema!r}"
        )

    for idx, sample in enumerate(doc.get("samples", [])):
        if not isinstance(sample, dict):
            warnings.append(f"samples[{idx}] skipped: not an object")
            continue
        try:
            out.append(normalize_ekf_sample(sample))
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationDatasetSchemaError(
                f"samples[{idx}]: EKF sample invalid: {exc}"
            ) from exc

    if not out:
        raise CalibrationDatasetSchemaError("EKF dataset has no usable samples")
    if schema == EKF_FORMAL_SCHEMA:
        warnings.append(
            "formal v2: MSE of fuse_head_temperature vs measured_head_air_temp_c"
        )
    else:
        warnings.append("demo_only: synthetic EKF targets from built-in truth params")
    return out, warnings
