"""First-phase CHTD HVAC scale calibration dataset adapter (prepared JSON schema).

Bridges bench/vehicle time-series into ``chtd_hvac_scale_adapter`` PSO objectives.
Does not run PSO, modify ``thermal.py``, golden, or runtime wiring.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.calibration.chtd_hvac_scale_adapter import (
    apply_chtd_hvac_scale_params,
    build_chtd_hvac_scale_parameter_specs,
)
from hvac_sim.calibration.schema import CalibrationSchemaError, ParameterSpec
from hvac_sim.chtd.bus_index import N_U, N_X_STATES, U_INDEX, X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.physical_params import build_physical_capacity_params
from hvac_sim.chtd.thermal import compute_chtd_delta

PREPARED_SCHEMA = "prepared_chtd_first_phase_calibration_v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_MD = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_FIRST_PHASE_DATASET_SCHEMA.md"
)
DEFAULT_SAMPLE_JSON = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_FIRST_PHASE_SAMPLE_DATASET.json"
)

SCENARIO_TYPES = (
    "cold_start_heating",
    "steady_cooling",
    "steady_heating",
    "heat_soak_cooldown",
    "solar_load",
)

# P0 (required after normalize) measured zone keys — short names in JSON.
P0_MEASURED_ZONE_KEYS = (
    "HeadFd",
    "HeadFp",
    "FeetFd",
    "FeetFp",
    "CabinFd",
    "CabinFp",
)

OPTIONAL_MEASURED_ZONE_KEYS = (
    "WinFd",
    "WinFp",
    "RoofTemp",
    "ConsoleTemp",
    "CabinFrntTemp",
    "HeadSd",
    "HeadSp",
    "FeetSd",
    "FeetSp",
    "CabinSd",
    "CabinSp",
)

ZONE_KEY_TO_X_NAME: Dict[str, str] = {
    "HeadFd": "HeadTempFd",
    "HeadFp": "HeadTempFp",
    "HeadSd": "HeadTempSd",
    "HeadSp": "HeadTempSp",
    "FeetFd": "FeetTempFd",
    "FeetFp": "FeetTempFp",
    "FeetSd": "FeetTempSd",
    "FeetSp": "FeetTempSp",
    "CabinFd": "CabinTempFd",
    "CabinFp": "CabinTempFp",
    "CabinSd": "CabinTempSd",
    "CabinSp": "CabinTempSp",
    "WinFd": "WinTempFd",
    "WinFp": "WinTempFp",
    "RoofTemp": "RoofTemp",
    "ConsoleTemp": "ConsoleTemp",
    "CabinFrntTemp": "CabinFrntTemp",
}

P0_TMA_FIELDS = (
    "FrntFdvTma",
    "FrntFdfTma",
    "FrntFpvTma",
    "FrntFpfTma",
    "RearSdvTma",
    "RearSdfTma",
    "RearSpvTma",
    "RearSpfTma",
    "FrntDefTmaEst",
)

# AFE G1/G2 front + 2nd-row duct flows (m³/h); third row excluded from first phase.
P0_FLOW_FIELDS = (
    "FrntFdvFlow",
    "FrntFdfFlow",
    "FrntFpvFlow",
    "FrntFpfFlow",
    "FrntSdvFlow",
    "FrntSdfFlow",
    "RearSdvFlow",
    "RearSdfFlow",
    "FrntSpvFlow",
    "FrntSpfFlow",
    "RearSpvFlow",
    "RearSpfFlow",
)

P0_QUALITY_FLAG_KEYS = (
    "has_afe_calibrated_flow",
    "has_tma_sensor",
    "has_zone_thermocouples",
    "has_solar",
    "steady_state_segment",
    "exclude_from_fit",
)

ROOT_P0_KEYS = ("schema", "vehicle", "source", "sample_rate_hz", "cases")
CASE_P0_KEYS = ("case_id", "scenario_type", "timestamps_s", "ambient", "hvac_inputs", "measured_zone_temps", "quality_flags")
AMBIENT_P0_KEYS = ("amb_t_c", "vehicle_speed_kph")


class CHTDFirstPhaseDatasetError(CalibrationSchemaError):
    """Invalid prepared first-phase CHTD calibration dataset."""


def _path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CHTDFirstPhaseDatasetError(f"{path}: must be an object")
    return value


def _require_list(value: Any, *, path: str, min_len: int = 1) -> List[Any]:
    if not isinstance(value, list):
        raise CHTDFirstPhaseDatasetError(f"{path}: must be a list")
    if len(value) < min_len:
        raise CHTDFirstPhaseDatasetError(
            f"{path}: must contain at least {min_len} element(s), got {len(value)}"
        )
    return value


def _as_float_series(value: Any, *, path: str, length: Optional[int] = None) -> List[float]:
    if isinstance(value, (int, float)):
        series = [float(value)]
    elif isinstance(value, list):
        if not value:
            raise CHTDFirstPhaseDatasetError(f"{path}: empty numeric series")
        try:
            series = [float(v) for v in value]
        except (TypeError, ValueError) as exc:
            raise CHTDFirstPhaseDatasetError(f"{path}: non-numeric value in series") from exc
    else:
        raise CHTDFirstPhaseDatasetError(f"{path}: must be numeric or list of numerics")
    if length is not None and len(series) != length:
        raise CHTDFirstPhaseDatasetError(
            f"{path}: length {len(series)} != expected {length}"
        )
    return series


def _require_numeric_series(
    container: Mapping[str, Any],
    key: str,
    *,
    path: str,
    length: Optional[int] = None,
) -> List[float]:
    if key not in container:
        raise CHTDFirstPhaseDatasetError(f"{path}: missing required field {key!r}")
    return _as_float_series(container[key], path=_path(path, key), length=length)


def validate_chtd_first_phase_dataset(data: Mapping[str, Any]) -> None:
    """Validate P0 fields; raise ``CHTDFirstPhaseDatasetError`` with clear paths."""
    root = _require_mapping(data, path="root")
    missing_root = [k for k in ROOT_P0_KEYS if k not in root]
    if missing_root:
        raise CHTDFirstPhaseDatasetError(
            f"root: missing required field(s): {', '.join(missing_root)}"
        )
    if str(root["schema"]) != PREPARED_SCHEMA:
        raise CHTDFirstPhaseDatasetError(
            f"root.schema must be {PREPARED_SCHEMA!r}, got {root['schema']!r}"
        )
    if not str(root["vehicle"]).strip():
        raise CHTDFirstPhaseDatasetError("root.vehicle: required non-empty string")
    if not str(root["source"]).strip():
        raise CHTDFirstPhaseDatasetError("root.source: required non-empty string")
    try:
        float(root["sample_rate_hz"])
    except (TypeError, ValueError) as exc:
        raise CHTDFirstPhaseDatasetError("root.sample_rate_hz: must be numeric") from exc

    cases = _require_list(root.get("cases"), path="root.cases", min_len=1)
    for idx, case in enumerate(cases):
        cp = _path("root.cases", f"[{idx}]")
        c = _require_mapping(case, path=cp)
        missing_case = [k for k in CASE_P0_KEYS if k not in c]
        if missing_case:
            raise CHTDFirstPhaseDatasetError(
                f"{cp}: missing required field(s): {', '.join(missing_case)}"
            )
        if c["scenario_type"] not in SCENARIO_TYPES:
            raise CHTDFirstPhaseDatasetError(
                f"{cp}.scenario_type: must be one of {SCENARIO_TYPES}, got {c['scenario_type']!r}"
            )
        n = len(_require_list(c["timestamps_s"], path=_path(cp, "timestamps_s"), min_len=2))
        amb = _require_mapping(c["ambient"], path=_path(cp, "ambient"))
        missing_amb = [k for k in AMBIENT_P0_KEYS if k not in amb]
        if missing_amb:
            raise CHTDFirstPhaseDatasetError(
                f"{cp}.ambient: missing required field(s): {', '.join(missing_amb)}"
            )
        _require_numeric_series(amb, "amb_t_c", path=_path(cp, "ambient"), length=n)
        _require_numeric_series(amb, "vehicle_speed_kph", path=_path(cp, "ambient"), length=n)

        hvac = _require_mapping(c["hvac_inputs"], path=_path(cp, "hvac_inputs"))
        for tma_key in P0_TMA_FIELDS:
            _require_numeric_series(hvac, tma_key, path=_path(cp, "hvac_inputs"), length=n)
        for flow_key in P0_FLOW_FIELDS:
            _require_numeric_series(hvac, flow_key, path=_path(cp, "hvac_inputs"), length=n)

        measured = _require_mapping(c["measured_zone_temps"], path=_path(cp, "measured_zone_temps"))
        for zkey in P0_MEASURED_ZONE_KEYS:
            _require_numeric_series(measured, zkey, path=_path(cp, "measured_zone_temps"), length=n)

        flags = _require_mapping(c["quality_flags"], path=_path(cp, "quality_flags"))
        missing_flags = [k for k in P0_QUALITY_FLAG_KEYS if k not in flags]
        if missing_flags:
            raise CHTDFirstPhaseDatasetError(
                f"{cp}.quality_flags: missing required field(s): {', '.join(missing_flags)}"
            )

        if "x0" in c:
            x0 = c["x0"]
            if not isinstance(x0, list) or len(x0) != N_X_STATES:
                raise CHTDFirstPhaseDatasetError(
                    f"{cp}.x0: must be a list of length {N_X_STATES}"
                )


def _series_or_fill(
    container: Mapping[str, Any],
    key: str,
    *,
    length: int,
    fill_value: float,
    fallbacks: List[str],
    path: str,
) -> List[float]:
    if key in container and container[key] is not None:
        raw = container[key]
        if isinstance(raw, list):
            if len(raw) == length:
                return [float(v) for v in raw]
            if len(raw) == 1:
                fallbacks.append(f"{path}.{key}: broadcast scalar to length {length}")
                return [float(raw[0])] * length
        else:
            fallbacks.append(f"{path}.{key}: broadcast scalar to length {length}")
            return [float(raw)] * length
    fallbacks.append(f"{path}.{key}: filled with {fill_value}")
    return [float(fill_value)] * length


def _trim_or_pad_series(series: List[float], length: int, *, path: str, fallbacks: List[str]) -> List[float]:
    if len(series) == length:
        return series
    if len(series) > length:
        fallbacks.append(f"{path}: trimmed from {len(series)} to {length}")
        return series[:length]
    fallbacks.append(f"{path}: padded from {len(series)} to {length}")
    return series + [series[-1]] * (length - len(series))


def normalize_chtd_first_phase_dataset(
    data: Mapping[str, Any],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Normalize lengths/units; record every fallback in ``normalization_log``."""
    if strict:
        validate_chtd_first_phase_dataset(data)

    fallbacks: List[str] = []
    root = copy.deepcopy(dict(data))
    root["schema"] = PREPARED_SCHEMA
    root.setdefault("vehicle", "unknown")
    root.setdefault("source", "unspecified")
    root.setdefault("sample_rate_hz", 1.0)

    cases_out: List[Dict[str, Any]] = []
    for idx, case in enumerate(_require_list(root.get("cases", []), path="root.cases", min_len=1)):
        cp = _path("root.cases", f"[{idx}]")
        c = dict(case)
        n = len(_require_list(c.get("timestamps_s", []), path=_path(cp, "timestamps_s"), min_len=1))

        amb = dict(c.get("ambient") or {})
        amb_t = _series_or_fill(amb, "amb_t_c", length=n, fill_value=20.0, fallbacks=fallbacks, path=_path(cp, "ambient"))
        if "raw_amb_t_c" not in amb or amb["raw_amb_t_c"] is None:
            fallbacks.append(f"{cp}.ambient.raw_amb_t_c: defaulted to amb_t_c")
            amb["raw_amb_t_c"] = list(amb_t)
        else:
            amb["raw_amb_t_c"] = _trim_or_pad_series(
                _as_float_series(amb["raw_amb_t_c"], path=_path(cp, "ambient.raw_amb_t_c")),
                n,
                path=_path(cp, "ambient.raw_amb_t_c"),
                fallbacks=fallbacks,
            )
        amb["amb_t_c"] = _trim_or_pad_series(amb_t, n, path=_path(cp, "ambient.amb_t_c"), fallbacks=fallbacks)
        spd_kph = _series_or_fill(
            amb, "vehicle_speed_kph", length=n, fill_value=0.0, fallbacks=fallbacks, path=_path(cp, "ambient")
        )
        amb["vehicle_speed_kph"] = _trim_or_pad_series(
            spd_kph, n, path=_path(cp, "ambient.vehicle_speed_kph"), fallbacks=fallbacks
        )
        fallbacks.append(f"{cp}.ambient.vehicle_speed_kph: stored as kph; converted to m/s at u-build")
        amb["solar_driver_w_m2"] = _trim_or_pad_series(
            _series_or_fill(amb, "solar_driver_w_m2", length=n, fill_value=0.0, fallbacks=fallbacks, path=_path(cp, "ambient")),
            n,
            path=_path(cp, "ambient.solar_driver_w_m2"),
            fallbacks=fallbacks,
        )
        amb["solar_passenger_w_m2"] = _trim_or_pad_series(
            _series_or_fill(
                amb, "solar_passenger_w_m2", length=n, fill_value=0.0, fallbacks=fallbacks, path=_path(cp, "ambient")
            ),
            n,
            path=_path(cp, "ambient.solar_passenger_w_m2"),
            fallbacks=fallbacks,
        )
        c["ambient"] = amb

        hvac = dict(c.get("hvac_inputs") or {})
        hvac_out: Dict[str, List[float]] = {}
        for tma_key in P0_TMA_FIELDS:
            series = _series_or_fill(
                hvac,
                tma_key,
                length=n,
                fill_value=amb_t[0],
                fallbacks=fallbacks,
                path=_path(cp, "hvac_inputs"),
            )
            hvac_out[tma_key] = _trim_or_pad_series(
                series, n, path=_path(cp, f"hvac_inputs.{tma_key}"), fallbacks=fallbacks
            )
        for flow_key in P0_FLOW_FIELDS:
            series = _series_or_fill(
                hvac, flow_key, length=n, fill_value=0.0, fallbacks=fallbacks, path=_path(cp, "hvac_inputs")
            )
            hvac_out[flow_key] = _trim_or_pad_series(
                series, n, path=_path(cp, f"hvac_inputs.{flow_key}"), fallbacks=fallbacks
            )
        c["hvac_inputs"] = hvac_out

        measured = dict(c.get("measured_zone_temps") or {})
        measured_out: Dict[str, List[float]] = {}
        for zkey in P0_MEASURED_ZONE_KEYS + OPTIONAL_MEASURED_ZONE_KEYS:
            if zkey in measured and measured[zkey] is not None:
                measured_out[zkey] = _trim_or_pad_series(
                    _as_float_series(measured[zkey], path=_path(cp, f"measured_zone_temps.{zkey}")),
                    n,
                    path=_path(cp, f"measured_zone_temps.{zkey}"),
                    fallbacks=fallbacks,
                )
        for zkey in P0_MEASURED_ZONE_KEYS:
            if zkey not in measured_out:
                fallbacks.append(f"{cp}.measured_zone_temps.{zkey}: filled with amb_t_c")
                measured_out[zkey] = list(amb_t)
        c["measured_zone_temps"] = measured_out

        flags = dict(c.get("quality_flags") or {})
        for fk in P0_QUALITY_FLAG_KEYS:
            if fk not in flags:
                fallbacks.append(f"{cp}.quality_flags.{fk}: defaulted false")
                flags[fk] = False
        c["quality_flags"] = flags

        if "x0" not in c or c["x0"] is None:
            fallbacks.append(f"{cp}.x0: synthesized from measured zones + amb fill")
            c["x0"] = _synthesize_x0_from_measured(measured_out, amb_t[0])
        c["timestamps_s"] = _trim_or_pad_series(
            [float(t) for t in c["timestamps_s"]],
            n,
            path=_path(cp, "timestamps_s"),
            fallbacks=fallbacks,
        )
        cases_out.append(c)

    root["cases"] = cases_out
    root["normalization_log"] = fallbacks
    root["normalized_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    validate_chtd_first_phase_dataset(root)
    return root


def _synthesize_x0_from_measured(
    measured: Mapping[str, Sequence[float]],
    fill_c: float,
) -> List[float]:
    x = np.full(N_X_STATES, float(fill_c))
    for zkey, xname in ZONE_KEY_TO_X_NAME.items():
        if zkey in measured and measured[zkey]:
            x[X_INDEX[xname]] = float(measured[zkey][0])
    return x.tolist()


def _build_u_at_step(case: Mapping[str, Any], step: int) -> np.ndarray:
    amb = case["ambient"]
    hvac = case["hvac_inputs"]
    u = np.zeros(N_U, dtype=float)
    amb_t = float(amb["amb_t_c"][step])
    u[U_INDEX["AmbT"]] = amb_t
    u[U_INDEX["RawAmbT"]] = float(amb.get("raw_amb_t_c", amb["amb_t_c"])[step])
    u[U_INDEX["VehSpd"]] = float(amb["vehicle_speed_kph"][step]) / 3.6
    u[U_INDEX["SolarFd"]] = float(amb.get("solar_driver_w_m2", [0.0])[step])
    u[U_INDEX["SolarFp"]] = float(amb.get("solar_passenger_w_m2", [0.0])[step])
    for key in P0_TMA_FIELDS:
        if key in U_INDEX and key in hvac:
            u[U_INDEX[key]] = float(hvac[key][step])
    for key in P0_FLOW_FIELDS:
        if key in U_INDEX and key in hvac:
            u[U_INDEX[key]] = float(hvac[key][step])
    return u


def _state_from_measured(
    case: Mapping[str, Any],
    step: int,
    *,
    fill_c: Optional[float] = None,
) -> np.ndarray:
    if step == 0 and case.get("x0") is not None:
        return np.asarray(case["x0"], dtype=float)
    fill = float(fill_c if fill_c is not None else case["ambient"]["amb_t_c"][step])
    x = np.full(N_X_STATES, fill)
    measured = case["measured_zone_temps"]
    for zkey, xname in ZONE_KEY_TO_X_NAME.items():
        if zkey in measured:
            x[X_INDEX[xname]] = float(measured[zkey][step])
    return x


def _resolve_target_zones(target_zones: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not target_zones:
        return P0_MEASURED_ZONE_KEYS
    out: List[str] = []
    for z in target_zones:
        zstr = str(z)
        if zstr in ZONE_KEY_TO_X_NAME or zstr in X_INDEX:
            out.append(zstr)
        else:
            raise CHTDFirstPhaseDatasetError(f"unknown target zone {zstr!r}")
    return tuple(out)


def _zone_to_x_index(zone: str) -> int:
    if zone in X_INDEX:
        return X_INDEX[zone]
    if zone in ZONE_KEY_TO_X_NAME:
        return X_INDEX[ZONE_KEY_TO_X_NAME[zone]]
    raise CHTDFirstPhaseDatasetError(f"cannot map zone {zone!r} to state index")


def evaluate_chtd_scale_objective(
    scale_values: Union[np.ndarray, Mapping[str, float], Sequence[float]],
    dataset: Mapping[str, Any],
    parameter_specs: Sequence[ParameterSpec],
    *,
    base_params: Optional[CHTDParams] = None,
    step_mode: str = "one_step",
    target_zones: Optional[Sequence[str]] = None,
    include_optional_scales: bool = True,
) -> Dict[str, Any]:
    """Compute MSE loss and per-zone residual summary for scale vector."""
    if base_params is None:
        base_params, _ = build_physical_capacity_params()
    params = apply_chtd_hvac_scale_params(
        base_params,
        scale_values,
        include_optional=include_optional_scales,
    )
    zones = _resolve_target_zones(target_zones)
    sq_errors: List[float] = []
    per_zone: Dict[str, List[float]] = {_zone_short_name(z): [] for z in zones}
    n_terms = 0

    for case in dataset["cases"]:
        flags = case.get("quality_flags", {})
        if flags.get("exclude_from_fit"):
            continue
        n_steps = len(case["timestamps_s"])
        if step_mode == "one_step":
            step_range = range(n_steps - 1)
        elif step_mode == "multi_step":
            step_range = range(n_steps - 1)
        else:
            raise CHTDFirstPhaseDatasetError(f"unknown step_mode {step_mode!r}")

        x = _state_from_measured(case, 0)
        for k in step_range:
            if step_mode == "one_step":
                x = _state_from_measured(case, k)
            u = _build_u_at_step(case, k)
            delta = compute_chtd_delta(x, u, params, AS_FOUND)
            x_pred = x + delta
            for z in zones:
                xi = _zone_to_x_index(z)
                meas_key = _zone_short_name(z)
                target = float(case["measured_zone_temps"][meas_key][k + 1])
                pred = float(x_pred[xi])
                err = pred - target
                sq_errors.append(err * err)
                per_zone[meas_key].append(err)
                n_terms += 1
            if step_mode == "multi_step":
                x = x_pred

    mse = float(np.mean(sq_errors)) if sq_errors else float("inf")
    summary = {
        zone: {
            "n": len(errs),
            "mse": float(np.mean(np.square(errs))) if errs else None,
            "mean_residual_c": float(np.mean(errs)) if errs else None,
            "max_abs_residual_c": float(np.max(np.abs(errs))) if errs else None,
        }
        for zone, errs in per_zone.items()
    }
    return {
        "loss": mse,
        "mse": mse,
        "n_terms": n_terms,
        "step_mode": step_mode,
        "target_zones": list(zones),
        "per_zone_residual_summary": summary,
    }


def _zone_short_name(zone: str) -> str:
    if zone in ZONE_KEY_TO_X_NAME:
        return zone
    inv = {v: k for k, v in ZONE_KEY_TO_X_NAME.items()}
    if zone in inv:
        return inv[zone]
    raise CHTDFirstPhaseDatasetError(f"unknown zone {zone!r}")


def build_chtd_scale_objective_from_dataset(
    dataset: Mapping[str, Any],
    parameter_specs: Sequence[ParameterSpec],
    *,
    base_params: Optional[CHTDParams] = None,
    step_mode: str = "one_step",
    target_zones: Optional[Sequence[str]] = None,
    include_optional_scales: bool = True,
) -> Tuple[Callable[[np.ndarray], float], Dict[str, Any]]:
    """Return PSO-ready ``f(scale_vector) -> loss`` plus static metadata."""
    names = [spec.name for spec in parameter_specs]

    def objective(vec: np.ndarray) -> float:
        vec = np.asarray(vec, dtype=float).reshape(-1)
        if vec.shape[0] != len(names):
            raise ValueError(f"expected {len(names)} scales, got {vec.shape[0]}")
        mapping = {name: float(vec[i]) for i, name in enumerate(names)}
        result = evaluate_chtd_scale_objective(
            mapping,
            dataset,
            parameter_specs,
            base_params=base_params,
            step_mode=step_mode,
            target_zones=target_zones,
            include_optional_scales=include_optional_scales,
        )
        return float(result["loss"])

    meta = {
        "parameter_names": names,
        "step_mode": step_mode,
        "target_zones": list(_resolve_target_zones(target_zones)),
        "schema": dataset.get("schema"),
        "n_cases": len(dataset.get("cases", [])),
        "classification": "smoke_objective_from_prepared_dataset",
    }
    return objective, meta


def build_sample_first_phase_dataset() -> Dict[str, Any]:
    """Synthetic demo dataset (not vehicle sign-off)."""
    n = 4
    t = [float(i) for i in range(n)]
    amb = [-5.0, -4.0, -2.0, 0.0]
    feet = [-5.0, -2.5, 1.0, 3.5]
    head = [-5.0, -3.0, 0.5, 2.0]
    cabin = [-5.0, -3.5, 0.0, 1.5]
    return {
        "schema": PREPARED_SCHEMA,
        "vehicle": "synthetic_m8_demo",
        "source": "generated_sample_not_vehicle_data",
        "sample_rate_hz": 1.0,
        "cases": [
            {
                "case_id": "demo_cold_start_heating",
                "scenario_type": "cold_start_heating",
                "timestamps_s": t,
                "ambient": {
                    "amb_t_c": amb,
                    "vehicle_speed_kph": [0.0] * n,
                },
                "hvac_inputs": {
                    "FrntFdvTma": [35.0] * n,
                    "FrntFdfTma": [42.0] * n,
                    "FrntFpvTma": [35.0] * n,
                    "FrntFpfTma": [42.0] * n,
                    "RearSdvTma": [34.0] * n,
                    "RearSdfTma": [34.0] * n,
                    "RearSpvTma": [34.0] * n,
                    "RearSpfTma": [34.0] * n,
                    "FrntDefTmaEst": [30.0] * n,
                    **{fk: [120.0 if "Fdf" in fk or "Fpf" in fk else 80.0] * n for fk in P0_FLOW_FIELDS},
                },
                "measured_zone_temps": {
                    "HeadFd": head,
                    "HeadFp": head,
                    "FeetFd": feet,
                    "FeetFp": feet,
                    "CabinFd": cabin,
                    "CabinFp": cabin,
                },
                "quality_flags": {
                    "has_afe_calibrated_flow": False,
                    "has_tma_sensor": False,
                    "has_zone_thermocouples": False,
                    "has_solar": False,
                    "steady_state_segment": False,
                    "exclude_from_fit": False,
                },
            }
        ],
    }


def build_schema_markdown() -> str:
    lines = [
        "# CHTD First-Phase Calibration Dataset Schema",
        "",
        f"Schema id: `{PREPARED_SCHEMA}`",
        "",
        "## Root (P0)",
        "",
        "| field | type | required |",
        "|-------|------|----------|",
        "| schema | string | yes |",
        "| vehicle | string | yes |",
        "| source | string | yes |",
        "| sample_rate_hz | number | yes |",
        "| cases | array | yes (≥1) |",
        "",
        "## Case (P0)",
        "",
        "| field | notes |",
        "|-------|-------|",
        "| case_id | unique string |",
        f"| scenario_type | one of {', '.join(SCENARIO_TYPES)} |",
        "| timestamps_s | ≥2 samples, uniform step ≈ 1/sample_rate_hz |",
        "| x0 | optional float[28] |",
        "| ambient.amb_t_c | °C series |",
        "| ambient.vehicle_speed_kph | kph (converted to m/s internally) |",
        "| ambient.raw_amb_t_c | optional |",
        "| ambient.solar_* | optional W/m² |",
        f"| hvac_inputs TMA | {', '.join(P0_TMA_FIELDS)} |",
        f"| hvac_inputs flow | {', '.join(P0_FLOW_FIELDS)} (m³/h) |",
        f"| measured_zone_temps P0 | {', '.join(P0_MEASURED_ZONE_KEYS)} |",
        f"| measured_zone_temps optional | {', '.join(OPTIONAL_MEASURED_ZONE_KEYS)} |",
        f"| quality_flags | {', '.join(P0_QUALITY_FLAG_KEYS)} |",
        "",
        "## Normalization",
        "",
        "- All series trimmed/padded to `len(timestamps_s)`",
        "- Missing optional fields filled; entries appended to `normalization_log`",
        "- Third-row ducts/zones excluded from first-phase scale adapter",
        "",
        "## Objective hook",
        "",
        "`build_chtd_scale_objective_from_dataset(dataset, parameter_specs)` returns",
        "a callable compatible with `chtd_hvac_scale_adapter` 8–10 scalar PSO.",
        "",
    ]
    return "\n".join(lines)


def prepare_first_phase_dataset_from_raw(
    raw: Mapping[str, Any],
    *,
    strict_validate: bool = False,
) -> Dict[str, Any]:
    """Normalize raw draft JSON; does not run PSO."""
    draft = copy.deepcopy(dict(raw))
    draft.setdefault("schema", PREPARED_SCHEMA)
    return normalize_chtd_first_phase_dataset(draft, strict=strict_validate)


def write_first_phase_dataset_artifacts(
    dataset: Mapping[str, Any],
    *,
    json_path: Optional[Union[str, Path]] = None,
    schema_md_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    json_out = Path(json_path) if json_path is not None else DEFAULT_SAMPLE_JSON
    md_out = Path(schema_md_path) if schema_md_path is not None else DEFAULT_SCHEMA_MD
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(build_schema_markdown(), encoding="utf-8")
    return {"json": json_out, "schema_md": md_out}
