"""PSO calibration runner for AFE airflow and CHTD thermal targets (CLI backend)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.afe.fan import FanSet
from hvac_sim.afe.flow import FlowInputs, afe_calc
from hvac_sim.afe.params import AFEParams
from hvac_sim.afe.resistance import FlapSet
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.thermal import one_step_chtd
from hvac_sim.ekf.head_temp_filter import fuse_head_temperature
from hvac_sim.config.param_loader import (
    InitialParamsDocument,
    InitialParamRecord,
    bundled_initial_params_path,
    resolve_initial_params,
    runtime_params_from_document,
)

from .data_prep import AIRFLOW_SCHEMA
from .dataset_schemas import (
    AIRFLOW_FORMAL_SCHEMA,
    CHTD_FORMAL_SCHEMA,
    EKF_FORMAL_SCHEMA,
    CalibrationDatasetSchemaError,
    airflow_measurements_from_dataset,
    chtd_one_step_samples_from_dataset,
    ekf_samples_from_dataset,
    validate_dataset_for_pso_target,
)
from .de_calibration import (
    AFEMeasurement,
    CHTDMeasurement,
    _AFE_CALIB_FIELDS,
    _CHTD_CALIB_FIELDS,
    _afe_residuals,
    _chtd_residuals,
)
from .pso import PSOOptions, PSOResult, pso_optimize
from .schema import ParameterSpec, specs_to_bounds, specs_to_names


class CalibrationRunnerError(ValueError):
    """Invalid calibration runner input."""


AFE_DEMO_CALIB_FIELDS = ["EvaRessCo", "FdvFlapRessCo", "FdfFlapRessCo"]
CHTD_DEMO_CALIB_FIELDS = ["HoodMassAtb", "CabinFrntMassAtb", "ConsoleMassAtb"]
EKF_CALIB_FIELDS = [
    "surface_to_air_offset_c",
    "k_air_speed",
    "k_radiation",
    "process_var",
    "measurement_var",
]

_AFE_BOUNDS = (0.1, 5000.0)
_CHTD_BOUNDS = (0.1, 1e6)
_EKF_BOUNDS: Dict[str, Tuple[float, float]] = {
    "surface_to_air_offset_c": (0.0, 8.0),
    "k_air_speed": (-5.0, 5.0),
    "k_radiation": (-1.0, 1.0),
    "process_var": (0.01, 10.0),
    "measurement_var": (0.01, 20.0),
}
_EKF_DEFAULTS: Dict[str, float] = {
    "surface_to_air_offset_c": 2.0,
    "k_air_speed": 0.0,
    "k_radiation": 0.0,
    "process_var": 1.0,
    "measurement_var": 1.0,
}

_DEMO_AIRFLOW_SCHEMA = "afe_calibration_demo_v1"
_DEMO_CHTD_SCHEMA = "chtd_calibration_demo_v1"
_DEMO_EKF_SCHEMA = "ekf_calibration_demo_v1"

# Truth params for synthetic demo targets (smoke PSO only).
_EKF_DEMO_TRUTH: Dict[str, float] = {
    "surface_to_air_offset_c": 2.5,
    "k_air_speed": 0.12,
    "k_radiation": -0.04,
    "process_var": 0.6,
    "measurement_var": 1.8,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _record_map(doc: InitialParamsDocument) -> Dict[str, InitialParamRecord]:
    return {rec.name: rec for rec in doc.parameters}


def _model_default(name: str, model: str) -> float:
    if model == "airflow":
        return float(getattr(AFEParams(), name))
    if model == "chtd":
        return float(getattr(CHTDParams(), name))
    raise CalibrationRunnerError(f"unknown model for default lookup: {model}")


def parameter_specs_for_fields(
    doc: InitialParamsDocument,
    field_names: Sequence[str],
    *,
    target_model: str,
    bounds: Tuple[float, float],
) -> List[ParameterSpec]:
    """Build PSO parameter specs from initial-guess document + model defaults."""
    by_name = _record_map(doc)
    lo, hi = bounds
    specs: List[ParameterSpec] = []
    for name in field_names:
        rec = by_name.get(name)
        if rec is not None:
            initial = float(rec.value)
            source = rec.source or "initial_params"
            unit = rec.unit
            note = rec.note
        else:
            initial = _model_default(name, target_model)
            source = "model_default"
            unit = ""
            note = "not in initial params; using model default"
        specs.append(
            ParameterSpec(
                name=name,
                initial=initial,
                lower=lo,
                upper=hi,
                unit=unit,
                description=note,
                target_model=target_model,
                source=source,
            )
        )
    return specs


def parameter_specs_for_ekf(
    field_names: Optional[Sequence[str]] = None,
) -> List[ParameterSpec]:
    """Build PSO parameter specs for EKF / IR head-temperature fusion."""
    names = list(field_names or EKF_CALIB_FIELDS)
    specs: List[ParameterSpec] = []
    for name in names:
        if name not in _EKF_BOUNDS:
            raise CalibrationRunnerError(f"unknown EKF calibration field: {name}")
        lo, hi = _EKF_BOUNDS[name]
        initial = float(_EKF_DEFAULTS.get(name, (lo + hi) * 0.5))
        specs.append(
            ParameterSpec(
                name=name,
                initial=initial,
                lower=lo,
                upper=hi,
                unit="",
                description="EKF / IR fusion scalar",
                target_model="ekf",
                source="ekf_defaults",
            )
        )
    return specs


def _apply_fields_to_dataclass(
    base: Any,
    field_names: Sequence[str],
    values: np.ndarray,
) -> Any:
    return replace(base, **dict(zip(field_names, values)))


def _afe_base_from_doc(doc: InitialParamsDocument, specs: Sequence[ParameterSpec]) -> AFEParams:
    base = runtime_params_from_document(doc, source="pso_initial_document").afe_params
    updates = {spec.name: spec.initial for spec in specs}
    known = {f.name for f in fields(AFEParams)}
    return replace(base, **{k: v for k, v in updates.items() if k in known})


def _chtd_base_from_doc(doc: InitialParamsDocument, specs: Sequence[ParameterSpec]) -> CHTDParams:
    base = runtime_params_from_document(doc, source="pso_initial_document").chtd_params
    updates = {spec.name: spec.initial for spec in specs}
    known = {f.name for f in fields(CHTDParams)}
    return replace(base, **{k: v for k, v in updates.items() if k in known})


def build_demo_airflow_dataset() -> Dict[str, Any]:
    """Synthetic AFE measurements for smoke PSO (not vehicle calibration)."""
    base = AFEParams()
    flaps = FlapSet()
    fans = FanSet()
    inp = FlowInputs(flaps=flaps, n_F1=0.55, n_F2=0.50, n_S=0.35, dp1=0.0, temp_C=25.0)
    ref = afe_calc(inp, base, fans)
    sample = {
        "flaps": {},
        "fan_set": {},
        "n_F1": inp.n_F1,
        "n_F2": inp.n_F2,
        "n_S": inp.n_S,
        "temp_C": inp.temp_C,
        "dp1": inp.dp1,
        "Qm_measured": ref.Qm,
        "Q1p_measured": ref.Q1_prime,
        "Q2p_measured": ref.Q2_prime,
        "w_Qm": 0.2,
        "w_Q1p": 1.0,
        "w_Q2p": 1.0,
    }
    return {
        "schema": _DEMO_AIRFLOW_SCHEMA,
        "classification": "demo_only",
        "purpose": "afe_resistance_smoke_pso",
        "not_formal_calibration": True,
        "samples": [sample],
        "notes": [
            "Synthetic targets from default AFEParams at fixed fan/flap set",
            "demo_only — not a measured vehicle dataset",
        ],
    }


def build_demo_chtd_dataset() -> Dict[str, Any]:
    """Single-step CHTD smoke target from bundled golden case 1 (read-only)."""
    golden_path = _repo_root() / "simulink_conversion_package" / "golden_cases" / "chtd_cases.json"
    if not golden_path.is_file():
        raise CalibrationRunnerError(f"CHTD demo requires golden file: {golden_path}")
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    case = payload["cases"][0]
    x0 = np.asarray(case["x0"], dtype=float)
    if "t_new" in case:
        t_new = np.asarray(case["t_new"], dtype=float).tolist()
    else:
        t_new = (x0 + np.asarray(case["xStep"], dtype=float)).tolist()
    return {
        "schema": _DEMO_CHTD_SCHEMA,
        "classification": "demo_only",
        "purpose": "chtd_one_step_smoke_pso",
        "not_formal_calibration": True,
        "golden_source": str(golden_path),
        "samples": [
            {
                "x0": case["x0"],
                "u": case["u"],
                "t_new": t_new,
                "name": case.get("name", "case_1"),
            }
        ],
        "notes": [
            "MSE between one_step_chtd(x0,u) and golden t_new for smoke only",
            "demo_only — not a measured thermal time-series calibration",
        ],
    }


def build_demo_ekf_dataset() -> Dict[str, Any]:
    """Synthetic IR + model head temps for EKF smoke PSO (not vehicle calibration)."""
    truth = _EKF_DEMO_TRUTH
    scenarios = [
        (22.0, 35.0, 0.3, 24.0),
        (23.0, 36.5, 0.5, 25.0),
        (21.5, 34.0, 0.2, 23.5),
        (22.8, 35.8, 0.45, 24.2),
    ]
    samples: List[Dict[str, Any]] = []
    for model_c, ir_c, air_speed, mrt in scenarios:
        fused = fuse_head_temperature(
            model_c,
            ir_c,
            0.95,
            process_var=truth["process_var"],
            measurement_var=truth["measurement_var"],
            surface_to_air_offset_c=truth["surface_to_air_offset_c"],
            air_speed_m_s=air_speed,
            mean_radiant_temp_c=mrt,
            k_air_speed=truth["k_air_speed"],
            k_radiation=truth["k_radiation"],
        )
        samples.append(
            {
                "model_head_temp_c": model_c,
                "ir_surface_temp_c": ir_c,
                "measured_head_air_temp_c": fused.fused_temp_c,
                "confidence": 0.95,
                "air_speed_m_s": air_speed,
                "mrt_c": mrt,
                "quality_flags": {"demo": True},
            }
        )
    return {
        "schema": _DEMO_EKF_SCHEMA,
        "classification": "demo_only",
        "purpose": "ekf_ir_fusion_smoke_pso",
        "not_formal_calibration": True,
        "samples": samples,
        "notes": [
            "Synthetic targets from built-in truth EKF/IR params",
            "demo_only — requires IR surface + ventilated thermocouple head air truth",
        ],
    }


def _afe_measurement_from_sample(sample: Mapping[str, Any]) -> AFEMeasurement:
    return AFEMeasurement(
        flaps=FlapSet(),
        fan_set=FanSet(),
        n_F1=float(sample.get("n_F1", 0.5)),
        n_F2=float(sample.get("n_F2", 0.5)),
        n_S=float(sample.get("n_S", 0.3)),
        temp_C=float(sample.get("temp_C", 25.0)),
        dp1=float(sample.get("dp1", 0.0)),
        Qm_measured=sample.get("Qm_measured"),
        Q1_measured=sample.get("Q1_measured"),
        Q1p_measured=sample.get("Q1p_measured"),
        Q2p_measured=sample.get("Q2p_measured"),
        Q_rear_measured=sample.get("Q_rear_measured"),
        w_Qm=float(sample.get("w_Qm", 0.0)),
        w_Q1=float(sample.get("w_Q1", 0.0)),
        w_Q1p=float(sample.get("w_Q1p", 1.0)),
        w_Q2p=float(sample.get("w_Q2p", 1.0)),
        w_rear=float(sample.get("w_rear", 0.0)),
    )


def load_calibration_dataset(
    dataset_path: Optional[Union[str, Path]],
    *,
    target: str,
) -> Tuple[Dict[str, Any], str, List[str]]:
    """Return (dataset document, classification, warnings)."""
    warnings: List[str] = []
    if dataset_path is None:
        if target == "airflow":
            doc = build_demo_airflow_dataset()
            return doc, "demo_only", list(doc.get("notes", []))
        if target == "chtd":
            doc = build_demo_chtd_dataset()
            return doc, "demo_only", list(doc.get("notes", []))
        if target == "ekf":
            doc = build_demo_ekf_dataset()
            return doc, "demo_only", list(doc.get("notes", []))
        raise CalibrationRunnerError(f"unsupported target: {target}")

    path = Path(dataset_path)
    if not path.is_file():
        raise CalibrationRunnerError(f"dataset not found: {path}")

    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise CalibrationRunnerError("dataset root must be a JSON object")

    schema = str(doc.get("schema", ""))
    try:
        if target == "airflow":
            if schema == _DEMO_AIRFLOW_SCHEMA:
                return doc, "demo_only", list(doc.get("notes", []))
            classification = validate_dataset_for_pso_target(doc, target="airflow")
            return doc, classification, warnings

        if target == "chtd":
            if schema == _DEMO_CHTD_SCHEMA:
                return doc, "demo_only", list(doc.get("notes", []))
            if schema == "chtd_cases" or "cases" in doc:
                warnings.append("using golden-style CHTD cases as demo calibration input")
                cases = doc.get("cases", [])
                if not cases:
                    raise CalibrationRunnerError("CHTD dataset has no cases")
                wrapped_samples = []
                for c in cases[:3]:
                    x0_arr = np.asarray(c["x0"], dtype=float)
                    if "t_new" in c:
                        t_new = np.asarray(c["t_new"], dtype=float).tolist()
                    else:
                        t_new = (x0_arr + np.asarray(c["xStep"], dtype=float)).tolist()
                    wrapped_samples.append(
                        {
                            "x0": c["x0"],
                            "u": c["u"],
                            "t_new": t_new,
                            "name": c.get("name"),
                        }
                    )
                return (
                    {
                        "schema": "chtd_golden_wrapper_v1",
                        "classification": "demo_only",
                        "samples": wrapped_samples,
                    },
                    "demo_only",
                    warnings,
                )
            classification = validate_dataset_for_pso_target(doc, target="chtd")
            return doc, classification, warnings

        if target == "ekf":
            if schema == _DEMO_EKF_SCHEMA:
                return doc, "demo_only", list(doc.get("notes", []))
            classification = validate_dataset_for_pso_target(doc, target="ekf")
            return doc, classification, warnings
    except CalibrationDatasetSchemaError as exc:
        raise CalibrationRunnerError(str(exc)) from exc

    raise CalibrationRunnerError(f"unsupported target: {target}")


def _chtd_one_step_demo_loss(
    params_vec: np.ndarray,
    samples: Sequence[Mapping[str, Any]],
    base_params: CHTDParams,
    calib_fields: List[str],
) -> float:
    p = _apply_fields_to_dataclass(base_params, calib_fields, params_vec)
    total = 0.0
    for sample in samples:
        x0 = np.asarray(sample["x0"], dtype=float)
        u = np.asarray(sample["u"], dtype=float)
        target = np.asarray(sample["t_new"], dtype=float)
        try:
            pred = one_step_chtd(x0, u, p)
        except Exception:
            return 1e10
        diff = pred - target
        total += float(np.sum(diff**2))
    return total


def _ekf_params_dict(
    field_names: Sequence[str],
    values: np.ndarray,
) -> Dict[str, float]:
    return {name: float(v) for name, v in zip(field_names, values)}


def _ekf_fusion_mse(
    params_vec: np.ndarray,
    samples: Sequence[Mapping[str, Any]],
    field_names: Sequence[str],
) -> float:
    params = _ekf_params_dict(field_names, params_vec)
    total = 0.0
    count = 0
    for sample in samples:
        try:
            result = fuse_head_temperature(
                float(sample["model_temp_c"]),
                sample.get("ir_surface_temp_c"),
                sample.get("confidence"),
                process_var=params["process_var"],
                measurement_var=params["measurement_var"],
                surface_to_air_offset_c=params["surface_to_air_offset_c"],
                air_speed_m_s=float(sample.get("air_speed_m_s") or 0.0),
                mean_radiant_temp_c=sample.get("mrt_c"),
                k_air_speed=params["k_air_speed"],
                k_radiation=params["k_radiation"],
            )
            if not result.valid:
                return 1e10
            target = float(sample["measured_head_air_temp_c"])
            diff = float(result.fused_temp_c) - target
            total += diff * diff
            count += 1
        except Exception:
            return 1e10
    if count == 0:
        return 1e10
    return total / count


def run_airflow_pso(
    *,
    dataset: Mapping[str, Any],
    initial_doc: InitialParamsDocument,
    calib_fields: Optional[Sequence[str]] = None,
    pso_options: PSOOptions,
) -> Tuple[PSOResult, List[ParameterSpec], Dict[str, float], List[str]]:
    fields_list = list(calib_fields or _AFE_CALIB_FIELDS)
    specs = parameter_specs_for_fields(
        initial_doc,
        fields_list,
        target_model="airflow",
        bounds=_AFE_BOUNDS,
    )
    base = _afe_base_from_doc(initial_doc, specs)
    warnings: List[str] = []

    schema = str(dataset.get("schema", ""))
    if schema == _DEMO_AIRFLOW_SCHEMA:
        measurements = [_afe_measurement_from_sample(s) for s in dataset["samples"]]
    elif schema in (AIRFLOW_SCHEMA, AIRFLOW_FORMAL_SCHEMA):
        measurements, parse_warnings = airflow_measurements_from_dataset(dataset)
        warnings.extend(parse_warnings)
    else:
        samples = dataset.get("samples", [])
        if not samples:
            raise CalibrationRunnerError("airflow dataset has no samples")
        measurements = [_afe_measurement_from_sample(s) for s in samples]

    names = specs_to_names(specs)
    objective = lambda v: _afe_residuals(
        np.asarray(v, dtype=float),
        measurements,
        base,
        names,
    )
    result = pso_optimize(objective, specs_to_bounds(specs), pso_options)
    best_params = {
        name: float(value) for name, value in zip(names, result.best_x)
    }
    return result, specs, best_params, warnings


def run_chtd_pso(
    *,
    dataset: Mapping[str, Any],
    initial_doc: InitialParamsDocument,
    calib_fields: Optional[Sequence[str]] = None,
    pso_options: PSOOptions,
) -> Tuple[PSOResult, List[ParameterSpec], Dict[str, float], List[str]]:
    fields_list = list(calib_fields or _CHTD_CALIB_FIELDS)
    specs = parameter_specs_for_fields(
        initial_doc,
        fields_list,
        target_model="chtd",
        bounds=_CHTD_BOUNDS,
    )
    base = _chtd_base_from_doc(initial_doc, specs)
    names = specs_to_names(specs)
    warnings: List[str] = []
    schema = str(dataset.get("schema", ""))
    if schema == CHTD_FORMAL_SCHEMA:
        samples, norm_warnings = chtd_one_step_samples_from_dataset(dataset)
        warnings.extend(norm_warnings)
    else:
        samples = list(dataset.get("samples", []))
    if not samples:
        raise CalibrationRunnerError("CHTD dataset has no samples")

    first = samples[0]
    use_one_step = schema in (_DEMO_CHTD_SCHEMA, "chtd_golden_wrapper_v1", CHTD_FORMAL_SCHEMA) or (
        isinstance(first, dict) and "t_new" in first
    )

    if use_one_step:
        warnings.append("CHTD smoke objective: one_step vs golden t_new (demo only)")
        objective = lambda v: _chtd_one_step_demo_loss(
            np.asarray(v, dtype=float),
            samples,
            base,
            names,
        )
    elif isinstance(first, dict) and "t_vec" in first and "T_measured" in first:
        from hvac_sim.chtd.thermal import CHTDInputs

        warnings.append("CHTD time-series objective via simulate_chtd (legacy API)")
        measurements: List[CHTDMeasurement] = []
        for sample in samples:
            t_vec = np.asarray(sample["t_vec"], dtype=float)
            T_meas = np.asarray(sample["T_measured"], dtype=float)
            inputs_list = [CHTDInputs(**row) for row in sample["inputs_list"]]
            measurements.append(
                CHTDMeasurement(
                    t_vec=t_vec,
                    inputs_list=inputs_list,
                    T_measured=T_meas,
                    T0=float(sample.get("T0", 20.0)),
                )
            )
        objective = lambda v: _chtd_residuals(
            np.asarray(v, dtype=float),
            measurements,
            base,
            names,
        )
    else:
        raise CalibrationRunnerError(
            "CHTD samples must include t_new (one-step demo) or "
            "t_vec/T_measured/inputs_list (legacy series)"
        )

    result = pso_optimize(objective, specs_to_bounds(specs), pso_options)
    best_params = {
        name: float(value) for name, value in zip(names, result.best_x)
    }
    return result, specs, best_params, warnings


def run_ekf_pso(
    *,
    dataset: Mapping[str, Any],
    calib_fields: Optional[Sequence[str]] = None,
    pso_options: PSOOptions,
) -> Tuple[PSOResult, List[ParameterSpec], Dict[str, float], List[str]]:
    fields_list = list(calib_fields or EKF_CALIB_FIELDS)
    specs = parameter_specs_for_ekf(fields_list)
    warnings: List[str] = []
    schema = str(dataset.get("schema", ""))
    if schema in (_DEMO_EKF_SCHEMA, EKF_FORMAL_SCHEMA):
        samples, parse_warnings = ekf_samples_from_dataset(dataset)
        warnings.extend(parse_warnings)
    else:
        raise CalibrationRunnerError(
            f"EKF dataset schema must be {_DEMO_EKF_SCHEMA!r} or {EKF_FORMAL_SCHEMA!r}"
        )

    names = specs_to_names(specs)
    objective = lambda v: _ekf_fusion_mse(
        np.asarray(v, dtype=float),
        samples,
        names,
    )
    result = pso_optimize(objective, specs_to_bounds(specs), pso_options)
    best_params = {
        name: float(value) for name, value in zip(names, result.best_x)
    }
    return result, specs, best_params, warnings


@dataclass
class CalibrationPSOResult:
    target_type: str
    initial_params_source: str
    initial_params_classification: str
    parameter_names: List[str]
    bounds: List[Tuple[float, float]]
    best_params: Dict[str, float]
    best_loss: float
    iterations: int
    classification: str
    warnings: List[str]
    dataset_schema: str = ""
    converged: bool = False
    history: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema": "calibration_pso_result_v1",
            "target_type": self.target_type,
            "initial_params_source": self.initial_params_source,
            "initial_params_classification": self.initial_params_classification,
            "parameter_names": list(self.parameter_names),
            "bounds": [[float(lo), float(hi)] for lo, hi in self.bounds],
            "best_params": self.best_params,
            "best_loss": float(self.best_loss),
            "iterations": int(self.iterations),
            "classification": self.classification,
            "warnings": list(self.warnings),
            "dataset_schema": self.dataset_schema,
            "converged": bool(self.converged),
        }
        if self.history is not None:
            payload["history"] = [float(v) for v in self.history]
        return payload


def run_calibration_pso(
    *,
    target: str,
    dataset_path: Optional[Union[str, Path]] = None,
    initial_params_path: Optional[Union[str, Path]] = None,
    max_iter: int = 50,
    swarm_size: int = 20,
    seed: int = 42,
    calib_fields: Optional[Sequence[str]] = None,
) -> CalibrationPSOResult:
    """Run PSO for airflow, chtd, or ekf target and return structured result."""
    target_key = target.strip().lower()
    if target_key not in ("airflow", "chtd", "ekf"):
        raise CalibrationRunnerError("--target must be 'airflow', 'chtd', or 'ekf'")

    initial_doc, initial_source = resolve_initial_params(initial_params_path)
    dataset, classification, load_warnings = load_calibration_dataset(
        dataset_path,
        target=target_key,
    )
    warnings = list(load_warnings)

    opts = PSOOptions(swarm_size=swarm_size, max_iter=max_iter, seed=seed)

    if target_key == "airflow":
        demo_fields = AFE_DEMO_CALIB_FIELDS if classification == "demo_only" else None
        fields_use = list(calib_fields or demo_fields or _AFE_CALIB_FIELDS)
        pso_result, specs, best_params, run_warnings = run_airflow_pso(
            dataset=dataset,
            initial_doc=initial_doc,
            calib_fields=fields_use,
            pso_options=opts,
        )
    elif target_key == "chtd":
        demo_fields = CHTD_DEMO_CALIB_FIELDS if classification == "demo_only" else None
        fields_use = list(calib_fields or demo_fields or _CHTD_CALIB_FIELDS)
        pso_result, specs, best_params, run_warnings = run_chtd_pso(
            dataset=dataset,
            initial_doc=initial_doc,
            calib_fields=fields_use,
            pso_options=opts,
        )
    else:
        fields_use = list(calib_fields or EKF_CALIB_FIELDS)
        pso_result, specs, best_params, run_warnings = run_ekf_pso(
            dataset=dataset,
            calib_fields=fields_use,
            pso_options=opts,
        )

    warnings.extend(run_warnings)
    if classification == "demo_only":
        warnings.append("classification=demo_only: output is smoke/demo, not vehicle calibration")

    param_names = specs_to_names(specs)
    bounds = specs_to_bounds(specs)

    return CalibrationPSOResult(
        target_type=target_key,
        initial_params_source=initial_source,
        initial_params_classification=initial_doc.classification,
        parameter_names=param_names,
        bounds=bounds,
        best_params=best_params,
        best_loss=float(pso_result.best_loss),
        iterations=int(pso_result.n_iter),
        classification=classification,
        warnings=warnings,
        dataset_schema=str(dataset.get("schema", "")),
        converged=bool(pso_result.converged),
        history=list(pso_result.history),
    )
