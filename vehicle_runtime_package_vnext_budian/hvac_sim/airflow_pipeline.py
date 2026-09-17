"""AFE / AFE_Calc → PMV vent-flow adapter (T10-7).

Runtime volumetric flows are **not** FTE inputs. They are computed from the in-house
AFE resistance network + fan model (``hvac_sim.afe``), then mapped to the fields
consumed by ``run_comfort_pipeline`` / ``AirSpeedInputs``.

Calibration CSV wind speeds (m/s) are **not** used here; see
``hvac_sim.calibration.data_prep`` for probe → validation datasets only.

Priority:
1. Explicit ``PmvVentFlows`` / flow dict (integration or replay)
2. ``afe_calc`` when ``FlowInputs`` (+ fans/params) are supplied and converged
3. Packaged runtime-example fallback flows (estimated, not measured)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from hvac_sim.air_speed import AirSpeedInputs
from hvac_sim.afe.fan import FanSet
from hvac_sim.afe.flow import AFECalcResult, FlowInputs, afe_calc
from hvac_sim.afe.params import AFEParams
from hvac_sim.afe.resistance import FlapSet

# Side-defrost split when only face flows are known (matches calibration/wind_speed).
FACE_VENT_SHARE = 0.93
SIDE_DEFROST_SHARE = 0.07

# AFE_Calc Q_Out indices (see simulink_conversion_package/golden_cases/afe_calc_cases.json).
_Q_TOTAL = 0
_Q_FD_DEF = 9
_Q_FD_SV = 10
_Q_FD_V = 11
_Q_FD_F = 12
_Q_FP_DEF = 13
_Q_FP_SV = 14
_Q_FP_V = 15
_Q_FP_F = 16
_Q_SDV = 17
_Q_SDF = 18
_Q_SPV = 19
_Q_SPF = 20

M3S_TO_M3H = 3600.0

PMV_AIRFLOW_FIELD_NAMES: Tuple[str, ...] = (
    "driver_face_flow_m3h",
    "passenger_face_flow_m3h",
    "driver_floor_flow_m3h",
    "passenger_floor_flow_m3h",
    "driver_defrost_flow_m3h",
    "passenger_defrost_flow_m3h",
)

OPTIONAL_SIDE_FLOW_FIELDS: Tuple[str, ...] = (
    "driver_side_defrost_flow_m3h",
    "passenger_side_defrost_flow_m3h",
)

REAR_FOOT_FLOW_FIELDS: Tuple[str, ...] = (
    "rear_driver_foot_flow_m3h",
    "rear_passenger_foot_flow_m3h",
)

_RUNTIME_EXAMPLES_DIR = (
    Path(__file__).resolve().parents[2]
    / "simulink_conversion_package"
    / "python_targets"
    / "runtime_examples"
)


@dataclass(frozen=True)
class PmvVentFlows:
    """Minimal vent flows for PMV / CHTD u-bus (volumetric, m³/h)."""

    driver_face_flow_m3h: float
    passenger_face_flow_m3h: float
    driver_floor_flow_m3h: float
    passenger_floor_flow_m3h: float
    driver_defrost_flow_m3h: float
    passenger_defrost_flow_m3h: float
    driver_side_defrost_flow_m3h: Optional[float] = None
    passenger_side_defrost_flow_m3h: Optional[float] = None
    rear_driver_foot_flow_m3h: Optional[float] = None
    rear_passenger_foot_flow_m3h: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        data = {name: getattr(self, name) for name in PMV_AIRFLOW_FIELD_NAMES}
        for name in OPTIONAL_SIDE_FLOW_FIELDS + REAR_FOOT_FLOW_FIELDS:
            data[name] = getattr(self, name)
        return data


@dataclass(frozen=True)
class AirflowEstimateResult:
    """Result of ``estimate_airflow_for_pmv``."""

    flows: PmvVentFlows
    air_speed_inputs: AirSpeedInputs
    source: str
    afe_converged: Optional[bool] = None
    warnings: tuple[str, ...] = ()
    provenance: Dict[str, Any] = field(default_factory=dict)


def _safe_m3h(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, float(value))


def _m3s_to_m3h(q_m3s: float) -> float:
    return _safe_m3h(q_m3s * M3S_TO_M3H)


def side_defrost_total_m3h_from_face(
    driver_face_flow_m3h: float,
    passenger_face_flow_m3h: float,
) -> float:
    """Total side-defrost flow from face vents via 7% / 93% rule [m³/h]."""
    face_total = _safe_m3h(driver_face_flow_m3h) + _safe_m3h(passenger_face_flow_m3h)
    if face_total <= 0.0:
        return 0.0
    return face_total * SIDE_DEFROST_SHARE / FACE_VENT_SHARE


def split_side_defrost_per_seat(side_total_m3h: float) -> Tuple[float, float]:
    """Split side-defrost total equally between driver and passenger [m³/h]."""
    half = _safe_m3h(side_total_m3h) * 0.5
    return half, half


def apply_side_defrost_rule_if_missing(flows: PmvVentFlows) -> PmvVentFlows:
    """Fill side-defrost flows from face totals when not already set."""
    if (
        flows.driver_side_defrost_flow_m3h is not None
        and flows.passenger_side_defrost_flow_m3h is not None
    ):
        return flows
    total = side_defrost_total_m3h_from_face(
        flows.driver_face_flow_m3h,
        flows.passenger_face_flow_m3h,
    )
    drv, psn = split_side_defrost_per_seat(total)
    return PmvVentFlows(
        driver_face_flow_m3h=flows.driver_face_flow_m3h,
        passenger_face_flow_m3h=flows.passenger_face_flow_m3h,
        driver_floor_flow_m3h=flows.driver_floor_flow_m3h,
        passenger_floor_flow_m3h=flows.passenger_floor_flow_m3h,
        driver_defrost_flow_m3h=flows.driver_defrost_flow_m3h,
        passenger_defrost_flow_m3h=flows.passenger_defrost_flow_m3h,
        driver_side_defrost_flow_m3h=(
            flows.driver_side_defrost_flow_m3h
            if flows.driver_side_defrost_flow_m3h is not None
            else drv
        ),
        passenger_side_defrost_flow_m3h=(
            flows.passenger_side_defrost_flow_m3h
            if flows.passenger_side_defrost_flow_m3h is not None
            else psn
        ),
        rear_driver_foot_flow_m3h=flows.rear_driver_foot_flow_m3h,
        rear_passenger_foot_flow_m3h=flows.rear_passenger_foot_flow_m3h,
    )


def pmv_vent_flows_from_mapping(raw: Mapping[str, Any]) -> PmvVentFlows:
    """Build ``PmvVentFlows`` from explicit flow dict (runtime-style keys accepted)."""
    key_aliases = {
        "driver_face_flow_m3h": ("driver_face_flow_m3h", "driver_face_flow"),
        "passenger_face_flow_m3h": (
            "passenger_face_flow_m3h",
            "passenger_face_flow",
        ),
        "driver_floor_flow_m3h": ("driver_floor_flow_m3h", "driver_floor_flow"),
        "passenger_floor_flow_m3h": (
            "passenger_floor_flow_m3h",
            "passenger_floor_flow",
        ),
        "driver_defrost_flow_m3h": (
            "driver_defrost_flow_m3h",
            "driver_defrost_flow",
        ),
        "passenger_defrost_flow_m3h": (
            "passenger_defrost_flow_m3h",
            "passenger_defrost_flow",
        ),
    }

    def _get(canonical: str) -> float:
        for key in key_aliases[canonical]:
            if key in raw and raw[key] is not None:
                return _safe_m3h(float(raw[key]))
        return 0.0

    def _optional(*keys: str) -> Optional[float]:
        for key in keys:
            if key in raw and raw[key] is not None:
                return _safe_m3h(float(raw[key]))
        return None

    return PmvVentFlows(
        driver_face_flow_m3h=_get("driver_face_flow_m3h"),
        passenger_face_flow_m3h=_get("passenger_face_flow_m3h"),
        driver_floor_flow_m3h=_get("driver_floor_flow_m3h"),
        passenger_floor_flow_m3h=_get("passenger_floor_flow_m3h"),
        driver_defrost_flow_m3h=_get("driver_defrost_flow_m3h"),
        passenger_defrost_flow_m3h=_get("passenger_defrost_flow_m3h"),
        driver_side_defrost_flow_m3h=_optional(
            "driver_side_defrost_flow_m3h", "driver_side_defrost_flow"
        ),
        passenger_side_defrost_flow_m3h=_optional(
            "passenger_side_defrost_flow_m3h", "passenger_side_defrost_flow"
        ),
        rear_driver_foot_flow_m3h=_optional(
            "rear_driver_foot_flow_m3h", "rear_driver_foot_flow"
        ),
        rear_passenger_foot_flow_m3h=_optional(
            "rear_passenger_foot_flow_m3h", "rear_passenger_foot_flow"
        ),
    )


def pmv_vent_flows_from_afe_q_out(q_out: np.ndarray) -> PmvVentFlows:
    """Map AFE_Calc ``Q_out`` (m³/s) to PMV vent flows (m³/h)."""
    q = np.asarray(q_out, dtype=float)
    if q.shape[0] < 21:
        raise ValueError(f"AFE Q_out must have at least 21 elements, got {q.shape}")

    return PmvVentFlows(
        driver_face_flow_m3h=_m3s_to_m3h(q[_Q_FD_V]),
        passenger_face_flow_m3h=_m3s_to_m3h(q[_Q_FP_V]),
        driver_floor_flow_m3h=_m3s_to_m3h(q[_Q_FD_F]),
        passenger_floor_flow_m3h=_m3s_to_m3h(q[_Q_FP_F]),
        driver_defrost_flow_m3h=_m3s_to_m3h(q[_Q_FD_DEF]),
        passenger_defrost_flow_m3h=_m3s_to_m3h(q[_Q_FP_DEF]),
        driver_side_defrost_flow_m3h=_m3s_to_m3h(q[_Q_FD_SV]),
        passenger_side_defrost_flow_m3h=_m3s_to_m3h(q[_Q_FP_SV]),
        rear_driver_foot_flow_m3h=None,
        rear_passenger_foot_flow_m3h=None,
    )


def pmv_vent_flows_from_afe_result(result: AFECalcResult) -> PmvVentFlows:
    """Map a converged or unconverged ``AFECalcResult`` to vent flows."""
    return pmv_vent_flows_from_afe_q_out(result.Q_out)


def vent_flows_to_air_speed_inputs(
    flows: PmvVentFlows,
    *,
    apply_side_defrost_rule: bool = True,
) -> AirSpeedInputs:
    """Convert vent flows to ``AirSpeedInputs`` for local air-speed estimation.

    Windshield defrost uses ``driver_defrost_flow_m3h`` / ``passenger_defrost_flow_m3h``.
    Side-defrost vents (u[30–33]) are not yet a separate air-speed channel; the 93/7
    rule fills ``PmvVentFlows`` side fields for CHTD bus mapping only.
    """
    resolved = (
        apply_side_defrost_rule_if_missing(flows)
        if apply_side_defrost_rule
        else flows
    )
    return AirSpeedInputs(
        driver_face_flow=resolved.driver_face_flow_m3h,
        passenger_face_flow=resolved.passenger_face_flow_m3h,
        driver_floor_flow=resolved.driver_floor_flow_m3h,
        passenger_floor_flow=resolved.passenger_floor_flow_m3h,
        driver_defrost_flow=resolved.driver_defrost_flow_m3h,
        passenger_defrost_flow=resolved.passenger_defrost_flow_m3h,
        rear_driver_foot_flow=resolved.rear_driver_foot_flow_m3h,
        rear_passenger_foot_flow=resolved.rear_passenger_foot_flow_m3h,
        flow_unit="m3h",
    )


def vent_flows_to_chtd_u_flow_dict(flows: PmvVentFlows) -> Dict[str, float]:
    """Map PMV vent flows to ``Bus_CHTD_u`` flow slot names (m³/h)."""
    resolved = apply_side_defrost_rule_if_missing(flows)
    out: Dict[str, float] = {
        "FrntFdvFlow": resolved.driver_face_flow_m3h,
        "FrntFpvFlow": resolved.passenger_face_flow_m3h,
        "FrntFdfFlow": resolved.driver_floor_flow_m3h,
        "FrntFpfFlow": resolved.passenger_floor_flow_m3h,
        "FrntFdDefFlow": resolved.driver_defrost_flow_m3h,
        "FrntFpDefFlow": resolved.passenger_defrost_flow_m3h,
    }
    if resolved.driver_side_defrost_flow_m3h is not None:
        out["FrntSdvFlow"] = resolved.driver_side_defrost_flow_m3h
    if resolved.passenger_side_defrost_flow_m3h is not None:
        out["FrntSpvFlow"] = resolved.passenger_side_defrost_flow_m3h
    return out


def _load_runtime_example_fallback(
    scenario: str = "neutral_runtime_input.json",
) -> PmvVentFlows:
    path = _RUNTIME_EXAMPLES_DIR / scenario
    if not path.is_file():
        raise FileNotFoundError(f"runtime example not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return pmv_vent_flows_from_mapping(raw)


def estimate_airflow_for_pmv(
    *,
    explicit_flows: Optional[Union[PmvVentFlows, Mapping[str, Any]]] = None,
    afe_inputs: Optional[FlowInputs] = None,
    afe_params: Optional[AFEParams] = None,
    fan_set: Optional[FanSet] = None,
    fallback_scenario: str = "neutral_runtime_input.json",
    apply_side_defrost_rule: bool = True,
) -> AirflowEstimateResult:
    """Estimate PMV vent flows and ``AirSpeedInputs``.

    Resolution order: explicit flows → AFE_Calc (if converged) → runtime example fallback.
    """
    warnings: list[str] = []
    provenance: Dict[str, Any] = {"schema": "airflow_estimate_v1"}

    if explicit_flows is not None:
        flows = (
            explicit_flows
            if isinstance(explicit_flows, PmvVentFlows)
            else pmv_vent_flows_from_mapping(explicit_flows)
        )
        if apply_side_defrost_rule:
            flows = apply_side_defrost_rule_if_missing(flows)
        provenance["source"] = "explicit"
        return AirflowEstimateResult(
            flows=flows,
            air_speed_inputs=vent_flows_to_air_speed_inputs(
                flows, apply_side_defrost_rule=False
            ),
            source="explicit",
            warnings=tuple(warnings),
            provenance=provenance,
        )

    if afe_inputs is not None:
        params = afe_params if afe_params is not None else AFEParams()
        fans = fan_set if fan_set is not None else FanSet()
        result = afe_calc(afe_inputs, params, fans)
        provenance["afe"] = {
            "converged": result.converged,
            "iterations": result.iterations,
            "Qm": result.Qm,
        }
        if result.converged:
            flows = pmv_vent_flows_from_afe_result(result)
            if apply_side_defrost_rule:
                flows = apply_side_defrost_rule_if_missing(flows)
            return AirflowEstimateResult(
                flows=flows,
                air_speed_inputs=vent_flows_to_air_speed_inputs(
                    flows, apply_side_defrost_rule=False
                ),
                source="afe_calc",
                afe_converged=True,
                warnings=tuple(warnings),
                provenance=provenance,
            )
        warnings.append(
            "AFE_Calc did not converge; falling back to runtime example flows."
        )

    try:
        flows = _load_runtime_example_fallback(fallback_scenario)
    except FileNotFoundError:
        flows = PmvVentFlows(
            driver_face_flow_m3h=90.0,
            passenger_face_flow_m3h=85.0,
            driver_floor_flow_m3h=40.0,
            passenger_floor_flow_m3h=38.0,
            driver_defrost_flow_m3h=0.0,
            passenger_defrost_flow_m3h=0.0,
        )
        warnings.append(
            f"Runtime example missing ({fallback_scenario}); using built-in neutral estimate."
        )
    else:
        warnings.append(
            f"AFE unavailable or not requested; using flows from {fallback_scenario}."
        )

    if apply_side_defrost_rule:
        flows = apply_side_defrost_rule_if_missing(flows)
    provenance["source"] = "runtime_example_fallback"
    provenance["fallback_scenario"] = fallback_scenario

    return AirflowEstimateResult(
        flows=flows,
        air_speed_inputs=vent_flows_to_air_speed_inputs(
            flows, apply_side_defrost_rule=False
        ),
        source="runtime_example_fallback",
        afe_converged=False if afe_inputs is not None else None,
        warnings=tuple(warnings),
        provenance=provenance,
    )


def afe_smoke_estimate(
    flaps: Optional[FlapSet] = None,
    *,
    n_F1: float = 1000.0,
    n_F2: float = 1000.0,
    n_S: float = 1000.0,
) -> AirflowEstimateResult:
    """Smoke helper: run default AFE_Calc and return PMV-oriented flows."""
    flap_set = flaps if flaps is not None else FlapSet()
    return estimate_airflow_for_pmv(
        afe_inputs=FlowInputs(flaps=flap_set, n_F1=n_F1, n_F2=n_F2, n_S=n_S),
    )
