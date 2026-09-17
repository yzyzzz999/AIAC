"""AFE v2 → CHTD u-bus preview adapter (experimental, not runtime).

Maps M8 dual-layer v2 coarse flows and a Tma source dict onto ``u[54]`` for
offline CHTD preview. Does **not** modify ``runtime_adapter`` or ``thermal.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np

from hvac_sim.afe.calibration_data import load_physical_outlet_map
from hvac_sim.chtd.bus_index import N_U, U_INDEX
from hvac_sim.defrost import estimate_tma_def

PREVIEW_SCHEMA = "afe_v2_chtd_u_preview_v1"
CLASSIFICATION = "experimental_preview_not_runtime"

# Coarse v2 flow keys expected from ``m8_dual_layer_v2_result``.
_V2_FLOW_KEYS = (
    "q_defrost_m3h",
    "q_face_m3h",
    "q_foot_m3h",
    "q_rear_face_m3h",
    "q_rear_foot_m3h",
)

# CHTD u slots this adapter intentionally writes.
FLOW_U_SLOTS = (
    "FrntFdDefFlow",
    "FrntFpDefFlow",
    "FrntFdvFlow",
    "FrntFpvFlow",
    "FrntFdfFlow",
    "FrntFpfFlow",
    "RearSdvFlow",
    "RearSpvFlow",
    "RearSdfFlow",
    "RearSpfFlow",
)

TMA_U_SLOTS = (
    "AmbT",
    "RawAmbT",
    "FrntDefTmaEst",
    "FrntFdvTma",
    "FrntFpvTma",
    "FrntFdfTma",
    "FrntFpfTma",
    "RearSdvTma",
    "RearSpvTma",
    "RearSdfTma",
    "RearSpfTma",
)

# Type alias — diagnostic JSON dict from ``run_afe_v2_diagnostic``.
AFEV2DiagnosticResult = Mapping[str, Any]
AfeV2ReplayCaseResult = Mapping[str, Any]

_FRONT_FACE_OUTLETS = ("frnt_face_fl", "frnt_face_flc", "frnt_face_frc", "frnt_face_fr")
_FRONT_FACE_DRIVER = ("frnt_face_fl", "frnt_face_flc")
_FRONT_FACE_PASSENGER = ("frnt_face_frc", "frnt_face_fr")
_REAR_FACE_LEFT = "row2_face_bp_left"
_REAR_FACE_CENTER = "row2_face_console"
_REAR_FACE_RIGHT = "row2_face_bp_right"


@dataclass
class AfeV2ToChtdPreviewResult:
    u: np.ndarray
    populated_slots: Dict[str, float]
    missing_slots: List[str]
    estimated_slots: List[str]
    fallback_used: Dict[str, str]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": PREVIEW_SCHEMA,
            "classification": CLASSIFICATION,
            "not_runtime_active": True,
            "runtime_adapter_default_unchanged": True,
            "u": self.u.tolist(),
            "populated_slots": dict(self.populated_slots),
            "missing_slots": list(self.missing_slots),
            "estimated_slots": list(self.estimated_slots),
            "fallback_used": dict(self.fallback_used),
            "provenance": dict(self.provenance),
        }


def _finite_nonneg(value: Any, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return max(v, 0.0)


def extract_m8_v2_flows(source: Mapping[str, Any]) -> Dict[str, float]:
    """Extract coarse M8 v2 flows from diagnostic or replay case dict."""
    if "m8_dual_layer_v2_result" in source:
        block = source["m8_dual_layer_v2_result"]
    elif "diagnostic" in source and isinstance(source["diagnostic"], Mapping):
        block = source["diagnostic"].get("m8_dual_layer_v2_result", {})
    else:
        block = source

    if not isinstance(block, Mapping):
        raise ValueError("AFE v2 source missing m8_dual_layer_v2_result")

    flows: Dict[str, float] = {}
    for key in _V2_FLOW_KEYS:
        if key in block:
            flows[key] = _finite_nonneg(block[key], key)
    if not flows:
        raise ValueError("m8_dual_layer_v2_result has no coarse flow fields")
    return flows


def _outlet_percent_weights(
    outlet_percent: Optional[Mapping[str, float]],
) -> Dict[str, Any]:
    """Derive split weights from anchor-style outlet_percent if present."""
    if not outlet_percent:
        return {
            "face_driver_fraction": 0.5,
            "face_passenger_fraction": 0.5,
            "foot_driver_fraction": 0.5,
            "foot_passenger_fraction": 0.5,
            "defrost_driver_fraction": 0.5,
            "defrost_passenger_fraction": 0.5,
            "rear_face_left_fraction": 1.0 / 3.0,
            "rear_face_center_fraction": 1.0 / 3.0,
            "rear_face_right_fraction": 1.0 / 3.0,
            "rear_foot_left_fraction": 0.5,
            "rear_foot_right_fraction": 0.5,
            "method": "equal_split_placeholder",
        }

    pct = {k: max(float(v), 0.0) for k, v in outlet_percent.items()}

    fd_face = sum(pct.get(k, 0.0) for k in _FRONT_FACE_DRIVER)
    fp_face = sum(pct.get(k, 0.0) for k in _FRONT_FACE_PASSENGER)
    face_total = fd_face + fp_face
    if face_total <= 1e-9:
        fd_frac, fp_frac = 0.5, 0.5
    else:
        fd_frac, fp_frac = fd_face / face_total, fp_face / face_total

    fd_foot = pct.get("frnt_foot_fl", 0.0)
    fp_foot = pct.get("frnt_foot_fr", 0.0)
    foot_total = fd_foot + fp_foot
    if foot_total <= 1e-9:
        foot_fd, foot_fp = 0.5, 0.5
    else:
        foot_fd, foot_fp = fd_foot / foot_total, fp_foot / foot_total

    def_main = pct.get("defrost_main", 0.0)
    def_side = pct.get("defrost_side", 0.0)
    def_total = def_main + def_side
    if def_total <= 1e-9:
        def_fd, def_fp = 0.5, 0.5
    else:
        # physical_outlet_map default: combined defrost split 50/50 Fd/Fp
        def_fd, def_fp = 0.5, 0.5

    rl = pct.get(_REAR_FACE_LEFT, 0.0)
    rc = pct.get(_REAR_FACE_CENTER, 0.0)
    rr = pct.get(_REAR_FACE_RIGHT, 0.0)
    rear_face_total = rl + rc + rr
    if rear_face_total <= 1e-9:
        rl_f, rc_f, rr_f = 1 / 3, 1 / 3, 1 / 3
    else:
        rl_f, rc_f, rr_f = rl / rear_face_total, rc / rear_face_total, rr / rear_face_total

    rf_l = pct.get("row2_foot_left", 0.0)
    rf_r = pct.get("row2_foot_right", 0.0)
    rf_total = rf_l + rf_r
    if rf_total <= 1e-9:
        rf_l_f, rf_r_f = 0.5, 0.5
    else:
        rf_l_f, rf_r_f = rf_l / rf_total, rf_r / rf_total

    return {
        "face_driver_fraction": fd_frac,
        "face_passenger_fraction": fp_frac,
        "foot_driver_fraction": foot_fd,
        "foot_passenger_fraction": foot_fp,
        "defrost_driver_fraction": def_fd,
        "defrost_passenger_fraction": def_fp,
        "rear_face_left_fraction": rl_f,
        "rear_face_center_fraction": rc_f,
        "rear_face_right_fraction": rr_f,
        "rear_foot_left_fraction": rf_l_f,
        "rear_foot_right_fraction": rf_r_f,
        "method": "physical_outlet_percent_weights",
    }


def map_v2_coarse_flows_to_chtd_flows(
    flows: Mapping[str, float],
    *,
    outlet_percent: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Map v2 coarse flows to CHTD u flow slot values (m³/h)."""
    weights = _outlet_percent_weights(outlet_percent)

    q_def = _finite_nonneg(flows.get("q_defrost_m3h", 0.0), "q_defrost_m3h")
    q_face = _finite_nonneg(flows.get("q_face_m3h", 0.0), "q_face_m3h")
    q_foot = _finite_nonneg(flows.get("q_foot_m3h", 0.0), "q_foot_m3h")
    q_rf = _finite_nonneg(flows.get("q_rear_face_m3h", 0.0), "q_rear_face_m3h")
    q_rfoot = _finite_nonneg(flows.get("q_rear_foot_m3h", 0.0), "q_rear_foot_m3h")

    rl = weights["rear_face_left_fraction"]
    rc = weights["rear_face_center_fraction"]
    rr = weights["rear_face_right_fraction"]
    # Console flow allocated 50/50 to Sdv (left+half center) / Spv (right+half center).
    rear_sdv_frac = rl + 0.5 * rc
    rear_spv_frac = rr + 0.5 * rc

    mapped = {
        "FrntFdDefFlow": q_def * weights["defrost_driver_fraction"],
        "FrntFpDefFlow": q_def * weights["defrost_passenger_fraction"],
        "FrntFdvFlow": q_face * weights["face_driver_fraction"],
        "FrntFpvFlow": q_face * weights["face_passenger_fraction"],
        "FrntFdfFlow": q_foot * weights["foot_driver_fraction"],
        "FrntFpfFlow": q_foot * weights["foot_passenger_fraction"],
        "RearSdvFlow": q_rf * rear_sdv_frac,
        "RearSpvFlow": q_rf * rear_spv_frac,
        "RearSdfFlow": q_rfoot * weights["rear_foot_left_fraction"],
        "RearSpfFlow": q_rfoot * weights["rear_foot_right_fraction"],
    }

    prov = {
        "flow_mapping": "v2_coarse_to_chtd_u",
        "split_weights": weights,
        "zone_split_rules_ref": "physical_outlet_map.json",
        "rear_face_console_split": "center_50_50_to_sdv_spv",
        "inputs_m3h": dict(flows),
        "outputs_m3h": dict(mapped),
    }
    return mapped, prov


def _resolve_tma_values(
    tma_source: Mapping[str, Any],
    *,
    provenance: MutableMapping[str, Any],
    fallback_used: MutableMapping[str, str],
    estimated_slots: List[str],
) -> Dict[str, float]:
    amb_t = float(tma_source["amb_t"])
    eva_t = float(tma_source["eva_t"]) if tma_source.get("eva_t") is not None else amb_t

    driver_face = float(tma_source["driver_face_tma"])
    passenger_face = float(tma_source["passenger_face_tma"])
    driver_foot = float(tma_source["driver_foot_tma"])
    passenger_foot = float(tma_source["passenger_foot_tma"])
    rear_face = float(tma_source["rear_face_tma"])

    if tma_source.get("rear_foot_tma") is not None:
        rear_foot = float(tma_source["rear_foot_tma"])
        rear_foot_method = "rear_foot_tma"
    else:
        rear_foot = rear_face
        rear_foot_method = "rear_face_tma_fallback"
        fallback_used["RearSdfTma"] = rear_foot_method
        fallback_used["RearSpfTma"] = rear_foot_method
        estimated_slots.extend(["RearSdfTma", "RearSpfTma"])

    if tma_source.get("defrost_tma") is not None:
        defrost_tma = float(tma_source["defrost_tma"])
        defrost_method = "defrost_tma_explicit"
    else:
        tma_def = estimate_tma_def(
            eva_temp_c=eva_t,
            hct_temp_c=tma_source.get("hct"),
            posn_fdh=tma_source.get("posn_fdh"),
            fallback="eva",
        )
        defrost_tma = float(tma_def.tma_def_c)
        defrost_method = tma_def.method
        provenance["FrntDefTmaEst"] = {
            "method": tma_def.method,
            "tmadef_approx": tma_def.tmadef_approx,
            "diagnostics": dict(tma_def.diagnostics),
        }
        estimated_slots.append("FrntDefTmaEst")
        if tma_def.tmadef_approx:
            fallback_used["FrntDefTmaEst"] = "estimate_tma_def_eva_fallback"

    provenance["tma_mapping"] = {
        "FrntFdvTma": "driver_face_tma",
        "FrntFpvTma": "passenger_face_tma",
        "FrntFdfTma": "driver_foot_tma",
        "FrntFpfTma": "passenger_foot_tma",
        "RearSdvTma": "rear_face_tma",
        "RearSpvTma": "rear_face_tma",
        "RearSdfTma": rear_foot_method,
        "RearSpfTma": rear_foot_method,
        "FrntDefTmaEst": defrost_method,
        "AmbT": "amb_t",
        "RawAmbT": "amb_t",
    }

    return {
        "AmbT": amb_t,
        "RawAmbT": amb_t,
        "FrntDefTmaEst": defrost_tma,
        "FrntFdvTma": driver_face,
        "FrntFpvTma": passenger_face,
        "FrntFdfTma": driver_foot,
        "FrntFpfTma": passenger_foot,
        "RearSdvTma": rear_face,
        "RearSpvTma": rear_face,
        "RearSdfTma": rear_foot,
        "RearSpfTma": rear_foot,
    }


def build_chtd_u_preview_from_afe_v2(
    afe_v2_result: Union[AFEV2DiagnosticResult, AfeV2ReplayCaseResult],
    tma_source: Mapping[str, Any],
    *,
    outlet_percent: Optional[Mapping[str, float]] = None,
) -> AfeV2ToChtdPreviewResult:
    """Build ``u[54]`` candidate from AFE v2 diagnostic/replay + Tma source dict."""
    _required_tma = (
        "driver_face_tma",
        "passenger_face_tma",
        "driver_foot_tma",
        "passenger_foot_tma",
        "rear_face_tma",
        "amb_t",
    )
    for key in _required_tma:
        if key not in tma_source:
            raise ValueError(f"tma_source missing required key: {key}")

    diagnostic = (
        afe_v2_result["diagnostic"]
        if "diagnostic" in afe_v2_result and "m8_dual_layer_v2_result" not in afe_v2_result
        else afe_v2_result
    )
    if not isinstance(diagnostic, Mapping):
        raise ValueError("invalid AFE v2 result: expected mapping")

    flows = extract_m8_v2_flows(diagnostic)
    if outlet_percent is None:
        outlet_percent = tma_source.get("outlet_percent")  # type: ignore[assignment]
        if outlet_percent is None and "layer_assignment" in diagnostic:
            outlet_percent = None

    flow_values, flow_prov = map_v2_coarse_flows_to_chtd_flows(
        flows,
        outlet_percent=outlet_percent,
    )

    u = np.zeros(N_U, dtype=float)
    populated: Dict[str, float] = {}
    fallback_used: Dict[str, str] = {}
    estimated_slots: List[str] = []
    provenance: Dict[str, Any] = {
        "source": "afe_v2_to_chtd_preview",
        "not_wired_to_runtime_pipeline": True,
        "physical_outlet_map_loaded": bool(load_physical_outlet_map()),
        "v2_flow_source": "m8_dual_layer_v2_result",
        **flow_prov,
    }

    for slot, value in flow_values.items():
        u[U_INDEX[slot]] = float(value)
        populated[slot] = float(value)

    tma_values = _resolve_tma_values(
        tma_source,
        provenance=provenance,
        fallback_used=fallback_used,
        estimated_slots=estimated_slots,
    )
    for slot, value in tma_values.items():
        u[U_INDEX[slot]] = float(value)
        populated[slot] = float(value)

    expected = set(FLOW_U_SLOTS) | set(TMA_U_SLOTS)
    missing = sorted(name for name in expected if name not in populated)

    if "case_id" in afe_v2_result:
        provenance["case_id"] = afe_v2_result["case_id"]

    return AfeV2ToChtdPreviewResult(
        u=u,
        populated_slots=populated,
        missing_slots=missing,
        estimated_slots=sorted(set(estimated_slots)),
        fallback_used=fallback_used,
        provenance=provenance,
    )


def build_previews_from_replay_output(
    replay_doc: Mapping[str, Any],
    tma_source: Mapping[str, Any],
    *,
    outlet_percent: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Build u previews for every case in an AFE v2 replay output document."""
    cases = replay_doc.get("cases")
    if not isinstance(cases, list):
        raise ValueError("replay document missing cases list")

    previews: List[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        preview = build_chtd_u_preview_from_afe_v2(
            case,
            tma_source,
            outlet_percent=outlet_percent,
        )
        item = preview.to_dict()
        item["case_id"] = case.get("case_id")
        previews.append(item)
    return previews


def default_tma_source_from_amb(*, amb_t: float = 24.0, eva_t: Optional[float] = 16.0) -> Dict[str, float]:
    """Minimal Tma source for previews when only ambient is known."""
    eva = float(eva_t) if eva_t is not None else amb_t
    return {
        "driver_face_tma": eva,
        "passenger_face_tma": eva,
        "driver_foot_tma": eva,
        "passenger_foot_tma": eva,
        "rear_face_tma": eva,
        "amb_t": amb_t,
        "eva_t": eva,
    }
