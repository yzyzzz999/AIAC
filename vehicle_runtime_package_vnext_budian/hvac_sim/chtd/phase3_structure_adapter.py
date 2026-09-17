"""Phase3 CHTD structure fixes — adapter / params bundle layer only.

Does not modify ``thermal.py``. Provenance: engineering_capacity_preview_not_vehicle_calibration.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from hvac_sim.chtd.bus_index import U_INDEX
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.afe.actuator_based_airflow import apply_voltage_actuator_distribution
from hvac_sim.validation.chtd_foot_flow_gating_fix import (
    FOOT_FLOW_KEYS,
    apply_actuator_gated_flow,
    apply_leakage_ratio,
    foot_open_normalized,
)
from hvac_sim.validation.chtd_global_calibration_phase1 import _scale_lut_params

PHASE3_PROVENANCE = "engineering_capacity_preview_not_vehicle_calibration"
CLASSIFICATION = PHASE3_PROVENANCE

FACE_DOMINANT_THRESHOLD = 0.75
FOOT_RESTORE_THRESHOLD = 0.50
FOOT_DOMINANCE_MARGIN = 0.10

FRONT_DRV_FLOW_KEYS = ("FrntFdvFlow", "FrntFdfFlow", "FrntFdDefFlow")
FRONT_PSG_FLOW_KEYS = ("FrntFpvFlow", "FrntFpfFlow", "FrntFpDefFlow")

HEAD_SOLAR_LUTS = (
    "CHTD_FdSolarRadCo_M",
    "CHTD_FpSolarRadCo_M",
    "CHTD_SdSolarRadCo_M",
    "CHTD_SpSolarRadCo_M",
    "CHTD_TdSolarRadCo_M",
    "CHTD_TpSolarRadCo_M",
)
SHELL_SOLAR_LUTS = (
    "CHTD_HoodSolarRadCo_M",
    "CHTD_RoofSolarRadCo_M",
    "CHTD_ConsoleSolarRadCo_M",
    "CHTD_CabinFdSolarRadCo_M",
    "CHTD_CabinFpSolarRadCo_M",
    "CHTD_CabinFrntSolarRadCo_M",
    "CHTD_WinFdSolarRadCo_M",
    "CHTD_WinFpSolarRadCo_M",
    "CHTD_WinSdSolarRadCo_M",
    "CHTD_WinSpSolarRadCo_M",
)

PHASE3_CAPACITY_MASS_KEYS = tuple(
    name
    for name in (
        "CHTD_HoodMassAtb_P",
        "CHTD_CabinFrntMassAtb_P",
        "CHTD_ConsoleMassAtb_P",
        "CHTD_RoofMassAtb_P",
        "CHTD_CabinFdMassAtb_P",
        "CHTD_CabinFpMassAtb_P",
        "CHTD_CabinSdMassAtb_P",
        "CHTD_CabinSpMassAtb_P",
        "CHTD_CabinTdMassAtb_P",
        "CHTD_CabinTpMassAtb_P",
        "CHTD_WinFdMassAtb_P",
        "CHTD_WinFpMassAtb_P",
        "CHTD_WinSdMassAtb_P",
        "CHTD_WinSpMassAtb_P",
        "CHTD_WinTdMassAtb_P",
        "CHTD_WinTpMassAtb_P",
    )
)


@dataclass
class Phase3StructureConfig:
    """Structure fix toggles (no PSO)."""

    variant_id: str = "phase2_baseline"
    use_actuator_based_distribution: bool = False
    use_voltage_actuator_scores: bool = True
    use_foot_leakage_routing: bool = False
    foot_leakage_ratio: float = 0.10
    use_capacity_bundle: bool = False
    capacity_scale: float = 100.0
    use_solar_shell_routing: bool = False
    head_solar_scale: float = 0.0
    shell_solar_scale: float = 2.0
    ac_mode_ventila_posn_used: bool = False

    def provenance(self) -> Dict[str, Any]:
        return {
            "schema": "phase3_structure_config_v1",
            "classification": PHASE3_PROVENANCE,
            "variant_id": self.variant_id,
            "use_actuator_based_distribution": self.use_actuator_based_distribution,
            "use_voltage_actuator_scores": self.use_voltage_actuator_scores,
            "use_foot_leakage_routing": self.use_foot_leakage_routing,
            "foot_leakage_ratio": self.foot_leakage_ratio,
            "use_capacity_bundle": self.use_capacity_bundle,
            "capacity_scale": self.capacity_scale,
            "use_solar_shell_routing": self.use_solar_shell_routing,
            "head_solar_scale": self.head_solar_scale,
            "shell_solar_scale": self.shell_solar_scale,
            "ac_mode_ventila_posn_used": self.ac_mode_ventila_posn_used,
        }


def _side_total(u: np.ndarray, keys: Tuple[str, ...]) -> float:
    return float(sum(float(u[U_INDEX[k]]) for k in keys))


def apply_actuator_based_distribution(
    u: np.ndarray,
    *,
    face_vent_v: float,
    foot_vent_v: float,
    defrost_vent_v: float,
    airflow_target: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Redistribute front HVAC flows by actuator weights; AC_ModeVentilaPosn ignored."""
    out = u.copy()
    fn = max(foot_open_normalized(face_vent_v), 0.0) if math.isfinite(face_vent_v) else 0.0
    ftn = max(foot_open_normalized(foot_vent_v), 0.0) if math.isfinite(foot_vent_v) else 0.0
    dn = max(foot_open_normalized(defrost_vent_v), 0.0) if math.isfinite(defrost_vent_v) else 0.0
    w_sum = fn + ftn + dn
    prov: Dict[str, Any] = {
        "distribution_method": "actuator_weighted",
        "ac_mode_ventila_posn_used": False,
        "face_weight": fn,
        "foot_weight": ftn,
        "defrost_weight": dn,
    }
    if w_sum < 1e-6:
        prov["status"] = "zero_actuator_weights_unchanged"
        return out, prov

    wf, wt, wd = fn / w_sum, ftn / w_sum, dn / w_sum
    total_drv = _side_total(out, FRONT_DRV_FLOW_KEYS)
    total_psg = _side_total(out, FRONT_PSG_FLOW_KEYS)
    if airflow_target is not None and math.isfinite(airflow_target) and airflow_target > 0:
        total_drv = 0.5 * float(airflow_target)
        total_psg = 0.5 * float(airflow_target)
        prov["total_flow_source"] = "airflow_target"
    else:
        prov["total_flow_source"] = "u_bus_preserving_side_total"

    out[U_INDEX["FrntFdvFlow"]] = total_drv * wf
    out[U_INDEX["FrntFdfFlow"]] = total_drv * wt
    out[U_INDEX["FrntFdDefFlow"]] = total_drv * wd
    out[U_INDEX["FrntFpvFlow"]] = total_psg * wf
    out[U_INDEX["FrntFpfFlow"]] = total_psg * wt
    out[U_INDEX["FrntFpDefFlow"]] = total_psg * wd
    prov["status"] = "redistributed"
    prov["driver_total_m3h"] = total_drv
    prov["passenger_total_m3h"] = total_psg
    return out, prov


def apply_foot_leakage_routing(
    u: np.ndarray,
    *,
    face_vent_v: float,
    foot_vent_v: float,
    leakage_ratio: float,
    foot_restore_threshold: float = FOOT_RESTORE_THRESHOLD,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Face-dominant: foot path = leakage only; high foot actuator restores foot path."""
    face_norm = foot_open_normalized(face_vent_v) if math.isfinite(face_vent_v) else 0.0
    foot_norm = foot_open_normalized(foot_vent_v) if math.isfinite(foot_vent_v) else 0.0
    prov = {
        "face_norm": face_norm,
        "foot_norm": foot_norm,
        "leakage_ratio": leakage_ratio,
    }
    face_dominant = face_norm >= FACE_DOMINANT_THRESHOLD and face_norm > foot_norm + FOOT_DOMINANCE_MARGIN
    if foot_norm >= foot_restore_threshold:
        out = apply_actuator_gated_flow(
            u, foot_vent_v=foot_vent_v, closed_threshold=0.3, span=0.3
        )
        prov["routing"] = "foot_actuator_high_full_path"
        return out, prov
    if face_dominant:
        out = apply_leakage_ratio(u, leakage_ratio=leakage_ratio)
        prov["routing"] = "face_dominant_leakage_only"
        return out, prov
    prov["routing"] = "unchanged"
    return u.copy(), prov


def build_phase3_capacity_params(
    base: CHTDParams,
    *,
    scale: float = 100.0,
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """Scale cabin/shell/win/hood lumped masses (adapter bundle, not vehicle calibration)."""
    p = copy.deepcopy(base)
    applied: Dict[str, float] = {}
    for key in PHASE3_CAPACITY_MASS_KEYS:
        if hasattr(p, key):
            old = float(getattr(p, key))
            new = old * float(scale)
            setattr(p, key, new)
            applied[key] = new
    meta = {
        "provenance": PHASE3_PROVENANCE,
        "capacity_scale": scale,
        "mass_keys_scaled": len(applied),
        "sample": {k: applied[k] for k in list(applied)[:4]},
    }
    return p, meta


def apply_solar_shell_routing_params(
    base: CHTDParams,
    *,
    head_solar_scale: float = 0.0,
    shell_solar_scale: float = 2.0,
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """Zero/minimize direct head solar; boost shell/win/roof/hood solar LUTs."""
    p = base
    p = _scale_lut_params(p, HEAD_SOLAR_LUTS, head_solar_scale)
    p = _scale_lut_params(p, SHELL_SOLAR_LUTS, shell_solar_scale)
    return p, {
        "provenance": PHASE3_PROVENANCE,
        "head_solar_scale": head_solar_scale,
        "shell_solar_scale": shell_solar_scale,
        "head_luts": list(HEAD_SOLAR_LUTS),
        "shell_luts": list(SHELL_SOLAR_LUTS),
    }


def apply_phase3_structure_to_params(
    base: CHTDParams,
    cfg: Phase3StructureConfig,
) -> Tuple[CHTDParams, Dict[str, Any]]:
    p = base
    meta: Dict[str, Any] = {"param_layers": []}
    if cfg.use_capacity_bundle:
        p, cap_meta = build_phase3_capacity_params(p, scale=cfg.capacity_scale)
        meta["param_layers"].append(cap_meta)
    if cfg.use_solar_shell_routing:
        p, sol_meta = apply_solar_shell_routing_params(
            p,
            head_solar_scale=cfg.head_solar_scale,
            shell_solar_scale=cfg.shell_solar_scale,
        )
        meta["param_layers"].append(sol_meta)
    return p, meta


def apply_phase3_u_routing(
    u: np.ndarray,
    *,
    face_vent_v: float,
    foot_vent_v: float,
    defrost_vent_v: float,
    airflow_target: Optional[float],
    cfg: Phase3StructureConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply Phase3 u-bus routing (distribution + foot leakage)."""
    prov: Dict[str, Any] = {"ac_mode_ventila_posn_used": cfg.ac_mode_ventila_posn_used}
    out = u.copy()
    if cfg.use_actuator_based_distribution:
        if cfg.use_voltage_actuator_scores:
            out, dprov = apply_voltage_actuator_distribution(
                out,
                face_v=face_vent_v,
                foot_v=foot_vent_v,
                defrost_v=defrost_vent_v,
                airflow_target=airflow_target,
                defrost_leakage_only=cfg.use_foot_leakage_routing,
                defrost_leak_ratio=cfg.foot_leakage_ratio,
            )
        else:
            out, dprov = apply_actuator_based_distribution(
                out,
                face_vent_v=face_vent_v,
                foot_vent_v=foot_vent_v,
                defrost_vent_v=defrost_vent_v,
                airflow_target=airflow_target,
            )
        prov["distribution"] = dprov
    elif cfg.use_foot_leakage_routing:
        out, fprov = apply_foot_leakage_routing(
            out,
            face_vent_v=face_vent_v,
            foot_vent_v=foot_vent_v,
            leakage_ratio=cfg.foot_leakage_ratio,
        )
        prov["foot_routing"] = fprov
    return out, prov
