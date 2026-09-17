"""M8 dual-layer AFE v2 draft model layer (decode/mapping only).

Provides structured intake, mode-layer, and Tma mapping datatypes for future
``target_vehicle_dual_layer`` solver work. **Not wired to** ``afe_calc``.

Customer structure update (2026-06-07): outlet layers confirmed — face/defrost upper,
foot/rear lower; ``face=mixed`` retained only in historical fitting reports (superseded).

Intake mechanism (decode-only):
- **Left motor** controls: fresh_air_door + left_25_recirc_leaf
- **Right motor** controls: middle_50_recirc_leaf + right_25_recirc_leaf
- Recirc leaves area ratio **1:2:1**; upper/lower fresh-recirc split pending calibration

Blowers: front HVAC blower + rear booster blower (not coaxial fan model).
Heat: heater core / water PTC pending (not air PTC).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from hvac_sim.afe.calibration_data import load_m8_dual_layer_structure_confirmed
from hvac_sim.defrost import estimate_tma_def

_RECIRC_STATES: Tuple[int, ...] = (0, 25, 50, 75, 100)
_RECIRC_DOOR_AREAS: Tuple[int, int, int] = (1, 2, 1)  # small : middle : large
_RECIRC_DOOR_TOTAL_AREA = sum(_RECIRC_DOOR_AREAS)

_MODE_ALIASES: Dict[str, str] = {
    "1": "V",
    "2": "V_F",
    "3": "V_D_F",
    "4": "F",
    "5": "F_D",
    "6": "D",
    "7": "V_D",
    "8": "OFF",
    "吹面": "V",
    "吹面吹脚": "V_F",
    "吹面吹脚除霜": "V_D_F",
    "吹脚": "F",
    "吹脚除霜": "F_D",
    "除霜": "D",
    "吹面除霜": "V_D",
}

# Left (main) intake motor voltage anchors → recirc / fresh / dual_layer.
_LEFT_INTAKE_MOTOR_ANCHORS: Tuple[Tuple[float, str, str], ...] = (
    (0.42, "recirc", "full_recirc_left_motor_anchor"),
    (4.24, "fresh", "fresh_air_mode_left_anchor"),
    (2.5, "dual_layer", "dual_layer_middle_position_left_anchor"),
)
_LEFT_MOTOR_MATCH_TOLERANCE_V = 0.45

# Right dual-layer motor voltage anchors → discrete leaf index (legacy CAN proxy).
_RIGHT_MOTOR_LEAF_ANCHORS: Tuple[Tuple[float, int, str], ...] = (
    (4.56, 100, "dual_layer_stall_full_recirc_bias"),
    (2.33, 25, "dual_layer_25pct_anchor"),
    (2.33, 50, "dual_layer_50pct_anchor"),
    (2.90, 75, "dual_layer_75pct_outer_bias"),
)

LeftMotorState = str  # recirc | fresh | dual_layer | unknown

_HEATING_DUAL_LAYER_MODES: Tuple[str, ...] = ("F_D", "V_D_F", "F")

_CUSTOMER_CONFIRMED_TAG = "customer_confirmed_2026_06_07"

# Confirmed outlet → layer (2026-06-07 customer feedback).
_CONFIRMED_OUTLET_LAYERS: Dict[str, str] = {
    "front_face": "upper_confirmed",
    "defrost_main": "upper_confirmed",
    "defrost_side": "upper_confirmed",
    "front_foot": "lower_confirmed",
    "rear_face": "lower_confirmed",
    "rear_foot": "lower_confirmed",
}

LEFT_MOTOR_CONTROLS: Tuple[str, ...] = ("fresh_air_door", "left_25_recirc_leaf")
RIGHT_MOTOR_CONTROLS: Tuple[str, ...] = ("middle_50_recirc_leaf", "right_25_recirc_leaf")


@dataclass(frozen=True)
class M8IntakeState:
    circle_mode_posn_pct: float
    circle_prior_posn_v: float
    recirc_fraction_total: float
    upper_recirc_fraction: float
    lower_recirc_fraction: float
    upper_fresh_fraction: float
    lower_fresh_fraction: float
    confidence: str
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class M8LayerFlowState:
    q_upper_m3h: float
    q_lower_m3h: float
    q_front_face_m3h: float
    q_front_foot_m3h: float
    q_defrost_m3h: float
    q_rear_face_m3h: float
    q_rear_foot_m3h: float
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class M8LayerTemperatureState:
    t_upper_driver_c: Optional[float]
    t_upper_passenger_c: Optional[float]
    t_lower_driver_c: Optional[float]
    t_lower_passenger_c: Optional[float]
    t_rear_channel_c: Optional[float]
    t_defrost_c: Optional[float]
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class M8ModeLayerDistribution:
    mode_code: str
    defrost_layer: str
    face_layer: str
    foot_layer: str
    rear_layer: str
    rear_face_policy: str = "lower_confirmed"
    rear_foot_policy: str = "lower_confirmed"
    dual_layer_active: Optional[bool] = None
    heating_mode: Optional[bool] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


def confirmed_outlet_layer_assignments() -> Dict[str, str]:
    """Return customer-confirmed outlet → layer map (2026-06-07)."""
    return dict(_CONFIRMED_OUTLET_LAYERS)


def intake_motor_control_ownership() -> Dict[str, Any]:
    """Document which intake elements each motor drives."""
    return {
        "left_motor_controls": list(LEFT_MOTOR_CONTROLS),
        "right_motor_controls": list(RIGHT_MOTOR_CONTROLS),
        "recirc_leaves_area_ratio": "1:2:1",
        "customer_confirmed": _CUSTOMER_CONFIRMED_TAG,
    }


@dataclass(frozen=True)
class M8RecircLeafState:
    leaf_small_left_open: bool
    leaf_middle_open: bool
    leaf_small_right_open: bool
    area_ratio_open: float
    total_recirc_fraction_cmd: float
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class M8IntakeMechanismState:
    left_motor_role: str
    left_motor_state: LeftMotorState
    right_motor_role: str
    recirc_leaf_state: M8RecircLeafState
    fresh_opening_policy: str
    upper_lower_split_policy: str
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecircThreeDoorState:
    state_index: int
    total_recirc_fraction: float
    open_doors: Tuple[str, ...]
    open_area_ratio: float
    door_area_ratio: Tuple[int, int, int]
    provenance: Dict[str, Any] = field(default_factory=dict)


def _finite_or_raise(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _normalize_mode(mode_code: Union[str, int]) -> str:
    key = str(mode_code).strip()
    if key in _MODE_ALIASES:
        return _MODE_ALIASES[key]
    return key.upper().replace(" ", "_")


def _snap_recirc_fraction_cmd(recirc_fraction_cmd: float) -> int:
    """Snap 0–100 % (or 0–1 fraction) command to discrete 0/25/50/75/100."""
    cmd = _finite_or_raise(recirc_fraction_cmd, "recirc_fraction_cmd")
    if 0.0 <= cmd <= 1.0 and not math.isclose(cmd, round(cmd), rel_tol=0, abs_tol=1e-9):
        cmd *= 100.0
    cmd = max(0.0, min(100.0, cmd))
    return min(_RECIRC_STATES, key=lambda s: abs(s - cmd))


def decode_recirc_leaf_state(recirc_fraction_cmd: Union[int, float]) -> M8RecircLeafState:
    """Decode right-motor recirc leaf command to three-leaf 1:2:1 opening state."""
    state_index = _snap_recirc_fraction_cmd(float(recirc_fraction_cmd))
    areas = _RECIRC_DOOR_AREAS
    total = _RECIRC_DOOR_TOTAL_AREA

    if state_index == 0:
        leaf_left = leaf_mid = leaf_right = False
        combo_note = "none_open"
        side_equivalent = False
        pending = False
    elif state_index == 25:
        leaf_left, leaf_mid, leaf_right = True, False, False
        combo_note = "small_leaf_open_side_equivalent"
        side_equivalent = True
        pending = False
    elif state_index == 50:
        leaf_left, leaf_mid, leaf_right = False, True, False
        combo_note = "middle_only"
        side_equivalent = False
        pending = False
    elif state_index == 75:
        leaf_left, leaf_mid, leaf_right = True, True, False
        combo_note = "middle_plus_small_leaf"
        side_equivalent = False
        pending = False
    else:
        leaf_left = leaf_mid = leaf_right = True
        combo_note = "all_leaves_open_full_recirc_path"
        side_equivalent = False
        pending = False

    open_area = (
        (areas[0] if leaf_left else 0)
        + (areas[1] if leaf_mid else 0)
        + (areas[2] if leaf_right else 0)
    )
    area_ratio_open = open_area / total

    return M8RecircLeafState(
        leaf_small_left_open=leaf_left,
        leaf_middle_open=leaf_mid,
        leaf_small_right_open=leaf_right,
        area_ratio_open=area_ratio_open,
        total_recirc_fraction_cmd=state_index / 100.0,
        provenance={
            "method": "three_leaf_discrete_area_model",
            "door_area_ratio": "1:2:1",
            "state_index": state_index,
            "door_combination_note": combo_note,
            "side_equivalent": side_equivalent if state_index == 25 else False,
            "door_combination_pending_confirm": pending,
            "left_motor_controls": list(LEFT_MOTOR_CONTROLS),
            "right_motor_controls": list(RIGHT_MOTOR_CONTROLS),
            "expected_area_ratio_open": state_index / 100.0,
            "matches_discrete_index": math.isclose(
                area_ratio_open, state_index / 100.0, rel_tol=0, abs_tol=1e-9
            ),
        },
    )


def decode_left_intake_motor_state(
    voltage: float,
    *,
    tolerance_v: float = _LEFT_MOTOR_MATCH_TOLERANCE_V,
) -> Tuple[LeftMotorState, Dict[str, Any]]:
    """Decode left (main) intake motor voltage → recirc / fresh / dual_layer."""
    v = _finite_or_raise(voltage, "voltage")
    best_anchor_v, best_state, best_label = min(
        _LEFT_INTAKE_MOTOR_ANCHORS,
        key=lambda row: abs(row[0] - v),
    )
    dist = abs(v - best_anchor_v)
    matched = dist <= tolerance_v
    state: LeftMotorState = best_state if matched else "unknown"
    prov = {
        "method": "left_main_intake_motor_voltage_anchor",
        "requested_voltage_v": v,
        "matched_anchor_v": best_anchor_v,
        "matched_anchor_label": best_label,
        "anchor_distance_v": dist,
        "tolerance_v": tolerance_v,
        "left_motor_role": "main_intake_state_selector",
    }
    return state, prov


def decode_recirc_three_door_model(state_index: int) -> RecircThreeDoorState:
    """Legacy wrapper — maps discrete index via ``decode_recirc_leaf_state``."""
    leaf = decode_recirc_leaf_state(int(state_index))
    open_doors: Tuple[str, ...] = ()
    if leaf.leaf_small_left_open:
        open_doors += ("small_left",)
    if leaf.leaf_middle_open:
        open_doors += ("middle",)
    if leaf.leaf_small_right_open:
        open_doors += ("small_right",)
    return RecircThreeDoorState(
        state_index=int(state_index),
        total_recirc_fraction=leaf.total_recirc_fraction_cmd,
        open_doors=open_doors,
        open_area_ratio=leaf.area_ratio_open,
        door_area_ratio=_RECIRC_DOOR_AREAS,
        provenance=dict(leaf.provenance),
    )


def _infer_left_from_mode_pct(mode_pct: float) -> LeftMotorState:
    """Legacy proxy when explicit left motor voltage is unavailable."""
    idx = _snap_recirc_fraction_cmd(mode_pct)
    if idx == 0:
        return "fresh"
    if idx == 100:
        return "recirc"
    return "dual_layer"


def _snap_right_motor_to_leaf_index(
    right_motor_voltage_v: float,
    circle_mode_posn_pct: float,
) -> Tuple[int, Dict[str, Any]]:
    """Map right-motor voltage to discrete leaf index with mode_pct tie-break."""
    v = _finite_or_raise(right_motor_voltage_v, "right_motor_voltage_v")
    mode_pct = _finite_or_raise(circle_mode_posn_pct, "circle_mode_posn_pct")

    best: Optional[Tuple[float, int, str]] = None
    ties: list[Tuple[float, int, str]] = []
    for anchor_v, state, label in _RIGHT_MOTOR_LEAF_ANCHORS:
        dist = abs(v - anchor_v)
        if best is None or dist < best[0]:
            best = (dist, state, label)
            ties = [(dist, state, label)]
        elif dist == best[0]:
            ties.append((dist, state, label))

    assert best is not None
    snap_meta: Dict[str, Any] = {
        "snap_method": "nearest_right_motor_voltage_anchor",
        "no_linear_extrapolation": True,
        "requested_right_motor_v": v,
        "anchor_distance_v": best[0],
        "matched_anchor_label": best[2],
    }

    if len(ties) > 1:
        bucket = _snap_recirc_fraction_cmd(mode_pct)
        candidates = {t[1] for t in ties}
        if bucket in candidates:
            state_index = bucket
            snap_meta["tie_break"] = "circle_mode_posn_pct_bucket"
            snap_meta["tie_candidates"] = sorted(candidates)
        else:
            state_index = min(candidates)
            snap_meta["tie_break"] = "lowest_state_index"
            snap_meta["tie_candidates"] = sorted(candidates)
    else:
        state_index = best[1]

    snap_meta["snapped_leaf_state_index"] = state_index
    return state_index, snap_meta


def build_m8_intake_mechanism_state(
    circle_mode_posn_pct: float,
    circle_prior_posn_v: float,
    *,
    left_motor_voltage_v: Optional[float] = None,
    state_index_override: Optional[int] = None,
) -> M8IntakeMechanismState:
    """Assemble left/right motor roles and leaf state from CAN proxy signals."""
    mode_pct = _finite_or_raise(circle_mode_posn_pct, "circle_mode_posn_pct")
    right_v = _finite_or_raise(circle_prior_posn_v, "circle_prior_posn_v")

    if left_motor_voltage_v is not None:
        left_state, left_prov = decode_left_intake_motor_state(left_motor_voltage_v)
        left_prov["source_signal"] = "explicit_left_motor_voltage_v"
    else:
        left_state = _infer_left_from_mode_pct(mode_pct)
        left_prov = {
            "method": "inferred_from_circle_mode_posn_pct",
            "inferred_left_state": left_state,
            "note": "pass left_motor_voltage_v for measured left-motor decode",
        }

    right_snap_meta: Dict[str, Any] = {}
    if state_index_override is not None:
        leaf_index = int(state_index_override)
        leaf_cmd_source = "explicit_state_index_override"
    elif left_state == "recirc":
        leaf_index = 100
        leaf_cmd_source = "forced_full_recirc_from_left_recirc_mode"
    elif left_state == "fresh":
        leaf_index = 0
        leaf_cmd_source = "forced_zero_recirc_from_left_fresh_mode"
    elif left_motor_voltage_v is not None:
        leaf_index = _snap_recirc_fraction_cmd(mode_pct)
        leaf_cmd_source = "circle_mode_posn_pct_leaf_cmd_with_left_dual_layer"
    else:
        leaf_index, right_snap_meta = _snap_right_motor_to_leaf_index(right_v, mode_pct)
        leaf_cmd_source = "right_motor_voltage_with_mode_pct_tie_break"

    leaf = decode_recirc_leaf_state(leaf_index)

    if left_state == "dual_layer":
        fresh_policy = "fixed_at_left_motor_middle_position_in_dual_layer"
        split_policy = "pending_customer_confirmation"
    elif left_state == "fresh":
        fresh_policy = "full_fresh_opening_left_motor_endstop"
        split_policy = "uniform_both_layers_no_split"
    elif left_state == "recirc":
        fresh_policy = "minimal_fresh_opening_left_motor_endstop"
        split_policy = "uniform_both_layers_no_split"
    else:
        fresh_policy = "unknown_pending_left_motor_calibration"
        split_policy = "pending_customer_confirmation"

    prov: Dict[str, Any] = {
        "method": "m8_intake_mechanism_hypothesis",
        "customer_confirmed": _CUSTOMER_CONFIRMED_TAG,
        "left_motor_controls": list(LEFT_MOTOR_CONTROLS),
        "right_motor_controls": list(RIGHT_MOTOR_CONTROLS),
        "left_main_intake_selector": True,
        "right_recirc_leaf_selector": True,
        "fresh_opening_fixed_in_dual_layer": left_state == "dual_layer",
        "upper_lower_split_pending_customer_confirmation": split_policy.startswith("pending"),
        "left_motor": left_prov,
        "right_motor_voltage_v": right_v,
        "right_leaf_cmd_source": leaf_cmd_source,
        "snapped_leaf_state_index": leaf_index,
        "circle_mode_posn_pct": mode_pct,
        **right_snap_meta,
    }
    if left_motor_voltage_v is not None:
        prov["left_motor_voltage_v"] = float(left_motor_voltage_v)

    return M8IntakeMechanismState(
        left_motor_role="main_intake_state_selector",
        left_motor_state=left_state,
        right_motor_role="recirc_door_leaf_selector",
        recirc_leaf_state=leaf,
        fresh_opening_policy=fresh_policy,
        upper_lower_split_policy=split_policy,
        provenance=prov,
    )


_MODE_LAYER_TABLE: Dict[str, Dict[str, str]] = {
    "V": {
        "face_layer": "upper_confirmed",
        "defrost_layer": "closed_or_minimal",
        "foot_layer": "closed_or_minimal",
    },
    "V_F": {
        "face_layer": "upper_confirmed",
        "defrost_layer": "closed_or_minimal",
        "foot_layer": "lower_confirmed",
    },
    "V_D": {
        "face_layer": "upper_confirmed",
        "defrost_layer": "upper_confirmed",
        "foot_layer": "closed_or_minimal",
    },
    "V_D_F": {
        "face_layer": "upper_confirmed",
        "defrost_layer": "upper_confirmed",
        "foot_layer": "lower_confirmed",
    },
    "F": {
        "face_layer": "closed_or_minimal",
        "defrost_layer": "upper_intentional_bleed",
        "foot_layer": "lower_confirmed",
    },
    "F_D": {
        "face_layer": "closed_or_minimal",
        "defrost_layer": "upper_confirmed",
        "foot_layer": "lower_confirmed",
    },
    "D": {
        "face_layer": "closed_or_minimal",
        "defrost_layer": "upper_confirmed",
        "foot_layer": "closed_or_minimal",
    },
    "OFF": {
        "face_layer": "closed_or_minimal",
        "defrost_layer": "closed_or_minimal",
        "foot_layer": "closed_or_minimal",
    },
}


def _placeholder_layer_fractions(
    mechanism: M8IntakeMechanismState,
    lut: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, float, float, float, str, Dict[str, Any]]:
    """Resolve per-layer fractions — LUT draft only when dual_layer; never as physics fact."""
    left = mechanism.left_motor_state
    leaf_idx = int(round(mechanism.recirc_leaf_state.total_recirc_fraction_cmd * 100))

    if left == "recirc":
        meta = {
            "layer_fractions_source": "uniform_full_recirc_both_layers",
            "upper_lower_split_pending_customer_confirmation": False,
        }
        return 1.0, 1.0, 0.0, 0.0, "uniform_recirc_mode", meta

    if left == "fresh":
        meta = {
            "layer_fractions_source": "uniform_full_fresh_both_layers",
            "upper_lower_split_pending_customer_confirmation": False,
        }
        return 0.0, 0.0, 1.0, 1.0, "uniform_fresh_mode", meta

    row = lut[str(leaf_idx)]
    meta = {
        "layer_fractions_source": "placeholder_lut_draft_not_sign_off",
        "upper_lower_split_pending_customer_confirmation": True,
        "upper_fresh_fraction_status": "placeholder_pending",
        "lower_recirc_fraction_status": "placeholder_pending",
        "not_hardcoded_upper_100pct_fresh_lower_100pct_recirc_as_physics": True,
        "lut_reference_state_index": leaf_idx,
    }
    return (
        float(row["upper_recirc_ratio"]),
        float(row["lower_recirc_ratio"]),
        float(row["upper_fresh_ratio"]),
        float(row["lower_fresh_ratio"]),
        "dual_layer_placeholder_lut",
        meta,
    )


def is_heating_mode(
    driver_temp_door_posn: float,
    passenger_temp_door_posn: float,
    *,
    threshold: float = 0.55,
) -> bool:
    """Heating vs cooling proxy from blend-door normalized positions."""
    avg = 0.5 * (
        _finite_or_raise(driver_temp_door_posn, "driver_temp_door_posn")
        + _finite_or_raise(passenger_temp_door_posn, "passenger_temp_door_posn")
    )
    return avg >= threshold


def compute_dual_layer_active(
    mode_code: Union[str, int],
    heating_mode: bool,
    *,
    include_foot_only: bool = True,
) -> bool:
    """Dual-layer path separation active only in selected heating modes."""
    code = _normalize_mode(mode_code)
    if not heating_mode:
        return False
    modes = set(_HEATING_DUAL_LAYER_MODES)
    if not include_foot_only:
        modes.discard("F")
    return code in modes


def effective_intake_layer_fractions(
    intake: M8IntakeState,
    dual_layer_active: bool,
) -> Tuple[float, float, float, float]:
    """Return (upper_recirc, lower_recirc, upper_fresh, lower_fresh) for solver."""
    if dual_layer_active:
        return (
            intake.upper_recirc_fraction,
            intake.lower_recirc_fraction,
            intake.upper_fresh_fraction,
            intake.lower_fresh_fraction,
        )
    tr = intake.recirc_fraction_total
    fr = 1.0 - tr
    return tr, tr, fr, fr


def decode_m8_intake_state(
    circle_mode_posn_pct: float,
    circle_prior_posn_v: float,
    mode_code: Optional[Union[str, int]] = None,
    *,
    left_motor_voltage_v: Optional[float] = None,
    state_index_override: Optional[int] = None,
) -> M8IntakeState:
    """Decode M8 intake via left/right motor mechanism hypothesis (decode-only)."""
    doc = load_m8_dual_layer_structure_confirmed()
    intake = doc["intake_actuators"]
    lut = doc["intake_state_lut"]

    mode_pct = _finite_or_raise(circle_mode_posn_pct, "circle_mode_posn_pct")
    prior_v = _finite_or_raise(circle_prior_posn_v, "circle_prior_posn_v")

    mechanism = build_m8_intake_mechanism_state(
        mode_pct,
        prior_v,
        left_motor_voltage_v=left_motor_voltage_v,
        state_index_override=state_index_override,
    )
    leaf = mechanism.recirc_leaf_state
    state_index = int(round(leaf.total_recirc_fraction_cmd * 100))

    ur, lr, uf, lf, layer_source, layer_meta = _placeholder_layer_fractions(mechanism, lut)

    if mechanism.left_motor_state == "dual_layer":
        confidence = "intake_mechanism_hypothesis_draft"
    elif mechanism.left_motor_state in ("recirc", "fresh"):
        confidence = "left_motor_mode_uniform_layers_draft"
    else:
        confidence = "intake_mechanism_unknown_left_motor_draft"

    prov: Dict[str, Any] = {
        "method": "m8_v2_intake_mechanism_hypothesis",
        "source": doc.get("source"),
        "customer_update": "2026-06-07_dual_motor_intake_mechanism",
        "customer_confirmed": _CUSTOMER_CONFIRMED_TAG,
        **intake_motor_control_ownership(),
        "circle_mode_signal": intake["circle_mode_signal"],
        "circle_mode_role": "intake_mode_pct_or_leaf_cmd_proxy",
        "circle_prior_signal": intake["circle_prior_signal"],
        "circle_prior_role": "right_recirc_leaf_motor_voltage_proxy",
        "left_main_intake_selector": True,
        "right_recirc_leaf_selector": True,
        "fresh_opening_fixed_in_dual_layer": (
            mechanism.fresh_opening_policy
            == "fixed_at_left_motor_middle_position_in_dual_layer"
        ),
        "upper_lower_split_pending_customer_confirmation": layer_meta.get(
            "upper_lower_split_pending_customer_confirmation", True
        ),
        "layer_split_rule_id": intake["layer_split_rule_id"],
        "layer_split_note": layer_meta.get("layer_fractions_source", "pending"),
        "layer_fractions_resolution": layer_source,
        "linked_motors": intake["linked_motors"],
        "not_wired_to": doc.get("not_wired_to"),
        "intake_mechanism": {
            "left_motor_role": mechanism.left_motor_role,
            "left_motor_state": mechanism.left_motor_state,
            "right_motor_role": mechanism.right_motor_role,
            "fresh_opening_policy": mechanism.fresh_opening_policy,
            "upper_lower_split_policy": mechanism.upper_lower_split_policy,
            **mechanism.provenance,
        },
        "recirc_leaf_state": {
            "leaf_small_left_open": leaf.leaf_small_left_open,
            "leaf_middle_open": leaf.leaf_middle_open,
            "leaf_small_right_open": leaf.leaf_small_right_open,
            "area_ratio_open": leaf.area_ratio_open,
            "total_recirc_fraction_cmd": leaf.total_recirc_fraction_cmd,
            **leaf.provenance,
        },
        "three_door_model": leaf.provenance,
        "total_recirc_fraction_discrete": leaf.total_recirc_fraction_cmd,
        **layer_meta,
    }
    if mode_code is not None:
        prov["mode_code"] = _normalize_mode(mode_code)

    return M8IntakeState(
        circle_mode_posn_pct=mode_pct,
        circle_prior_posn_v=prior_v,
        recirc_fraction_total=leaf.total_recirc_fraction_cmd,
        upper_recirc_fraction=ur,
        lower_recirc_fraction=lr,
        upper_fresh_fraction=uf,
        lower_fresh_fraction=lf,
        confidence=confidence,
        provenance=prov,
    )


def decode_m8_mode_layer_distribution(
    mode_code: Union[str, int],
    *,
    heating_mode: Optional[bool] = None,
) -> M8ModeLayerDistribution:
    """Decode outlet-to-layer assignment (customer confirmed 2026-06-07)."""
    doc = load_m8_dual_layer_structure_confirmed()
    code = _normalize_mode(mode_code)
    table = _MODE_LAYER_TABLE.get(code, _MODE_LAYER_TABLE["OFF"])

    face_layer = table["face_layer"]
    defrost_layer = table["defrost_layer"]
    foot_layer = table["foot_layer"]
    rear_layer = "lower_confirmed"
    rear_face_policy = "lower_confirmed"
    rear_foot_policy = "lower_confirmed"

    dual_active: Optional[bool] = None
    if heating_mode is not None:
        dual_active = compute_dual_layer_active(code, heating_mode)

    prov: Dict[str, Any] = {
        "method": "m8_v2_mode_layer_customer_confirmed",
        "source": doc.get("source"),
        "customer_confirmed": _CUSTOMER_CONFIRMED_TAG,
        "customer_update": "2026-06-07_face_upper_foot_rear_lower",
        "mode_code": code,
        "confirmed_outlet_layers": confirmed_outlet_layer_assignments(),
        "rear_face_policy": rear_face_policy,
        "rear_foot_policy": rear_foot_policy,
        "face_mixed_superseded": {
            "status": "historical_smoke_fit_only",
            "note": "face=mixed policy-scan result superseded by customer structure feedback",
            "not_physical_structure": True,
        },
        "blower_model": {
            "front": "front_hvac_blower",
            "rear": "rear_booster_blower",
            "rejected": "coaxial_single_motor_model",
        },
        "heat_model": {
            "heater_core": "shared_core_upper_lower_channels_pending_detail",
            "water_ptc": "pending",
            "rejected": "air_ptc",
        },
        "isolated_until": doc["layer_separation"]["isolated_until"],
        "not_wired_to": doc.get("not_wired_to"),
    }

    if code == "F":
        prov["foot_only_lower"] = True
        prov["defrost_intentional_bleed"] = True
    elif code in ("F_D", "V_D_F"):
        prov["foot_defrost_split"] = True
        prov["foot_defrost_note"] = assign_note(doc, code)

    if heating_mode is not None:
        prov["heating_mode"] = heating_mode
        prov["dual_layer_active"] = dual_active

    prov["temperature_policy"] = {
        "rear_face_sensor": doc["temperature_sensors"]["rear_face"]["placement"],
        "rear_foot_sensor": doc["temperature_sensors"]["rear_foot"]["available"],
        "defrost_sensor": doc["temperature_sensors"]["defrost"]["available"],
        "rear_face_shared": doc["temperature_sensors"]["rear_face"]["shared_across_outlets"],
    }

    return M8ModeLayerDistribution(
        mode_code=code,
        defrost_layer=defrost_layer,
        face_layer=face_layer,
        foot_layer=foot_layer,
        rear_layer=rear_layer,
        rear_face_policy=rear_face_policy,
        rear_foot_policy=rear_foot_policy,
        dual_layer_active=dual_active,
        heating_mode=heating_mode,
        provenance=prov,
    )


def assign_note(doc: Mapping[str, Any], code: str) -> str:
    assign = doc["mode_layer_assignment"]
    fd = assign.get("foot_defrost", {})
    return str(fd.get("note", f"{code} foot+defrost split"))


def build_m8_tma_mapping(
    *,
    drvr_face_act_t_c: Optional[float] = None,
    pass_face_act_t_c: Optional[float] = None,
    drvr_foot_act_t_c: Optional[float] = None,
    pass_foot_act_t_c: Optional[float] = None,
    sec_row_face_act_t_c: Optional[float] = None,
    eva_temp_c: Optional[float] = None,
    hct_temp_c: Optional[float] = None,
    posn_fdh: Optional[float] = None,
    mode_code: Optional[Union[str, int]] = None,
) -> M8LayerTemperatureState:
    """Map CAN/SIG actual vent temperatures to M8 layer temperature state."""
    doc = load_m8_dual_layer_structure_confirmed()
    prov: Dict[str, Any] = {
        "method": "m8_v2_tma_mapping",
        "source": doc.get("source"),
        "not_wired_to": doc.get("not_wired_to"),
        "heat_source_note": "heater_core_water_ptc_pending_not_air_ptc",
        "signals": {
            "front_face_driver": "AC_DrvrFaceVentActT",
            "front_face_passenger": "AC_PassFaceVentActT",
            "front_foot_driver": "AC_DrvrFootVentActT",
            "front_foot_passenger": "AC_PassFootVentActT",
            "rear_face_channel": "AC_SecRowFaceVentActT",
        },
    }
    if mode_code is not None:
        prov["mode_code"] = _normalize_mode(mode_code)

    t_defrost: Optional[float] = None
    defrost_prov: Dict[str, Any] = {}
    if eva_temp_c is not None:
        tma_def = estimate_tma_def(
            eva_temp_c=float(eva_temp_c),
            hct_temp_c=hct_temp_c,
            posn_fdh=posn_fdh,
        )
        t_defrost = tma_def.tma_def_c
        defrost_prov = {
            "method": "estimate_tma_def",
            "tmadef_approx": tma_def.tmadef_approx,
            "diagnostics": dict(tma_def.diagnostics),
        }
    prov["defrost"] = {
        "sensor_available": doc["temperature_sensors"]["defrost"]["available"],
        **defrost_prov,
    }

    rear_foot_note = doc["temperature_sensors"]["rear_foot"]["note"]
    prov["rear_foot"] = {
        "independent_sensor": False,
        "policy": "shared_rear_channel_estimate",
        "layer_policy": "lower_confirmed",
        "note": rear_foot_note,
    }
    if sec_row_face_act_t_c is not None and drvr_foot_act_t_c is None and pass_foot_act_t_c is None:
        prov["rear_foot"]["estimate_source"] = "sec_row_face_channel_proxy_only"

    return M8LayerTemperatureState(
        t_upper_driver_c=drvr_face_act_t_c,
        t_upper_passenger_c=pass_face_act_t_c,
        t_lower_driver_c=drvr_foot_act_t_c,
        t_lower_passenger_c=pass_foot_act_t_c,
        t_rear_channel_c=sec_row_face_act_t_c,
        t_defrost_c=t_defrost,
        provenance=prov,
    )


def placeholder_m8_layer_flow_state(
    *,
    reason: str = "solver_not_implemented",
) -> M8LayerFlowState:
    """Return NaN flow placeholders until AFE v2 resistance network exists."""
    nan = float("nan")
    return M8LayerFlowState(
        q_upper_m3h=nan,
        q_lower_m3h=nan,
        q_front_face_m3h=nan,
        q_front_foot_m3h=nan,
        q_defrost_m3h=nan,
        q_rear_face_m3h=nan,
        q_rear_foot_m3h=nan,
        provenance={
            "status": "placeholder",
            "reason": reason,
            "not_wired_to": "afe_calc",
            "no_silent_flow_assignment": True,
        },
    )


def rear_temperature_sharing_policy_v2() -> Dict[str, Any]:
    """Confirmed second-row Tma sharing rules for M8 v2 draft."""
    doc = load_m8_dual_layer_structure_confirmed()
    rear = doc["temperature_sensors"]["rear_face"]
    foot = doc["temperature_sensors"]["rear_foot"]
    return {
        "sensor_placement": rear["placement"],
        "sensor_placement_cn": rear["placement_cn"],
        "shared_tma_across_outlets": rear["shared_across_outlets"],
        "outlets_shared": list(rear["outlets_shared"]),
        "rear_foot_independent_tma": foot["available"],
        "rear_foot_policy": "lower_confirmed_shared_channel_estimate",
        "rear_face_policy": "lower_confirmed",
        "provenance": {
            "source": doc.get("source"),
            "confirmed_by": doc.get("confirmed_by"),
            "module": "m8_dual_layer_model",
            "customer_update": "2026-06-07_rear_lower_confirmed",
            "customer_confirmed": _CUSTOMER_CONFIRMED_TAG,
        },
    }
