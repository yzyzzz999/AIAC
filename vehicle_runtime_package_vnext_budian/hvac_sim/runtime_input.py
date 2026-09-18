"""Canonical runtime-input types and legacy dictionary normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from hvac_sim.occupant import ImageModuleInputs, SeatOccupantInput
from hvac_sim.runtime_parameter_mode import CHTD_PARAM_MODE_DEFAULT, resolve_chtd_param_mode


DEFAULT_RH_PERCENT = 50.0


@dataclass(frozen=True)
class RuntimeFeatureFlags:
    enable_ir_fusion: bool = False
    enable_image_occupancy: bool = True
    enable_post_glass_defrost: bool = False
    enable_geometry_merge: bool = False
    enable_corrected_chtd: bool = False
    enable_vehicle_adapted_params: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "enable_ir_fusion": self.enable_ir_fusion,
            "enable_image_occupancy": self.enable_image_occupancy,
            "enable_post_glass_defrost": self.enable_post_glass_defrost,
            "enable_geometry_merge": self.enable_geometry_merge,
            "enable_corrected_chtd": self.enable_corrected_chtd,
            "enable_vehicle_adapted_params": self.enable_vehicle_adapted_params,
        }

    @property
    def disabled_feature_names(self) -> List[str]:
        return [
            name
            for name, enabled in (
                ("ir_fusion", self.enable_ir_fusion),
                ("image_occupancy", self.enable_image_occupancy),
                ("post_glass_defrost", self.enable_post_glass_defrost),
                ("geometry_merge", self.enable_geometry_merge),
                ("corrected_chtd", self.enable_corrected_chtd),
                ("vehicle_adapted_params", self.enable_vehicle_adapted_params),
            )
            if not enabled
        ]

    @property
    def active_feature_names(self) -> List[str]:
        all_names = {
            "ir_fusion",
            "image_occupancy",
            "post_glass_defrost",
            "geometry_merge",
            "corrected_chtd",
            "vehicle_adapted_params",
        }
        return sorted(all_names - set(self.disabled_feature_names))


@dataclass
class RuntimeSignalInput:
    amb_t_c: float
    raw_amb_t_c: Optional[float] = None
    rh_percent: float = DEFAULT_RH_PERCENT
    vehicle_speed_kph: Optional[float] = None
    solar_driver_w_m2: Optional[float] = None
    solar_passenger_w_m2: Optional[float] = None
    eva_t_c: Optional[float] = None
    hct_c: Optional[float] = None
    posn_fdh: Optional[float] = None
    blend_request: Optional[float] = None
    frnt_def_tma_est_c: Optional[float] = None
    windshield_glass_temp_c: Optional[float] = None
    ac_mode_ventila_posn: Optional[int] = None
    win_shd_t_est_c: Optional[float] = None
    ict_c: Optional[float] = None
    driver_face_flow: Optional[float] = None
    passenger_face_flow: Optional[float] = None
    driver_floor_flow: Optional[float] = None
    passenger_floor_flow: Optional[float] = None
    driver_defrost_flow: Optional[float] = None
    passenger_defrost_flow: Optional[float] = None
    measured_driver_head_air_temp_c: Optional[float] = None
    measured_passenger_head_air_temp_c: Optional[float] = None
    tma_c: Dict[str, float] = field(default_factory=dict)
    flows: Dict[str, float] = field(default_factory=dict)
    image_inputs: Optional[ImageModuleInputs] = None
    use_ir_head_fusion: bool = False
    enable_image_occupancy: bool = True
    enable_post_glass_defrost: bool = False
    enable_corrected_chtd: bool = False
    apply_geometry: bool = False
    geometry_path: Optional[str] = None
    apply_runtime_policy: bool = True
    use_next_state: bool = True
    params_bundle_path: Optional[str] = None
    use_initial_calibration_params: bool = False
    chtd_param_mode: str = CHTD_PARAM_MODE_DEFAULT
    bypass_models: bool = False
    driver_air_temp_override_c: Optional[float] = None
    passenger_air_temp_override_c: Optional[float] = None
    driver_mrt_override_c: Optional[float] = None
    passenger_mrt_override_c: Optional[float] = None
    driver_air_speed_override_m_s: Optional[float] = None
    passenger_air_speed_override_m_s: Optional[float] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RuntimeSignalInput":
        return runtime_mapping_to_signal_input(raw)


def first_present(raw: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return None


def as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def mean_optional(*values: Any) -> Optional[float]:
    finite = [value for item in values if (value := as_optional_float(item)) is not None]
    return sum(finite) / len(finite) if finite else None


def resolve_runtime_feature_flags(raw: Mapping[str, Any]) -> RuntimeFeatureFlags:
    def flag(*keys: str, default: bool = False) -> bool:
        for key in keys:
            if key in raw and raw[key] is not None:
                return bool(raw[key])
        return default

    vehicle_params = flag(
        "enable_vehicle_adapted_params",
        "use_initial_calibration_params",
    ) or first_present(
        raw,
        "params_bundle_path",
        "initial_params_path",
        "param_bundle_path",
    ) is not None
    return RuntimeFeatureFlags(
        enable_ir_fusion=flag("enable_ir_fusion", "use_ir_head_fusion"),
        enable_image_occupancy=flag("enable_image_occupancy", default=True),
        enable_post_glass_defrost=flag("enable_post_glass_defrost"),
        enable_geometry_merge=flag("enable_geometry_merge", "apply_geometry"),
        enable_corrected_chtd=flag("enable_corrected_chtd"),
        enable_vehicle_adapted_params=bool(vehicle_params),
    )


def feature_flags_from_signal(signal: RuntimeSignalInput) -> RuntimeFeatureFlags:
    return RuntimeFeatureFlags(
        enable_ir_fusion=bool(signal.use_ir_head_fusion),
        enable_image_occupancy=bool(signal.enable_image_occupancy),
        enable_post_glass_defrost=bool(signal.enable_post_glass_defrost),
        enable_geometry_merge=bool(signal.apply_geometry),
        enable_corrected_chtd=bool(signal.enable_corrected_chtd),
        enable_vehicle_adapted_params=bool(
            signal.use_initial_calibration_params or signal.params_bundle_path is not None
        ),
    )


def _parse_seat(raw: Any) -> SeatOccupantInput:
    if raw is None:
        return SeatOccupantInput()
    if not isinstance(raw, dict):
        raise ValueError("image_inputs seat entry must be an object")
    return SeatOccupantInput(
        occupied=raw.get("occupied"),
        age=raw.get("age"),
        gender=raw.get("gender"),
        height_cm=raw.get("height_cm"),
        weight_kg=raw.get("weight_kg"),
        bmi=raw.get("bmi"),
        clothing_clo=raw.get("clothing_clo"),
        clothing_class=raw.get("clothing_class"),
        activity_met=raw.get("activity_met"),
        ir_head_surface_temp_c=raw.get("ir_head_surface_temp_c"),
        confidence=raw.get("confidence"),
    )


def _parse_image_inputs(raw: Any) -> Optional[ImageModuleInputs]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("image_inputs must be an object when present")
    return ImageModuleInputs(
        driver=_parse_seat(raw.get("driver")),
        passenger=_parse_seat(raw.get("passenger")),
    )


def runtime_mapping_to_signal_input(raw: Mapping[str, Any]) -> RuntimeSignalInput:
    if "amb_t_c" not in raw:
        raise ValueError("runtime input requires amb_t_c")

    def numeric_map(name: str) -> Dict[str, float]:
        block = raw.get(name)
        if not isinstance(block, dict):
            return {}
        return {
            str(key): value
            for key, item in block.items()
            if (value := as_optional_float(item)) is not None
        }

    measured = raw.get("measured_head_air_temp_c")
    measured = measured if isinstance(measured, Mapping) else {}
    flags = resolve_runtime_feature_flags(raw)
    mode_position = first_present(raw, "ac_mode_ventila_posn", "mode_ventila_posn")
    params_path = first_present(
        raw,
        "params_bundle_path",
        "initial_params_path",
        "param_bundle_path",
    )
    return RuntimeSignalInput(
        amb_t_c=float(raw["amb_t_c"]),
        raw_amb_t_c=as_optional_float(
            first_present(raw, "raw_amb_t_c", "raw_ambient_temp_c")
        ),
        rh_percent=float(
            first_present(raw, "rh_percent", "rh", "relative_humidity_percent")
            or DEFAULT_RH_PERCENT
        ),
        vehicle_speed_kph=as_optional_float(
            first_present(raw, "vehicle_speed_kph", "veh_spd_kph", "veh_spd")
        ),
        solar_driver_w_m2=as_optional_float(
            first_present(raw, "solar_driver_w_m2", "solar_fd_w_m2", "solar_fd")
        ),
        solar_passenger_w_m2=as_optional_float(
            first_present(raw, "solar_passenger_w_m2", "solar_fp_w_m2", "solar_fp")
        ),
        eva_t_c=as_optional_float(
            first_present(
                raw,
                "eva_t_c",
                "ac_evap_temp_c",
                "ac_fevap_current_temp_c",
            )
        ),
        hct_c=as_optional_float(
            first_present(raw, "hct_c", "heater_core_outlet_temp_c")
        ),
        posn_fdh=as_optional_float(
            first_present(raw, "posn_fdh", "defrost_mix_position")
        ),
        blend_request=as_optional_float(
            first_present(raw, "blend_request", "defrost_blend_request")
        ),
        frnt_def_tma_est_c=as_optional_float(
            first_present(raw, "frnt_def_tma_est_c", "tma_def_c")
        ),
        windshield_glass_temp_c=as_optional_float(
            first_present(
                raw,
                "windshield_glass_temp_c",
                "glass_temp_c",
                "windshield_temp_state_c",
            )
        ),
        ac_mode_ventila_posn=int(mode_position) if mode_position is not None else None,
        win_shd_t_est_c=as_optional_float(
            first_present(raw, "win_shd_t_est_c", "windshield_temp_c")
        ),
        ict_c=as_optional_float(raw.get("ict_c")),
        driver_face_flow=as_optional_float(raw.get("driver_face_flow")),
        passenger_face_flow=as_optional_float(raw.get("passenger_face_flow")),
        driver_floor_flow=as_optional_float(raw.get("driver_floor_flow")),
        passenger_floor_flow=as_optional_float(raw.get("passenger_floor_flow")),
        driver_defrost_flow=as_optional_float(raw.get("driver_defrost_flow")),
        passenger_defrost_flow=as_optional_float(raw.get("passenger_defrost_flow")),
        measured_driver_head_air_temp_c=mean_optional(
            measured.get("driver_left"), measured.get("driver_right")
        ),
        measured_passenger_head_air_temp_c=mean_optional(
            measured.get("passenger_left"), measured.get("passenger_right")
        ),
        tma_c=numeric_map("tma"),
        flows=numeric_map("flows"),
        image_inputs=_parse_image_inputs(raw.get("image_inputs")),
        use_ir_head_fusion=flags.enable_ir_fusion,
        enable_image_occupancy=flags.enable_image_occupancy,
        enable_post_glass_defrost=flags.enable_post_glass_defrost,
        enable_corrected_chtd=flags.enable_corrected_chtd,
        apply_geometry=flags.enable_geometry_merge,
        geometry_path=(
            str(raw["geometry_path"])
            if raw.get("geometry_path") is not None
            else None
        ),
        apply_runtime_policy=bool(raw.get("apply_runtime_policy", True)),
        use_next_state=bool(raw.get("use_next_state", True)),
        params_bundle_path=str(params_path) if params_path is not None else None,
        use_initial_calibration_params=flags.enable_vehicle_adapted_params,
        chtd_param_mode=resolve_chtd_param_mode(raw),
        bypass_models=bool(raw.get("bypass_models", False)),
        driver_air_temp_override_c=as_optional_float(raw.get("driver_air_temp_override_c")),
        passenger_air_temp_override_c=as_optional_float(raw.get("passenger_air_temp_override_c")),
        driver_mrt_override_c=as_optional_float(raw.get("driver_mrt_override_c")),
        passenger_mrt_override_c=as_optional_float(raw.get("passenger_mrt_override_c")),
        driver_air_speed_override_m_s=as_optional_float(raw.get("driver_air_speed_override_m_s")),
        passenger_air_speed_override_m_s=as_optional_float(
            raw.get("passenger_air_speed_override_m_s")
        ),
    )
