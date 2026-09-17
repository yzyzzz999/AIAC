"""Vehicle comfort interface — maps cabin conditions to Fanger PMV/PPD."""

from __future__ import annotations

from dataclasses import dataclass

from .fanger import pmv_ppd


@dataclass(frozen=True)
class VehicleComfortInputs:
    """Cabin comfort inputs for Fanger PMV/PPD.

    All temperature fields are populated from CHTD model outputs (x_next after one_step_chtd),
    NOT from external thermocouple or sensor measurements.

    Runtime mapping (vehicle deployment path):
        air_temp_c            ← x_next[X_INDEX['HeadTempFd']]  (CHTD estimated, driver side)
                                FORBIDDEN source: TA_FdHeadTempLe / any thermocouple
        mean_radiant_temp_c   ← mean(x_next[HeadTempFd, CabinTempFd, WinTempFd, RoofTemp])
                                All components are CHTD model states, not measured temperatures
        air_velocity_m_s      ← derived from FTE duct airflow (SIG/AFE)
        relative_humidity_pct ← SIG rh_percent (e.g. SIG_AmbRelHumFb or VIU_RelHum)
                                MUST NOT use PMV_RH_DEFAULT_PCT=0 fallback in vehicle runtime

    Offline calibration / validation only (NOT runtime inputs):
        TA_FdHeadTempLe / TA_FdHeadTempRi  — thermocouple labels for offline validation
        TS_FrntRoofTemp                     — thermistor labels for offline validation
        V_FrntTemp                          — vehicle interior sensor for calibration reference
    """

    air_temp_c: float
    mean_radiant_temp_c: float
    air_velocity_m_s: float
    relative_humidity_pct: float
    metabolic_rate_met: float = 1.0
    clothing_insulation_clo: float = 0.5


@dataclass(frozen=True)
class VehicleComfortResult:
    """Fanger comfort indices."""

    pmv: float
    ppd: float


def compute_vehicle_pmv(inputs: VehicleComfortInputs) -> VehicleComfortResult:
    """Compute PMV/PPD for vehicle cabin comfort assessment."""
    pmv, ppd = pmv_ppd(
        ta=inputs.air_temp_c,
        tr=inputs.mean_radiant_temp_c,
        vel=inputs.air_velocity_m_s,
        rh=inputs.relative_humidity_pct,
        met=inputs.metabolic_rate_met,
        clo=inputs.clothing_insulation_clo,
    )
    return VehicleComfortResult(pmv=pmv, ppd=ppd)
