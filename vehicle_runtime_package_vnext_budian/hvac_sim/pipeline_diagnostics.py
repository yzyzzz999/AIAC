"""Optional runtime diagnostics for the comfort pipeline.

This module owns sampling and logging so numerical pipeline code never writes
directly to stdout or stderr. Applications opt in by configuring logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


LOGGER = logging.getLogger("hvac_sim.pmv.diagnostics")


@dataclass(frozen=True)
class PmvDiagnosticSnapshot:
    passenger_pmv: float
    passenger_ppd: float
    passenger_air_temp_c: float
    passenger_air_temp_source: str
    passenger_mrt_c: float
    passenger_air_speed_m_s: float
    relative_humidity_pct: float
    passenger_met: float
    passenger_clo: float
    passenger_head_raw_c: float
    passenger_cabin_raw_c: float
    passenger_window_raw_c: float
    passenger_roof_raw_c: float
    passenger_window_mrt_c: float
    passenger_roof_mrt_c: float
    driver_head_raw_c: float
    driver_feet_raw_c: float
    passenger_feet_raw_c: float
    driver_pmv: float
    mrt_hot_surface_threshold_c: float


class PeriodicPmvObserver:
    """Log the first snapshot and then every ``interval`` snapshots."""

    def __init__(self, interval: int = 20) -> None:
        self.interval = interval
        self._count = 0

    def observe(self, snapshot: PmvDiagnosticSnapshot) -> None:
        self._count += 1
        if self._count != 1 and self._count % self.interval != 0:
            return

        threshold = snapshot.mrt_hot_surface_threshold_c
        LOGGER.info(
            "PMV diagnostic #%d: passenger PMV=%+.3f PPD=%.1f%%; "
            "ta=%.2f°C (%s), tr=%.2f°C, v=%.3fm/s, RH=%.1f%%, "
            "met=%.2f, clo=%.2f; MRT raw head/cabin/window/roof="
            "%.2f/%.2f/%.2f/%.2f°C, effective window/roof=%.2f/%.2f°C "
            "(clamped=%s/%s); driver head/feet=%.2f/%.2f°C, "
            "passenger feet=%.2f°C, driver PMV=%+.3f",
            self._count,
            snapshot.passenger_pmv,
            snapshot.passenger_ppd,
            snapshot.passenger_air_temp_c,
            snapshot.passenger_air_temp_source,
            snapshot.passenger_mrt_c,
            snapshot.passenger_air_speed_m_s,
            snapshot.relative_humidity_pct,
            snapshot.passenger_met,
            snapshot.passenger_clo,
            snapshot.passenger_head_raw_c,
            snapshot.passenger_cabin_raw_c,
            snapshot.passenger_window_raw_c,
            snapshot.passenger_roof_raw_c,
            snapshot.passenger_window_mrt_c,
            snapshot.passenger_roof_mrt_c,
            snapshot.passenger_window_raw_c > threshold,
            snapshot.passenger_roof_raw_c > threshold,
            snapshot.driver_head_raw_c,
            snapshot.driver_feet_raw_c,
            snapshot.passenger_feet_raw_c,
            snapshot.driver_pmv,
        )


PMV_DIAGNOSTIC_OBSERVER = PeriodicPmvObserver()
