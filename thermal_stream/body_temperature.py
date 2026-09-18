"""Body temperature estimation from thermal camera temperature matrices."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BodyTemperatureReading:
    body_temp_c: Optional[float]
    valid: bool
    sample_count: int
    min_c: float
    max_c: float
    ts: float

    def to_payload(self) -> dict:
        return {
            "type": "body_temperature",
            "body_temp_c": None if self.body_temp_c is None else round(self.body_temp_c, 2),
            "valid": self.valid,
            "sample_count": self.sample_count,
            "range_c": [round(self.min_c, 1), round(self.max_c, 1)],
            "ts": round(self.ts, 3),
        }


class BodyTemperatureEstimator:
    """Estimate body temperature by averaging pixels within a human skin range."""

    def __init__(
        self,
        min_c: float = 35.0,
        max_c: float = 38.0,
        min_pixels: int = 25,
        ema_alpha: float = 0.35,
    ):
        self.min_c = min_c
        self.max_c = max_c
        self.min_pixels = min_pixels
        self.ema_alpha = ema_alpha
        self._smoothed: Optional[float] = None

    def estimate(self, temp_celsius: np.ndarray, ts: float) -> BodyTemperatureReading:
        finite_temp = temp_celsius[np.isfinite(temp_celsius)]
        if finite_temp.size == 0:
            return BodyTemperatureReading(None, False, 0, 0.0, 0.0, ts)

        mask = (finite_temp >= self.min_c) & (finite_temp <= self.max_c)
        samples = finite_temp[mask]
        if samples.size < self.min_pixels:
            self._smoothed = None
            return BodyTemperatureReading(
                None,
                False,
                int(samples.size),
                float(np.min(finite_temp)),
                float(np.max(finite_temp)),
                ts,
            )

        raw_mean = float(np.mean(samples))
        if self._smoothed is None:
            self._smoothed = raw_mean
        else:
            self._smoothed = (
                self.ema_alpha * raw_mean + (1.0 - self.ema_alpha) * self._smoothed
            )

        return BodyTemperatureReading(
            float(self._smoothed),
            True,
            int(samples.size),
            float(np.min(finite_temp)),
            float(np.max(finite_temp)),
            ts,
        )
