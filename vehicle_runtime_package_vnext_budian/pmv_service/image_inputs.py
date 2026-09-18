"""Image-service client and occupant mapping for PMV runtime inputs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests


LOGGER = logging.getLogger(__name__)
IMAGE_STATS_URL = "http://127.0.0.1:7860/stats"
IMAGE_FETCH_TIMEOUT_S = 0.5

CLO_ENUM_MAP = {
    0: 1.00,
    1: 0.70,
    2: 1.20,
    3: 1.50,
    4: 0.70,
    5: 0.60,
    6: 0.40,
    7: 0.25,
    8: 0.50,
    9: 0.70,
}
DEFAULT_CLO = 0.60


class ImageStatsClient:
    def __init__(self, url: str = IMAGE_STATS_URL) -> None:
        self.url = url
        self._connected = False

    def fetch(self, timeout: float = IMAGE_FETCH_TIMEOUT_S) -> Optional[Dict[str, Any]]:
        try:
            response = requests.get(self.url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except Exception:
            if self._connected:
                LOGGER.warning("Image API unreachable, using default clothing/met")
                self._connected = False
            return None
        if not isinstance(data, dict) or data.get("status") != "ok":
            return None
        if not self._connected:
            LOGGER.info("Image API connected, using real clothing/met")
            self._connected = True
        return data


DEFAULT_IMAGE_CLIENT = ImageStatsClient()


def fetch_image_stats(timeout: float = IMAGE_FETCH_TIMEOUT_S) -> Optional[Dict[str, Any]]:
    return DEFAULT_IMAGE_CLIENT.fetch(timeout)


def parse_seat(stats_seat: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(stats_seat, dict):
        return {"occupied": False}

    cloth_enum = stats_seat.get("cloth")
    try:
        clothing_clo = CLO_ENUM_MAP.get(int(cloth_enum)) if cloth_enum is not None else None
    except (TypeError, ValueError):
        clothing_clo = None

    gender_raw = stats_seat.get("gender")
    try:
        gender = "male" if int(gender_raw) == 1 else "female" if gender_raw is not None else None
    except (TypeError, ValueError):
        gender = None

    return {
        "occupied": True,
        "gender": gender,
        "age": stats_seat.get("age"),
        "height_cm": stats_seat.get("height"),
        "bmi": stats_seat.get("bmi"),
        "clothing_clo": clothing_clo,
        "cloth_enum": cloth_enum,
    }


def inject_image_to_pmv_input(
    pmv_input: Dict[str, Any],
    stats_data: Optional[Dict[str, Any]],
) -> None:
    if stats_data is None:
        pmv_input["image_inputs"] = {
            "driver": {"occupied": False},
            "passenger": {"occupied": False},
        }
        return
    pmv_input["image_inputs"] = {
        "driver": parse_seat(stats_data.get("driver")),
        "passenger": parse_seat(stats_data.get("passenger")),
    }
