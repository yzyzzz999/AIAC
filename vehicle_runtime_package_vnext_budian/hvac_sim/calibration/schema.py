"""Model-agnostic calibration schema (T9-7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


class CalibrationSchemaError(ValueError):
    """Invalid calibration schema or dataset."""


@dataclass(frozen=True)
class ParameterSpec:
    """One tunable parameter for PSO or other optimizers."""

    name: str
    initial: float
    lower: float
    upper: float
    unit: str = ""
    description: str = ""
    target_model: str = ""
    source: str = "manual"


@dataclass
class CalibrationDataset:
    """Prepared calibration samples consumed by model adapters."""

    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    target_type: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationDataset:
        if not isinstance(data, Mapping):
            raise CalibrationSchemaError("dataset must be a mapping")
        raw_samples = data.get("samples")
        if not isinstance(raw_samples, list):
            raise CalibrationSchemaError("'samples' must be a list")
        metadata = data.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise CalibrationSchemaError("'metadata' must be an object when present")
        return cls(
            samples=list(raw_samples),
            metadata=dict(metadata),
            source=str(data.get("source", "")),
            target_type=str(data.get("target_type", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "metadata": self.metadata,
            "source": self.source,
            "target_type": self.target_type,
        }

    def validate(self, *, require_non_empty: bool = True) -> None:
        if not isinstance(self.samples, list):
            raise CalibrationSchemaError("samples must be a list")
        if require_non_empty and len(self.samples) == 0:
            raise CalibrationSchemaError("samples must not be empty")
        for idx, sample in enumerate(self.samples):
            if not isinstance(sample, dict):
                raise CalibrationSchemaError(f"samples[{idx}] must be an object")


def parameter_spec_from_dict(data: Mapping[str, Any]) -> ParameterSpec:
    if not isinstance(data, Mapping):
        raise CalibrationSchemaError("parameter spec must be an object")
    try:
        return ParameterSpec(
            name=str(data["name"]),
            initial=float(data["initial"]),
            lower=float(data["lower"]),
            upper=float(data["upper"]),
            unit=str(data.get("unit", "")),
            description=str(data.get("description", "")),
            target_model=str(data.get("target_model", "")),
            source=str(data.get("source", "manual")),
        )
    except KeyError as exc:
        raise CalibrationSchemaError(f"missing parameter spec field: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise CalibrationSchemaError(f"invalid parameter spec value: {exc}") from exc


def load_parameter_specs(data: Sequence[Mapping[str, Any]]) -> List[ParameterSpec]:
    return [parameter_spec_from_dict(item) for item in data]


def specs_to_bounds(specs: Sequence[ParameterSpec]) -> List[Tuple[float, float]]:
    return [(spec.lower, spec.upper) for spec in specs]


def specs_to_initial_vector(specs: Sequence[ParameterSpec]) -> np.ndarray:
    return np.asarray([spec.initial for spec in specs], dtype=float)


def specs_to_names(specs: Sequence[ParameterSpec]) -> List[str]:
    return [spec.name for spec in specs]
