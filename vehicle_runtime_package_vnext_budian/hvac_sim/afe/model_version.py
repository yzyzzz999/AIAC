"""AFE model version boundary (AFE-T21-A).

Default runtime remains ``afe_as_found`` (existing resistance/fan topology).
``target_vehicle_dual_layer`` is interface-only until a signed-off network exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AFEModelVersion(str, Enum):
    """Selectable AFE topology variants."""

    AFE_AS_FOUND = "afe_as_found"
    TARGET_VEHICLE_DUAL_LAYER = "target_vehicle_dual_layer"


DEFAULT_AFE_MODEL_VERSION = AFEModelVersion.AFE_AS_FOUND


@dataclass(frozen=True)
class AFEModelVersionStatus:
    version: AFEModelVersion
    implemented: bool
    default: bool
    notes: str


def resolve_model_version(name: Optional[str] = None) -> AFEModelVersion:
    """Resolve version id; unknown values fall back to ``afe_as_found``."""
    if name is None or not str(name).strip():
        return DEFAULT_AFE_MODEL_VERSION
    key = str(name).strip().lower()
    for member in AFEModelVersion:
        if member.value == key:
            return member
    return DEFAULT_AFE_MODEL_VERSION


def get_model_version_status(name: Optional[str] = None) -> AFEModelVersionStatus:
    """Describe whether the requested variant is runnable."""
    version = resolve_model_version(name)
    if version == AFEModelVersion.TARGET_VEHICLE_DUAL_LAYER:
        return AFEModelVersionStatus(
            version=version,
            implemented=False,
            default=False,
            notes=(
                "Solver not implemented. Decode-only layer available via "
                "m8_dual_layer_decoder.py and m8_dual_layer_structure_confirmed.json. "
                "Do not guess internal split resistances."
            ),
        )
    return AFEModelVersionStatus(
        version=AFEModelVersion.AFE_AS_FOUND,
        implemented=True,
        default=version == DEFAULT_AFE_MODEL_VERSION,
        notes="Existing AFE topology (single OSA/REC paths, unchanged afe_calc defaults).",
    )


def assert_runnable_model_version(name: Optional[str] = None) -> AFEModelVersion:
    """Return version when implemented; raise for placeholder variants."""
    status = get_model_version_status(name)
    if not status.implemented:
        raise NotImplementedError(status.notes)
    return status.version
