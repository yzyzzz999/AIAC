"""CHTD per-zone implementation mode registry (T7-0).

Each of the 28 thermal states is tagged TRACE, FAST_APPROX, or GAP.
TRACE zones use dedicated formulas in compute_chtd_delta; all others use
the §8 generic first-order template via fast_approx.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .bus_index import CHTD_X_NAMES, N_X_STATES, X_INDEX


class ZoneImplementationMode(str, Enum):
    TRACE = "TRACE"
    FAST_APPROX = "FAST_APPROX"
    GAP = "GAP"


@dataclass(frozen=True)
class ZoneConfigEntry:
    name: str
    index: int
    mode: ZoneImplementationMode
    note: str = ""


def _entry(name: str, mode: ZoneImplementationMode, note: str = "") -> ZoneConfigEntry:
    return ZoneConfigEntry(name=name, index=X_INDEX[name], mode=mode, note=note)


CHTD_ZONE_CONFIG: tuple[ZoneConfigEntry, ...] = (
    _entry("HoodTemp", ZoneImplementationMode.TRACE, "T6D-0 §7"),
    _entry(
        "CabinFrntTemp",
        ZoneImplementationMode.TRACE,
        "T9 CabinFrntTemp_topology_spec",
    ),
    _entry("ConsoleTemp", ZoneImplementationMode.TRACE, "T9 ConsoleTemp_topology_spec"),
    _entry("HeadTempFd", ZoneImplementationMode.TRACE, "T8 HeadTempFd_topology_spec"),
    _entry("HeadTempFp", ZoneImplementationMode.TRACE, "T8 passenger HeadTempFp_topology_spec"),
    _entry("HeadTempSd", ZoneImplementationMode.TRACE, "T9 HeadTempSd_topology_spec"),
    _entry("HeadTempSp", ZoneImplementationMode.TRACE, "T6A-2 + T6B-0 q-bus"),
    _entry(
        "HeadTempTd",
        ZoneImplementationMode.TRACE,
        "T11 HeadTempTd_topology_spec; TdTp q-bus to HeadTp/Roof",
    ),
    _entry(
        "HeadTempTp",
        ZoneImplementationMode.TRACE,
        "T11 HeadTempTp_topology_spec; TdTp q-bus TODO=0",
    ),
    _entry("FeetTempFd", ZoneImplementationMode.TRACE, "T8 FeetTempFd_topology_spec"),
    _entry("FeetTempFp", ZoneImplementationMode.TRACE, "T8 passenger FeetTempFp_topology_spec"),
    _entry("FeetTempSd", ZoneImplementationMode.TRACE, "T9 FeetTempSd_topology_spec"),
    _entry("FeetTempSp", ZoneImplementationMode.TRACE, "T9 FeetTempSp_topology_spec"),
    _entry(
        "FeetTempTd",
        ZoneImplementationMode.TRACE,
        "T11 FeetTempTd_topology_spec",
    ),
    _entry(
        "FeetTempTp",
        ZoneImplementationMode.TRACE,
        "T11 FeetTempTp_topology_spec",
    ),
    _entry("CabinTempFd", ZoneImplementationMode.TRACE, "T6C-0 §5"),
    _entry("CabinTempFp", ZoneImplementationMode.TRACE, "T8 CabinTempFp_topology_spec §5"),
    _entry("CabinTempSd", ZoneImplementationMode.TRACE, "T9 CabinTempSd_topology_spec"),
    _entry("CabinTempSp", ZoneImplementationMode.TRACE, "T9 CabinTempSp_topology_spec"),
    _entry(
        "CabinTempTd",
        ZoneImplementationMode.TRACE,
        "T11 CabinTempTd_topology_spec",
    ),
    _entry(
        "CabinTempTp",
        ZoneImplementationMode.TRACE,
        "T11 CabinTempTp_topology_spec; HeadTp/FeetTp q-bus TODO=0",
    ),
    _entry("WinTempFd", ZoneImplementationMode.TRACE, "T8 WinTempFd_topology_spec"),
    _entry("WinTempFp", ZoneImplementationMode.TRACE, "T8 WinTempFp_topology_spec"),
    _entry("WinTempSd", ZoneImplementationMode.TRACE, "T9 WinTempSd_topology_spec"),
    _entry("WinTempSp", ZoneImplementationMode.TRACE, "T9 WinTempSp_topology_spec"),
    _entry(
        "WinTempTd",
        ZoneImplementationMode.TRACE,
        "T11 WinTempTd_topology_spec",
    ),
    _entry(
        "WinTempTp",
        ZoneImplementationMode.TRACE,
        "T11 WinTempTp_topology_spec; DEFECT D_WinTp HumFeetTpPower AS_FOUND",
    ),
    _entry("RoofTemp", ZoneImplementationMode.TRACE, "T8 RoofTemp_topology_spec"),
)

assert len(CHTD_ZONE_CONFIG) == N_X_STATES
assert [e.index for e in CHTD_ZONE_CONFIG] == list(range(N_X_STATES))
assert [e.name for e in CHTD_ZONE_CONFIG] == CHTD_X_NAMES

CHTD_ZONE_CONFIG_BY_NAME: dict[str, ZoneConfigEntry] = {
    e.name: e for e in CHTD_ZONE_CONFIG
}

TRACE_ZONE_INDICES: frozenset[int] = frozenset(
    e.index for e in CHTD_ZONE_CONFIG if e.mode is ZoneImplementationMode.TRACE
)

FAST_APPROX_ZONE_INDICES: frozenset[int] = frozenset(
    e.index
    for e in CHTD_ZONE_CONFIG
    if e.mode in (ZoneImplementationMode.FAST_APPROX, ZoneImplementationMode.GAP)
)
