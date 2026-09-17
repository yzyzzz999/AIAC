"""MF4 signal index extraction (asammdf optional)."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

SIGNAL_INDEX_SCHEMA = "mf4_signal_index_v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
TDC_DATA_DIR = _REPO_ROOT / "Experiment data" / "TDCdata"
TDC_DATA_EXTERNAL_CANDIDATE = Path(r"F:\TempWork\SeriesPMV\01_APP\ExperimentData\TDCdata")
DEFAULT_DBC_PATH = _REPO_ROOT / "files" / "CAN.dbc"
DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "vehicle_mf4_replay"
)


class MultipleMf4FilesError(ValueError):
    """More than one MF4 in TDCdata without explicit --mf4."""

    def __init__(self, files: Sequence[Path]) -> None:
        self.files = list(files)
        names = ", ".join(p.name for p in self.files)
        super().__init__(f"multiple MF4 files in TDCdata; specify --mf4: {names}")


@dataclass(frozen=True)
class ChannelSeries:
    """One MF4-like channel time series."""

    signal_name: str
    timestamps_s: np.ndarray
    samples: np.ndarray
    unit: str = ""


def asammdf_availability() -> Dict[str, Any]:
    """Report whether asammdf is importable and usable."""
    try:
        import asammdf  # noqa: WPS433

        return {
            "available": True,
            "status": "installed",
            "version": getattr(asammdf, "__version__", "unknown"),
        }
    except ImportError:
        return {
            "available": False,
            "status": "dependency_missing",
            "package": "asammdf",
            "install_hint": "pip install asammdf",
        }
    except BaseException as exc:  # noqa: BLE001 — incl. AttributeError from broken numpy/pandas/bottleneck
        return {
            "available": False,
            "status": "dependency_broken",
            "package": "asammdf",
            "error": f"{type(exc).__name__}: {exc}",
            "install_hint": "fix numpy/pandas/asammdf environment",
        }


MF4_SKIP_PREFIXES: Tuple[str, ...] = (
    "CAN_DataFrame",
    "CAN_ErrorFrame",
    "CAN_RemoteFrame",
)

MF4_DEFAULT_ALLOW_PREFIXES: Tuple[str, ...] = (
    "TA_",
    "AC_",
    "RSM_",
    "VIU_",
    "IPB_",
    "SIG_",
    "BCM_",
    "Driver",
    "Passenger",
)


def _should_skip_mf4_channel(
    name: str,
    *,
    allow_patterns: Optional[Sequence[str]] = None,
) -> bool:
    """Skip raw CAN bus frames and channels outside allow list."""
    if any(name.startswith(prefix) for prefix in MF4_SKIP_PREFIXES):
        return True
    if "." in name:
        root = name.split(".", 1)[0]
        if root in MF4_SKIP_PREFIXES:
            return True
    patterns = allow_patterns if allow_patterns is not None else MF4_DEFAULT_ALLOW_PREFIXES
    if patterns:
        return not any(name.startswith(p) or name == p for p in patterns)
    return False


def _samples_are_numeric(samples: np.ndarray) -> bool:
    arr = np.asarray(samples)
    if arr.size == 0:
        return False
    if not np.issubdtype(arr.dtype, np.number):
        return False
    finite = arr[np.isfinite(arr.astype(float, copy=False))]
    return finite.size > 0


def classify_likely_category(signal_name: str, unit: str = "") -> str:
    """Heuristic channel category for PMV replay planning."""
    name = signal_name.lower()
    unit_l = (unit or "").lower()

    if any(k in name for k in ("can", "viu_", "ac_", "rsm_", "bcm_", "sig_")):
        if "temp" in name or "t_" in name or unit_l in ("°c", "degc", "c"):
            return "temperature_sensor"
        if "solar" in name or "inten" in name:
            return "solar"
        if "hum" in name or "rh" in name:
            return "hvac"
        if "spd" in name or "speed" in name or "veh" in name:
            return "vehicle"
        if any(k in name for k in ("blow", "vent", "mode", "posn", "flow", "def")):
            return "hvac"
        return "CAN"

    if "temp" in name or unit_l in ("°c", "degc", "c", "celsius"):
        return "temperature_sensor"
    if "solar" in name or "irradi" in name:
        return "solar"
    if "rh" in name or "humid" in name:
        return "hvac"
    if "speed" in name or "vehspd" in name:
        return "vehicle"
    return "unknown"


def _finite_stats(samples: np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(samples, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def index_one_channel(channel: ChannelSeries) -> Dict[str, Any]:
    ts = np.asarray(channel.timestamps_s, dtype=float)
    samples = np.asarray(channel.samples, dtype=float)
    stats = _finite_stats(samples)
    time_start = float(ts[0]) if ts.size else None
    time_end = float(ts[-1]) if ts.size else None
    return {
        "signal_name": channel.signal_name,
        "unit": channel.unit or "",
        "sample_count": int(samples.size),
        "time_start": time_start,
        "time_end": time_end,
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["mean"],
        "likely_category": classify_likely_category(channel.signal_name, channel.unit),
    }


def build_signal_index(
    channels: Sequence[ChannelSeries],
    *,
    source_path: Optional[Union[str, Path]] = None,
    dependency_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rows = [index_one_channel(ch) for ch in channels]
    rows.sort(key=lambda r: r["signal_name"].lower())
    return {
        "schema": SIGNAL_INDEX_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_mf4": str(source_path) if source_path is not None else None,
        "channel_count": len(rows),
        "dependency": dict(dependency_meta or {}),
        "channels": rows,
    }


def write_signal_index(
    index: Mapping[str, Any],
    *,
    csv_path: Union[str, Path],
    json_path: Union[str, Path],
) -> Dict[str, str]:
    csv_out = Path(csv_path)
    json_out = Path(json_path)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "signal_name",
        "unit",
        "sample_count",
        "time_start",
        "time_end",
        "min",
        "max",
        "mean",
        "likely_category",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in index.get("channels", []):
            writer.writerow({k: row.get(k) for k in fieldnames})

    json_out.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"csv": str(csv_out.resolve()), "json": str(json_out.resolve())}


def _read_mf4_with_asammdf(
    mf4_path: Path,
    *,
    allow_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[ChannelSeries], Dict[str, Any]]:
    import logging

    from asammdf import MDF  # noqa: WPS433

    logging.getLogger("asammdf").setLevel(logging.CRITICAL)

    channels: List[ChannelSeries] = []
    skipped: Dict[str, int] = {
        "filtered_name": 0,
        "duplicate_name": 0,
        "non_numeric": 0,
        "read_error": 0,
        "empty": 0,
    }
    with MDF(str(mf4_path)) as mdf:
        for name in sorted(mdf.channels_db.keys()):
            if _should_skip_mf4_channel(str(name), allow_patterns=allow_patterns):
                skipped["filtered_name"] += 1
                continue
            entries = mdf.channels_db[name]
            if len(entries) != 1:
                skipped["duplicate_name"] += 1
                continue
            group, index = entries[0]
            try:
                sig = mdf.get(str(name), group=group, index=index)
            except Exception:
                skipped["read_error"] += 1
                continue
            if sig is None or len(sig.samples) == 0:
                skipped["empty"] += 1
                continue
            samples = np.asarray(sig.samples, dtype=float)
            if not _samples_are_numeric(samples):
                skipped["non_numeric"] += 1
                continue
            ts = np.asarray(sig.timestamps, dtype=float)
            if ts.size and ts[0] > 1e8:
                ts = ts - ts[0]
            channels.append(
                ChannelSeries(
                    signal_name=str(name),
                    timestamps_s=ts,
                    samples=samples,
                    unit=str(getattr(sig, "unit", "") or ""),
                )
            )
    dep = asammdf_availability()
    dep["mf4_path"] = str(mf4_path)
    dep["channels_read"] = len(channels)
    dep["channels_skipped"] = skipped
    return channels, dep


def list_mf4_channel_names(
    mf4_path: Union[str, Path],
    *,
    allow_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """List channel names in MF4 without reading sample payloads (fast inventory)."""
    path = Path(mf4_path)
    dep = asammdf_availability()
    if not dep["available"]:
        return [], {**dep, "status": "asammdf_unavailable", "mf4_path": str(path)}
    if not path.is_file():
        return [], {"status": "mf4_not_found", "mf4_path": str(path)}
    try:
        from asammdf import MDF  # noqa: WPS433

        names: List[str] = []
        with MDF(str(path)) as mdf:
            for name in sorted(mdf.channels_db.keys()):
                sname = str(name)
                if _should_skip_mf4_channel(sname, allow_patterns=allow_patterns):
                    continue
                names.append(sname)
        dep["mf4_path"] = str(path)
        dep["channel_name_count"] = len(names)
        return names, dep
    except Exception as exc:  # noqa: BLE001
        return [], {
            **dep,
            "status": "mf4_list_error",
            "error": f"{type(exc).__name__}: {exc}",
            "mf4_path": str(path),
        }


def extract_mf4_channels(
    mf4_path: Union[str, Path],
    *,
    channels: Optional[Sequence[ChannelSeries]] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    skip_read: bool = False,
) -> Tuple[List[ChannelSeries], Dict[str, Any]]:
    """Load MF4 channels or use injected channels (tests/mock)."""
    if channels is not None:
        return list(channels), {"status": "injected_channels", "channel_count": len(channels)}

    if skip_read:
        return [], {
            "status": "skip_read",
            "available": False,
            "note": "MF4 read skipped by caller",
        }

    path = Path(mf4_path)
    if not path.is_file():
        return [], {
            "status": "mf4_not_found",
            "path": str(path),
            **asammdf_availability(),
        }

    dep = asammdf_availability()
    if not dep["available"]:
        return [], {
            **dep,
            "mf4_path": str(path),
            "note": "MF4 not read; install asammdf or pass mock channels",
        }

    try:
        return _read_mf4_with_asammdf(path, allow_patterns=allow_patterns)
    except Exception as exc:
        return [], {
            **dep,
            "status": "mf4_read_error",
            "error": f"{type(exc).__name__}: {exc}",
            "mf4_path": str(path),
        }


def build_mock_summer_cooldown_channels(
    *,
    duration_s: float = 2400.0,
    sample_s: float = 1.0,
) -> List[ChannelSeries]:
    """Synthetic MF4-like channels for tests (summer cabin cooldown)."""
    times = np.arange(0.0, duration_s + sample_s, sample_s)
    n = times.size
    u = np.clip(times / (duration_s * 0.6), 0.0, 1.0)
    cabin = 38.0 - 12.0 * u + 0.3 * np.sin(times / 120.0)
    amb = 32.0 + 0.1 * np.sin(times / 300.0)
    solar_l = np.clip(800.0 - 600.0 * u, 0.0, None)
    solar_r = solar_l * 0.95
    rh = np.full(n, 55.0)
    veh_spd = np.where(times < 120.0, 0.0, 40.0)
    head = cabin - 1.0
    feet = cabin - 2.5
    face_tma = 18.0 + 2.0 * (1.0 - u)
    foot_tma = 16.0 + 1.5 * (1.0 - u)
    win = cabin + 3.0

    def ch(name: str, samples: np.ndarray, unit: str = "°C") -> ChannelSeries:
        return ChannelSeries(signal_name=name, timestamps_s=times.copy(), samples=samples, unit=unit)

    return [
        ch("SIG_AcFrntIcTempFb", cabin),
        ch("SIG_AmbTempFb", amb),
        ch("SIG_AmbLeSolarIntenFb", solar_l, "W/m2"),
        ch("SIG_AmbRiSolarIntenFb", solar_r, "W/m2"),
        ch("SIG_AmbRelHumFb", rh, "%"),
        ch("VIU_VehSpd", veh_spd, "km/h"),
        ch("DriverHeadTempMeas", head),
        ch("PassengerHeadTempMeas", head - 0.5),
        ch("DriverFeetTempMeas", feet),
        ch("SIG_AcFdDrvFdvTmaFb", face_tma),
        ch("SIG_AcFdDrvFdfTmaFb", foot_tma),
        ch("SIG_AmbWindshieldTempFb", win),
    ]


def get_tdc_data_dir(*, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve TDC data root: explicit > env TDC_DATA_DIR > external candidate > repo."""
    if override is not None:
        return Path(override)
    env = os.environ.get("TDC_DATA_DIR")
    if env:
        return Path(env)
    if TDC_DATA_EXTERNAL_CANDIDATE.is_dir():
        return TDC_DATA_EXTERNAL_CANDIDATE
    return TDC_DATA_DIR


def discover_tdcdata_files(
    *,
    suffix: str,
    tdc_dir: Optional[Path] = None,
    recursive: bool = True,
) -> List[Path]:
    """Discover ``*{suffix}`` under TDC data dir (recursive by default)."""
    root = get_tdc_data_dir(override=tdc_dir)
    if not root.is_dir():
        return []
    if recursive:
        return sorted(root.rglob(f"*{suffix}"), key=lambda p: (str(p.parent).lower(), p.name.lower()))
    return sorted(root.glob(f"*{suffix}"), key=lambda p: p.name.lower())


def _date_folder(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return ""
    parts = rel.parts
    return parts[0] if len(parts) > 1 else ""


def file_entry(
    path: Path,
    *,
    root: Optional[Path] = None,
    processed_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    stat = path.stat()
    resolved = str(path.resolve())
    rel = ""
    date_folder = ""
    if root is not None:
        try:
            rel = str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
            date_folder = _date_folder(path, root)
        except ValueError:
            rel = resolved
    processed_set = {str(Path(p).resolve()) for p in (processed_paths or [])}
    return {
        "name": path.name,
        "date_folder": date_folder,
        "relative_path": rel,
        "path": resolved,
        "size_bytes": int(stat.st_size),
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "processed": resolved in processed_set,
    }


def build_mf4_file_inventory(
    *,
    tdc_dir: Optional[Path] = None,
    mf4_path: Optional[Path] = None,
    blf_path: Optional[Path] = None,
    processed_paths: Optional[Sequence[str]] = None,
    recursive: bool = True,
) -> Dict[str, Any]:
    """Inventory TDCdata measurement files recursively (no MF4 read)."""
    root = get_tdc_data_dir(override=tdc_dir)
    mf4_files = discover_tdcdata_files(suffix=".mf4", tdc_dir=root, recursive=recursive)
    blf_files = discover_tdcdata_files(suffix=".blf", tdc_dir=root, recursive=recursive)
    date_folders = sorted({row["date_folder"] for row in [file_entry(p, root=root) for p in mf4_files] if row["date_folder"]})
    return {
        "schema": "mf4_file_inventory_v2",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tdcdata_dir": str(root.resolve()),
        "tdcdata_exists": root.is_dir(),
        "recursive_search": recursive,
        "date_folders": date_folders,
        "mf4_count": len(mf4_files),
        "blf_count": len(blf_files),
        "mf4_files": [
            file_entry(p, root=root, processed_paths=processed_paths) for p in mf4_files
        ],
        "blf_files": [
            file_entry(p, root=root, processed_paths=processed_paths) for p in blf_files
        ],
        "mf4_path_resolved": str(mf4_path.resolve()) if mf4_path and mf4_path.is_file() else None,
        "blf_path_resolved": str(blf_path.resolve()) if blf_path and blf_path.is_file() else None,
        "asammdf": asammdf_availability(),
    }


def format_tdcdata_file_list(*, tdc_dir: Optional[Path] = None) -> str:
    """Human-readable listing for ``--list-files``."""
    inv = build_mf4_file_inventory(tdc_dir=tdc_dir)
    lines = [
        "TDCdata file inventory (recursive)",
        f"Directory: {inv['tdcdata_dir']}",
        f"Date folders: {', '.join(inv.get('date_folders', [])) or '(flat)'}",
        f"MF4 count: {inv.get('mf4_count', 0)} | BLF count: {inv.get('blf_count', 0)}",
        "",
        "MF4 files:",
    ]
    if inv["mf4_files"]:
        current_folder = ""
        for row in inv["mf4_files"]:
            folder = row.get("date_folder") or "(root)"
            if folder != current_folder:
                lines.append(f"  [{folder}]")
                current_folder = folder
            lines.append(f"    - {row['name']}  ({row['size_mb']} MB)")
            lines.append(f"      {row.get('relative_path') or row['path']}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("BLF files:")
    if inv["blf_files"]:
        for row in inv["blf_files"]:
            lines.append(f"  - {row['name']}  ({row['size_mb']} MB)")
            lines.append(f"    {row.get('relative_path') or row['path']}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"asammdf available: {inv['asammdf'].get('available')}")
    return "\n".join(lines)


def resolve_mf4_path(
    explicit: Optional[Union[str, Path]] = None,
    *,
    tdc_dir: Optional[Path] = None,
) -> Tuple[Path, List[Path]]:
    """Resolve MF4 path: explicit ``--mf4`` or sole discovered MF4."""
    discovered = discover_tdcdata_files(suffix=".mf4", tdc_dir=tdc_dir)
    if explicit is not None:
        path = Path(explicit)
        if path.is_file():
            return path.resolve(), discovered
        raise FileNotFoundError(f"MF4 not found: {path}")
    if not discovered:
        root = get_tdc_data_dir(override=tdc_dir)
        raise FileNotFoundError(f"no .mf4 files under {root}")
    if len(discovered) == 1:
        return discovered[0].resolve(), discovered
    raise MultipleMf4FilesError(discovered)


def resolve_blf_path(
    explicit: Optional[Union[str, Path]] = None,
    *,
    tdc_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve BLF path: explicit, or sole / can_recording*.blf in TDCdata."""
    discovered = discover_tdcdata_files(suffix=".blf", tdc_dir=tdc_dir)
    if explicit is not None:
        path = Path(explicit)
        return path.resolve() if path.is_file() else None
    if not discovered:
        return None
    preferred = [p for p in discovered if p.name.lower().startswith("can_recording")]
    if len(preferred) == 1:
        return preferred[0].resolve()
    if len(discovered) == 1:
        return discovered[0].resolve()
    return discovered[0].resolve()


def run_mf4_signal_index_phase(
    mf4_path: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    channels: Optional[Sequence[ChannelSeries]] = None,
) -> Dict[str, Any]:
    """Phase 1: build and write MF4 signal index artifacts."""
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    loaded, dep = extract_mf4_channels(mf4_path, channels=channels)
    index = build_signal_index(loaded, source_path=mf4_path, dependency_meta=dep)
    paths = write_signal_index(
        index,
        csv_path=out_dir / "mf4_signal_index.csv",
        json_path=out_dir / "mf4_signal_index.json",
    )
    return {"index": index, "paths": paths, "dependency": dep}
