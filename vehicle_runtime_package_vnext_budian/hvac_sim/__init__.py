"""HVAC simulation package: AFE resistance network, flow solver, CHTD thermal model."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hvac-sim")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
