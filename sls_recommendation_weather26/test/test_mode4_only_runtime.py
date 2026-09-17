"""单一 mode4 + weather26 运行边界回归。"""
from __future__ import annotations

import inspect
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.inference_engine import InferenceEngine
from src.core.models.base_model_loader import RandomForestModelLoader
from src.main_controller import _build_parser


class Mode4OnlyRuntimeTest(unittest.TestCase):
    def test_supported_cli_is_accepted(self):
        args = _build_parser().parse_args([
            "--run-mode", "mode4",
            "--model-package-mode", "weather26",
            "--shadow-mode",
        ])
        self.assertEqual(args.run_mode, "mode4")
        self.assertEqual(args.model_package_mode, "weather26")
        self.assertTrue(args.shadow_mode)

    def test_legacy_cli_modes_are_rejected(self):
        for mode in ("mode1", "mode2", "mode3"):
            with self.subTest(mode=mode), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    _build_parser().parse_args(["--run-mode", mode])
                self.assertEqual(raised.exception.code, 2)

    def test_legacy_model_packages_are_rejected(self):
        for package in ("auto", "delivery", "legacy", "steady26"):
            with self.subTest(package=package):
                with self.assertRaises(ValueError):
                    RandomForestModelLoader(package_mode=package)

    def test_inference_engine_has_no_legacy_preference_argument(self):
        parameters = inspect.signature(InferenceEngine).parameters
        self.assertNotIn("preference_module", parameters)

    def test_removed_modules_are_absent(self):
        root = Path(__file__).resolve().parents[1] / "src"
        removed = (
            root / "preference_learning" / "preference_module.py",
            root / "preference_learning" / "mlp_preference_module.py",
            root / "preference_learning" / "stability_detector.py",
            root / "services" / "data_collection" / "user_action_collector_v2.py",
            root / "services" / "data_collection" / "sqlite_storage.py",
        )
        self.assertFalse(any(path.exists() for path in removed))


if __name__ == "__main__":
    unittest.main()
