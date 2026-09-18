import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import person_info_sync as sync


class MappingTests(unittest.TestCase):
    def test_age_and_cloth_are_mapped_to_strings(self):
        self.assertEqual(sync.map_age(3), "60-74")
        self.assertEqual(sync.map_age("4"), "75+")
        self.assertEqual(sync.map_cloth(5), "长袖衬衫")
        self.assertEqual(sync.map_cloth("0"), "西装外套")

    def test_pmv_score_uses_ppd_percentage(self):
        self.assertEqual(sync.format_pmv_score(0.2, 35.4), "舒适[64.6]")
        self.assertEqual(sync.format_pmv_score(0.8, 12.3), "轻微偏热[87.7]")
        self.assertEqual(sync.format_pmv_score(-1.2, 10), "偏冷[90.0]")
        self.assertEqual(sync.format_pmv_score(2.4, 1), "极热[99.0]")

    def test_build_seats_payload_maps_stats_and_redis_data(self):
        stats = {
            "driver": {
                "identity_id": "drv-1",
                "gender": 1,
                "age": 1,
                "cloth": 6,
                "height": 175.0,
                "bmi": 22.5,
            },
            "passenger": None,
        }
        pmv = {
            "comfort": {
                "driver": {"pmv": 1.2, "ppd": 20.0},
                "passenger": {"pmv": None, "ppd": None},
            }
        }

        seats = sync.build_seats(
            stats,
            pmv,
            {"type": "body_temperature", "body_temp_c": 36.42, "valid": True},
        )

        self.assertEqual(
            seats,
            {
                "driver_side": {
                    "User_Id": "drv-1",
                    "sex": "男性",
                    "age": "18-40",
                    "cloth": "短袖",
                    "bmi": 22.5,
                    "tmperature": 36.42,
                    "pmv_score": "偏热[80.0]",
                },
                "passenger_side": {
                    "User_Id": "-1",
                    "sex": None,
                    "age": None,
                    "cloth": None,
                    "bmi": None,
                    "tmperature": None,
                    "pmv_score": None,
                },
            },
        )

    def test_build_seats_accepts_pmv_api_shape(self):
        stats = {
            "driver": {"identity_id": "drv-1", "gender": 1, "age": 2, "cloth": 8, "bmi": 20.1},
            "passenger": None,
        }
        pmv = {
            "driver": {"pmv": -0.8, "ppd": 30.0},
            "passenger": {"pmv": None, "ppd": None},
            "run_index": 123,
            "status": "ok",
        }

        seats = sync.build_seats(stats, pmv)

        self.assertEqual(seats["driver_side"]["age"], "41-59")
        self.assertEqual(seats["driver_side"]["cloth"], "长袖")
        self.assertEqual(seats["driver_side"]["pmv_score"], "轻微偏冷[70.0]")
        self.assertIsNone(seats["driver_side"]["tmperature"])

    def test_build_seats_ignores_invalid_body_temperature(self):
        stats = {
            "driver": {"identity_id": "drv-1", "gender": 1, "age": 2, "cloth": 8, "bmi": 20.1},
            "passenger": None,
        }
        pmv = {"driver": {"pmv": 0.0, "ppd": 5.0}}

        seats = sync.build_seats(
            stats,
            pmv,
            {"type": "body_temperature", "body_temp_c": None, "valid": False},
        )

        self.assertIsNone(seats["driver_side"]["tmperature"])


class ConfigTests(unittest.TestCase):
    def test_default_config_uses_local_stats_and_redis(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = sync.load_config(Path(tmp) / "missing.yaml", env={})

        self.assertEqual(cfg.stats_url, "http://127.0.0.1:7680/stats")
        self.assertEqual(cfg.redis_host, "127.0.0.1")
        self.assertEqual(cfg.redis_port, 6379)
        self.assertEqual(cfg.redis_key, "pmv_thermal_comfort")
        self.assertEqual(cfg.pmv_url, "http://127.0.0.1:7861/pmv")
        self.assertEqual(cfg.body_temperature_url, "http://127.0.0.1:7864/body-temperature")
        self.assertEqual(
            cfg.update_person_info_url,
            "http://192.168.0.104:20001/api/airConditioningControl/updatePersonInfo",
        )

    def test_config_yaml_and_env_override_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "\n".join(
                    [
                        "stats_url: http://127.0.0.1:9999/stats",
                        "update_person_info_url: http://127.0.0.1:20001/update",
                        "redis:",
                        "  host: 127.0.0.2",
                        "  port: 6380",
                        "  key: custom_key",
                        "poll_interval_seconds: 2.5",
                        "pmv_url: http://127.0.0.1:9999/pmv",
                        "body_temperature_url: http://127.0.0.1:9999/body-temperature",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = sync.load_config(
                path,
                env={
                    "PERSON_SYNC_STATS_URL": "http://env.local/stats",
                    "PERSON_SYNC_BODY_TEMPERATURE_URL": "http://env.local/body-temperature",
                    "PERSON_SYNC_REDIS_PORT": "6381",
                },
            )

        self.assertEqual(cfg.stats_url, "http://env.local/stats")
        self.assertEqual(cfg.redis_host, "127.0.0.2")
        self.assertEqual(cfg.redis_port, 6381)
        self.assertEqual(cfg.redis_key, "custom_key")
        self.assertEqual(cfg.pmv_url, "http://127.0.0.1:9999/pmv")
        self.assertEqual(cfg.body_temperature_url, "http://env.local/body-temperature")
        self.assertEqual(cfg.poll_interval_seconds, 2.5)


class ChangeDetectionTests(unittest.TestCase):
    def test_run_once_posts_only_when_seats_change(self):
        cfg = sync.Config(pmv_url="")
        sent = []
        stats = {
            "driver": {"identity_id": "1", "gender": 0, "age": 0, "cloth": 0, "bmi": 20},
            "passenger": None,
        }
        pmv_raw = json.dumps({"comfort": {"driver": {"pmv": 0, "ppd": 0}}})

        service = sync.PersonInfoSync(
            cfg,
            fetch_stats=lambda: stats,
            redis_get=lambda key: pmv_raw,
            fetch_body_temperature=lambda: {
                "type": "body_temperature",
                "body_temp_c": 36.58,
                "valid": True,
            },
            post_json=lambda url, payload: sent.append(payload),
        )

        self.assertTrue(service.run_once())
        self.assertFalse(service.run_once())
        self.assertEqual(len(sent), 1)
        self.assertIn("timestamp", sent[0])
        self.assertEqual(sent[0]["seats"]["driver_side"]["age"], "12-17")
        self.assertEqual(sent[0]["seats"]["driver_side"]["tmperature"], 36.58)

        stats["driver"]["age"] = 1
        self.assertTrue(service.run_once())
        self.assertEqual(len(sent), 2)

    def test_run_once_prefers_pmv_url_over_redis(self):
        cfg = sync.Config(pmv_url="http://127.0.0.1:7861/pmv")
        sent = []
        service = sync.PersonInfoSync(
            cfg,
            fetch_stats=lambda: {
                "driver": {"identity_id": "1", "gender": 1, "age": 1, "cloth": 6, "bmi": 20},
                "passenger": None,
            },
            fetch_pmv=lambda: {"driver": {"pmv": 0.9, "ppd": 10}},
            fetch_body_temperature=lambda: {"body_temp_c": 36.33, "valid": True},
            redis_get=lambda key: (_ for _ in ()).throw(AssertionError("redis should not be used")),
            post_json=lambda url, payload: sent.append(payload),
        )

        self.assertTrue(service.run_once())
        self.assertEqual(sent[0]["seats"]["driver_side"]["pmv_score"], "轻微偏热[90.0]")
        self.assertEqual(sent[0]["seats"]["driver_side"]["tmperature"], 36.33)

    def test_run_once_keeps_syncing_when_body_temperature_fails(self):
        cfg = sync.Config(pmv_url="http://127.0.0.1:7861/pmv")
        sent = []
        service = sync.PersonInfoSync(
            cfg,
            fetch_stats=lambda: {
                "driver": {"identity_id": "1", "gender": 1, "age": 1, "cloth": 6, "bmi": 20},
                "passenger": None,
            },
            fetch_pmv=lambda: {"driver": {"pmv": 0.0, "ppd": 5}},
            fetch_body_temperature=lambda: (_ for _ in ()).throw(ConnectionRefusedError("boom")),
            post_json=lambda url, payload: sent.append(payload),
        )

        self.assertTrue(service.run_once())
        self.assertIsNone(sent[0]["seats"]["driver_side"]["tmperature"])

    def test_run_once_error_includes_fetch_stage(self):
        cfg = sync.Config(stats_url="http://127.0.0.1:7680/stats")
        service = sync.PersonInfoSync(
            cfg,
            fetch_stats=lambda: (_ for _ in ()).throw(ConnectionRefusedError("boom")),
            redis_get=lambda key: "{}",
            post_json=lambda url, payload: None,
        )

        with self.assertRaisesRegex(RuntimeError, "fetch stats failed.*127.0.0.1:7680"):
            service.run_once()

    def test_run_once_error_includes_post_stage(self):
        cfg = sync.Config(update_person_info_url="http://127.0.0.1:20001/update")
        service = sync.PersonInfoSync(
            cfg,
            fetch_stats=lambda: {"driver": None, "passenger": None},
            fetch_pmv=lambda: {},
            redis_get=lambda key: "{}",
            post_json=lambda url, payload: (_ for _ in ()).throw(ConnectionRefusedError("boom")),
        )

        with self.assertRaisesRegex(RuntimeError, "post updatePersonInfo failed.*127.0.0.1:20001"):
            service.run_once()


if __name__ == "__main__":
    unittest.main()
