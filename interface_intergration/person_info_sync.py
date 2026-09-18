import argparse
import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib import request


AGE_MAP = {
    0: "12-17",
    1: "18-40",
    2: "41-59",
    3: "60-74",
    4: "75+",
}

CLOTH_MAP = {
    0: "西装外套",
    1: "薄夹克",
    2: "长款大衣",
    3: "羽绒服",
    4: "长袖针织毛衣",
    5: "长袖衬衫",
    6: "短袖",
    7: "背心",
    8: "长袖",
    9: "连帽卫衣",
}

GENDER_MAP = {
    0: "女性",
    1: "男性",
}


@dataclass
class Config:
    stats_url: str = "http://127.0.0.1:7680/stats"
    pmv_url: str = "http://127.0.0.1:7861/pmv"
    body_temperature_url: str = "http://127.0.0.1:7864/body-temperature"
    update_person_info_url: str = (
        "http://192.168.0.104:20001/api/airConditioningControl/updatePersonInfo"
    )
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_key: str = "pmv_thermal_comfort"
    poll_interval_seconds: float = 1.0
    request_timeout_seconds: float = 3.0


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def map_age(value: Any) -> str | None:
    return AGE_MAP.get(_coerce_int(value))


def map_cloth(value: Any) -> str | None:
    return CLOTH_MAP.get(_coerce_int(value))


def map_gender(value: Any) -> str | None:
    return GENDER_MAP.get(_coerce_int(value))


def format_pmv_score(pmv: Any, ppd: Any) -> str | None:
    if pmv is None or ppd is None:
        return None
    try:
        pmv_value = float(pmv)
        ppd_value = float(ppd)
    except (TypeError, ValueError):
        return None

    label = pmv_label(pmv_value)
    if label is None:
        return None
    score = (1.0 - ppd_value / 100.0) * 100.0
    return f"{label}[{score:.1f}]"


def pmv_label(pmv: float) -> str | None:
    if -0.5 <= pmv <= 0.5:
        return "舒适"
    if 0.5 < pmv <= 1:
        return "轻微偏热"
    if 1 < pmv <= 1.5:
        return "偏热"
    if 1.5 < pmv <= 2:
        return "热"
    if 2 < pmv <= 3:
        return "极热"
    if -1 <= pmv < -0.5:
        return "轻微偏冷"
    if -1.5 <= pmv < -1:
        return "偏冷"
    if -2 <= pmv < -1.5:
        return "冷"
    if -3 <= pmv < -2:
        return "极冷"
    return None


def normalize_body_temperature(body_temperature_data: dict[str, Any] | None) -> float | None:
    if not body_temperature_data or not body_temperature_data.get("valid"):
        return None
    value = body_temperature_data.get("body_temp_c")
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def build_seats(
    stats: dict[str, Any],
    pmv_data: dict[str, Any],
    body_temperature_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comfort = normalize_pmv_data(pmv_data)
    body_temperature = normalize_body_temperature(body_temperature_data)
    return {
        "driver_side": build_seat(stats.get("driver"), comfort.get("driver"), body_temperature),
        "passenger_side": build_seat(
            stats.get("passenger"), comfort.get("passenger"), body_temperature
        ),
    }


def normalize_pmv_data(pmv_data: dict[str, Any]) -> dict[str, Any]:
    comfort = pmv_data.get("comfort")
    if isinstance(comfort, dict):
        return comfort

    result: dict[str, Any] = {}
    for seat in ("driver", "passenger"):
        value = pmv_data.get(seat)
        result[seat] = value if isinstance(value, dict) else {}
    return result


def build_seat(
    person: dict[str, Any] | None,
    comfort: dict[str, Any] | None,
    body_temperature: float | None = None,
) -> dict[str, Any]:
    if not person:
        return {
            "User_Id": "-1",
            "sex": None,
            "age": None,
            "cloth": None,
            "bmi": None,
            "tmperature": None,
            "pmv_score": None,
        }

    comfort = comfort or {}
    return {
        "User_Id": str(person.get("identity_id") or "-1"),
        "sex": map_gender(person.get("gender")),
        "age": map_age(person.get("age")),
        "cloth": map_cloth(person.get("cloth")),
        "bmi": person.get("bmi"),
        "tmperature": body_temperature,
        "pmv_score": format_pmv_score(comfort.get("pmv"), comfort.get("ppd")),
    }


def load_config(path: str | Path = "config.yaml", env: dict[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    data = _read_simple_yaml(Path(path))
    redis_data = data.get("redis") if isinstance(data.get("redis"), dict) else {}

    cfg = Config(
        stats_url=str(data.get("stats_url") or Config.stats_url),
        pmv_url=str(data.get("pmv_url") if data.get("pmv_url") is not None else Config.pmv_url),
        body_temperature_url=str(
            data.get("body_temperature_url")
            if data.get("body_temperature_url") is not None
            else Config.body_temperature_url
        ),
        update_person_info_url=str(
            data.get("update_person_info_url") or Config.update_person_info_url
        ),
        redis_host=str(redis_data.get("host") or Config.redis_host),
        redis_port=int(redis_data.get("port") or Config.redis_port),
        redis_key=str(redis_data.get("key") or Config.redis_key),
        poll_interval_seconds=float(
            data.get("poll_interval_seconds") or Config.poll_interval_seconds
        ),
        request_timeout_seconds=float(
            data.get("request_timeout_seconds") or Config.request_timeout_seconds
        ),
    )

    cfg.stats_url = env.get("PERSON_SYNC_STATS_URL", cfg.stats_url)
    cfg.pmv_url = env.get("PERSON_SYNC_PMV_URL", cfg.pmv_url)
    cfg.body_temperature_url = env.get(
        "PERSON_SYNC_BODY_TEMPERATURE_URL", cfg.body_temperature_url
    )
    cfg.update_person_info_url = env.get(
        "PERSON_SYNC_UPDATE_URL", cfg.update_person_info_url
    )
    cfg.redis_host = env.get("PERSON_SYNC_REDIS_HOST", cfg.redis_host)
    cfg.redis_port = int(env.get("PERSON_SYNC_REDIS_PORT", cfg.redis_port))
    cfg.redis_key = env.get("PERSON_SYNC_REDIS_KEY", cfg.redis_key)
    cfg.poll_interval_seconds = float(
        env.get("PERSON_SYNC_POLL_INTERVAL_SECONDS", cfg.poll_interval_seconds)
    )
    cfg.request_timeout_seconds = float(
        env.get("PERSON_SYNC_REQUEST_TIMEOUT_SECONDS", cfg.request_timeout_seconds)
    )
    return cfg


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    result: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith(" ") and line.endswith(":"):
            section_name = line[:-1].strip()
            current_section = {}
            result[section_name] = current_section
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        target = current_section if raw_line.startswith(" ") and current_section is not None else result
        target[key.strip()] = _parse_scalar(value.strip())
    return result


def _parse_scalar(value: str) -> Any:
    if value in ("null", "None", "~"):
        return None
    if value in ("true", "false"):
        return value == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def http_get_json(url: str, timeout: float) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def redis_get(host: str, port: int, key: str, timeout: float) -> str | None:
    command = f"*2\r\n$3\r\nGET\r\n${len(key.encode('utf-8'))}\r\n{key}\r\n".encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(command)
        return _read_redis_bulk_string(sock)


def _read_redis_bulk_string(sock: socket.socket) -> str | None:
    first_line = _readline(sock)
    if first_line == b"$-1\r\n":
        return None
    if not first_line.startswith(b"$"):
        raise RuntimeError(f"Unexpected Redis response: {first_line!r}")
    length = int(first_line[1:].strip())
    data = b""
    while len(data) < length + 2:
        chunk = sock.recv(length + 2 - len(data))
        if not chunk:
            raise RuntimeError("Redis connection closed before full response")
        data += chunk
    return data[:length].decode("utf-8")


def _readline(sock: socket.socket) -> bytes:
    data = b""
    while not data.endswith(b"\r\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise RuntimeError("Redis connection closed before line end")
        data += chunk
    return data


def timestamp_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


class PersonInfoSync:
    def __init__(
        self,
        config: Config,
        fetch_stats: Callable[[], dict[str, Any]] | None = None,
        fetch_pmv: Callable[[], dict[str, Any]] | None = None,
        fetch_body_temperature: Callable[[], dict[str, Any]] | None = None,
        redis_get: Callable[[str], str | None] | None = None,
        post_json: Callable[[str, dict[str, Any]], Any] | None = None,
    ):
        self.config = config
        self.fetch_stats = fetch_stats or (
            lambda: http_get_json(config.stats_url, config.request_timeout_seconds)
        )
        self.fetch_pmv = fetch_pmv or (
            lambda: http_get_json(config.pmv_url, config.request_timeout_seconds)
        )
        self.fetch_body_temperature = fetch_body_temperature or (
            lambda: http_get_json(config.body_temperature_url, config.request_timeout_seconds)
        )
        self.redis_get = redis_get or (
            lambda key: globals()["redis_get"](
                config.redis_host,
                config.redis_port,
                key,
                config.request_timeout_seconds,
            )
        )
        self.post_json = post_json or (
            lambda url, payload: http_post_json(
                url,
                payload,
                config.request_timeout_seconds,
            )
        )
        self.last_sent_seats_json: str | None = None

    def run_once(self) -> bool:
        try:
            stats = self.fetch_stats()
        except Exception as exc:
            raise RuntimeError(f"fetch stats failed from {self.config.stats_url}: {exc}") from exc

        try:
            if self.config.pmv_url:
                pmv_data = self.fetch_pmv()
            else:
                raw_pmv = self.redis_get(self.config.redis_key)
                pmv_data = json.loads(raw_pmv) if raw_pmv else {}
        except Exception as exc:
            if self.config.pmv_url:
                raise RuntimeError(f"fetch PMV API failed from {self.config.pmv_url}: {exc}") from exc
            else:
                redis_addr = f"{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_key}"
                raise RuntimeError(f"fetch Redis PMV failed from {redis_addr}: {exc}") from exc

        body_temperature_data = None
        if self.config.body_temperature_url:
            try:
                body_temperature_data = self.fetch_body_temperature()
            except Exception as exc:
                print(
                    "person info sync warning: "
                    f"fetch body temperature failed from {self.config.body_temperature_url}: {exc}"
                )

        seats = build_seats(stats, pmv_data, body_temperature_data)
        seats_json = json.dumps(seats, sort_keys=True, ensure_ascii=False)
        if seats_json == self.last_sent_seats_json:
            return False

        payload = {
            "timestamp": timestamp_now(),
            "seats": seats,
        }
        try:
            self.post_json(self.config.update_person_info_url, payload)
        except Exception as exc:
            url = self.config.update_person_info_url
            raise RuntimeError(f"post updatePersonInfo failed to {url}: {exc}") from exc

        self.last_sent_seats_json = seats_json
        return True

    def run_forever(self) -> None:
        while True:
            try:
                sent = self.run_once()
                if sent:
                    print("person info updated")
            except KeyboardInterrupt:
                print("person info sync stopped")
                return
            except Exception as exc:  # Keep the daemon alive across transient services.
                print(f"person info sync error: {exc}")
            time.sleep(self.config.poll_interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync person info to AI air-conditioner API.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--once", action="store_true", help="Run one sync iteration and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    service = PersonInfoSync(cfg)
    if args.once:
        try:
            return 0 if service.run_once() else 2
        except Exception as exc:
            print(f"person info sync error: {exc}")
            return 1
    service.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
