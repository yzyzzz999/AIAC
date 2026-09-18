# PMV HTTP API 使用指南

## 架构

```
CAN (can2~can5，含 can5 CAN FD 头部温度帧)
  → can0_service (解析+物理量缩放)
  → Unix Socket (../can_service/sock/can0_bus.sock)
  → pmv_socket_consumer.py (兼容 CLI / 编排入口)
  → pmv_service.socket_client (协议、心跳、重连)
  → pmv_service.input_builder + image_inputs (纯输入映射)
  → hvac_sim.runtime_input + runtime_adapter (规范化、状态/总线构造)
  → hvac_sim.pipeline (CHTD / 旁路调度)
  → hvac_sim.pipeline_pmv (单座 PMV/PPD)
  → pmv_service.api (稳定 HTTP 契约)
  → HTTP API :7861 (每秒更新)
```

`pmv_socket_consumer.py` 保留既有导入名和命令行参数，具体职责位于上述小模块中。
核心计算模块不执行网络、文件或控制台 I/O；周期诊断由 logging observer 注入。

| 项 | 值 |
|---|-----|
| API 地址 | `http://localhost:7861/pmv` |
| 默认端口 | 7861 (`--api-port` 可配置) |
| 更新频率 | 每秒 (GET 覆盖写，只保留最新值) |
| PMV 范围 | **-3 (冷) ~ 0 (中性) ~ +3 (热)** (ISO 7730) |
| 运行模式 | 实车 consumer 默认直接 PMV 旁路；离线入口支持 CHTD |

---

## 一键启停

```bash
# 开发模式
bash start_pmv_service.sh

# 自定义端口
API_PORT=8080 bash start_pmv_service.sh

# 停止
bash stop_pmv_service.sh
```

日志: `/tmp/pmv_service_logs/`, PID: `/tmp/pmv_service_pids/`

---

## 快速读取

```bash
curl -s http://localhost:7861/pmv | python3 -m json.tool
```

---

## API 响应格式

```bash
curl -s http://localhost:7861/pmv | jq '.'
```

```json
{
  "driver": {
    "pmv": 1.151,
    "ppd": 35.4,
    "head_temp_c": 29.61,
    "feet_temp_c": 29.69,
    "mrt_c": 27.2,
    "rh_percent": 45.0,
    "air_speed_m_s": 0.31
  },
  "passenger": {
    "pmv": 1.236,
    "ppd": 38.7,
    "head_temp_c": 29.99,
    "feet_temp_c": 29.69,
    "mrt_c": 27.2,
    "rh_percent": 45.0,
    "air_speed_m_s": 0.31
  },
  "cabin_temp_c": 29.75,
  "amb_temp_c": 27.4,
  "timestamp": "2026-06-19T08:43:01.711013Z",
  "status": "ok",
  "run_index": 12
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `driver.pmv` | float | 主驾 PMV (-3~+3) |
| `driver.ppd` | float | 主驾 PPD 不满意率 (%) |
| `driver.head_temp_c` | float/null | PMV 实际使用的主驾空气温度；实车旁路时来自头部测温平均值 (°C) |
| `driver.feet_temp_c` | float/null | 兼容字段，来自 CHTD 模型状态；旁路模式下可能不随每次计算变化 |
| `driver.mrt_c` | float/null | PMV 使用的平均辐射温度；实车旁路当前以车内温度为代理 (°C) |
| `driver.rh_percent` | float/null | PMV 使用的相对湿度 (%) |
| `driver.air_speed_m_s` | float/null | PMV 使用的局部空气流速估算值 (m/s) |
| `passenger.*` | float | 副驾对应字段 |
| `cabin_temp_c` | float/null | 当前兼容映射为主驾 PMV 空气温度；命名偏差留待独立修复 |
| `amb_temp_c` | float | 外温/环境温度 (°C) |
| `timestamp` | string | UTC ISO 时间戳 |
| `status` | string | `"ok"` / `"error"` |
| `run_index` | int | PMV 运行计数 |

---

## 例程

### 1. 主副驾舒适度

```bash
curl -s http://localhost:7861/pmv | jq '{driver: .driver, passenger: .passenger}'
```

### 2. Python 读取

```python
import requests
d = requests.get("http://localhost:7861/pmv").json()
drv = d["driver"]
print(f"PMV drv={drv['pmv']:+.2f} pass={d['passenger']['pmv']:+.2f} | "
      f"Head={drv['head_temp_c']:.1f}°C Cabin={d['cabin_temp_c']:.1f}°C "
      f"Amb={d['amb_temp_c']:.1f}°C | Status={d['status']}")
```

### 3. 持续监控

```bash
watch -n 1 'curl -s http://localhost:7861/pmv | jq "{driver, passenger, cabin_temp_c, amb_temp_c}"'
```

### 4. 存活检查

```bash
curl -s http://localhost:7861/pmv | jq '.run_index'
sleep 3
curl -s http://localhost:7861/pmv | jq '.run_index'
# run_index 增长 → 服务正常
```

### 5. 实时仪表盘

```bash
python live_pmv_display.py --api-port 7861
```

---

## PMV 值含义 (±3, ISO 7730)

| PMV | 体感 | 舒适 |
|-----|------|------|
| -3 ~ -2 | 很冷 | ✗ |
| -2 ~ -0.5 | 稍冷 | ✗ |
| **-0.5 ~ +0.5** | **中性** | **✓** |
| +0.5 ~ +2 | 稍暖 | ✗ |
| +2 ~ +3 | 很热 | ✗ |

---

## 消费者启动参数

```bash
python pmv_socket_consumer.py \
    --api-port 7861 \           # HTTP API 端口 (默认 7861)
    --interval 1.0 \             # PMV 运行间隔(秒)
    --socket <path> \            # Unix socket 路径
    --output-dir /tmp/pmv_out \  # 每帧存文件 (可选)
    --log-level INFO             # DEBUG/INFO/WARNING/ERROR/CRITICAL
```

机器可读的 HTTP 响应契约见 `../config/schemas/pmv_http_api_schema.json`。
