# PMV HTTP API 使用指南

## 架构

```
CAN (can2, 9个CAN ID, 14个DBC信号)
  → can0_service (解析+物理量缩放)
  → Unix Socket (../can_service/sock/can0_bus.sock)
  → pmv_socket_consumer.py (信号映射, PMV计算)
  → HTTP API :7861 (每秒更新)
```

| 项 | 值 |
|---|-----|
| API 地址 | `http://localhost:7861/pmv` |
| 默认端口 | 7861 (`--api-port` 可配置) |
| 更新频率 | 每秒 (GET 覆盖写，只保留最新值) |
| PMV 范围 | **-3 (冷) ~ 0 (中性) ~ +3 (热)** (ISO 7730) |
| 运行模式 | `safe_preview` (非量产标定) |

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
    "feet_temp_c": 29.69
  },
  "passenger": {
    "pmv": 1.236,
    "ppd": 38.7,
    "head_temp_c": 29.99,
    "feet_temp_c": 29.69
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
| `driver.head_temp_c` | float | 主驾头部温度 (°C) |
| `driver.feet_temp_c` | float | 主驾脚部温度 (°C) |
| `passenger.*` | float | 副驾对应字段 |
| `cabin_temp_c` | float | 前排座舱温度 (°C) |
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
    --output-dir /tmp/pmv_out    # 每帧存文件 (可选)
```
