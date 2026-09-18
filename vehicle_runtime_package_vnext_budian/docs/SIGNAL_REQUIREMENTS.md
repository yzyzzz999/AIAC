# PMV 运行时信号需求

## 必需信号

| 字段 | 单位 | 说明 |
|---|---|---|
| `amb_t_c` | °C | 首次运行必需 |

## 可选信号与默认值

| 字段 | 单位 | 默认或回退 |
|---|---|---|
| `rh_percent` | % | 50 |
| `vehicle_speed_kph` | km/h | 0 |
| `solar_driver_w_m2` / `solar_passenger_w_m2` | W/m² | 0 |
| `eva_t_c` | °C | 环境温度 |
| `ict_c` | °C | 用于初始化和在线状态校正 |
| `tma` | °C | 缺失时使用现有模型 fallback |
| `flows` | m³/h | 缺失时为 0 或由适配器构造 |
| `measured_head_air_temp_c` | °C | 主副驾独立；缺失侧回退模型状态 |
| `image_inputs` | — | 缺失时使用默认乘员参数 |

在线旁路额外使用总风量和三路风门位置估算主副驾局部风速。只有完整且有限、总风量大于零的输入才生成 override。

完整机器可读定义见 `../config/schemas/runtime_input_schema.json`。
