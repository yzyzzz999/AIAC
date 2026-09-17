# B1: SIG/FTE 运行时信号需求

**文档版本**: vehicle_runtime_package_vnext_v1
**日期**: 2026-06-15

## 强制要求

运行时输入只能来自 SIG/FTE 工程量模块。禁止直接使用 CAN 原始信号（有例外，见 B2）。

## 信号清单

| 运行时字段 | SIG/FTE 来源 | 单位 | 必须 | 默认值 |
|------------|-------------|------|------|--------|
| `amb_t_c` | SIG.AmbientTemperature_C | °C | 是 | — |
| `rh_percent` | SIG.RelativeHumidity_pct | % | 否 | 50.0 |
| `vehicle_speed_kph` | SIG.VehicleSpeed_kph | km/h | 否 | 0.0 |
| `solar_driver_w_m2` | SIG.SolarIrradiance_Driver | W/m² | 否 | 0.0 |
| `solar_passenger_w_m2` | SIG.SolarIrradiance_Passenger | W/m² | 否 | 0.0 |
| `eva_t_c` | FTE.EvapOutletTemp_C | °C | 否 | — |
| `hct_c` | FTE.HeaterCoreTemp_C | °C | 否 | — |
| `driver_face_flow` | FTE.FrntDrvFaceFlow_m3h | m³/h | 否 | — |
| `passenger_face_flow` | FTE.FrntPsgFaceFlow_m3h | m³/h | 否 | — |
| `driver_floor_flow` | FTE.FrntDrvFloorFlow_m3h | m³/h | 否 | — |
| `passenger_floor_flow` | FTE.FrntPsgFloorFlow_m3h | m³/h | 否 | — |
| `driver_defrost_flow` | FTE.FrntDrvDefrostFlow_m3h | m³/h | 否 | — |
| `passenger_defrost_flow` | FTE.FrntPsgDefrostFlow_m3h | m³/h | 否 | — |

## 说明

- `ict_c` (舱内设定温度) 当前通过 CAN 例外 `AC_FrntInCarT` 获得，SIG 替换待接入
- `image_inputs` 可选，用于乘员信息（占用状态、衣着、代谢率）
- `x_init` 可选，28 元素 CHTD 状态向量，用于热启动

## CAN 例外

见 `CAN_SIGNAL_EXCEPTION_REPORT.md` (B2)
