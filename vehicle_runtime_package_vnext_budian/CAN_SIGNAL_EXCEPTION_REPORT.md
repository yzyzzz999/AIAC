# B2: CAN 信号例外报告

**文档版本**: vehicle_runtime_package_vnext_v1
**日期**: 2026-06-15
**状态**: 14 个 CAN 信号仍需直接接入（SIG/FTE 替代未完成）

## 例外清单

| CAN 信号 | 运行时字段 | 单位 | SIG/FTE 替代 | 部署影响 |
|----------|-----------|------|-------------|---------|
| `VIU_AmbT` | `amb_t_c` | °C | SIG.AmbientTemperature_C | 必须 — 无默认值 |
| `RH_pct` | `rh_percent` | % | SIG.RelativeHumidity_pct | 缺失时默认 50% |
| `IPB_VehicleSpeed` | `vehicle_speed_kph` | km/h | SIG.VehicleSpeed_kph | 缺失时默认 0 km/h |
| `RSM_LeSolarInten` | `solar_driver_w_m2` | W/m² | SIG.SolarIrradiance_Driver | 缺失时默认 0 W/m² |
| `RSM_RiSolarInten` | `solar_passenger_w_m2` | W/m² | SIG.SolarIrradiance_Passenger | 缺失时默认 0 W/m² |
| `AC_Forward_AirFlowTarget` | `flows.total_m3h` | m³/h | FTE.TotalAirflowTarget_m3h | 影响风量计算 |
| `AC_DrvrFaceVentActT` | `driver_face_flow` | m³/h | FTE.FrntDrvFaceFlow_m3h | 头部区域必需 |
| `AC_PassFaceVentActT` | `passenger_face_flow` | m³/h | FTE.FrntPsgFaceFlow_m3h | 头部区域必需 |
| `AC_DrvrFootVentActT` | `driver_floor_flow` | m³/h | FTE.FrntDrvFloorFlow_m3h | 脚部区域必需 |
| `AC_PassFootVentActT` | `passenger_floor_flow` | m³/h | FTE.FrntPsgFloorFlow_m3h | 脚部区域必需 |
| `AC_BLOW_FaceVentilaPosn` | `tma.face_posn_v` | V | FTE.FaceVentPosition_V | 模式检测 |
| `AC_FrantFootVentPosn` | `tma.foot_posn_v` | V | FTE.FootVentPosition_V | 模式检测 |
| `AC_DefrostVentilaPosn` | `tma.defrost_posn_v` | V | FTE.DefrostVentPosition_V | 模式检测 |
| `AC_FrntInCarT` | `ict_c` | °C | SIG.InCarTemp_C | 可选，内部估算 |

## 例外原因

当前 SIG/FTE 模块尚未完成以下信号的完整接入：

1. 执行器位置电压 (V)：`AC_BLOW_FaceVentilaPosn`、`AC_FrantFootVentPosn`、`AC_DefrostVentilaPosn`
   - 用于 v5 airflow 模式检测（actuator_first 策略）
   - SIG/FTE 替代信号定义中，待接入

2. 流量工程量 (m³/h)：6 个风口流量
   - 当前通过执行器电压反算，FTE 直接流量测量精度更高

## 量产接入路径

量产前需完成：
- [ ] SIG.AmbientTemperature_C → `amb_t_c`（替代 VIU_AmbT）
- [ ] FTE.FaceVentPosition_V 等 3 个执行器位置信号
- [ ] FTE 6 个风口流量工程量

在 SIG/FTE 完成接入前，上述 CAN 信号通过运行时适配器（`runtime_adapter.py`）直接映射。
