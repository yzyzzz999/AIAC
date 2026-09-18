# CAN 运行时信号说明

当前在线服务由 `can0_service_v1.3.8` 完成 DBC 解码和物理量缩放，PMV consumer 只接收工程量。

| CAN/DBC 信号 | PMV 字段或用途 | 缺失行为 |
|---|---|---|
| `VIU_AmbT` | `amb_t_c` | 首次运行必须存在；之后保留最近有效值 |
| `RSM_RelHum` | `rh_percent` | 默认 50% |
| `IPB_VehicleSpeed` | `vehicle_speed_kph` | 默认 0 km/h |
| `RSM_LeSolarInten` / `RSM_RiSolarInten` | 左右太阳辐照 | 默认 0 W/m² |
| `AC_FEvapCurrentTemp` | `eva_t_c` | 可选 |
| `AC_FrntInCarT` | `ict_c`；旁路 MRT 代理 | 初始状态需要 |
| `AC_Drvr/PassFaceVentActT` | `tma` 出风温度 | 可选 |
| `AC_Forward_AirFlowTarget` | 总风量 | 缺失或无效时不生成风速 override |
| 三路前排风门位置 | 风量分支比例 | 缺失或非有限时不生成风速 override |
| `TA_Fd/FpHeadTempLe/Ri` | 主副驾实测头部空气温度 | 单侧有效时采用单侧；两侧缺失时回退模型状态 |
| `TS_FrntWidTemp` | 前挡温度诊断输入 | 可选 |

Socket 协议层会丢弃非数值和非有限值，不用无效新值覆盖已有有效缓存。
