# 运行时输入边界

## 禁止直接传入公共离线入口的真值字段

`run_vehicle_pmv.py` 会标记以下原始验证字段：

- `TA_FdHeadTempLe`、`TA_FpHeadTempLe`
- `TA_FdFloorTemp1/2`、`TA_FpFloorTemp1/2`
- `TA_CarbinFrntTempLe/Ri`
- `measured_head_temperature`
- `measured_feet_temperature`
- `measured_cabin_temperature`

这些名称属于原始 CAN/验证数据边界，不应由普通离线调用方直接注入模型。

## 在线 consumer 的受控例外

在线链路会在 `pmv_service.input_builder` 中读取四路头部测点，完成有限值检查和左右平均后，只把规范化字段 `measured_head_air_temp_c` 传给 pipeline。该字段是当前实车 PMV 空气温度来源，不等同于允许任意调用方绕过输入边界传入原始真值键。

脚温和 cabin 真值仍不作为 PMV 直接输入。`feet_temp_c` 仅作为兼容输出字段。
