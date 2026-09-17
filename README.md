# AIAC

AIAC 车载空调智能控制系统的集成代码仓库。

## 服务

- `can_service`: CAN 总线读写与 Socket 服务
- `0612_seatfixed`: 座舱视觉感知服务
- `vehicle_runtime_package_vnext_budian`: PMV 与热舒适计算
- `interface_intergration`: 人员信息同步
- `sls_recommendation_weather26`: 空调推荐与偏好学习
- `thermal_stream`: 热成像视频流

## 本地配置

仓库不包含运行时数据、用户数据、模型二进制文件和本机配置。部署前请复制并按环境修改：

- `interface_intergration/config.example.yaml` → `interface_intergration/config.yaml`
- `thermal_stream/.env.example` → `thermal_stream/.env`
- `sls_recommendation_weather26/config.example.json` → `sls_recommendation_weather26/config.json`
- `0612_seatfixed/.env.example` → `0612_seatfixed/.env`

模型文件需要单独部署到各服务预期的 `models/` 目录。

## 启停

```bash
bash start_all.sh
bash stop_all.sh
```
