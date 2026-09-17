# AIAC 项目恢复指南

**当前状态：已停用（2026-08-03，采集数据前关闭）**
**代码备份：** `/home/data/AIAC_project_20260803.zip`

---

## 一、恢复步骤

### 1. 恢复 systemd 服务（开机自动启动 CAN 层）

```bash
sudo systemctl enable aiac-fix-can-names.service aiac-can-setup.service aiac-can0-service.service
sudo systemctl start aiac-fix-can-names.service aiac-can-setup.service aiac-can0-service.service
```

检查状态：
```bash
sudo systemctl status aiac-fix-can-names.service aiac-can-setup.service aiac-can0-service.service
```

### 2. 恢复 crontab 定时任务

在 `crontab -e` 中添加以下两行（注意保留已有的 `can_parser` 那一行）：

```cron
@reboot sleep 60 && /bin/bash /home/data/AIAC/start_all.sh >> /home/data/AIAC/.startup_logs/auto_start.log 2>&1
*/2 * * * * /bin/bash /home/data/AIAC/watchdog.sh
```

或者在命令行直接追加：
```bash
(crontab -l 2>/dev/null; echo '@reboot sleep 60 && /bin/bash /home/data/AIAC/start_all.sh >> /home/data/AIAC/.startup_logs/auto_start.log 2>&1'; echo '*/2 * * * * /bin/bash /home/data/AIAC/watchdog.sh') | crontab -
```

### 3. 手动启动全部服务

```bash
bash /home/data/AIAC/start_all.sh
```

### 4. 检查状态

```bash
bash /home/data/AIAC/start_all.sh --status
```

---

## 二、架构概览（4 个核心服务 + 仪表盘）

| 序号 | 服务 | 目录 | 端口/Socket |
|------|------|------|-------------|
| 1 | CAN 总线服务 | `can_service/` | `/home/data/AIAC/can_service/sock/can0_bus.sock` |
| 2 | 视觉感知服务 | `0612_seatfixed/` | HTTP `:7860` |
| 3 | PMV 舒适度服务 | `vehicle_runtime_package_vnext_budian/` | HTTP `:7861` |
| 4 | 推荐服务 | `sls_recommendation_weather26/` | 无端口（消费型） |
| 5 | 实时仪表盘 | `live_dashboard.py`（根目录） | HTTP `:7862` |

**Python 环境：** conda env `AIAC`，位于 `/home/data/miniconda3/envs/AIAC/`

---

## 三、自动保护机制（恢复后生效）

| 机制 | 触发条件 | 行为 |
|------|----------|------|
| systemd `aiac-can0-service` | CAN0 进程退出 | `Restart=always`，5 秒后自动拉起 |
| crontab `@reboot` | 系统启动 60 秒后 | 执行 `start_all.sh` |
| crontab `*/2` watchdog | 每 2 分钟 | 检查 6 项健康指标，任意失败自动重启 |

---

## 四、手动停止（如需再次关闭）

```bash
bash /home/data/AIAC/stop_all.sh
```

---

## 五、已确认关停清单（2026-08-03）

- [x] 所有 AIAC 进程已停止
- [x] crontab 中 AIAC 的 `@reboot` 和 `*/2` watchdog 已移除
- [x] systemd 服务已全部 disable + stop（aiac-can-setup / aiac-can0-service / aiac-fix-can-names）
- [x] 代码全量备份已生成：`/home/data/AIAC_project_20260803.zip`
