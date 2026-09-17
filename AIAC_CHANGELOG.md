# AIAC 变更日志（模拟 git）

本文件模拟 git 提交历史，记录 /home/data/AIAC 项目及其运行环境的全部变更。
规则：每次变更追加一个 commit 块，**最新提交在最上方**；commit id 取内容摘要前 7 位；
回滚方式与验证结果必须随提交一并记录。

```
================================================================
commit 5d2c8f1  (HEAD -> main)
Author: Kimi <kimi-remote@aibox>
Date:   Mon Aug 03 16:22:00 2026 +0800

    fix(faceid): 合并同一人双身份 4c479dde5d944fa9 → b553d99c5831476b

    背景: 下午女测试在人脸库中存在两份档案 —— 4c479d(7-29创建,44863帧)
    与 b553d99c(6-22创建,68375帧), embedding相似度0.2949低于dedup
    阈值0.35, 自动去重从未触发。今天15:38识别为4c479d并训练新偏好,
    15:47又切到b553d99c用7/30旧模型发26°C, 同一人新旧偏好打架。
    主驾身高读数两ID一致(171.1), 用户确认当天实为两女一男三人。

    Changed:
      0612_seatfixed/data/gallery.pkl
        按样本数加权合并512维centroid(68375+44863=113238帧),
        保留b553d99c(更老), 删除4c479d, gallery 17→16
        (人脸服务经 scripts/stop.sh + start.sh 重启生效)
      models/user_mlps/b553d99c5831476b/
        用4c479d今天15:45新训练的MLP(24.5/23.5/风量2偏好)覆盖
        7/30旧模型, 删除4c479d的MLP目录

    Backup:
      /home/data/AIAC/backups/faceid_merge_20260803_1610/
        (原gallery.pkl + 两ID的MLP目录)
      回滚: 恢复gallery.pkl后 bash scripts/restart.sh;
            恢复两个MLP目录即可

    Verify:
      gallery: 16条, b553d99c n_samples=113238, 4c479d已删除
      人脸服务: 重启正常(worker PID 50971/api PID 51087), API应答恢复
      待观察: 她下次上车主驾应稳定识别为b553d99c, 加载今日新偏好
================================================================
commit b3f8e12
Author: Kimi <kimi-remote@aibox>
Date:   Mon Aug 03 15:25:00 2026 +0800

    fix(preference): P4 采集期补纯RF推理, 修复0样本卡死(P3副作用)

    现象(2026-08-03 14:29-14:47 实测): 用户接管→COLLECTING后14分钟
    0样本, 训练永不触发, 系统永远停在人工观察阶段。
    原因: P3在人工观察期清空_last_command_result, 而样本生成要求
    rf_result非空(作为base预测), 条件永不满足 → P3引入的回归。

    Changed (git 仓库 sls_recommendation_weather26, commit f9b0a65):
      src/main_controller.py
        + _run_rf_inference_for_sampling(): 采集期每5s补一次纯RF推理
          (不发送/不登记命令/CAN链路不健康时跳过), 样本base改为当前
          工况实时RF输出(比旧版用接管前命令更准)
      src/preference_learning/preference_layer_manager.py
        + state 只读属性(主循环采集路由用)
      test_p4_sampling.py (新增白盒测试)

    Backup:
      /home/data/AIAC/sls_recommendation_weather26/backups/p4_sampling_20260803_1502/
      回滚: 恢复该目录两个文件后重启推荐服务

    Verify:
      白盒: test_p4 5/5 PASS, P2回归 7/7 PASS
      车端全链路(15:11-15:20, 用户0b1587d84025427a):
        15:11:31 FaceID识别 → JOINT_ACTIVE
        15:11:49 人工调温 → 接管检测正常触发(P2)
        15:11:54 接管确认 → COLLECTING
        15:16:55 采集58条样本/301s → TRAINING (P4生效, 修复前为0)
        15:17:12 训练完成(1.25→0.0) → JOINT_ACTIVE
        15:17:13起 联合推理每5s下发24.5/23.0/4/2, CAN逐条确认,
                 无phantom二次接管(P3生效)
      遗留: updated_score=0.0 退化拟合 + MLP Δ=+2.5偏差 → 模型质量
            问题(P5, 与7/29-7/31历史分析一致), 机制层面已全通
================================================================
commit 8a6dc45
Author: Kimi <kimi-remote@aibox>
Date:   Mon Aug 03 12:26:00 2026 +0800

    fix(preference): P3 训练后过期命令基线导致秒级 phantom 接管

    现象(2026-08-03 12:06 实测): 第二轮人工接管→采集300s→训练成功后,
    恢复 JOINT_ACTIVE 仅 1 秒即报 CAN 不一致(driver_temp+wind_speed),
    5 秒后二次接管确认 → 再次 COLLECTING, 形成
    训练→推理→phantom接管→再训练 死循环, AI 永远学不进去。

    根因(日志+源码双重实锤):
      接管检测基线 _expected_command 在 COLLECTING/TRAINING 全程不清除。
      恢复 JOINT_ACTIVE 时仍保留接管前最后一条 AI 命令(12:00:36 发的
      19.5°C/风量6)且 ack 门已开放; 她的人工设定(24.0/风量3)与之对比
      必现 phantom 不一致 —— 不一致字段恰好只有 driver_temp+wind_speed,
      passenger/mode 因旧命令恰好相等而"一致"。同时主循环传入的
      rf_result=_last_command_result 也是这条过期命令。

    Changed (git 仓库 sls_recommendation_weather26, commit 672464a):
      src/preference_learning/preference_layer_manager.py
        + _clear_command_baseline(): TRAINING→JOINT_ACTIVE 及两处
          训练失败回退 RF_ONLY 时清空命令基线
        + _handle_joint_active / _handle_rf_only 门卫: 仅在本阶段
          真实发送过命令(note_command_sent 登记)后才执行接管检测
      src/main_controller.py
        + 人工观察/采集期间清空 _last_command_result
      test_p3_stale_baseline.py (新增白盒测试)

    Backup:
      /home/data/AIAC/sls_recommendation_weather26/backups/p3_stale_baseline_20260803_1216/
      回滚: 恢复该目录两个文件后重启推荐服务

    Verify:
      白盒: test_p3 6/6 PASS(含12:06场景复现), P2回归 7/7 PASS
      部署: md5 2fd956dc(manager)/9a0e7ebf(controller) 与编译验证一致
      服务: 12:24 重启(PID 17469), 启动8/8正常, CAN socket已连接
      车端: 待测试人员上车复测 —— 调温→采集→训练→恢复推理后
            不再秒级接管, AI值稳定在学到的偏好附近
================================================================
commit 2e7a9c4
Author: Kimi <kimi-remote@aibox>
Date:   Sun Aug 03 11:52:00 2026 +0800

    fix(preference): P2 接管检测失效修复 - 60s冷却窗早退 + ack死锁

    现象(2026-08-03 11:27 实测): 第二名测试人员上车被FaceID识别后,
    AI立即下发其MLP偏好值19°C并每5s重发; 她调温后被"立马顶回",
    日志中接管确认计时全程未触发。

    根因(双重叠加):
      (1) preference_layer_manager._handle_joint_active 在用户切换/
          训练后60s冷却窗内直接早退, 完全跳过接管检测, 而AI照常
          每5s发送 → 冷却窗内人工调整被无限顶回;
      (2) _detect_takeover 的 ack 门控: AI命令未获CAN回读确认前
          不判接管, 用户在回显前调温则确认门永久等不到匹配,
          超时(8s)仅记日志不开放 → 接管检测永久失效。

    Changed (git 仓库 sls_recommendation_weather26):
      aba7026  fix(preference): P2 接管检测修复
        + ACK_MISMATCH_FALLBACK_SECONDS=4.0: ack挂起且差异持续
          >=4s 直接按人工接管处理(回显实测~2s, 4s早于下一次5s调度)
        - 移除 _handle_joint_active 的 _post_train_cooldown 早退
      096152d  fix(git): .gitignore 锚定根目录模式
        原 preference_learning/ 未锚定, 误忽略 src/preference_learning/
        8个核心代码文件; 补录并顺带收录 test_p1/p2 白盒测试

    Backup:
      /home/data/AIAC/backups/takeover_ack_fix_20260803_1145/
      回滚: 恢复该目录下 preference_layer_manager.py 后重启推荐服务

    Verify:
      - P2白盒测试 7/7 PASS (/home/test/test_p2_takeover_fix.py)
      - P1白盒 9/9 PASS, test_mode4_safety_regressions EXIT=0
      - 服务 11:49 重启, mode4启用, 启动完成, 无Traceback
      - 部署MD5一致: df5e867b2b6202d9e7891b766bf6d2c7
================================================================
commit 5c8f3b2
Author: Kimi <kimi-remote@aibox>
Date:   Sun Aug 02 17:21:14 2026 +0800

    chore(git): sls_recommendation_weather26 纳入本地 git 版本管理

    在 /home/data/AIAC/sls_recommendation_weather26 执行 git init
    (分支 main), .gitignore 遵循 sls_recommendation 仓库约定:
    只追踪 src/ 源码、config.json、tests、models/feature_columns.json
    (共 25 个文件), 排除 logs/ data/ models/ 交付包与 zip 大文件。
    仓库级提交身份: Kimi <kimi-remote@aibox> (git config --local)。

    已重放两个真实 commit (与下方模拟记录一一对应):
      d3d3eed  chore: 基线导入 weather26 (mode4 运行版, P1 修复前)
      84ddd91  fix(recommendation): P1 门控 (对应 7b4d1e8)
    基线代码取自 backups/can_link_guard_20260802_1705/。

    分工约定:
      - 本 changelog 继续记录"环境级"变更 (systemd、脚本、目录级调整)
      - weather26 代码变更以后直接用 git 管理 (git log/diff 追溯)
================================================================
commit 7b4d1e8
Author: Kimi <kimi-remote@aibox>
Date:   Sun Aug 02 17:08:26 2026 +0800

    fix(recommendation): P1 CAN断流期推理发送门控 + 恢复窗温度跳变校验

    背景:
      P0(9f3e2a1)消除了 can0_service 的 ~96s 重启循环, 但车上/台架
      仍可能因其他原因出现 CAN 断流。断流期 CAN 特征过期后被
      config 默认值填充, RF+MLP 在 OOD 输入下输出极端结果
      (7/29-7/31 实测 Δ 最高 +5.5/+3.0°C), 发送到 0x387 后
      覆盖 HMI 人工设定。

    Changed:
      sls_recommendation_weather26/src/main_controller.py (+95/-6 行)
        + __init__: 链路状态/恢复观察窗(10s)/跳变阈值(2°C) 5个成员
        + _main_loop: 每拍 _update_can_link_state() 跟踪连接沿
        + _update_can_link_state(): 记录断开/恢复时刻
        + _can_link_healthy_for_send(): 链路已连接 且 0x33A 四路
          反馈信号全部新鲜才允许发送
        + _send_jump_suspect(): 恢复 10s 观察窗内与上次成功命令
          相比主/副驾温度跳变 >2°C 判定为断流污染推理, 丢弃
        * _run_base_inference / _run_joint_inference_mode4:
          发送路径接入上述两道门控, 拦截时不更新接管判断基准

    Backup:
      /home/data/AIAC/backups/can_link_guard_20260802_1705/main_controller.py
      回滚: cp 该文件覆盖 src/main_controller.py 后重启推荐服务

    Verify:
      - py_compile 本地/远程通过
      - test_mode4_safety_regressions.py EXIT=0
      - 新增白盒测试 /home/test/test_p1_link_guard.py: 9/9 PASS
        (连接沿检测/健康门控/跳变校验/观察窗边界)
      - 服务 17:08:26 重启, mode4 启用, 启动完成, 运行无 Traceback
      - 部署 MD5 一致: 0daacb73d619cca6ecd56eda9a7d9b55
================================================================
commit 9f3e2a1
Author: Kimi <kimi-remote@aibox>
Date:   Sun Aug 02 16:47:34 2026 +0800

    fix(can): 修复 aiac-can0-service 每 ~96s 被 systemd 误杀重启

    根因:
      unit 配置 Type=forking, 但 can0_service_v1.3.6.py 是前台长驻
      进程并不 daemonize, systemd 每 90s (TimeoutStartUSec 默认值)
      判定启动超时 → SIGTERM → Restart=on-failure + RestartSec=5s
      拉起 → 无限循环。修复前 NRestarts 仅当天即达 25 次;
      7/29-7/31 三天日志统计重启 2064 次, 每次产生 ~10s CAN 断流,
      导致 sls_recommendation_weather26 在断流窗口用默认特征值
      推理并发送极端温度/风量 (最高 Δ=+5.5/+3.0°C), 覆盖 HMI
      人工设定。

    Changed:
      /etc/systemd/system/aiac-can0-service.service
        - Type=forking        Restart=on-failure
        + Type=simple         Restart=always
      (其余字段不变; 已执行 daemon-reload + reset-failed + restart)

    Backup:
      /etc/systemd/system/aiac-can0-service.service.bak_20260802
      回滚: sudo cp 该文件还原 && systemctl daemon-reload
            && systemctl restart aiac-can0-service

    Verify (2026-08-02 16:47:34 ~ 17:01:10, 持续 13.5 min):
      - NRestarts: 0 (修复前半天 25 次)
      - SubState: 全程 running (修复前永远卡在 activating/start)
      - CAN 断流事件: 0 次新增 (仅 16:47:34 计划内重启 1 次;
        修复前每 ~96s 必有一次)
      - socket 客户端连接时长: >800s 持续增长 (修复前每 ~96s 归零)
        recommendation_client / result_sender / pmv_consumer /
        face_recognition 全部稳定在线

    Refs:
      根因分析: 见本次会话产出的
      《CAN重启与HMI覆盖问题_根因分析报告.md》(2026-08-02)
================================================================
```

<!-- 新提交请插入到本行上方、代码块内部 -->

