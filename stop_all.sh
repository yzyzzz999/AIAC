#!/usr/bin/env bash
# ==============================================================================
# AIAC 一键停止脚本
# ==============================================================================
# 按反向顺序停止所有服务（热成像流 → 仪表盘 → 推荐 → 接口同步 → PMV → 视觉 → CAN），
# 确保依赖关系不被破坏。
#
# 用法:
#   bash stop_all.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAN_SERVICE_PATTERN='can0_service_v[0-9.]+\.py'

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_step()  { echo -e "\n${CYAN}──  $*${NC}"; }

kill_by_pidfile() {
    local pidfile="$1" label="$2"
    if [ -f "$pidfile" ]; then
        local pid
        pid="$(cat "$pidfile" 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log_info "停止 ${label} (PID=$pid)..."
            kill "$pid" 2>/dev/null || true
            for i in $(seq 1 10); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            kill -9 "$pid" 2>/dev/null || true
            log_info "${label} 已停止 ✓"
        fi
        rm -f "$pidfile"
    fi
}

kill_by_pattern() {
    local pattern="$1" label="$2"
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log_info "停止 ${label} (PIDs: $pids)..."
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 2
        pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
        log_info "${label} 已停止 ✓"
    fi
}

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     AIAC 系统一键停止               ║"
echo "╚══════════════════════════════════════╝"

# ---- 7. 停止热成像视频流 ------------------------------------------------
log_step "1/7  停止热成像视频流"
if [ -f "${SCRIPT_DIR}/thermal_stream/stop.sh" ]; then
    bash "${SCRIPT_DIR}/thermal_stream/stop.sh"
else
    kill_by_pattern "[t]hermal_stream/main.py" "热成像视频流"
fi

# ---- 6. 停止仪表盘 ------------------------------------------------------
log_step "2/7  停止仪表盘"
kill_by_pidfile "/tmp/dashboard.pid" "仪表盘"
kill_by_pattern "live_dashboard" "仪表盘(残留)"

# ---- 5. 停止推荐服务 ----------------------------------------------------
log_step "3/7  停止推荐服务"
kill_by_pidfile "/tmp/recommendation.pid" "推荐服务"
kill_by_pattern "main_controller" "推荐服务(残留)"

# ---- 4. 停止接口同步服务 ------------------------------------------------
log_step "4/7  停止接口同步服务"
# 兼容此前通过 systemd-run 启动的实例。
systemctl --user stop interface-integration-person-info-sync.service >/dev/null 2>&1 || true
kill_by_pidfile "/tmp/interface_integration.pid" "接口同步服务"
kill_by_pattern "[p]erson_info_sync.py" "接口同步服务(残留)"

# ---- 3. 停止 PMV 服务 ---------------------------------------------------
log_step "5/7  停止 PMV 服务"
if [ -f "${SCRIPT_DIR}/vehicle_runtime_package_vnext_budian/stop_pmv_service.sh" ]; then
    bash "${SCRIPT_DIR}/vehicle_runtime_package_vnext_budian/stop_pmv_service.sh"
else
    kill_by_pidfile "/tmp/pmv_service_pids/pmv_consumer.pid" "PMV Consumer"
    kill_by_pattern "pmv_socket_consumer" "PMV Consumer(残留)"
fi

# ---- 2. 停止视觉感知服务 ------------------------------------------------
log_step "6/7  停止视觉感知服务"
if [ -f "${SCRIPT_DIR}/0612_seatfixed/scripts/stop.sh" ]; then
    bash "${SCRIPT_DIR}/0612_seatfixed/scripts/stop.sh"
else
    kill_by_pidfile "${SCRIPT_DIR}/0612_seatfixed/worker.pid" "视觉 Worker"
    kill_by_pidfile "${SCRIPT_DIR}/0612_seatfixed/server.pid" "视觉 API"
fi

# ---- 1. 停止 CAN 服务 ---------------------------------------------------
log_step "7/7  停止 CAN 服务"
kill_by_pidfile "/tmp/can0_service.pid" "can0 Service"
kill_by_pattern "$CAN_SERVICE_PATTERN" "can0 Service(残留)"
# root 启动的 can0 服务需要 sudo 才能停止（sudoers 白名单）
if pgrep -f "$CAN_SERVICE_PATTERN" >/dev/null 2>&1; then
    sudo -n pkill -f "$CAN_SERVICE_PATTERN" 2>/dev/null || log_warn "can0 服务以 root 运行但 sudo 停止失败，请手动处理"
fi

# ---- 清理共享内存和 socket -----------------------------------------------
log_step "清理残留"
for shm in vr_frame vr_result; do
    [ -e "/dev/shm/${shm}" ] && rm -f "/dev/shm/${shm}" 2>/dev/null && log_info "已清理 /dev/shm/${shm}"
done
[ -S /tmp/vr_cmd.sock ] && rm -f /tmp/vr_cmd.sock && log_info "已清理 /tmp/vr_cmd.sock"
if pgrep -f "$CAN_SERVICE_PATTERN" >/dev/null 2>&1; then
    log_warn "can0 服务仍在运行，保留 CAN socket，避免破坏活动监听"
elif [ -S "${SCRIPT_DIR}/can_service/sock/can0_bus.sock" ]; then
    rm -f "${SCRIPT_DIR}/can_service/sock/can0_bus.sock"
    log_info "已清理残留 CAN socket"
fi

echo ""
log_info "全部服务已停止 ✓"
echo ""
