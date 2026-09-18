#!/usr/bin/env bash
# ==============================================================================
# AIAC 一键启动脚本
# ==============================================================================
# 自动依次启动全部 7 个服务，等待每个服务就绪后再启动下一个，
# 并在最后汇总健康状态。
#
# 用法:
#   bash start_all.sh              # 正式模式（推理+记录+下发控制空调）
#   bash start_all.sh --status     # 仅检查各服务状态
#
# 停止: bash stop_all.sh
# ==============================================================================
set -euo pipefail

# ---- 配置 ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="AIAC"
LOG_BASE="${SCRIPT_DIR}/.startup_logs"

# 各服务端口 / 就绪标记
CAN_SOCKET="${SCRIPT_DIR}/can_service/sock/can0_bus.sock"
CAN_SERVICE_NAME="can0_service_v1.3.8.py"
CAN_SERVICE_PATH="${SCRIPT_DIR}/can_service/${CAN_SERVICE_NAME}"
VISION_API_PORT="${VISION_API_PORT:-7860}"
VISION_API_URL="http://localhost:${VISION_API_PORT}/health"
VISION_STARTUP_TIMEOUT=120

PMV_API_PORT="${PMV_API_PORT:-7861}"
PMV_API_URL="http://localhost:${PMV_API_PORT}/pmv"
PMV_STARTUP_TIMEOUT=60

INTERFACE_INTEGRATION_PIDFILE="/tmp/interface_integration.pid"

THERMAL_STREAM_PORT="${THERMAL_STREAM_PORT:-7864}"
THERMAL_STREAM_URL="http://localhost:${THERMAL_STREAM_PORT}/"
THERMAL_STREAM_STARTUP_TIMEOUT=30

DASHBOARD_PORT="${DASHBOARD_PORT:-7862}"
DASHBOARD_PIDFILE="/tmp/dashboard.pid"

RECOMMENDATION_LOG="${SCRIPT_DIR}/sls_recommendation_weather26/logs/recommendation.log"
RECOMMENDATION_STARTUP_TIMEOUT=120

# ---- 工具函数 ------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S')  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S')  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S')  $*"; }
log_step()  { echo -e "\n${CYAN}════════════════════════════════════════${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}════════════════════════════════════════${NC}"; }

# 查找 conda python
find_python() {
    local python_bin
    for cand in \
        "/home/data/miniconda3/envs/${CONDA_ENV}/bin/python3" \
        "/home/data/miniconda3/envs/${CONDA_ENV}/bin/python" \
        "$HOME/miniconda3/envs/${CONDA_ENV}/bin/python3" \
        "$HOME/miniconda3/envs/${CONDA_ENV}/bin/python" \
    ; do
        if [ -f "$cand" ]; then
            echo "$cand"
            return 0
        fi
    done
    echo "python3"
}

PYPATH="$(find_python)"

wait_for_http() {
    local url="$1" timeout="$2" label="$3"
    log_info "等待 ${label} 就绪 (${url}) ..."
    local elapsed=0
    while [ $elapsed -lt "$timeout" ]; do
        if curl -fsSL --max-time 3 "$url" >/dev/null 2>&1; then
            log_info "${label} HTTP 就绪 ✓  (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    echo ""
    log_error "${label} HTTP 健康检查超时 (${timeout}s)"
    return 1
}

wait_for_socket() {
    local socket_path="$1" timeout="$2" label="$3"
    log_info "等待 ${label} 就绪 (${socket_path}) ..."
    local elapsed=0
    while [ $elapsed -lt "$timeout" ]; do
        if [ -S "$socket_path" ]; then
            log_info "${label} 就绪 ✓  (${elapsed}s)"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        echo -n "."
    done
    echo ""
    log_error "${label} Socket 未在 ${timeout}s 内出现"
    return 1
}

wait_for_log() {
    local logfile="$1" pattern="$2" timeout="$3" label="$4"
    log_info "等待 ${label} 就绪 (日志: ${logfile}) ..."
    local elapsed=0
    while [ $elapsed -lt "$timeout" ]; do
        if [ -f "$logfile" ] && grep -q "$pattern" "$logfile" 2>/dev/null; then
            log_info "${label} 就绪 ✓  (${elapsed}s)"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        echo -n "."
    done
    echo ""
    log_error "${label} 日志标记超时 (${timeout}s) — 未找到: ${pattern}"
    if [ -f "$logfile" ]; then
        log_error "最后 20 行日志:"
        tail -20 "$logfile" | sed 's/^/  /'
    fi
    return 1
}

# ---- 步骤 ----------------------------------------------------------------

# 等待 can2-can5 全部 link UP（总线/ECU 上电可能需要时间，带重试）
wait_for_can_up() {
    local timeout=60 interval=5 elapsed=0 last_missing=""
    while [ $elapsed -lt "$timeout" ]; do
        local missing=()
        for iface in can2 can3 can4 can5; do
            if ip link show "$iface" >/dev/null 2>&1; then
                local state
                state=$(ip link show "$iface" 2>/dev/null | grep -oP '(?<=state )\w+' || true)
                [ "$state" = "UP" ] || missing+=("${iface}:${state:-unknown}")
            else
                missing+=("${iface}:missing")
            fi
        done
        if [ ${#missing[@]} -eq 0 ]; then
            log_info "CAN 接口 can2-can5 全部在线 ✓"
            return 0
        fi
        local cur
        cur=$(IFS=,; echo "${missing[*]}")
        if [ "$cur" != "$last_missing" ]; then
            log_warn "等待 CAN 接口就绪 (${elapsed}s/${timeout}s): ${cur}"
            last_missing="$cur"
        else
            echo -n "."
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    echo ""
    log_error "CAN 接口未在 ${timeout}s 内全部 UP: ${last_missing}"
    log_error "可能原因: 总线无其他节点(ECU 未上电) / PCAN 设备未就绪 / 接线或终端电阻问题"
    log_error "CAN 配置日志最后 20 行:"
    tail -20 "${LOG_BASE}/can_setup.log" | sed 's/^/  /'
    return 1
}

# 真实探测 CAN socket 是否可连接（防止残留 socket 文件造成假阳性）
can_socket_ok() {
    [ -S "$CAN_SOCKET" ] || return 1
    "$PYPATH" - "$CAN_SOCKET" >/dev/null 2>&1 <<'PYEOF'
import socket, sys
try:
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(2)
    s.connect(sys.argv[1])
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PYEOF
}

start_can() {
    log_step "1/7  启动 CAN 服务"
    mkdir -p "$LOG_BASE"

    # 1a. CAN 接口配置
    cd "${SCRIPT_DIR}/can_service"
    log_info "运行 can_setup.sh ..."
    if ! bash can_setup.sh >> "${LOG_BASE}/can_setup.log" 2>&1; then
        log_error "CAN 接口配置失败，查看: ${LOG_BASE}/can_setup.log"
        tail -20 "${LOG_BASE}/can_setup.log"
        return 1
    fi
    log_info "CAN 接口配置完成 ✓"

    # 1b. 启动 can0 服务 (后台)。进程存在不代表 Unix Socket 仍可连接；
    # stop_all 曾可能留下一个监听着已删除 inode 的旧进程，需要先清理再拉起。
    local start_can_service=1
    if pgrep -f "$CAN_SERVICE_NAME" >/dev/null 2>&1; then
        if can_socket_ok; then
            log_warn "can0 服务已在运行且 Socket 正常，跳过启动"
            start_can_service=0
        else
            log_warn "can0 进程存在但 Socket 不可连接，正在清理失效进程后重启"
            pkill -TERM -f "$CAN_SERVICE_NAME" 2>/dev/null || true
            sudo -n pkill -TERM -f "$CAN_SERVICE_NAME" 2>/dev/null || true
            local elapsed=0
            while pgrep -f "$CAN_SERVICE_NAME" >/dev/null 2>&1 && [ "$elapsed" -lt 5 ]; do
                sleep 1
                elapsed=$((elapsed + 1))
            done
            if pgrep -f "$CAN_SERVICE_NAME" >/dev/null 2>&1; then
                log_warn "失效的 can0 进程未及时退出，正在强制停止"
                pkill -KILL -f "$CAN_SERVICE_NAME" 2>/dev/null || true
                sudo -n pkill -KILL -f "$CAN_SERVICE_NAME" 2>/dev/null || true
                sleep 1
                if pgrep -f "$CAN_SERVICE_NAME" >/dev/null 2>&1; then
                    log_error "失效的 can0 进程无法停止，请检查进程权限"
                    return 1
                fi
            fi
            rm -f "$CAN_SOCKET"
        fi
    fi

    if [ "$start_can_service" = 1 ]; then
        # 存在 sudoers NOPASSWD 白名单时自动以 root 启动（socket chown root:test 需要）
        if sudo -n -l 2>/dev/null | grep -qF "$CAN_SERVICE_NAME"; then
            log_info "启动 can0_service (root, sudoers 白名单) ..."
            nohup sudo -n -- "$(readlink -f "$PYPATH")" -u "$CAN_SERVICE_PATH" \
                >> "${LOG_BASE}/can0_service.log" 2>&1 </dev/null &
        else
            log_info "启动 can0_service (当前用户, 无 sudoers 白名单) ..."
            nohup "$PYPATH" -u "$CAN_SERVICE_PATH" \
                >> "${LOG_BASE}/can0_service.log" 2>&1 </dev/null &
        fi
        echo $! > /tmp/can0_service.pid
        STARTED_CAN=1
        log_info "can0 PID=$(cat /tmp/can0_service.pid)"
    fi

    # 等待 socket 出现
    wait_for_socket "$CAN_SOCKET" 30 "CAN Socket" || return 1

    if ! can_socket_ok; then
        log_error "CAN Socket 文件已出现，但无法建立连接"
        return 1
    fi

    # socket 文件可能残留自上次崩溃，需确认进程真实存活
    if ! pgrep -f "$CAN_SERVICE_NAME" >/dev/null 2>&1; then
        log_error "can0 服务进程已退出，查看: ${LOG_BASE}/can0_service.log"
        tail -30 "${LOG_BASE}/can0_service.log" | sed 's/^/  /'
        return 1
    fi

    # 验证 can2-can5 全部在线（带重试，等待 ECU/总线就绪）
    wait_for_can_up
}

start_vision() {
    log_step "2/7  启动视觉感知服务"

    cd "${SCRIPT_DIR}/0612_seatfixed"

    # 使用自带 start.sh (内部已处理 PID、日志、就绪等待)
    if bash scripts/start.sh >> "${LOG_BASE}/vision_startup.log" 2>&1; then
        STARTED_VISION=1
        log_info "视觉感知服务启动成功 ✓"
    else
        log_error "视觉感知服务启动失败，查看: ${LOG_BASE}/vision_startup.log"
        return 1
    fi

    # 额外验证 HTTP 健康（GPU 初始化较慢，给足时间）
    wait_for_http "$VISION_API_URL" "$VISION_STARTUP_TIMEOUT" "Vision API"
}

start_pmv() {
    log_step "3/7  启动 PMV 服务"

    cd "${SCRIPT_DIR}/vehicle_runtime_package_vnext_budian"

    # 清除残留 PID 文件（start_pmv_service.sh 内部有判断避免重复启动）
    local pmv_pidfile="/tmp/pmv_service_pids/pmv_consumer.pid"
    if [ -f "$pmv_pidfile" ]; then
        local old_pid
        old_pid="$(cat "$pmv_pidfile" 2>/dev/null || true)"
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            log_warn "PMV 服务已在运行 (PID=$old_pid)，跳过启动"
        else
            rm -f "$pmv_pidfile"
        fi
    fi

    if bash start_pmv_service.sh >> "${LOG_BASE}/pmv_startup.log" 2>&1; then
        STARTED_PMV=1
        log_info "PMV 服务启动成功 ✓"
    else
        log_error "PMV 服务启动失败，查看: ${LOG_BASE}/pmv_startup.log"
        return 1
    fi

    # PMV HTTP 就绪验证（start_pmv_service.sh 已有 1s 存活检查，这里再等 HTTP）
    wait_for_http "$PMV_API_URL" "$PMV_STARTUP_TIMEOUT" "PMV API"
}

start_interface_integration() {
    log_step "4/7  启动接口同步服务"

    if pgrep -f "[p]erson_info_sync.py" >/dev/null 2>&1; then
        log_warn "接口同步服务已在运行，跳过启动"
        return 0
    fi

    cd "${SCRIPT_DIR}/interface_intergration"
    log_info "启动 person_info_sync ..."
    nohup "$PYPATH" -u person_info_sync.py --config config.yaml \
        >> "${LOG_BASE}/interface_integration.log" 2>&1 </dev/null &
    local interface_pid=$!
    echo "$interface_pid" > "$INTERFACE_INTEGRATION_PIDFILE"
    STARTED_INTERFACE=1
    sleep 2

    if kill -0 "$interface_pid" 2>/dev/null; then
        log_info "接口同步服务启动成功 ✓  PID=$interface_pid"
    else
        log_error "接口同步服务启动失败，查看: ${LOG_BASE}/interface_integration.log"
        return 1
    fi
}

start_recommendation() {
    log_step "5/7  启动推荐服务"

    cd "${SCRIPT_DIR}/sls_recommendation_weather26"

    if pgrep -f "main_controller" >/dev/null 2>&1; then
        log_warn "推荐服务已在运行，跳过启动"
        return 0
    fi

    # 截断就绪标记日志，避免 wait_for_log 匹配到历史残留内容
    : > "${LOG_BASE}/recommendation.log"

    # 应用自身日志过大时轮转，防止磁盘占满
    local app_log="${SCRIPT_DIR}/sls_recommendation_weather26/logs/recommendation.log"
    if [ -f "$app_log" ] && [ "$(stat -c%s "$app_log")" -gt $((200*1024*1024)) ]; then
        mv "$app_log" "${app_log}.1"
        log_info "应用日志已轮转: ${app_log} -> ${app_log}.1"
    fi

    #local rec_args=(--model-package-mode steady26)
    local rec_args=(--model-package-mode weather26 --run-mode mode4)
    log_info "启动 main_controller (weather26 + 偏好学习 mode4: 接管→收集→训练→联合推理) ..."
    nohup "$PYPATH" -u -m src.main_controller \
        "${rec_args[@]}" \
        >> "${LOG_BASE}/recommendation.log" 2>&1 </dev/null &
    local rec_pid=$!
    echo "$rec_pid" > /tmp/recommendation.pid
    STARTED_REC=1
    log_info "推荐服务 PID=$rec_pid"

    # 等待日志中出现启动完成标记
    wait_for_log "${LOG_BASE}/recommendation.log" \
        "系统启动完成，开始运行" \
        "$RECOMMENDATION_STARTUP_TIMEOUT" \
        "Recommendation"

    # 给 2s 缓冲
    sleep 2
}

start_dashboard() {
    log_step "6/7  启动实时仪表盘"

    cd "${SCRIPT_DIR}"

    if [ -f "$DASHBOARD_PIDFILE" ]; then
        local old_pid
        old_pid="$(cat "$DASHBOARD_PIDFILE" 2>/dev/null || true)"
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            log_warn "仪表盘已在运行 (PID=$old_pid, port=${DASHBOARD_PORT})，跳过启动"
            return 0
        fi
        rm -f "$DASHBOARD_PIDFILE"
    fi

    log_info "启动 live_dashboard (port ${DASHBOARD_PORT}) ..."
    nohup "$PYPATH" -u live_dashboard.py --port "${DASHBOARD_PORT}" \
        >> "${LOG_BASE}/dashboard.log" 2>&1 </dev/null &
    local dash_pid=$!
    echo "$dash_pid" > "$DASHBOARD_PIDFILE"
    STARTED_DASH=1
    sleep 2

    if kill -0 "$dash_pid" 2>/dev/null; then
        log_info "仪表盘启动成功 ✓  http://<IP>:${DASHBOARD_PORT}"
    else
        log_error "仪表盘启动失败，查看: ${LOG_BASE}/dashboard.log"
        return 1
    fi
}

start_thermal_stream() {
    log_step "7/7  启动热成像视频流"

    if pgrep -f "[t]hermal_stream/main.py" >/dev/null 2>&1; then
        log_warn "热成像视频流已在运行，跳过启动"
    else
        if bash "${SCRIPT_DIR}/thermal_stream/start.sh" \
            >> "${LOG_BASE}/thermal_stream_startup.log" 2>&1; then
            STARTED_THERMAL=1
            log_info "热成像视频流进程已启动"
        else
            log_error "热成像视频流启动失败，查看: ${LOG_BASE}/thermal_stream_startup.log"
            return 1
        fi
    fi

    wait_for_http "$THERMAL_STREAM_URL" "$THERMAL_STREAM_STARTUP_TIMEOUT" "Thermal Stream"
}

# ---- 状态检查 ------------------------------------------------------------

check_status() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  AIAC 服务状态"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    check_one "CAN 接口 (can2-can5)" \
        "ip link show can2 >/dev/null 2>&1 && ip link show can3 >/dev/null 2>&1 && ip link show can4 >/dev/null 2>&1 && ip link show can5 >/dev/null 2>&1"

    check_one "CAN Socket" \
        "can_socket_ok"

    check_one "can0 Service" \
        "pgrep -f 'can0_service_v1.3.8.py' >/dev/null 2>&1"

    check_one "Vision API (${VISION_API_PORT})" \
        "curl -fsSL --max-time 3 ${VISION_API_URL} >/dev/null 2>&1"

    check_one "PMV API (${PMV_API_PORT})" \
        "curl -fsSL --max-time 3 ${PMV_API_URL} >/dev/null 2>&1"

    check_one "Interface Integration" \
        "pgrep -f '[p]erson_info_sync.py' >/dev/null 2>&1"

    check_one "Recommendation" \
        "pgrep -f 'main_controller' >/dev/null 2>&1"

    check_one "Dashboard (${DASHBOARD_PORT})" \
        "curl -fsSL --max-time 3 http://localhost:${DASHBOARD_PORT}/health >/dev/null 2>&1"

    check_one "Thermal Stream (${THERMAL_STREAM_PORT})" \
        "curl -fsSL --max-time 3 ${THERMAL_STREAM_URL} >/dev/null 2>&1"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

check_one() {
    local label="$1" cmd="$2"
    if eval "$cmd"; then
        echo -e "  ${GREEN}✓${NC} $label"
    else
        echo -e "  ${RED}✗${NC} $label"
    fi
}

# ---- 失败回滚 ------------------------------------------------------------

STARTED_CAN=0; STARTED_VISION=0; STARTED_PMV=0; STARTED_INTERFACE=0
STARTED_REC=0; STARTED_DASH=0; STARTED_THERMAL=0

rollback_started() {
    log_warn "回滚：停止本次启动的服务 ..."
    if [ "$STARTED_THERMAL" = 1 ]; then
        bash "${SCRIPT_DIR}/thermal_stream/stop.sh" >> "${LOG_BASE}/thermal_stream_startup.log" 2>&1 || true
    fi
    if [ "$STARTED_DASH" = 1 ]; then
        kill "$(cat "$DASHBOARD_PIDFILE" 2>/dev/null)" 2>/dev/null || true
        rm -f "$DASHBOARD_PIDFILE"
    fi
    if [ "$STARTED_REC" = 1 ]; then
        kill "$(cat /tmp/recommendation.pid 2>/dev/null)" 2>/dev/null || true
        pkill -f "src.main_controller" 2>/dev/null || true
        rm -f /tmp/recommendation.pid
    fi
    if [ "$STARTED_INTERFACE" = 1 ]; then
        kill "$(cat "$INTERFACE_INTEGRATION_PIDFILE" 2>/dev/null)" 2>/dev/null || true
        rm -f "$INTERFACE_INTEGRATION_PIDFILE"
    fi
    if [ "$STARTED_PMV" = 1 ]; then
        bash "${SCRIPT_DIR}/vehicle_runtime_package_vnext_budian/stop_pmv_service.sh" >> "${LOG_BASE}/pmv_startup.log" 2>&1 || true
    fi
    if [ "$STARTED_VISION" = 1 ]; then
        bash "${SCRIPT_DIR}/0612_seatfixed/scripts/stop.sh" >> "${LOG_BASE}/vision_startup.log" 2>&1 || true
    fi
    if [ "$STARTED_CAN" = 1 ]; then
        kill "$(cat /tmp/can0_service.pid 2>/dev/null)" 2>/dev/null || true
        pkill -f "can0_service_v1.3.8.py" 2>/dev/null || true
        # root 启动的 can0 服务需要 sudo 才能停止（sudoers 白名单）
        sudo -n pkill -f can0_service_v1.3.8.py 2>/dev/null || true
        rm -f /tmp/can0_service.pid
    fi
}

# ---- 主入口 --------------------------------------------------------------

mkdir -p "$LOG_BASE"

# 大日志轮转，防止磁盘被占满
if [ -f "${LOG_BASE}/can0_service.log" ] && [ "$(stat -c%s "${LOG_BASE}/can0_service.log")" -gt $((200*1024*1024)) ]; then
    mv "${LOG_BASE}/can0_service.log" "${LOG_BASE}/can0_service.log.1"
    log_info "日志已轮转: can0_service.log -> can0_service.log.1"
fi

case "${1:-}" in
    --status|-s)
        check_status
        exit 0
        ;;
    --stop)
        bash "${SCRIPT_DIR}/stop_all.sh"
        exit $?
        ;;
    --help|-h)
        echo "用法: bash start_all.sh [--status|--help]"
        echo "  (无参数)    正式模式：推理+记录+下发控制空调"
        echo "  --status    仅检查各服务状态"
        exit 0
        ;;
esac

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     AIAC 系统一键启动               ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  Python  : $PYPATH"
echo "  模型    : weather26 (26维随机森林) + 偏好学习mode4 (接管→收集→训练→联合推理)"
echo "  模式    : 正式模式 (推理+记录+下发控制)"
echo "  启动日志: $LOG_BASE/"
echo ""

FAILED=0

start_can || { FAILED=1; log_error "CAN 服务启动失败，终止。"; }

if [ $FAILED -eq 0 ]; then
    start_vision || { FAILED=1; log_error "视觉服务启动失败，终止。"; }
else
    log_warn "跳过视觉服务（前置步骤失败）"
fi

if [ $FAILED -eq 0 ]; then
    start_pmv || { FAILED=1; log_error "PMV 服务启动失败，终止。"; }
else
    log_warn "跳过 PMV 服务（前置步骤失败）"
fi

if [ $FAILED -eq 0 ]; then
    start_interface_integration || { FAILED=1; log_error "接口同步服务启动失败，终止。"; }
else
    log_warn "跳过接口同步服务（前置步骤失败）"
fi

if [ $FAILED -eq 0 ]; then
    start_recommendation || { FAILED=1; log_error "推荐服务启动失败，终止。"; }
else
    log_warn "跳过推荐服务（前置步骤失败）"
fi

# 仪表盘是可选的（失败不阻止主流程）
if [ $FAILED -eq 0 ]; then
    start_dashboard || log_warn "仪表盘启动失败（主服务不受影响）"
else
    log_warn "跳过仪表盘（前置步骤失败）"
fi

# 热成像流是可选的（失败不阻止主流程）
if [ $FAILED -eq 0 ]; then
    start_thermal_stream || log_warn "热成像视频流启动失败（主服务不受影响）"
else
    log_warn "跳过热成像视频流（前置步骤失败）"
fi

# 启动失败时回滚本次启动的服务，避免留下半启动状态
if [ $FAILED -ne 0 ]; then
    rollback_started
fi

echo ""
if [ $FAILED -eq 0 ]; then
    log_info "全部服务启动完成！"
    check_status
    echo ""
    echo "  停止所有服务: bash stop_all.sh"
    echo "  查看状态:     bash start_all.sh --status"
    echo "  启动日志:     ls ${LOG_BASE}/"
    echo ""
else
    log_error "部分服务启动失败，请查看日志: ${LOG_BASE}/"
    check_status
    exit 1
fi
