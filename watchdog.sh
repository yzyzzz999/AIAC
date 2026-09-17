#!/usr/bin/env bash
# ==============================================================================
# AIAC 看门狗脚本
# ==============================================================================
# 定期检查所有服务健康状态，任意服务异常则自动重启。
# 通过 crontab 定时触发，内置锁防止并发执行。
#
# 用法:
#   bash watchdog.sh            # 检查并自动修复
#   bash watchdog.sh --report   # 仅检查，不修复，输出状态
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/.startup_logs/watchdog.log"
LOCK_FILE="/tmp/aiac_watchdog.lock"
CAN_SOCKET="${SCRIPT_DIR}/can_service/sock/can0_bus.sock"
VISION_API_URL="http://localhost:7860/health"
PMV_API_URL="http://localhost:7861/pmv"
MAX_LOG_LINES=5000

REPORT_ONLY=false
if [ "${1:-}" = "--report" ]; then
    REPORT_ONLY=true
fi

# ---- 日志 ----------------------------------------------------------------
log_msg() {
    local level="$1" msg="$2"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${ts}] [${level}] ${msg}" | tee -a "$LOG_FILE"
}

# 日志轮转
if [ -f "$LOG_FILE" ]; then
    _lines=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "$_lines" -gt "$MAX_LOG_LINES" ]; then
        tail -n "$MAX_LOG_LINES" "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
    fi
fi

# ---- 锁 -----------------------------------------------------------------
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    # 已有 watchdog 在运行，静默退出
    exit 0
fi

# ---- 健康检查 ------------------------------------------------------------
FAILED_SERVICES=()

check_one() {
    local label="$1" cmd="$2"
    if eval "$cmd"; then
        return 0
    else
        FAILED_SERVICES+=("$label")
        return 1
    fi
}

check_all() {
    FAILED_SERVICES=()
    check_one "CAN接口(can2-can5)" \
        "ip link show can2 >/dev/null 2>&1 && ip link show can3 >/dev/null 2>&1"
    check_one "CAN_Socket" \
        "[ -S ${CAN_SOCKET} ]"
    check_one "can0_Service" \
        "pgrep -f 'can0_service_v1.3.6.py' >/dev/null 2>&1"
    check_one "Vision_API(7860)" \
        "curl -fsSL --max-time 5 ${VISION_API_URL} >/dev/null 2>&1"
    check_one "PMV_API(7861)" \
        "curl -fsSL --max-time 5 ${PMV_API_URL} >/dev/null 2>&1"
    check_one "Recommendation" \
        "pgrep -f 'main_controller' >/dev/null 2>&1"
}

# ---- 主逻辑 --------------------------------------------------------------
check_all

if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
    # 全部健康，只记录到日志（不输出到终端，减少 cron 邮件噪音）
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [OK] 全部 6 项检查通过" >> "$LOG_FILE"
    exit 0
fi

log_msg "WARN" "检测到 ${#FAILED_SERVICES[@]} 项异常: ${FAILED_SERVICES[*]}"

if $REPORT_ONLY; then
    for s in "${FAILED_SERVICES[@]}"; do
        echo "  FAIL: $s"
    done
    exit 1
fi

# 自动修复：调用 start_all.sh
log_msg "INFO" "触发自动重启 (start_all.sh) ..."
bash "${SCRIPT_DIR}/start_all.sh" >> "$LOG_FILE" 2>&1
_rc=$?

if [ $_rc -eq 0 ]; then
    log_msg "INFO" "自动重启完成，重新检查状态..."
    sleep 5
    check_all
    if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
        log_msg "OK" "自动恢复成功，所有服务已正常运行"
    else
        log_msg "ERROR" "自动重启后仍有异常: ${FAILED_SERVICES[*]}"
    fi
else
    log_msg "ERROR" "start_all.sh 返回码=$_rc，自动重启失败"
fi

exit $_rc
