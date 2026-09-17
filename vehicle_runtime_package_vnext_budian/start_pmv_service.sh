#!/usr/bin/env bash
# ==============================================================================
# PMV Consumer 启动脚本
# ==============================================================================
#
# 前置: can0_service + BLF 回放已由其他脚本启动
# 本脚本仅启动 PMV consumer: Socket → PMV → HTTP API
#
# 用法:
#   bash start_pmv_service.sh
#
# 环境变量:
#   API_PORT        默认 7861
#   PMV_INTERVAL    默认 1.0 (秒)
#   SOCKET_PATH     默认 ../can_service/sock/can0_bus.sock
#   PMV_PARAM_LOG   默认 /tmp/pmv_service_logs/pmv_param_trace.log
#
# 日志: /tmp/pmv_service_logs/pmv_consumer.log
# 参数: /tmp/pmv_service_logs/pmv_param_trace.log
# PID:  /tmp/pmv_service_pids/pmv_consumer.pid
# ==============================================================================
set -euo pipefail

API_PORT="${API_PORT:-7861}"
PMV_INTERVAL="${PMV_INTERVAL:-1.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOCKET_PATH="${SOCKET_PATH:-${SCRIPT_DIR}/../can_service/sock/can0_bus.sock}"
LOG_DIR="/tmp/pmv_service_logs"
PID_DIR="/tmp/pmv_service_pids"
PID_FILE="$PID_DIR/pmv_consumer.pid"
LOG_FILE="$LOG_DIR/pmv_consumer.log"
PARAM_LOG_FILE="${PMV_PARAM_LOG:-$LOG_DIR/pmv_param_trace.log}"

mkdir -p "$LOG_DIR" "$PID_DIR"

source /home/data/miniconda3/etc/profile.d/conda.sh
conda activate AIAC

if [ -f "$PID_FILE" ]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        echo "PMV consumer already running, PID=$old_pid"
        echo "Monitor: tail -f $LOG_FILE"
        exit 0
    fi
    echo "Removing stale PID file: $PID_FILE"
    rm -f "$PID_FILE"
fi

if command -v ss >/dev/null 2>&1 && ss -ltn | grep -q ":${API_PORT} "; then
    echo "ERROR: API port ${API_PORT} is already in use. Refusing to overwrite PID file."
    exit 1
fi

echo "Starting PMV consumer..."
echo "  Socket:   $SOCKET_PATH"
echo "  API port: $API_PORT"
echo "  Interval: ${PMV_INTERVAL}s"
echo "  Log:      $LOG_FILE"
echo "  ParamLog: $PARAM_LOG_FILE"

: > "$LOG_FILE"
: > "$PARAM_LOG_FILE"
nohup python -u "$SCRIPT_DIR/pmv_socket_consumer.py" \
    --socket "$SOCKET_PATH" \
    --interval "$PMV_INTERVAL" \
    --api-port "$API_PORT" \
    --param-log "$PARAM_LOG_FILE" \
    >> "$LOG_FILE" 2>&1 </dev/null &

PMV_PID=$!
sleep 1
if ! kill -0 "$PMV_PID" 2>/dev/null; then
    echo "ERROR: PMV consumer exited during startup. See: $LOG_FILE"
    tail -n 80 "$LOG_FILE" || true
    exit 1
fi

echo "$PMV_PID" > "$PID_FILE"
echo "PMV consumer started, PID=$PMV_PID"
echo "Monitor: tail -f $LOG_FILE"
echo "Params:  tail -f $PARAM_LOG_FILE"
echo "Stop:    bash ${SCRIPT_DIR}/stop_pmv_service.sh"
