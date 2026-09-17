#!/usr/bin/env bash
# ==============================================================================
# PMV Consumer 停止脚本
# ==============================================================================
#
# 仅停止 PMV consumer。can0_service 和 BLF 回放不受影响。
# 日志保留在 /tmp/pmv_service_logs/。
#
# 用法: bash stop_pmv_service.sh
# ==============================================================================
set -euo pipefail

PID_DIR="/tmp/pmv_service_pids"
PID_FILE="$PID_DIR/pmv_consumer.pid"

echo "=== Stopping PMV Consumer ==="

if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Stopping PID=$pid..."
        kill "$pid" 2>/dev/null || true
        for i in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        kill -9 "$pid" 2>/dev/null || true
        echo "  Stopped"
    else
        echo "  PID=$pid already dead"
    fi
    rm -f "$PID_FILE"
fi

pkill -f "pmv_socket_consumer" 2>/dev/null && echo "  Cleaned leftover" || true

echo "=== Done ==="
