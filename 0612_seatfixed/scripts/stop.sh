#!/bin/bash
# scripts/stop.sh
# 停止人脸识别服务（API + GPU 推理进程）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="${PROJECT_DIR}"

echo "[INFO] 正在停止服务..."

# 停止 API 进程
for pf in "server.pid" "worker.pid"; do
    PID_FILE="${PID_DIR}/${pf}"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[INFO] 停止进程 PID=$PID ($pf)..."
            kill "$PID"
            sleep 2
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID" 2>/dev/null || true
            fi
            echo "[OK] 进程 $pf 已停止"
        else
            echo "[WARN] PID=$PID 进程不存在，可能已停止"
        fi
        rm -f "$PID_FILE"
    fi
done

# 清理共享内存和 socket
for shm in vr_frame vr_result; do
    [ -e "/dev/shm/${shm}" ] && rm -f "/dev/shm/${shm}" || true
done
[ -S /tmp/vr_cmd.sock ] && rm -f /tmp/vr_cmd.sock || true

echo "[OK] 停止完成"
