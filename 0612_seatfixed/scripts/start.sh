#!/bin/bash
# scripts/start.sh
# 启动人脸识别服务（GPU 推理进程 + API 进程）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
PID_DIR="${PROJECT_DIR}"
ENV_FILE="${PROJECT_DIR}/.env"
WORKER_PID_FILE="${PID_DIR}/worker.pid"
API_PID_FILE="${PID_DIR}/server.pid"

mkdir -p "$LOG_DIR"

# 加载环境变量
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# 兜底默认值
PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda}"

echo "========================================"
echo "  Vehicle Face Recognition Service"
echo "========================================"
echo "  Host     : $HOST"
echo "  Port     : $PORT"
echo "  Device   : $DEVICE"
echo "  Log      : $LOG_DIR/"
echo "========================================"

cd "$PROJECT_DIR"

# 优先 conda 环境的 python3，不管用户是否已 activate
_PYTHON=""
for _try in \
    "/home/data/miniconda3/envs/AIAC/bin/python3" \
    "/home/data/miniconda3/envs/AIAC/bin/python" \
    "$HOME/miniconda3/envs/AIAC/bin/python3" \
    "$HOME/miniconda3/envs/AIAC/bin/python" \
; do
    if [ -f "$_try" ]; then
        _PYTHON="$_try"
        break
    fi
done
# 兜底：当前 shell 的 python3
if [ -z "$_PYTHON" ]; then
    _PYTHON="$(command -v python3 2>/dev/null || echo "")"
fi
if [ -z "$_PYTHON" ]; then
    _PYTHON="$(command -v python 2>/dev/null || echo "")"
fi
if [ -z "$_PYTHON" ]; then
    _PYTHON="python3"
fi

# 检查 Python 依赖
$_PYTHON -c "import fastapi, uvicorn, insightface" 2>/dev/null || {
    echo "[ERROR] 缺少依赖，请先安装: pip install -r requirements.txt"
    exit 1
}

# ── 停止旧进程 ────────────────────────────────────────────────────────
for pf in "$WORKER_PID_FILE" "$API_PID_FILE"; do
    if [ -f "$pf" ]; then
        OLD_PID=$(cat "$pf")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "[INFO] 停止旧进程 PID=$OLD_PID ($pf)..."
            kill -9 "$OLD_PID" 2>/dev/null || true
            for _i in $(seq 1 10); do
                kill -0 "$OLD_PID" 2>/dev/null || break
                sleep 0.5
            done
        fi
        rm -f "$pf"
    fi
done

# ── 清理残留共享内存（防止上次异常退出后残留）─────────────────────────
for shm_name in vr_frame vr_result; do
    if [ -e "/dev/shm/${shm_name}" ]; then
        rm -f "/dev/shm/${shm_name}" 2>/dev/null || true
    fi
done
[ -S /tmp/vr_cmd.sock ] && rm -f /tmp/vr_cmd.sock || true

# ── 1. 启动 GPU 推理进程 ──────────────────────────────────────────────
echo "[INFO] 启动 GPU 推理进程..."
nohup $_PYTHON -m service.inference_worker \
    > "$LOG_DIR/worker.log" 2>&1 &

echo $! > "$WORKER_PID_FILE"
WORKER_PID=$(cat "$WORKER_PID_FILE")
echo "  GPU 推理进程 PID=$WORKER_PID"

# ── 等待 worker 完全就绪（日志中出现就绪标记）─────────────────────
echo -n "  等待 worker 就绪..."
WORKER_LOG="$LOG_DIR/inference_worker.log"
# 清理旧日志，确保不会匹配到上一次的启动记录
rm -f "$WORKER_LOG"
_WAIT_COUNT=0
while true; do
    _WAIT_COUNT=$((_WAIT_COUNT + 1))
    if [ -e "$WORKER_LOG" ] && grep -q "GPU 推理进程就绪" "$WORKER_LOG" 2>/dev/null; then
        echo " OK ($((_WAIT_COUNT))s)"
        break
    fi
    sleep 1
    echo -n "."
    if [ "$_WAIT_COUNT" -ge 60 ]; then
        echo " TIMEOUT"
        echo "[ERROR] GPU 推理进程未在 60 秒内就绪，请查看日志: $LOG_DIR/worker.log"
        tail -20 "$LOG_DIR/worker.log"
        exit 1
    fi
done

# ── 2. 启动 API 进程 ──────────────────────────────────────────────────
echo "[INFO] 启动 API 进程..."
nohup $_PYTHON -m uvicorn api.server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 60 \
    > "$LOG_DIR/server.log" 2>&1 &

echo $! > "$API_PID_FILE"
API_PID=$(cat "$API_PID_FILE")

sleep 3

if kill -0 "$API_PID" 2>/dev/null && kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "[OK] 服务已启动"
    echo "  GPU 推理进程  PID=$WORKER_PID  | 日志: $LOG_DIR/worker.log"
    echo "  API 进程      PID=$API_PID    | 日志: $LOG_DIR/server.log"
    echo "  API 文档: http://localhost:$PORT/docs"
    echo "  健康检查: http://localhost:$PORT/health"
    echo "  预览:     http://localhost:$PORT/preview"
else
    echo "[ERROR] 服务启动失败:"
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "  GPU 推理进程退出，日志:"
        tail -20 "$LOG_DIR/worker.log"
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "  API 进程退出，日志:"
        tail -20 "$LOG_DIR/server.log"
    fi
    exit 1
fi
