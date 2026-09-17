#!/bin/bash
# deploy/entrypoint.sh
# Docker 容器入口脚本

set -e

echo "============================================"
echo "  Vehicle Face Recognition Service"
echo "============================================"
echo "  API URL: http://\$HOST:\$PORT"
echo "  Gallery: \$GALLERY_FILE"
echo "============================================"

# 等待依赖服务
sleep 2

# 启动 GPU 推理进程
echo "[INFO] 启动 GPU 推理进程..."
python3 -m service.inference_worker &
WORKER_PID=$!

# 等待 IPC 就绪
echo -n "  等待 IPC 就绪..."
for i in $(seq 1 30); do
    if [ -e "/dev/shm/vr_result" ] && [ -S /tmp/vr_cmd.sock ]; then
        echo " OK (${i}s)"
        break
    fi
    sleep 1
    echo -n "."
    if [ "$i" -eq 30 ]; then
        echo " TIMEOUT"
        exit 1
    fi
done

# 启动 API 服务
exec python3 -m uvicorn \
    api.server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 60 \
    --log-level info
