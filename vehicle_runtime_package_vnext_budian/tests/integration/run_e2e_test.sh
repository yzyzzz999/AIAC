#!/usr/bin/env bash
# 端到端回放测试: BLF → vcan2 → can0_service → socket → PMV consumer
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BLF_DIR="/home/data/data_collection/outputs/records_no_hw/runs"
SOCKET_PATH="${PROJECT_DIR}/../can_service/sock/can0_bus.sock"
CHANNEL="vcan2"

# 默认选中第一个 run
RUN_ID="${1:-run_0001}"
BLF_FILE=""
for f in "$BLF_DIR/$RUN_ID"/*.blf; do
    BLF_FILE="$f"
    break
done

if [ -z "$BLF_FILE" ] || [ ! -f "$BLF_FILE" ]; then
    echo "ERROR: BLF not found for $RUN_ID in $BLF_DIR/$RUN_ID/"
    echo "Available runs:"
    ls "$BLF_DIR/"
    exit 1
fi

echo "=== E2E Replay Test ==="
echo "BLF:  $BLF_FILE"
echo "Channel: $CHANNEL"
echo "Socket: $SOCKET_PATH"
echo

# 1. Setup vcan2
echo "[1/4] Setting up $CHANNEL..."
sudo ip link set "$CHANNEL" down 2>/dev/null || true
sudo ip link add dev "$CHANNEL" type vcan 2>/dev/null || true
sudo ip link set "$CHANNEL" up
echo "  $CHANNEL is UP ($(sudo ip -d link show "$CHANNEL" | grep -o 'state [A-Z]*'))"

# 2. Clean old socket
echo "[2/4] Cleaning old socket..."
rm -f "$SOCKET_PATH"

# 3. Start can0_service on vcan2 (background)
echo "[3/4] Starting can0_service on $CHANNEL..."
source /home/data/miniconda3/etc/profile.d/conda.sh
conda activate AIAC
python "$PROJECT_DIR/../can_service/can0_service_v1.3.8.py" --channel "$CHANNEL" &
SERVICE_PID=$!
echo "  Service PID=$SERVICE_PID"
sleep 2

# Check socket is ready
for i in $(seq 1 10); do
    if [ -S "$SOCKET_PATH" ]; then
        echo "  Socket ready: $SOCKET_PATH"
        break
    fi
    sleep 1
done

if [ ! -S "$SOCKET_PATH" ]; then
    echo "ERROR: Socket not created, check service logs"
    kill $SERVICE_PID 2>/dev/null || true
    exit 1
fi

# 4. Start BLF replay (background)
echo "[4/4] Starting BLF replay..."
python "$PROJECT_DIR/tests/integration/blf_to_vcan.py" "$BLF_FILE" --channel "$CHANNEL" --speed 1.0 &
REPLAY_PID=$!
echo "  Replay PID=$REPLAY_PID"

# 5. Start PMV consumer (foreground)
echo
echo "=== Starting PMV Consumer ==="
echo "Press Ctrl+C to stop all"
echo
python "$PROJECT_DIR/pmv_socket_consumer.py" --socket "$SOCKET_PATH" --interval 1.0 --output-dir /tmp/pmv_e2e_test_out

# Cleanup
echo
echo "=== Shutting down ==="
kill $REPLAY_PID 2>/dev/null || true
kill $SERVICE_PID 2>/dev/null || true
wait 2>/dev/null
echo "Done."
