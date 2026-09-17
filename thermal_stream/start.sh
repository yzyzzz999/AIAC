#!/bin/bash
cd "$(dirname "$0")/.."

# Check if already running
PID=$(pgrep -f "thermal_stream/main.py" 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Server already running (PID: $PID)"
    echo "Use restart.sh to restart or stop.sh to stop"
    exit 1
fi

# Activate conda and start
source /home/data/miniconda3/etc/profile.d/conda.sh
conda activate AIAC

nohup python3 -u thermal_stream/main.py > /tmp/thermal_stream.log 2>&1 &
PID=$!
echo "Server started (PID: $PID)"
echo "Log: /tmp/thermal_stream.log"
echo "URL: http://$(hostname -I | awk '{print $1}'):7864/"
