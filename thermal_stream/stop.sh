#!/bin/bash
cd "$(dirname "$0")/.."

PID=$(pgrep -f "thermal_stream/main.py" 2>/dev/null)

if [ -z "$PID" ]; then
    echo "Server not running"
    exit 0
fi

echo "Stopping server (PID: $PID)..."
kill $PID 2>/dev/null
sleep 1

# Force kill if still running
if kill -0 $PID 2>/dev/null; then
    echo "Force killing..."
    kill -9 $PID 2>/dev/null
fi

echo "Server stopped"
