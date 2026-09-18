#!/usr/bin/env python3
"""将 BLF 录制的 CAN 数据回放到 vcan2，支持加速回放。"""
import sys
import time
import argparse
import signal
from pathlib import Path

import can

# 仅回放 PMV 相关的 CAN ID（减小 vcan 负载）
PMV_CAN_IDS = {0x18F, 0x338, 0x33E, 0x33F, 0x35B, 0x3B9}


def main():
    parser = argparse.ArgumentParser(description="BLF → vcan2 replay")
    parser.add_argument("blf", help="BLF file path")
    parser.add_argument("--channel", default="vcan2", help="VCAN channel (default: vcan2)")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--all-ids", action="store_true", help="Replay all CAN IDs (not just PMV-relevant)")
    parser.add_argument("--loop", action="store_true", help="Loop replay indefinitely")
    args = parser.parse_args()

    blf_path = Path(args.blf)
    if not blf_path.exists():
        print(f"ERROR: BLF file not found: {args.blf}")
        sys.exit(1)

    reader = can.BLFReader(str(blf_path))
    messages = list(reader)
    reader.stop()

    if not args.all_ids:
        messages = [m for m in messages if m.arbitration_id in PMV_CAN_IDS]

    if not messages:
        print("ERROR: No messages to replay (check --all-ids or BLF content)")
        sys.exit(1)

    ids_included = sorted(set(m.arbitration_id for m in messages))
    duration = messages[-1].timestamp - messages[0].timestamp
    print(f"Loaded {len(messages)} messages, {len(ids_included)} CAN IDs: {[f'0x{x:X}' for x in ids_included]}")
    print(f"Duration: {duration:.1f}s, Speed: {args.speed}x → replay in {duration/args.speed:.1f}s")
    print(f"Channel: {args.channel}")

    bus = can.Bus(interface="socketcan", channel=args.channel, bitrate=500000)
    print("Replay started...")
    start_wall = time.monotonic()
    t0 = messages[0].timestamp

    running = True

    def sig_handler(sig, frame):
        nonlocal running
        running = False
        print("\nStopping...")

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    sent = 0
    try:
        while running:
            for msg in messages:
                if not running:
                    break
                # 计算等待时间
                elapsed_real = time.monotonic() - start_wall
                elapsed_can = (msg.timestamp - t0) / args.speed
                wait = elapsed_can - elapsed_real
                if wait > 0:
                    time.sleep(min(wait, 0.05))  # cap sleep to avoid oversleep
                bus.send(msg)
                sent += 1
                if sent % 5000 == 0:
                    print(f"  Sent {sent}/{len(messages)} messages...")
            if not args.loop:
                break
            print(f"Loop replay restarting...")
            start_wall = time.monotonic()
    finally:
        bus.shutdown()
        print(f"Replay done. Sent {sent} messages.")


if __name__ == "__main__":
    main()
