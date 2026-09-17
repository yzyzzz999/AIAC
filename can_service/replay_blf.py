#!/usr/bin/env python3
"""Replay BLF file frames to multiple CAN interfaces with auto channel detection.

Usage:
    python3 replay_blf.py <file.blf>                    # auto-detect
    python3 replay_blf.py <file.blf> --can2 can2        # override can2 interface
    python3 replay_blf.py <file.blf> --can3 can3 --can4 can4
    python3 replay_blf.py <file.blf> --speed 2 --loop
"""

# ============================================================
# 运行参数配置 —— 直接改这里，然后 python replay_blf.py 即可
# ============================================================
BLF_FILE = "/home/data/data_collection/outputs/records_no_hw/runs/run_0002/can_recording_20260530_013358.blf"
CAN2_IFACE = "can2"       # can2 接口名（AC数据）
CAN3_IFACE = "can3"       # can3 接口名（副驾座椅）
CAN4_IFACE = "can4"       # can4 接口名（主驾座椅）
SPEED = 1.0               # 回放速度：1=原速, 0=全速, 2=2倍速
LOOP = True               # True=循环回放, False=播一次
DRY_RUN = False           # True=只显示通道映射不播放
# ============================================================

import argparse
import socket
import struct
import sys
import threading
import time
from collections import Counter
from can import BLFReader

# CAN ID signatures for auto-detection
AC_IDS = {
    0x387, 0x381, 0x33A, 0x33B, 0x33C, 0x35B, 0x3B9,
    0x33E, 0x33F, 0x338, 0x18F, 0x3BE, 0x371, 0x370, 0x541,
}
DRIVER_SEAT_IDS = {0x393, 0x552}   # 主驾座椅
PASSENGER_SEAT_IDS = {0x394, 0x553}  # 副驾座椅


def analyze_channels(messages):
    """Analyze each BLF channel and return {channel: set_of_can_ids}."""
    ch_ids = {}
    for m in messages:
        ch_ids.setdefault(m.channel, set()).add(m.arbitration_id)
    return ch_ids


def score_channel(can_ids, target_ids):
    """Return how many target IDs are present in this channel's CAN IDs."""
    return len(can_ids & target_ids)


def auto_map(channel_ids):
    """Map BLF channels to CAN interfaces based on CAN ID signatures.

    Returns {interface_name: blf_channel} or None if a channel can't be mapped.
    """
    mapping = {}
    used_channels = set()

    for iface, sig_ids in [("can2", AC_IDS), ("can3", PASSENGER_SEAT_IDS), ("can4", DRIVER_SEAT_IDS)]:
        best_ch = -1
        best_score = 0
        for ch, ids in channel_ids.items():
            if ch in used_channels:
                continue
            s = score_channel(ids, sig_ids)
            if s > best_score:
                best_score = s
                best_ch = ch
        if best_ch >= 0 and best_score > 0:
            mapping[iface] = best_ch
            used_channels.add(best_ch)

    # Fallback: assign unmapped channels to first available interface
    avail_ifaces = [n for n in ["can2", "can3", "can4"] if n not in mapping]
    for ch in sorted(channel_ids.keys()):
        if ch not in used_channels and avail_ifaces:
            mapping[avail_ifaces.pop(0)] = ch
            used_channels.add(ch)

    return mapping


def replay(s, messages, speed):
    """Replay messages to a CAN socket. Returns (sent_count, elapsed_seconds)."""
    base_ts = messages[0].timestamp
    base_real = time.monotonic()
    sent = 0

    for msg in messages:
        if speed > 0:
            target_elapsed = (msg.timestamp - base_ts) / speed
            real_elapsed = time.monotonic() - base_real
            sleep_time = target_elapsed - real_elapsed
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.01))

        can_id = msg.arbitration_id
        if msg.is_extended_id:
            can_id |= socket.CAN_EFF_FLAG
        data = msg.data.ljust(8, b'\x00')[:8]
        can_frame = struct.pack("=IB3x8s", can_id, len(msg.data), data)
        try:
            s.send(can_frame)
            sent += 1
        except OSError:
            pass

    return sent, time.monotonic() - base_real


def replay_loop(iface, messages, speed, loop, results):
    """Replay loop for a single CAN interface (runs in its own thread)."""
    try:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((iface,))
    except OSError as e:
        print(f"  [{iface}] ERROR: Cannot bind: {e}")
        results[iface] = {"error": str(e)}
        return

    duration = messages[-1].timestamp - messages[0].timestamp
    print(f"  [{iface}] {len(messages)} frames, {duration:.0f}s, speed={speed}x")

    loop_count = 0
    try:
        while True:
            sent, elapsed = replay(s, messages, speed)
            loop_count += 1
            fps = sent / elapsed if elapsed > 0 else 0
            print(f"  [{iface}] loop #{loop_count}: {sent} frames in {elapsed:.1f}s ({fps:.0f} fps)")
            if not loop:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        s.close()

    results[iface] = {"sent": sent, "loops": loop_count}


def main():
    parser = argparse.ArgumentParser(description="Replay BLF to multiple CAN interfaces")
    parser.add_argument("blf", nargs="?", default=BLF_FILE, help=f"Path to BLF file (default: {BLF_FILE})")
    parser.add_argument("--can2", default=CAN2_IFACE, help=f"CAN2 interface (default: {CAN2_IFACE})")
    parser.add_argument("--can3", default=CAN3_IFACE, help=f"CAN3 interface (default: {CAN3_IFACE})")
    parser.add_argument("--can4", default=CAN4_IFACE, help=f"CAN4 interface (default: {CAN4_IFACE})")
    parser.add_argument("--speed", type=float, default=SPEED, help=f"Speed multiplier (default: {SPEED})")
    parser.add_argument("--loop", action="store_true", default=LOOP, help=f"Loop replay (default: {LOOP})")
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN, help=f"Dry run only (default: {DRY_RUN})")
    args = parser.parse_args()

    print(f"Reading {args.blf}...")
    reader = BLFReader(args.blf)
    messages = list(reader)
    print(f"Loaded {len(messages)} frames, channels: {sorted(set(m.channel for m in messages))}")

    # Analyze and map channels
    ch_ids = analyze_channels(messages)
    print("\nChannel analysis:")
    for ch in sorted(ch_ids.keys()):
        ids = ch_ids[ch]
        ac_hits = ids & AC_IDS
        drv_hits = ids & DRIVER_SEAT_IDS
        pas_hits = ids & PASSENGER_SEAT_IDS
        tags = []
        if ac_hits: tags.append(f"AC({len(ac_hits)} IDs)")
        if drv_hits: tags.append(f"DriverSeat({len(drv_hits)} IDs)")
        if pas_hits: tags.append(f"PassSeat({len(pas_hits)} IDs)")
        other = len(ids) - len(ac_hits | drv_hits | pas_hits)
        if other: tags.append(f"other({other} IDs)")
        count = sum(1 for m in messages if m.channel == ch)
        print(f"  Channel {ch}: {count} frames, {', '.join(tags)}")

    mapping = auto_map(ch_ids)
    print("\nAuto-mapping:")
    for iface in ["can2", "can3", "can4"]:
        ch = mapping.get(iface)
        if ch is not None:
            count = sum(1 for m in messages if m.channel == ch)
            idx = ch_ids[ch] & (AC_IDS | DRIVER_SEAT_IDS | PASSENGER_SEAT_IDS)
            desc = []
            if idx & AC_IDS: desc.append("AC")
            if idx & DRIVER_SEAT_IDS: desc.append("主驾座椅")
            if idx & PASSENGER_SEAT_IDS: desc.append("副驾座椅")
            print(f"  {iface} ← channel {ch}  ({count} frames, {', '.join(desc) if desc else 'other'})")
        else:
            print(f"  {iface} ← (no channel mapped)")

    if args.dry_run:
        print("\nDry run - done.")
        return

    # Confirm
    print("\nPress Ctrl+C to stop.\n")

    # Start replay threads
    threads = []
    results = {}
    for iface in ["can2", "can3", "can4"]:
        ch = mapping.get(iface)
        if ch is None:
            continue
        ch_msgs = [m for m in messages if m.channel == ch]
        if not ch_msgs:
            continue
        t = threading.Thread(
            target=replay_loop,
            args=(iface, ch_msgs, args.speed, args.loop, results),
            daemon=True, name=f"replay-{iface}"
        )
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print("Done.")


if __name__ == "__main__":
    main()
