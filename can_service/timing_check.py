#!/usr/bin/env python3
"""
CAN vs Socket 时间差测量.
测量: CAN帧到达 → can0_service处理 → socket客户端收到 的端到端延迟.
"""

import socket
import json
import struct
import select
import time
import os
import sys
import argparse
from collections import defaultdict

SOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sock", "can0_bus.sock")
CAN_IFACES = ["can2", "can3", "can4"]

ALL_IDS = [
    0x18F, 0x338, 0x33A, 0x33B, 0x33C, 0x35B, 0x3B9,
    0x33E, 0x33F, 0x387, 0x381, 0x3BE, 0x371,
    0x370, 0x541, 0x552, 0x553,
]

ID_SIG = {
    0x18F: "IPB_VehicleSpeed",
    0x338: "AC_FrntInCarT",
    0x33A: "AC_ACSystemOnOffSts",
    0x33B: "AC_ACMaxSts",
    0x33C: "AC_TimeVentSt",
    0x35B: "VIU_AmbT",
    0x3B9: "BMS_PackSOCRange",
    0x33E: "AC_FEvapTargetTemp",
    0x33F: "AC_DrvrFaceVentTargetT",
    0x387: "CDC_ACSystemOnOffSet",
    0x381: "CDC_ACMaxSet",
    0x3BE: "VIU_DrvrDoorSt",
    0x371: "AC_DrvrFootVentActT",
    0x370: "AC_DrvrTempVentilaPosn",
    0x541: "AC_RBlowFaceVentPosn",
    0x552: "DSM_SLCUotPosn",
    0x553: "PSM_SLCUotPosn",
}


class C:
    HDR = '\033[1;36m'
    OK = '\033[1;32m'
    WARN = '\033[1;33m'
    FAIL = '\033[1;31m'
    DIM = '\033[2m'
    RST = '\033[0m'
    BLD = '\033[1m'


def open_can_sockets(ifaces):
    socks = []
    for iface in ifaces:
        try:
            sk = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            sk.bind((iface,))
            sk.setblocking(False)
            socks.append((iface, sk))
        except Exception as e:
            print(f"[CAN] bind {iface} failed: {e}")
    return socks


def recv_can_frames(can_socks):
    """返回 [(can_id, kernel_ts), ...]"""
    frames = []
    for iface, sk in can_socks:
        try:
            r, _, _ = select.select([sk], [], [], 0.001)
            if not r:
                continue
            f = sk.recv(16)
            cid = struct.unpack("=IB", f[:5])[0]
            # CAN_RAW socket recv includes timestamp in ancillary data,
            # but struct can_frame is 16 bytes; timestamp from SO_TIMESTAMPNS
            frames.append((cid, time.time()))
        except (BlockingIOError, OSError):
            pass
    return frames


def connect_socket(path, ids):
    sk = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sk.settimeout(10)
    sk.connect(path)
    sub = json.dumps({"client_id": "timing_check", "filters": ids})
    sk.sendall(sub.encode() + b"\n")
    sk.recv(4096)  # ACK
    sk.setblocking(False)
    return sk


class LatencyStats:
    def __init__(self, max_samples=500):
        self.samples = []
        self.max_samples = max_samples

    def add(self, v):
        self.samples.append(v)
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples:]

    def report(self):
        if not self.samples:
            return None
        s = sorted(self.samples)
        return {
            "cnt": len(s),
            "min": s[0],
            "max": s[-1],
            "avg": sum(s) / len(s),
            "p50": s[len(s) // 2],
            "p95": s[int(len(s) * 0.95)],
            "p99": s[int(len(s) * 0.99)] if len(s) >= 100 else s[-1],
        }


def main():
    parser = argparse.ArgumentParser(description="CAN vs Socket 时间差测量")
    parser.add_argument("--ids", type=str, default="",
                        help="逗号分隔的 CAN ID, 如 0x552,0x553")
    parser.add_argument("--duration", type=int, default=0,
                        help="运行秒数后退出 (0=无限)")
    parser.add_argument("--top", type=int, default=10,
                        help="显示延迟最高的N个ID (默认10)")
    args = parser.parse_args()

    if args.ids:
        target_ids = [int(x.strip(), 0) for x in args.ids.split(",")]
    else:
        target_ids = list(ALL_IDS)

    # ---- 连接 CAN ----
    can_socks = open_can_sockets(CAN_IFACES)
    if not can_socks:
        print("[CAN] 无可用接口, 仅测量 socket 内部时间戳偏差")
    else:
        print(f"[CAN] 已绑定: {', '.join(f for f, _ in can_socks)}")

    # ---- 连接 Socket ----
    try:
        sock = connect_socket(SOCK_PATH, target_ids)
        print(f"[Socket] 已连接 {SOCK_PATH}")
    except Exception as e:
        print(f"[Socket] 连接失败: {e}")
        sys.exit(1)

    # ---- 统计 ----
    per_id_latency = defaultdict(LatencyStats)   # socket内部时间戳延迟 (CAN ts → now)
    per_id_can_to_sock = defaultdict(list)        # CAN原始到达 → socket收到的时间差
    socket_internal = LatencyStats(500)
    # 用于匹配: {can_id: last_can_arrival_time}
    last_can_seen = {}

    buf = ""
    start = time.time()
    iteration = 0
    last_report = 0
    can_rx_total = 0
    sock_rx_total = 0

    print(f"\n{C.HDR}{'='*90}{C.RST}")
    print(f"  测量: CAN帧时间戳 → Socket客户端收到 的端到端延迟")
    print(f"  监控 {len(target_ids)} 个 CAN ID  |  按 Ctrl+C 停止")
    if args.duration:
        print(f"  运行时长: {args.duration}s")
    print(f"{C.HDR}{'='*90}{C.RST}\n")

    try:
        while True:
            iteration += 1

            # ---- 读 CAN 原始帧, 记录到达时间 ----
            for cid, ts in recv_can_frames(can_socks):
                if cid in target_ids:
                    last_can_seen[cid] = ts
                    can_rx_total += 1

            # ---- 读 Socket 消息 ----
            try:
                r, _, _ = select.select([sock], [], [], 0.005)
                if r:
                    data = sock.recv(65536)
                    if not data:
                        print(f"{C.FAIL}[Socket] 服务端断开{C.RST}")
                        break
                    buf += data.decode(errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # 心跳
                        if msg.get("type") == "heartbeat":
                            try:
                                sock.sendall(json.dumps({"type": "pong"}).encode() + b"\n")
                            except Exception:
                                pass
                            continue

                        if msg.get("type") != "data":
                            continue

                        now = time.time()
                        sock_rx_total += 1

                        # 方法1: socket消息内嵌的CAN时间戳 vs 当前时间
                        can_ts = msg.get("timestamp", 0)
                        if can_ts > 0:
                            delta1 = now - can_ts
                            socket_internal.add(delta1)
                            # 定位这个socket消息的CAN ID, 记录延迟
                        phys = msg.get("data", {})
                        for cid, sig in ID_SIG.items():
                            if sig in phys:
                                per_id_latency[cid].add(delta1)
                                # 方法2: CAN bus看到该帧 → socket收到 的时间差
                                if cid in last_can_seen:
                                    delta2 = now - last_can_seen[cid]
                                    per_id_can_to_sock[cid].append(delta2)
                                    if len(per_id_can_to_sock[cid]) > 100:
                                        per_id_can_to_sock[cid] = per_id_can_to_sock[cid][-100:]
                                break

            except (BlockingIOError, OSError):
                pass
            except Exception as e:
                print(f"[Socket] 读错误: {e}")
                break

            # ---- 定期输出 ----
            now = time.time()
            if now - last_report >= 2.0:
                last_report = now
                elapsed = now - start

                # 清屏
                print(f"\033[2J\033[H")

                print(f"{C.HDR}{'='*90}{C.RST}")
                print(f"  CAN vs Socket 延迟  |  {time.strftime('%H:%M:%S')}  |  "
                      f"运行 {elapsed:.0f}s  |  CAN帧:{can_rx_total}  Socket消息:{sock_rx_total}")
                print(f"{C.HDR}{'='*90}{C.RST}")

                # 总体统计 (socket内嵌时间戳方法)
                r = socket_internal.report()
                if r:
                    print(f"\n  {C.BLD}[总体] Socket消息内嵌CAN时间戳 → 收到 的延迟{C.RST}")
                    print(f"  {'':>6s} {'min':>8s} {'max':>8s} {'avg':>8s} {'p50':>8s} {'p95':>8s} {'p99':>8s}  samples")
                    color = C.OK if r["avg"] < 0.05 else (C.WARN if r["avg"] < 0.2 else C.FAIL)
                    print(f"  {color}{'':>6s} {r['min']*1000:>7.1f}ms {r['max']*1000:>7.1f}ms {r['avg']*1000:>7.1f}ms "
                          f"{r['p50']*1000:>7.1f}ms {r['p95']*1000:>7.1f}ms {r['p99']*1000:>7.1f}ms  "
                          f"({r['cnt']}){C.RST}")

                # 按ID统计
                if per_id_latency:
                    print(f"\n  {C.BLD}[按ID] 各CAN ID延迟统计 (按avg排序, 仅显示top{args.top}){C.RST}")
                    print(f"  {'ID':>10s} {'Name':30s} {'min':>8s} {'max':>8s} {'avg':>8s} {'p50':>8s} {'p95':>8s}  {'samples':>7s}")

                    id_avg = []
                    for cid, stats in per_id_latency.items():
                        rpt = stats.report()
                        if rpt:
                            id_avg.append((cid, rpt))
                    id_avg.sort(key=lambda x: x[1]["avg"], reverse=True)
                    id_avg = id_avg[:args.top]

                    for cid, rpt in id_avg:
                        name = ID_SIG.get(cid, "?")
                        color = C.OK if rpt["avg"] < 0.05 else (C.WARN if rpt["avg"] < 0.2 else C.FAIL)
                        print(f"  {color}0x{cid:03X} {name:30s} {rpt['min']*1000:>7.1f}ms {rpt['max']*1000:>7.1f}ms "
                              f"{rpt['avg']*1000:>7.1f}ms {rpt['p50']*1000:>7.1f}ms {rpt['p95']*1000:>7.1f}ms  "
                              f"{rpt['cnt']:>7d}{C.RST}")

                # 额外: 各can接口的帧率
                print(f"\n  {C.DIM}--- 上次刷新后经过 {now - last_report:.1f}s ---{C.RST}")

            if args.duration and elapsed >= args.duration:
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    # ---- 最终报告 ----
    print(f"\n\n{C.HDR}{'='*90}{C.RST}")
    print(f"  {C.BLD}最终统计 (运行 {time.time()-start:.1f}s){C.RST}")
    print(f"{C.HDR}{'='*90}{C.RST}")

    r = socket_internal.report()
    if r:
        print(f"\n  Socket消息延迟 (CAN时间戳 → 收到):")
        print(f"    min={r['min']*1000:.1f}ms  max={r['max']*1000:.1f}ms  avg={r['avg']*1000:.1f}ms  "
              f"p50={r['p50']*1000:.1f}ms  p95={r['p95']*1000:.1f}ms  p99={r['p99']*1000:.1f}ms  "
              f"samples={r['cnt']}")

    sock.close()
    for _, sk in can_socks:
        sk.close()
    print()


if __name__ == "__main__":
    main()
