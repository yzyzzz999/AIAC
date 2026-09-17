#!/usr/bin/env python3
"""
CAN Monitor: DBC vs Socket — 全信号实时对比工具.

用法:
    python monitor_compare.py [选项]

选项:
    --rate N        刷新间隔秒数 (默认: 2)
    --raw           对比原始值 (默认对比物理值)
    --ids 0x552,0x553  只显示指定 CAN ID
    --no-dbc        仅 Socket 模式, 不连 CAN 总线
    --can-only      仅 CAN 总线模式, 不连 Socket
    --once          单次快照后退出
    --csv           输出 CSV 格式
    --stats         退出时打印统计摘要
    --help          显示帮助

Ctrl+C 停止.
"""

import socket
import json
import struct
import select
import time
import sys
import os
import argparse

# ============================================================
# 配置
# ============================================================
DBC_PATH = "/home/data/data_collection/CAN.dbc"
SOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sock", "can0_bus.sock")
CAN_INTERFACES = ["can2", "can3", "can4"]

# 所有 service 解析的 CAN ID (与 PARSED_CAN_IDS 保持一致)
ALL_IDS = [
    0x18F, 0x338, 0x33A, 0x33B, 0x33C, 0x35B, 0x3B9,
    0x33E, 0x33F, 0x387, 0x381, 0x3BE, 0x371,
    0x370, 0x541, 0x552, 0x553,
]

# 每个 ID 的唯一标识信号 (用于 socket 消息中识别该帧)
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

# ============================================================
# 颜色
# ============================================================
class C:
    HEADER = '\033[1;36m'
    OK = '\033[1;32m'
    FAIL = '\033[1;31m'
    STALE = '\033[2m'
    WARN = '\033[1;33m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def _fmt_val(v):
    """紧凑格式化任意值."""
    if v is None:
        return "·"
    if isinstance(v, float):
        return f"{v:.1f}"
    s = str(v)
    return s[:14]


def _vals_match(a, b):
    """判断两个值是否近似相等."""
    try:
        return abs(float(a) - float(b)) < 0.01
    except (ValueError, TypeError):
        return str(a) == str(b)


def _load_dbc(path, ids):
    """加载 DBC 文件, 返回 {can_id: message}."""
    try:
        import cantools
    except ImportError:
        print("需要安装 cantools: pip install cantools")
        sys.exit(1)
    db = cantools.database.load_file(path)
    result = {}
    for cid in ids:
        try:
            result[cid] = db.get_message_by_frame_id(cid)
        except Exception:
            pass
    return result


class SocketClient:
    """can0_service Unix Socket 客户端, 处理粘包/断包/心跳/重连."""

    def __init__(self, path, ids, client_id="monitor"):
        self.path = path
        self.ids = ids
        self.client_id = client_id
        self.sk = None
        self._buf = ""
        self.last_data_ts = 0
        self.pong_missed = 0
        self.connected = False

    def connect(self):
        """建立连接并订阅."""
        if not os.path.exists(self.path):
            return False
        try:
            self.sk = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sk.settimeout(10)
            self.sk.connect(self.path)
            # 发送订阅消息
            sub = json.dumps({"client_id": self.client_id, "filters": self.ids})
            self.sk.sendall(sub.encode() + b"\n")
            # 读取 ACK（可能因服务器处理其他客户端而延迟）
            self.sk.recv(4096)
            self.sk.setblocking(False)
            self._buf = ""
            self.pong_missed = 0
            self.connected = True
            return True
        except Exception:
            self._close()
            return False

    def _close(self):
        self.connected = False
        try:
            if self.sk:
                self.sk.close()
        except Exception:
            pass
        self.sk = None

    def recv_messages(self):
        """非阻塞读取, 返回解析后的消息列表. 自动处理粘包/断包."""
        if not self.sk or not self.connected:
            return []
        messages = []
        try:
            r, _, _ = select.select([self.sk], [], [], 0.01)
            if not r:
                return messages
            data = self.sk.recv(65536)
            if not data:
                self._close()
                return messages
            self._buf += data.decode(errors="replace")
            # 按换行分割, 最后一段可能不完整留在 _buf
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    messages.append(msg)
                    self.last_data_ts = time.time()
                except json.JSONDecodeError:
                    pass  # 跳过损坏的 JSON
        except (BlockingIOError, OSError):
            pass
        except Exception:
            self._close()
        return messages

    def process_heartbeats(self, messages):
        """处理心跳消息, 自动回复 pong. 返回非心跳消息列表."""
        data_msgs = []
        for m in messages:
            if m.get("type") == "heartbeat":
                try:
                    self.sk.sendall(json.dumps({"type": "pong"}).encode() + b"\n")
                    self.pong_missed = 0
                except Exception:
                    pass
            elif m.get("type") == "data":
                data_msgs.append(m)
        return data_msgs

    def close(self):
        self._close()


class CANReader:
    """多接口 CAN 总线读取器."""

    def __init__(self, interfaces, ids):
        self.ids = set(ids)
        self.sockets = []
        for iface in interfaces:
            try:
                sk = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                sk.bind((iface,))
                sk.setblocking(False)
                self.sockets.append((iface, sk))
            except Exception as e:
                print(f"[CAN] 无法绑定 {iface}: {e}")

    def recv_frames(self):
        """非阻塞读取所有接口, 返回 [(can_id, data_bytes), ...]."""
        frames = []
        for iface, sk in self.sockets:
            try:
                r, _, _ = select.select([sk], [], [], 0.005)
                if not r:
                    continue
                f = sk.recv(16)
                cid = struct.unpack("=IB", f[:5])[0]
                dlc = struct.unpack("=IB", f[:5])[1]
                data = f[8:8 + dlc]
                if cid in self.ids:
                    frames.append((cid, data))
            except (BlockingIOError, OSError):
                pass
            except Exception:
                pass
        return frames

    def close(self):
        for _, sk in self.sockets:
            try:
                sk.close()
            except Exception:
                pass


class MonitorState:
    """每个 CAN ID 的状态存储."""

    def __init__(self):
        self.dbc_signals = {}    # {signal_name: value} from DBC decoded CAN frame
        self.sock_phys = {}      # {signal_name: value} from socket "data"
        self.sock_raw = {}       # {signal_name: value} from socket "raw"
        self.last_can_ts = 0.0
        self.last_sock_ts = 0.0
        self.can_count = 0
        self.sock_count = 0

    @property
    def can_age(self):
        return time.time() - self.last_can_ts if self.last_can_ts else 999

    @property
    def sock_age(self):
        return time.time() - self.last_sock_ts if self.last_sock_ts else 999


def main():
    parser = argparse.ArgumentParser(
        description="CAN Monitor: DBC vs Socket 实时对比工具",
        add_help=False,
    )
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--ids", type=str, default="")
    parser.add_argument("--no-dbc", action="store_true")
    parser.add_argument("--can-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--help", action="store_true")
    args = parser.parse_args()

    if args.help:
        print(__doc__)
        sys.exit(0)

    # 确定要监控的 ID 列表
    if args.ids:
        try:
            target_ids = [int(x.strip(), 0) for x in args.ids.split(",") if x.strip()]
        except ValueError:
            print(f"无效的 --ids 格式, 例如: --ids 0x552,0x553")
            sys.exit(1)
        # 验证 ID 是否在已知列表中
        for cid in target_ids:
            if cid not in ALL_IDS:
                print(f"警告: 0x{cid:03X} 不在 service 解析列表中")
    else:
        target_ids = list(ALL_IDS)

    raw_mode = args.raw
    no_dbc = args.no_dbc or args.can_only
    no_sock = args.can_only
    rate = args.rate
    once = args.once
    csv_mode = args.csv

    # 初始化
    states = {cid: MonitorState() for cid in target_ids}

    # 加载 DBC
    dbc_msgs = {}
    if not no_dbc:
        dbc_msgs = _load_dbc(DBC_PATH, target_ids)

    # 连接 CAN
    can_reader = None
    if not no_dbc:
        can_reader = CANReader(CAN_INTERFACES, target_ids)
        if not can_reader.sockets:
            print("[CAN] 无法连接任何 CAN 接口, 自动切换为 socket-only 模式")
            no_dbc = True

    # 连接 Socket
    sock_client = None
    if not no_sock:
        sock_client = SocketClient(SOCK_PATH, target_ids)
        if not sock_client.connect():
            print(f"[Socket] 无法连接 {SOCK_PATH}, 自动切换为 can-only 模式")
            no_sock = True

    # CSV 表头
    if csv_mode:
        headers = ["timestamp"]
        for cid in target_ids:
            msg = dbc_msgs.get(cid)
            if msg:
                for sig in msg.signals:
                    headers.append(f"0x{cid:03X}_{sig.name}_dbc")
                    headers.append(f"0x{cid:03X}_{sig.name}_sock")
                    headers.append(f"0x{cid:03X}_{sig.name}_match")
        print(",".join(headers))

    # 连接状态
    sock_ok = sock_client and sock_client.connected
    can_ok = can_reader and len(can_reader.sockets) > 0

    # ============================================================
    # 主循环
    # ============================================================
    last_print = 0
    iteration = 0
    print_err_once = True

    if not csv_mode:
        print(f"\n{'='*100}")
        parts = []
        if can_ok:
            parts.append(f"CAN: {','.join(CAN_INTERFACES)}")
        if sock_ok:
            parts.append(f"Socket: OK")
        else:
            parts.append(f"Socket: OFF")
        parts.append(f"{'raw' if raw_mode else 'physical'}")
        parts.append(f"{rate}s refresh")
        if args.ids:
            parts.append(f"IDs: {args.ids}")
        print(f"  Monitor  |  {'  |  '.join(parts)}")
        print(f"{'='*100}")

    try:
        while True:
            iteration += 1

            # ---- 读取 CAN 总线 ----
            if can_reader:
                for cid, data in can_reader.recv_frames():
                    st = states[cid]
                    st.can_count += 1
                    st.last_can_ts = time.time()
                    msg = dbc_msgs.get(cid)
                    if msg:
                        try:
                            if raw_mode:
                                decoded = msg.decode(data, decode_choices=False, scaling=False)
                            else:
                                decoded = msg.decode(data)
                            st.dbc_signals = decoded
                        except Exception:
                            pass

            # ---- 读取 Socket ----
            if sock_client:
                if not sock_client.connected:
                    # 尝试重连
                    if iteration % 20 == 0:
                        if sock_client.connect():
                            sock_ok = True
                            if not csv_mode:
                                print(f"  {C.OK}[Socket] 重连成功{C.RESET}")
                else:
                    msgs = sock_client.recv_messages()
                    data_msgs = sock_client.process_heartbeats(msgs)
                    for m in data_msgs:
                        phys = m.get("data", {})
                        raw_vals = m.get("raw", {})
                        # 通过唯一信号定位 CAN ID
                        for cid, sig in ID_SIG.items():
                            if cid not in states:
                                continue
                            if sig in phys or sig in raw_vals:
                                st = states[cid]
                                st.sock_count += 1
                                st.last_sock_ts = time.time()
                                st.sock_phys = phys
                                st.sock_raw = raw_vals
                                break

            # ---- 输出 ----
            now = time.time()
            if once or (now - last_print >= rate):
                last_print = now

                if csv_mode:
                    _print_csv(states, target_ids, dbc_msgs, raw_mode)
                else:
                    _print_table(states, target_ids, dbc_msgs, raw_mode,
                                 sock_client, can_ok, sock_ok)
                sys.stdout.flush()

                if once:
                    break

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass

    # ---- 统计摘要 ----
    if not csv_mode:
        print(f"\n{'='*100}")
        print(f"  统计摘要")
        print(f"{'='*100}")
        total_can = sum(s.can_count for s in states.values())
        total_sock = sum(s.sock_count for s in states.values())
        print(f"  CAN 帧总数: {total_can}")
        print(f"  Socket 消息总数: {total_sock}")

        for cid in target_ids:
            st = states[cid]
            dbc_msg = dbc_msgs.get(cid)
            name = dbc_msg.name if dbc_msg else f"0x{cid:03X}"
            if st.can_count > 0 or st.sock_count > 0:
                match_cnt = 0
                total_sigs = 0
                if dbc_msg and st.sock_phys:
                    for sig in dbc_msg.signals:
                        total_sigs += 1
                        dbc_v = st.dbc_signals.get(sig.name)
                        if raw_mode:
                            sock_v = st.sock_raw.get(sig.name)
                        else:
                            sock_v = st.sock_phys.get(sig.name)
                        if dbc_v is not None and sock_v is not None and _vals_match(dbc_v, sock_v):
                            match_cnt += 1

                match_str = f"  match={match_cnt}/{total_sigs}" if total_sigs > 0 else ""
                print(f"  0x{cid:03X} {name:35s}  CAN={st.can_count:5d}  SOCK={st.sock_count:5d}{match_str}")

    # 清理
    if can_reader:
        can_reader.close()
    if sock_client:
        sock_client.close()
    print()


def _print_table(states, target_ids, dbc_msgs, raw_mode, sock_client, can_ok, sock_ok):
    """打印表格形式的对比结果."""
    now = time.time()

    # 连接状态行
    if sock_client:
        if sock_client.connected:
            status = f"{C.OK}[SOCK: OK]{C.RESET}"
        else:
            status = f"{C.FAIL}[SOCK: 断开]{C.RESET}"
        if can_ok:
            status += f"  {C.OK}[CAN: OK]{C.RESET}"
        print(f"\n  {status}  {time.strftime('%H:%M:%S')}")
    else:
        print(f"\n  [SOCK: OFF]  {time.strftime('%H:%M:%S')}")

    for cid in target_ids:
        st = states[cid]
        dbc_msg = dbc_msgs.get(cid)
        if dbc_msg is None:
            continue

        can_age = st.can_age
        sock_age = st.sock_age
        name = dbc_msg.name
        sig_count = len(dbc_msg.signals)

        # 标题行颜色
        if can_age > 5 and sock_age > 5:
            stale = f" {C.STALE}STALE{C.RESET}"
        elif can_age > 5:
            stale = f" {C.WARN}CAN STALE{C.RESET}"
        elif sock_age > 5:
            stale = f" {C.WARN}SOCK STALE{C.RESET}"
        else:
            stale = ""

        print(f"\n  {C.HEADER}0x{cid:03X} {name}{stale}{C.RESET}  "
              f"(CAN:{can_age:.0f}s ago, SOCK:{sock_age:.0f}s ago, "
              f"sig={sig_count})")
        print(f"  {'Signal':36s} │ {'DBC':>14s} │ {'Socket':>14s} │")

        match_ok = 0
        match_total = 0
        for sig in dbc_msg.signals:
            sn = sig.name
            dbc_v = _fmt_val(st.dbc_signals.get(sn))
            if raw_mode:
                sock_v = st.sock_raw.get(sn)
            else:
                sock_v = st.sock_phys.get(sn)

            if sock_v is not None:
                sock_vs = _fmt_val(sock_v)
                match_total += 1
                if _vals_match(st.dbc_signals.get(sn, 0), sock_v):
                    match = f"{C.OK}✓{C.RESET}"
                    match_ok += 1
                else:
                    match = f"{C.FAIL}✗{C.RESET}"
            else:
                sock_vs = "·"
                match = " "

            # 超时数据变暗
            dim = C.DIM if (can_age > 5 or sock_age > 5) else ""
            print(f"  {dim}{sn:36s}{C.RESET} │ {dim}{dbc_v:>14s}{C.RESET} │ "
                  f"{dim}{sock_vs:>14s}{C.RESET} │ {dim}{match}{C.RESET}")

        # 匹配统计
        if match_total > 0:
            pct = match_ok / match_total * 100
            match_color = C.OK if pct >= 90 else (C.WARN if pct >= 50 else C.FAIL)
            print(f"  {'':36s}   {'':14s}   {'':14s}   "
                  f"{match_color}match: {match_ok}/{match_total} ({pct:.0f}%){C.RESET}")


def _print_csv(states, target_ids, dbc_msgs, raw_mode):
    """输出 CSV 格式."""
    now = time.time()
    row = [f"{now:.3f}"]
    for cid in target_ids:
        st = states[cid]
        dbc_msg = dbc_msgs.get(cid)
        if dbc_msg is None:
            continue
        for sig in dbc_msg.signals:
            sn = sig.name
            dbc_v = _fmt_val(st.dbc_signals.get(sn))
            if raw_mode:
                sock_v = st.sock_raw.get(sn)
            else:
                sock_v = st.sock_phys.get(sn)
            sock_vs = _fmt_val(sock_v) if sock_v is not None else ""
            if sock_v is not None:
                m = "1" if _vals_match(st.dbc_signals.get(sn, 0), sock_v) else "0"
            else:
                m = ""
            row.append(dbc_v)
            row.append(sock_vs)
            row.append(m)
    print(",".join(row))


if __name__ == "__main__":
    main()
