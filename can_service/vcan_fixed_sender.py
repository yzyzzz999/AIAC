#!/usr/bin/env python3
"""
Send fixed-value CAN frames on vcan/can2 — values extracted from BLF recording.
Each signal value is individually adjustable via JSON config file.

Usage:
    python vcan_fixed_sender.py                                 # extract from BLF, send
    python vcan_fixed_sender.py --channel can2 --rate 100       # 100ms per msg
    python vcan_fixed_sender.py --config my_values.json         # custom values
    python vcan_fixed_sender.py --list                          # list current values
    python vcan_fixed_sender.py --interactive                   # edit values live
"""
import argparse, json, os, socket, struct, sys, time, threading
import cantools

DBC = "/home/data/data_collection/CAN.dbc"
BLF = "/home/data/data_collection/outputs/records_no_hw/runs/run_0002/can_recording_20260530_013358.blf"
DEFAULT_CHANNEL = "can2"
CONFIG = "/home/data/test/xyj/vcan_fixed_config.json"


def extract_values_from_blf(blf_path, dbc):
    """Extract one frame per CAN ID from a BLF recording."""
    from can import BLFReader
    cfg = {}
    found = set()

    print(f"Reading BLF: {blf_path}")
    for msg in BLFReader(blf_path):
        cid = msg.arbitration_id
        if cid in found:
            continue
        try:
            dbc_msg = dbc.get_message_by_frame_id(cid)
            decoded = dbc_msg.decode(msg.data)
            sig_values = {}
            for sig in dbc_msg.signals:
                v = decoded[sig.name]
                if isinstance(v, (int, float, str, bool)):
                    sig_values[sig.name] = v
                else:
                    sig_values[sig.name] = str(v)
            cfg[f"0x{cid:03X}"] = sig_values
            found.add(cid)
        except Exception:
            pass
        if len(found) >= len(dbc.messages):
            break

    print(f"Extracted {len(found)}/{len(dbc.messages)} CAN IDs from BLF")
    return cfg


def encode_message(msg, values):
    """Encode a message with given signal values. Returns (data, success)."""
    data = {}
    for sig in msg.signals:
        v = values.get(sig.name)
        if v is None:
            return None
        data[sig.name] = v
    try:
        return msg.encode(data, scaling=True)
    except Exception:
        return None


def list_signals(dbc, config):
    for msg in sorted(dbc.messages, key=lambda m: m.frame_id):
        vals = config.get(f"0x{msg.frame_id:03X}", {})
        print(f"\n0x{msg.frame_id:03X} {msg.name} ({msg.length}B, {msg.cycle_time}ms):")
        for sig in msg.signals:
            v = vals.get(sig.name, "—")
            unit = f" {sig.unit}" if sig.unit else ""
            print(f"  {sig.name:30s} = {v}{unit}")


def sender_loop(channel, dbc, config, interval_ms, stop_event):
    """Send all messages at interval_ms rate."""
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        s.bind((channel,))
    except OSError as e:
        print(f"ERROR: can't bind {channel}: {e}")
        return

    pre_encoded = []
    for msg in dbc.messages:
        vals = config.get(f"0x{msg.frame_id:03X}", {})
        data = encode_message(msg, vals)
        if data:
            can_frame = struct.pack("=IB3x8s", msg.frame_id, msg.length, data.ljust(8, b"\x00"))
            pre_encoded.append((msg, can_frame))

    if not pre_encoded:
        print("ERROR: no messages to send (config empty?)")
        return

    cycle_us = interval_ms / 1000.0 / len(pre_encoded)
    print(f"Sending {len(pre_encoded)} msgs on {channel} every {interval_ms}ms total ({cycle_us*1000:.1f}ms each)")
    try:
        while not stop_event.is_set():
            for msg, frame in pre_encoded:
                try:
                    s.send(frame)
                except OSError:
                    pass
                stop_event.wait(cycle_us)
                if stop_event.is_set():
                    break
    finally:
        s.close()


def interactive_edit(dbc, config):
    """Simple interactive editor."""
    msgs = sorted(dbc.messages, key=lambda m: m.frame_id)
    idx = 0
    while True:
        msg = msgs[idx]
        msg_key = f"0x{msg.frame_id:03X}"
        signals = msg.signals
        sig_idx = 0

        while True:
            os.system("clear" if os.name != "nt" else "cls")
            sig = signals[sig_idx]
            cur_val = config.get(msg_key, {}).get(sig.name, "—")
            lo = sig.minimum
            hi = sig.maximum
            unit = f" {sig.unit}" if sig.unit else ""

            print(f"0x{msg.frame_id:03X} {msg.name}  [{idx+1}/{len(msgs)}]")
            print(f"  Signal: {sig.name}  [{sig_idx+1}/{len(signals)}]")
            if lo is not None and hi is not None:
                print(f"  Range: [{lo}, {hi}]{unit}")
            print(f"  Value: {cur_val}{unit}")
            print()
            print("  n/p: prev/next signal   N/P: prev/next msg")
            print("  Enter: edit   r: reset to BLF   s: save   q: quit")
            print()

            cmd = input("> ").strip().lower()
            if cmd == "q": return
            elif cmd == "n": sig_idx = (sig_idx + 1) % len(signals)
            elif cmd == "p": sig_idx = (sig_idx - 1) % len(signals)
            elif cmd == "N": idx = (idx + 1) % len(msgs); break
            elif cmd == "P": idx = (idx - 1) % len(msgs); break
            elif cmd == "s":
                with open(CONFIG, "w") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print(f"  Saved to {CONFIG}")
            elif cmd == "r":
                # Reload from BLF
                blf_vals = extract_values_from_blf(BLF, dbc)
                config.update(blf_vals)
                print("  Reloaded from BLF")
            elif cmd == "":
                v = input(f"  New value: ").strip()
                if v:
                    try:
                        config.setdefault(msg_key, {})[sig.name] = float(v)
                    except ValueError:
                        pass  # keep old value


def main():
    parser = argparse.ArgumentParser(description="Fixed-value CAN sender — values from BLF")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--rate", type=float, default=100, help="Total cycle ms for all messages")
    parser.add_argument("--config", default=CONFIG, help="JSON config file")
    parser.add_argument("--blf", default=BLF, help="BLF file to extract values from")
    parser.add_argument("--list", action="store_true", help="List all signal values")
    parser.add_argument("--interactive", action="store_true", help="Edit values live")
    parser.add_argument("--dbc", default=DBC)
    args = parser.parse_args()

    dbc = cantools.database.load_file(args.dbc)

    # Load or extract
    if os.path.exists(args.config):
        print(f"Loading config: {args.config}")
        with open(args.config) as f:
            config = json.load(f)
    else:
        print(f"Extracting values from BLF...")
        config = extract_values_from_blf(args.blf, dbc)
        with open(args.config, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.config}")

    if args.list:
        list_signals(dbc, config)
        return

    if args.interactive:
        interactive_edit(dbc, config)
        with open(args.config, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Saved.")
        return

    # Send
    stop = threading.Event()
    t = threading.Thread(target=sender_loop, args=(args.channel, dbc, config, args.rate, stop), daemon=True)
    t.start()
    print(f"\nRunning on {args.channel} @ {args.rate}ms/cycle. Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        t.join(timeout=2)
        print("Stopped.")


if __name__ == "__main__":
    main()
