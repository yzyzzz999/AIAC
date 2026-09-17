#!/usr/bin/env python3
"""Real-time PMV comfort display via HTTP API."""

import json
import time
import argparse

import requests
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.console import Console
from rich import box


def pmv_color(pmv: float) -> str:
    """ISO 7730 7-point scale colors."""
    if pmv is None:
        return "dim"
    if pmv <= -2.5:
        return "bright_blue"     # -3  冷
    elif pmv <= -1.5:
        return "blue"             # -2  凉
    elif pmv <= -0.5:
        return "cyan"             # -1  稍凉
    elif pmv < 0.5:
        return "bright_green"     #  0  中性
    elif pmv < 1.5:
        return "yellow"           # +1  稍暖
    elif pmv < 2.5:
        return "bright_yellow"    # +2  暖
    else:
        return "bright_red"       # +3  热


def pmv_label(pmv: float) -> str:
    """ISO 7730 7-point scale labels."""
    if pmv is None:
        return "N/A"
    if pmv <= -2.5:
        return "冷  −3"
    elif pmv <= -1.5:
        return "凉  −2"
    elif pmv <= -0.5:
        return "稍凉 −1"
    elif pmv < 0.5:
        return "舒适  0"
    elif pmv < 1.5:
        return "稍暖 +1"
    elif pmv < 2.5:
        return "暖   +2"
    else:
        return "热   +3"


def ppd_style(ppd: float) -> str:
    if ppd is None:
        return "dim"
    if ppd <= 10:
        return "bright_green"
    elif ppd <= 25:
        return "yellow"
    else:
        return "bright_red"


def build_dashboard(data: dict) -> Layout:
    driver = data.get("driver", {})
    passenger = data.get("passenger", {})

    d_pmv = driver.get("pmv")
    d_ppd = driver.get("ppd")
    d_head = driver.get("head_temp_c")
    d_feet = driver.get("feet_temp_c")

    p_pmv = passenger.get("pmv")
    p_ppd = passenger.get("ppd")
    p_head = passenger.get("head_temp_c")
    p_feet = passenger.get("feet_temp_c")

    cabin = data.get("cabin_temp_c")
    amb = data.get("amb_temp_c")

    layout = Layout()

    header = Text()
    header.append("座舱热舒适度 - PMV 实时监控", style="bold white")
    ts = data.get("timestamp", "")
    if ts:
        header.append(f"    {ts[:19]}", style="dim")
    run_idx = data.get("run_index", 0)
    status = data.get("status", "unknown")
    header.append(f"    run #{run_idx}", style="dim")
    status_style = "green" if status == "ok" else "red"
    header.append(f" [{status}]", style=status_style)

    layout.split_column(
        Layout(Panel(header, padding=(0, 2)), size=3),
        Layout(name="body"),
    )

    def bar(value, width=40):
        if value is None:
            return Text("—", style="dim")
        mid = width // 2
        norm = value / 3.0
        pos = int(mid + norm * mid)
        pos = max(0, min(width - 1, pos))
        chars = [" "] * width
        chars[mid] = "│"
        chars[pos] = "●"
        return Text("".join(chars), style=pmv_color(value))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold white")
    table.add_column("", width=14, style="dim")
    table.add_column("主驾 Driver", width=28, justify="center")
    table.add_column("副驾 Passenger", width=28, justify="center")

    d_pmv_t = Text(f"{d_pmv:+.2f}  {pmv_label(d_pmv)}", style=pmv_color(d_pmv)) if d_pmv is not None else Text("—", style="dim")
    p_pmv_t = Text(f"{p_pmv:+.2f}  {pmv_label(p_pmv)}", style=pmv_color(p_pmv)) if p_pmv is not None else Text("—", style="dim")
    table.add_row("PMV 预测平均投票", d_pmv_t, p_pmv_t)
    table.add_row("", bar(d_pmv), bar(p_pmv))

    d_ppd_t = Text(f"{d_ppd:.1f}%", style=ppd_style(d_ppd)) if d_ppd is not None else Text("—", style="dim")
    p_ppd_t = Text(f"{p_ppd:.1f}%", style=ppd_style(p_ppd)) if p_ppd is not None else Text("—", style="dim")
    table.add_row("PPD 预测不满意率", d_ppd_t, p_ppd_t)

    d_ht = Text(f"{d_head:.1f} °C", style="bold yellow") if d_head is not None else Text("—", style="dim")
    p_ht = Text(f"{p_head:.1f} °C", style="bold yellow") if p_head is not None else Text("—", style="dim")
    table.add_row("头部温度", d_ht, p_ht)

    d_ft = Text(f"{d_feet:.1f} °C", style="bold yellow") if d_feet is not None else Text("—", style="dim")
    p_ft = Text(f"{p_feet:.1f} °C", style="bold yellow") if p_feet is not None else Text("—", style="dim")
    table.add_row("脚部温度", d_ft, p_ft)

    cabin_text = Text()
    if cabin is not None:
        cabin_text.append(f"座舱: {cabin:.1f} °C", style="bold cyan")
    if amb is not None:
        if cabin is not None:
            cabin_text.append("    ")
        cabin_text.append(f"外温: {amb:.1f} °C", style="bold yellow")
    if cabin is None and amb is None:
        cabin_text.append("温度: N/A", style="dim")

    body_layout = Layout()
    body_layout.split_column(
        Layout(table),
        Layout(Panel(cabin_text, padding=(0, 2)), size=3),
    )

    layout["body"].update(body_layout)
    return layout


def disable_alternate_scroll():
    """Disable xterm alternate-scroll mode so mouse wheel in alternate screen
    does not generate arrow-key escape sequences."""
    print("\x1b[?1007l", end="", flush=True)


def restore_alternate_scroll():
    print("\x1b[?1007h", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="PMV 实时显示")
    parser.add_argument("--api-host", default="localhost")
    parser.add_argument("--api-port", type=int, default=7861)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    api_url = f"http://{args.api_host}:{args.api_port}/pmv"

    disable_alternate_scroll()

    console = Console()
    no_data_count = 0

    try:
        with Live(console=console, screen=True, refresh_per_second=4) as live:
            while True:
                try:
                    resp = requests.get(api_url, timeout=3)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    data = None

                if data and data.get("status") != "waiting":
                    no_data_count = 0
                    layout = build_dashboard(data)
                    live.update(layout)
                else:
                    no_data_count += 1
                    hint = "等待中..." if data else f"连接失败 ({no_data_count}s)"
                    msg = Panel(
                        Text(f"{hint}\nAPI '{api_url}'\n请确认 pmv_socket_consumer 是否正在运行", style="yellow"),
                        title="PMV 实时显示",
                    )
                    live.update(msg)

                time.sleep(args.interval)
    finally:
        restore_alternate_scroll()


if __name__ == "__main__":
    main()
