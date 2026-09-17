#!/usr/bin/env bash
set -euo pipefail

# Deterministic CAN naming by stable device path fragments.
# Final target:
#   onboard c310000.mttcan         -> can0
#   onboard c320000.mttcan         -> can1
#   USB PCAN port .1 (1-*.*.1)     -> can2
#   USB PCAN port .2 (1-*.*.2)     -> can3
#   USB PCAN port .3 (1-*.*.3)     -> can4
#   USB PCAN port .4 (1-*.*.4)     -> can5

find_target() {
    local devpath="$1"

    # Onboard mttcan – match by controller name (stable)
    if [[ "$devpath" == *"c310000.mttcan"* ]]; then echo "can0"; return 0; fi
    if [[ "$devpath" == *"c320000.mttcan"* ]]; then echo "can1"; return 0; fi

    # USB PCAN – match by downstream hub port suffix (robust across hub topology changes)
    # The pattern 1-*.*.X:1.0 identifies port X on any downstream USB hub
    if [[ "$devpath" =~ 1-[0-9]+\.[1]/1-[0-9]+\.[1]:1\.0 ]]; then echo "can2"; return 0; fi
    if [[ "$devpath" =~ 1-[0-9]+\.[2]/1-[0-9]+\.[2]:1\.0 ]]; then echo "can3"; return 0; fi
    if [[ "$devpath" =~ 1-[0-9]+\.[3]/1-[0-9]+\.[3]:1\.0 ]]; then echo "can4"; return 0; fi
    if [[ "$devpath" =~ 1-[0-9]+\.[4]/1-[0-9]+\.[4]:1\.0 ]]; then echo "can5"; return 0; fi

    return 1
}

# Stage 1: move mismatched interfaces to temporary names to break rename cycles.
declare -A TMP_TO_TARGET=()
idx=0
for sysif in /sys/class/net/can*; do
    [[ -e "$sysif" ]] || continue
    cur="$(basename "$sysif")"
    dev="$(readlink -f "$sysif/device" 2>/dev/null || true)"
    [[ -n "$dev" ]] || continue

    tgt="$(find_target "$dev" || true)"
    [[ -n "$tgt" ]] || continue

    if [[ "$cur" != "$tgt" ]]; then
        tmp="cantmp${idx}"
        idx=$((idx + 1))
        ip link set "$cur" down 2>/dev/null || true
        ip link set "$cur" name "$tmp" 2>/dev/null || true
        TMP_TO_TARGET["$tmp"]="$tgt"
    fi
done

# Stage 2: assign final names.
for tmp in "${!TMP_TO_TARGET[@]}"; do
    tgt="${TMP_TO_TARGET[$tmp]}"
    ip link set "$tmp" down 2>/dev/null || true
    ip link set "$tmp" name "$tgt" 2>/dev/null || true
    echo "Renamed: $tmp -> $tgt"
done

echo "CAN naming fix complete."
ip -details -brief link show type can 2>/dev/null || true
