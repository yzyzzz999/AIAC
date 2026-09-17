#!/bin/bash
# can_setup.sh - CAN/CANFD配置脚本

# ============================================================
# 接口独立配置
# ============================================================

# can0: Classical CAN 500k

CAN0_BITRATE=500000

# CAN bitrate for physical interfaces
CAN_BITRATE=500000

# can2: Classical CAN 500k
can2_MODE="CLASSICAL"

# can3: Classical CAN 500k
can3_MODE="CLASSICAL"

# can4: Classical CAN 500k
can4_MODE="CLASSICAL"

# can5: CAN FD 500k/2M (来自 CAN-Send-2.dbc)
can5_MODE="CANFD"
CAN5_ARB_BITRATE=500000
CAN5_DATA_BITRATE=2000000

# 需要配置的接口列表
CAN_INTERFACES=("can2" "can3" "can4")

# 排除列表：这些物理CAN接口不会被脚本认领（有其他用途的设备）
# 例如 can0/can1 上有其他CAN卡在用，就写 EXCLUDED_DEVICES=("can0" "can1")
EXCLUDED_DEVICES=("can0" "can1")

# 其他配置
RESTART_MS=100             # 自动重启时间(ms)
SAMPLE_POINT=0.875         # 仲裁段采样点位置
DATA_SAMPLE_POINT=0.75     # 数据段采样点位置（仅CAN FD）
# ============================================================
# 函数定义
# ============================================================

# 配置Classical CAN
configure_classical_can() {
    local interface=$1
    local bitrate=$2
    
    echo "  → Configuring Classical CAN (bitrate=$bitrate)"
    sudo ip link set $interface type can \
        bitrate $bitrate \
        restart-ms $RESTART_MS \
        sample-point $SAMPLE_POINT
    
    return $?
}

# 配置CAN FD（仲裁段/数据段）
configure_canfd() {
    local interface=$1
    local arbitration_bitrate=$2
    local data_bitrate=$3
    
    echo "  → Configuring CAN FD (arb=$arbitration_bitrate, data=$data_bitrate)"
    sudo ip link set $interface type can \
        bitrate $arbitration_bitrate \
        dbitrate $data_bitrate \
        fd on \
        restart-ms $RESTART_MS \
        sample-point $SAMPLE_POINT \
        dsample-point $DATA_SAMPLE_POINT
    
    return $?
}

# 检查接口是否是 vcan（虚拟CAN）
is_vcan() {
    local interface=$1
    local target
    target=$(readlink "/sys/class/net/${interface}" 2>/dev/null)
    if [[ "$target" == *virtual* ]]; then
        return 0
    else
        return 1
    fi
}

# 检查接口是否是物理CAN（有真实硬件设备）
is_physical_can() {
    local interface=$1
    if [ -L "/sys/class/net/${interface}/device" ]; then
        return 0
    else
        return 1
    fi
}

# 查找未被脚本管理的物理CAN接口（扫描所有 can*，排除 EXCLUDED_DEVICES 中的）
find_unclaimed_physical_can() {
    for iface in /sys/class/net/can*; do
        [ -e "$iface" ] || continue
        local name
        name=$(basename "$iface")
        # 必须是物理CAN
        [ -L "$iface/device" ] || continue
        # 排除已在目标列表中的
        local skip=false
        for t in "${CAN_INTERFACES[@]}"; do
            if [ "$name" == "$t" ]; then
                skip=true
                break
            fi
        done
        [ "$skip" = true ] && continue
        # 排除用户指定不碰的接口（如 can0/can1 有其他用途）
        for e in "${EXCLUDED_DEVICES[@]}"; do
            if [ "$name" == "$e" ]; then
                skip=true
                break
            fi
        done
        [ "$skip" = true ] && continue
        echo "$name"
    done
}

# 占用一个未管理的物理CAN接口，重命名为目标名称
claim_physical_can() {
    local target_name=$1
    local source_name=$2

    echo "  → Claiming physical CAN: $source_name → $target_name"
    sudo ip link set "$source_name" down 2>/dev/null
    if ! sudo ip link set "$source_name" name "$target_name" 2>/dev/null; then
        echo "  ✗ Failed to rename $source_name to $target_name"
        return 1
    fi
    echo "  ✓ Renamed to $target_name"
    return 0
}

# 删除VCAN并替换为物理CAN
replace_vcan_with_physical() {
    local vcan_if=$1
    local phys_if=$2

    echo "  → Physical CAN device detected: $phys_if"
    echo "  → Tearing down VCAN $vcan_if, switching to physical..."

    sudo ip link delete "$vcan_if" 2>/dev/null

    if [ "$phys_if" != "$vcan_if" ]; then
        claim_physical_can "$vcan_if" "$phys_if" || return 1
    fi
    return 0
}

# 检查接口是否存在
check_interface() {
    local interface=$1
    if ip link show $interface > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 完整的接口准备+配置（物理/VCAN自动切换）
setup_can_interface() {
    local interface=$1
    local mode=$2
    local bitrate=$3
    local data_bitrate=$4

    echo ""
    echo "========================================="
    echo "Preparing: $interface"

    # 1. 接口已存在 → 判断是不是 VCAN 被物理设备替代
    if check_interface "$interface"; then
        if is_vcan "$interface"; then
            local unclaimed
            unclaimed=($(find_unclaimed_physical_can))
            if [ ${#unclaimed[@]} -gt 0 ]; then
                echo "  VCAN $interface exists, but physical CAN ${unclaimed[0]} appeared"
                replace_vcan_with_physical "$interface" "${unclaimed[0]}"
            else
                echo "  Keeping VCAN $interface (no physical device found)"
            fi
        fi
    # 2. 接口不存在 → 尝试占用物理设备，否则创建 VCAN
    else
        local unclaimed
        unclaimed=($(find_unclaimed_physical_can))
        if [ ${#unclaimed[@]} -gt 0 ]; then
            claim_physical_can "$interface" "${unclaimed[0]}"
        else
            echo "  No physical CAN found, creating VCAN $interface..."
            sudo ip link add dev "$interface" type vcan
            echo "  ✓ VCAN $interface created"
        fi
    fi

    # 3. 配置接口
    if check_interface "$interface"; then
        if configure_interface "$interface" "$mode" "$bitrate" "$data_bitrate"; then
            CONFIGURED_INTERFACES+=("$interface")
        else
            FAILED_INTERFACES+=("$interface")
        fi
    else
        echo "  ✗ Failed to create or find $interface"
        FAILED_INTERFACES+=("$interface")
    fi
}

# 配置单个CAN接口
configure_interface() {
    local interface=$1
    local mode=$2
    local bitrate=$3
    local data_bitrate=$4

    echo ""
    echo "----------------------------------------"
    echo "Configuring: $interface"

    # 检测是否为虚拟CAN
    if is_vcan "$interface"; then
        echo "  Type: VCAN (virtual CAN) — skipping bitrate/mode config"
        # 虚拟CAN：直接启动，不配置比特率
        echo "  Step 1: Bringing $interface up..."
        if ! sudo ip link set $interface up 2>/dev/null; then
            echo "Failed to bring $interface up"
            return 1
        fi
        sleep 0.1
        if ip link show $interface | grep -q "UP"; then
            echo "  ✓ $interface (VCAN) configured successfully"
            ip -d link show $interface | grep -E "(can[0-9]|state|UP)"
            return 0
        else
            echo "  ✗ $interface is not UP"
            return 1
        fi
    fi

    # 实体CAN：按模式配置
    echo "  Type: Physical CAN"
    echo "  Mode: $mode"
    if [ "$mode" == "CLASSICAL" ]; then
        echo "  Bitrate: $bitrate bps"
    else
        echo "  Arbitration Bitrate: $bitrate bps"
        echo "  Data Bitrate: $data_bitrate bps"
    fi

    # 1. 关闭接口
    echo "  Step 1: Bringing $interface down..."
    if ! sudo ip link set $interface down 2>/dev/null; then
        echo "Failed to bring $interface down"
        return 1
    fi

    # 2. 根据模式配置参数
    echo "  Step 2: Configuring $interface..."
    if [ "$mode" == "CLASSICAL" ]; then
        if ! configure_classical_can $interface $bitrate; then
            echo "Failed to configure $interface (Classical CAN)"
            return 1
        fi
    elif [ "$mode" == "CANFD" ]; then
        if ! configure_canfd $interface $bitrate $data_bitrate; then
            echo "Failed to configure $interface (CAN FD)"
            return 1
        fi
    else
        echo "Unknown CAN mode: $mode"
        return 1
    fi

    # 3. 启动接口
    echo "  Step 3: Bringing $interface up..."
    if ! sudo ip link set $interface up 2>/dev/null; then
        echo "Failed to bring $interface up"
        return 1
    fi

    # 4. 验证配置
    sleep 0.1
    if ip -d link show $interface | grep -qE "state (UP|ERROR-ACTIVE|ERROR-WARNING|ERROR-PASSIVE)"; then
        echo "  ✓ $interface configured successfully"

        # 显示配置
        if [ "$mode" == "CLASSICAL" ]; then
            ip -d link show $interface | grep -E "(bitrate|state)"
        else
            ip -d link show $interface | grep -E "(bitrate|dbitrate|fd|state)"
        fi
        return 0
    else
        echo "  ✗ $interface is not UP"
        return 1
    fi
}

# 显示所有接口状态
show_all_status() {
    echo ""
    echo "========================================="
    echo "Final Status of All CAN Interfaces"
    echo "========================================="
    
    local any_configured=false
    
    for interface in "${CAN_INTERFACES[@]}"; do
        if check_interface $interface; then
            any_configured=true
            echo ""
            echo "--- $interface ---"
            if is_vcan "$interface"; then
                echo "  Type: VCAN"
            fi
            ip -d link show $interface | grep -E "(can[0-9]:|state|bitrate|dbitrate|fd)"
            
            # 显示统计信息
            echo "Statistics:"
            ip -s link show $interface | grep -A 3 "RX:" | tail -2 | sed 's/^/  /'
        fi
    done
    
    if [ "$any_configured" = false ]; then
        echo ""
        echo "No CAN interfaces found!"
    fi
}

# 显示配置
show_config_summary() {
    echo ""
    echo "========================================="
    echo "Configuration Summary"
    echo "========================================="
    echo "can2: Classical CAN 500k"
    echo "can3: Classical CAN 500k"
    echo "can4: Classical CAN 500k"
    echo "can5: CAN FD 500k/2M (仅物理接口，不创建vcan)"
    echo "========================================="
}
# ============================================================
# 主程序
# ============================================================

echo "========================================="
echo "CAN Interface Configuration Script"
echo "========================================="
show_config_summary

# 加载CAN内核模块
echo ""
echo "Loading CAN kernel modules..."
sudo modprobe can 2>/dev/null
sudo modprobe can_raw 2>/dev/null
sudo modprobe can_dev 2>/dev/null

# 检查并配置每个接口
FAILED_INTERFACES=()
CONFIGURED_INTERFACES=()

	# 配置各接口（自动处理物理/VCAN切换）
	setup_can_interface "can2" "$can2_MODE" "$CAN_BITRATE" ""
	setup_can_interface "can3" "$can3_MODE" "$CAN_BITRATE" ""
	setup_can_interface "can4" "$can4_MODE" "$CAN_BITRATE" ""

	# can5: 仅在有物理接口时配置（不创建vcan）
	echo ""
	echo "========================================="
	echo "Preparing: can5"
	if check_interface "can5"; then
	    configure_interface "can5" "$can5_MODE" "$CAN5_ARB_BITRATE" "$CAN5_DATA_BITRATE"
	    if [ $? -eq 0 ]; then
	        CONFIGURED_INTERFACES+=("can5")
	    else
	        FAILED_INTERFACES+=("can5")
	    fi
	else
	    echo "  can5 not found — skipping (no vcan fallback)"
	fi

# 显示最终状态
show_all_status

# 输出配置结果
echo ""
echo "========================================="
echo "Configuration Result"
echo "========================================="
echo "Successfully configured: ${#CONFIGURED_INTERFACES[@]} interface(s)"
if [ ${#CONFIGURED_INTERFACES[@]} -gt 0 ]; then
    for iface in "${CONFIGURED_INTERFACES[@]}"; do
        echo "   - $iface"
    done
fi

if [ ${#FAILED_INTERFACES[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED_INTERFACES[@]} interface(s)"
    for iface in "${FAILED_INTERFACES[@]}"; do
        echo "   - $iface"
    done
    exit 1
else
    echo ""
    echo "All CAN interfaces configured successfully!"
    exit 0
fi
