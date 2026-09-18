#!/usr/bin/env python3


"""

2026-9-16  v1.3.8
虚拟订阅ID：
    0xF001 (AI状态推送)：客户端 filters 中加入 "0xF001" 即可订阅，
           每秒推送 {"version":"1.0","type":"ai_state","timestamp":...,"data":{"ai":"on/off"}}

"""




"""
CAN总线接收模块
解析的ID：
    0x387 (VIU_ACSet1), 
    0x381 (VIU_ACSet2), 
    0x33A (AC_Sts1), 
    0x33B (AC_Sts2), 
    0x33C (AC_Sts3), 
    0x35B (VIU_35B), 
    0x3B9 (VIU_LIN1),
    0x33E (AC_TEMP2),
    0x33F (AC_TEMP3),
支持 Unix Socket 通信
"""


import can
import time
import os
import socket
import json
import threading
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict
import multiprocessing as mp
import signal  
import sys  
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib import request as urllib_request
from urllib import error as urllib_error


# ============================================================
# 软件版本信息
# ============================================================
SOFTWARE_VERSION = "1.3.8"
SOFTWARE_NAME = "CAN0 Bus Services"
BUILD_DATE = "2026-09-16"

# ============================================================
# 运行配置（在此修改）
# ============================================================
CAN_INTERFACES = ["can2", "can3", "can4", "can5"]   # CAN 接口列表
CPU_AFFINITY = None                          # 绑定的 CPU 核心，None 表示不绑定（例: 8）


SEAT_TRIGGER_STABLE_SECONDS = 5.0
SEAT_SCRIPT_TIMEOUT_SECONDS = 30
SEAT_TRIGGER_LOGGING_ENABLED = False
SEAT_PARSE_LOGGING_ENABLED = False




# Backend air-conditioning integration
ENABLE_BACKEND_AC_CONTROL = True          # 后台空调集成功能总开关
ENABLE_BACKEND_AC_LOG = True              # 后台空调打印总开关，关闭后以下分类日志均不打印
ENABLE_BACKEND_AC_RX_LOG = True          # 是否打印 /adjust 请求处理结果及 HTTP 访问日志
ENABLE_BACKEND_AC_RAW_JSON_LOG = True    # 是否打印后台下发的原始 JSON 报文
ENABLE_BACKEND_AC_TX_LOG = True          # 是否打印 /updateInfo 回显状态及发送成功信息
ENABLE_BACKEND_AC_GATE_LOG = False        # 是否打印 Socket 0x387 改写信息（最多每秒一次）
ENABLE_BACKEND_AC_DEBUG_FEEDBACK = False  # 是否启用 /updateInfo 回显模拟值
ENABLE_BACKEND_AC_DEBUG_CAN_OVERRIDE = False  # 是否强制用模拟值改写发往CAN的0x387并发送
ENABLE_BACKEND_AC_SOCKET_0X387_RAW_LOG = False  # 是否打印客户端转发到CAN前的0x387原始报文
ENABLE_BACKEND_AC_AI_STATE_LOG = True     # 是否打印 ai 状态切换日志（仅变化时打印一次，如 off -> on）
BACKEND_AC_DEBUG_AI = "off"                # 模拟回显 AI 空调总开关
BACKEND_AC_DEBUG_MODE = 2                 # 模拟回显空调吹风模式
BACKEND_AC_DEBUG_LEVEL = 2                # 模拟回显空调风量挡位
BACKEND_AC_DEBUG_DRIVER_TEMP = 18.5       # 模拟回显主驾温度
BACKEND_AC_DEBUG_PASSENGER_TEMP = 18.5    # 模拟回显副驾温度
BACKEND_AC_DEFAULT_MODE = 1               # 未收到0x387前默认回显空调吹风模式
BACKEND_AC_DEFAULT_LEVEL = 1              # 未收到0x387前默认回显空调风量挡位
BACKEND_AC_DEFAULT_DRIVER_TEMP = 24.5     # 未收到0x387前默认回显主驾温度
BACKEND_AC_DEFAULT_PASSENGER_TEMP = 24.5  # 未收到0x387前默认回显副驾温度

BACKEND_LISTEN_HOST = os.environ.get("BACKEND_LISTEN_HOST", "0.0.0.0")  # AIBox HTTP监听地址，0.0.0.0监听所有
BACKEND_LISTEN_PORT = int(os.environ.get("BACKEND_LISTEN_PORT", "20001"))
BACKEND_UPDATE_URL = os.environ.get(
    "BACKEND_UPDATE_URL",
    "http://192.168.0.48:20001/api/airConditioningControl/updateInfo",
)
BACKEND_UPDATE_INTERVAL_SECONDS = 1.0
BACKEND_HTTP_TIMEOUT_SECONDS = 0.8
BACKEND_MAX_BODY_BYTES = 64 * 1024

# AI 状态订阅推送（v1.3.8 新增）
VIRTUAL_ID_AI_STATE = 0xF001               # ai 状态虚拟订阅 ID（>0x7FF，不与真实标准帧ID冲突）
AI_STATE_PUBLISH_INTERVAL_SECONDS = 1.0    # 推送周期


@dataclass
class SeatCycleState:
    phase: str = "IDLE"
    occupied_start: Optional[float] = None
    leave_start: Optional[float] = None




# ============================================================
# 0. CAN ID 配置区
# ============================================================

class CAN_ID(IntEnum):
    VIU_AC_SET1 = 0x387    # VIU空调设置1
    VIU_AC_SET2 = 0x381    # VIU空调设置2
    AC_STS1 = 0x33A        # 空调状态1
    AC_STS2 = 0x33B        # 空调状态2
    AC_STS3 = 0x33C        # 空调状态3
    VIU_35B = 0x35B        # 环境温度/车窗状态
    VIU_LIN1 = 0x3B9       # LIN信号
    AC_TEMP2 = 0x33E       # 蒸发器温度/温湿度/制冷采暖等级
    AC_TEMP3 = 0x33F       # 吹面出风温度
    AC_TEMP1 = 0x338
    VIU_CHA_18F = 0x18F
    VIU_PASSIVE_SEC_SEAT_3BE = 0x3BE
    AC_TEMP5 = 0x371
    AC_TEMP4 = 0x370       # 风门位置
    AC_RHVAC = 0x541       # 后排空调/风门电压
    VIU_DSM_552 = 0x552    # 主驾座椅电机位置
    VIU_PSM_553 = 0x553    # 副驾座椅电机位置
    C_100 = 0x100           # CAN-Send-2: GPS/温度/湿度

# 需要解析的CAN ID列表
PARSED_CAN_IDS = [
    CAN_ID.VIU_AC_SET1,
    CAN_ID.VIU_AC_SET2,
    CAN_ID.AC_STS1,
    CAN_ID.AC_STS2,
    CAN_ID.AC_STS3,
    CAN_ID.VIU_35B,
    CAN_ID.VIU_LIN1,
    CAN_ID.AC_TEMP2,
    CAN_ID.AC_TEMP3,
    CAN_ID.AC_TEMP1,
    CAN_ID.VIU_CHA_18F,
    CAN_ID.VIU_PASSIVE_SEC_SEAT_3BE,
    CAN_ID.AC_TEMP5,
    CAN_ID.AC_TEMP4,
    CAN_ID.AC_RHVAC,
    CAN_ID.VIU_DSM_552,
    CAN_ID.VIU_PSM_553,
    CAN_ID.C_100,
]

# 显示选项
SHOW_ALL_RECEIVED = False
SHOW_ALL_SEND = False
SHOW_PARSED_ONLY = False
SHOW_RAW_ON_ERROR = True

# 打印控制：每接收多少帧打印一次（设为1表示每帧都打印）
PRINT_INTERVAL = 10

# Socket 路径
SOCKET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sock", "can0_bus.sock")


# ============================================================
# 1. RTE 信号接口整理模块
# ============================================================

class RTESignalMapper:
    """RTE 信号接口映射器 - 将 CAN 信号转换为 RTE 标准格式"""
    
    # RTE 信号名称映射 (CAN信号名 -> RTE信号名)
    SIGNAL_MAP = {
        # 空调设置 (0x387)
        "CDC_ACSystemOnOffSet": "AC_PwrReq",
        "CDC_RACSystemOnOffSet": "AC_PwrReq_Rr",
        "CDC_ACSet": "AC_ACReq",
        "CDC_FrontDefrostSet": "AC_FrontDefrostReq",
        "CDC_FHvacBlowLvSet": "AC_FanLvReq_Frnt",
        "CDC_RHvacBlowLvSet": "AC_FanLvReq_Rr",
        "CDC_FHvacModeSet": "AC_ModeReq_Frnt",
        "CDC_RHvacModeSet": "AC_ModeReq_Rr",
        "CDC_DriverTempCSet": "AC_TempReq_Drv",
        "CDC_PassengerTempCSet": "AC_TempReq_Pass",
        "CDC_RrTempCSet": "AC_TempReq_Rr",
        
        # 空调设置2 (0x381)
        "CDC_ACMaxSet": "AC_MaxACReq",
        "CDC_HeatMaxSet": "AC_MaxHeatReq",
        "CDC_HvacAutoModeSet": "AC_AutoModeReq",
        "CDC_RHvacAutoModeSet": "AC_AutoModeReq_Rr",
        "CDC_AutoDefogSet": "AC_AutoDefogReq",
        "CDC_ACFourZoneSync": "AC_FourZoneSyncReq",
        "CDC_TempExclusive": "AC_TempExclusiveReq",
        "CDC_CircModSet": "AC_CirculationReq",
        "CDC_IntelligentizePurSet": "AC_IntelligentPurReq",
        "CDC_ACParkAutoVentSet": "AC_ParkAutoVentReq",
        "CDC_IONset": "AC_IONReq",
        "CDC_UVCset": "AC_UVCReq",
        "CDC_AutoDrySet": "AC_AutoDryReq",
        "CDC_ACECOModReq": "AC_EcoModeReq",
        "CDC_VentSet": "AC_VentReq",
        "CDC_OdorRemoveSet": "AC_OdorRemoveReq",
        "CDC_ThirdRowHvacBlowLvSet": "AC_FanLvReq_Third",
        "CDC_ThirdRowHvacModeSet": "AC_ModeReq_Third",
        "CDC_ThirdRowTempCSet": "AC_TempReq_Third",
        "CDC_ThirdRowHvacAutoModeSet": "AC_AutoModeReq_Third",
        
        # 空调状态1 (0x33A)
        "AC_ACSystemOnOffSts": "AC_PwrSts",
        "AC_RACSystemOnOffSts": "AC_PwrSts_Rr",
        "AC_ACSts": "AC_ACSts",
        "AC_FrontDefrostSts": "AC_FrontDefrostSts",
        "AC_FBlowSpeedLevel": "AC_FanLvSts_Frnt",
        "AC_RBlowSpeedLevel": "AC_FanLvSts_Rr",
        "AC_FModeAdjustSts": "AC_ModeSts_Frnt",
        "AC_RModeAdjustSts": "AC_ModeSts_Rr",
        "AC_DriverTempC": "AC_TempSts_Drv",
        "AC_PassengerTempC": "AC_TempSts_Pass",
        "AC_RrTempC": "AC_TempSts_Rr",
        "AC_CircleModeSts": "AC_CirculationSts",
        "AC_ParkAutoVentSts": "AC_ParkAutoVentSts",
        "AC_FHvacBlowLvAutoSts": "AC_FanAutoSts_Frnt",
        "AC_RHvacBlowLvAutoSts": "AC_FanAutoSts_Rr",
        "AC_FBlowModeAutoSts": "AC_ModeAutoSts_Frnt",
        "AC_RBlowModeAutoSts": "AC_ModeAutoSts_Rr",
        "AC_ThirdRowACSystemOnOffSts": "AC_PwrSts_Third",
        "AC_ThirdRowHvacBlowLvAutoSts": "AC_FanAutoSts_Third",
        "AC_ThirdRowBlowModeAutoSts": "AC_ModeAutoSts_Third",
        
        # 空调状态2 (0x33B)
        "AC_ACMaxSts": "AC_MaxACSts",
        "AC_HeatMaxSts": "AC_MaxHeatSts",
        "AC_AutoModeSts": "AC_AutoSts",
        "AC_RAutoModeSts": "AC_AutoSts_Rr",
        "AC_AutoDefogSts": "AC_AutoDefogSts",
        "AC_ACFourZoneSyncSts": "AC_FourZoneSyncSts",
        "AC_TempExclusiveSts": "AC_TempExclusiveSts",
        "AC_AutoDefogRunSts": "AC_AutoDefogRunSts",
        "AC_FaultREASON": "AC_FaultReason",
        "AC_Pres_HP": "AC_Pres_HP",
        "AC_Pres_LPT_Chiller": "AC_Pres_LPT_Chiller",
        "AC_Pres_LPT_EVAP": "AC_Pres_LPT_EVAP",
        "AC_ThirdRowBlowSpeedLevel": "AC_FanLvSts_Third",
        "AC_ThirdRowModeAdjustSts": "AC_ModeSts_Third",
        "AC_ThirdRowTempC": "AC_TempSts_Third",
        "AC_ThirdRowAutoModeSts": "AC_AutoSts_Third",
        
        # 空调状态3 (0x33C)
        "AC_TimeVentSt": "AC_TimeVentSts",
        "AC_VentTimeFlag": "AC_VentTimeFlagSts",
        "AC_VentholdTime": "AC_VentHoldTimeSts",
        "AC_VentTime": "AC_VentTimeSts",
        "AC_BlowerVolt": "AC_BlowerVolt",
        "AC_RBlowerVolt": "AC_RBlowerVolt",
        "AC_UVCSts": "AC_UVCSts",
        "AC_UVC_Err": "AC_UVCErrSts",
        "AC_UVC_ErrValid": "AC_UVCErrValidSts",
        "AC_AutoDrySts": "AC_AutoDrySts",
        "AC_ACECOModSts": "AC_EcoModeSts",
        "AC_VentSts": "AC_VentSts",
        "AC_RemFaultSts": "AC_RemoteFaultSts",
        "AC_OdorRemoveSts": "AC_OdorRemoveSts",
        "AC_FilterRemain": "AC_FilterRemain",
        
        # 环境温度/车窗状态 (0x35B)
        "VIU_AmbTVld": "AmbientTemp_Valid",
        "VIU_AmbT": "AmbientTemp",
        "VIU_VehAntithftSt": "VehicleAntiTheft_Status",
        "CDC_SceneModePowerOnSts": "SceneMode_PowerStatus",
        "VIU_FLWinOpenDeg": "Window_LeftFront_OpenPercent",
        "VIU_FRWinOpenDeg": "Window_RightFront_OpenPercent",
        "VIU_RLWinOpenDeg": "Window_LeftRear_OpenPercent",
        "VIU_RRWinOpenDeg": "Window_RightRear_OpenPercent",
        
        # LIN信号 (0x3B9)
        "BMS_PackSOCRange": "BMS_PackSOCRange",
        "SSM_F_Position": "Sunshade_Position",
        "RSM_DewPointT": "DewPoint_Temperature",
        "RSM_RainFalLev": "Rainfall_Level",
        "RSM_WinT": "Window_Temperature",
        "RSM_RelHum": "Relative_Humidity",
        "RSM_LeSolarInten": "Left_Solar_Intensity",
        "RSM_RiSolarInten": "Right_Solar_Intensity",
        
        # 蒸发器温度/温湿度/制冷采暖等级 (0x33E)
        "AC_FEvapTargetTemp": "FrontEvap_TargetTemp",
        "AC_FEvapCurrentTemp": "FrontEvap_CurrentTemp",
        "AC_REvapTargetTemp": "RearEvap_TargetTemp",
        "AC_REvapCurrentTemp": "RearEvap_CurrentTemp",
        "AC_THS_RelHum": "THS_RelativeHumidity",
        "AC_THS_DewPointT": "THS_DewPointTemp",
        "AC_THS_T": "THS_Temperature",
        "AC_cabinCoolingLevel": "Cabin_CoolingLevel",
        "AC_cabinheatingLevel": "Cabin_HeatingLevel",
        
        # 吹面出风温度 (0x33F)
        "AC_DrvrFaceVentTargetT": "Driver_FaceVent_TargetTemp",
        "AC_PassFaceVentTargetT": "Passenger_FaceVent_TargetTemp",
        "AC_SecRowFaceVentTargetT": "SecondRow_FaceVent_TargetTemp",
        "AC_DrvrFaceVentActT": "Driver_FaceVent_ActualTemp",
        "AC_PassFaceVentActT": "Passenger_FaceVent_ActualTemp",
        "AC_SecRowFaceVentActT": "SecondRow_FaceVent_ActualTemp",
        "AC_ThrdRowVentTargetT": "ThirdRow_Vent_TargetTemp",
        "AC_ThrdRowVentActT": "ThirdRow_Vent_ActualTemp",
        
        # AC_Temp1 (0x338)
        "AC_FrntInCarT": "FrontInCarTemp",
        "AC_ReInCarT": "RearInCarTemp",
        "AC_FrntInCarTValid": "FrontInCarTemp_Valid",
        "AC_ReInCarTValid": "RearInCarTemp_Valid",
        "AC_OverheatdT": "OverheatTemp",
        "AC_Forward_BlwPwmOut": "ChillerOutletTemp",
        "AC_LPTSnsrT": "CompressorInletTemp",
        "AC_Second_BlwPwmOut": "SecondRowBlowerPWM",
        
        # VIU_CHA_18F (0x18F)
        "VDC_HV_State": "HighVoltage_State",
        "IPB_VehicleSpeedValid": "VehicleSpeed_Valid",
        "IPB_VehicleSpeed": "VehicleSpeed",
        
        # VIU_PassiveSecSeat_3BE (0x3BE)
        "VIU_DrvrDoorSt": "DriverDoor_Status",
        "VIU_PassDoorSt": "PassengerDoor_Status",
        "VIU_RLDoorSt": "RearLeftDoor_Status",
        "VIU_RRDoorSt": "RearRightDoor_Status",
        "VIU_TailgateSt": "Tailgate_Status",
        "VIU_HoodSts": "Hood_Status",
        "VIU_DriverSeatOccptSt": "DriverSeat_Occupancy",
        "VIU_PassSeatOccptSt": "PassengerSeat_Occupancy",
        "VIU_RLSecRowSeatOccptSt": "SecondRowLeft_Occupancy",
        "VIU_RMSecRowSeatOccptSt": "SecondRowMiddle_Occupancy",
        "VIU_RRSecRowSeatOccptSt": "SecondRowRight_Occupancy",
        "VIU_RLThrdRowSeatOccptSt": "ThirdRowLeft_Occupancy",
        "VIU_RRThrdRowSeatOccptSt": "ThirdRowRight_Occupancy",

        # VIU_DSM_552 (0x552) — 主驾座椅电机位置
        "DSM_SLCUotPosn": "DriverSeat_SlidePosition",
        "DSM_BackrestMotPosn": "DriverSeat_BackrestPosition",
        "DSM_CushHeiMotPosn": "DriverSeat_CushionHeight",
        "DSM_CushTiltMotPosn": "DriverSeat_CushionTilt",
        "DSM_CushExtnMotPosn": "DriverSeat_CushionExtension",

        # VIU_PSM_553 (0x553) — 副驾座椅电机位置
        "PSM_SLCUotPosn": "PassengerSeat_SlidePosition",
        "PSM_BackrestMotPosn": "PassengerSeat_BackrestPosition",
        "PSM_CushHeiMotPosn": "PassengerSeat_CushionHeight",
        "PSM_CushTiltMotPosn": "PassengerSeat_CushionTilt",
        "PSM_LegSupportMotPosn": "PassengerSeat_LegSupport",
        "PSM_FootSupportMotPosn": "PassengerSeat_FootSupport",
        "PSM_SpindleMotPosn": "PassengerSeat_SpindlePosition",

        # AC_Temp5 (0x371) — 吹脚温度
        "AC_DrvrFootVentTargetT": "Driver_FootVent_TargetTemp",
        "AC_PassFootVentTargetT": "Passenger_FootVent_TargetTemp",
        "AC_ThrdRowFootVentTargetT": "ThirdRow_FootVent_TargetTemp",
        "AC_DrvrFootVentActT": "Driver_FootVent_ActualTemp",
        "AC_PassFootVentActT": "Passenger_FootVent_ActualTemp",
        "AC_ThrdRowFootVentActT": "ThirdRow_FootVent_ActualTemp",
        "AC_Forward_AirFlowTarget": "Forward_AirFlow_Target",
        "AC_Second_AirFlowTarget": "Second_AirFlow_Target",

        # AC_Temp4 (0x370) — 风门位置
        "AC_PassTempVentilaPosn": "Passenger_TempVent_Position",
        "AC_DrvrTempVentilaPosn": "Driver_TempVent_Position",
        "AC_CircleModeVentilaPosn": "CircleModeVent_Position",
        "AC_SecRowTempVentilaPosn": "SecondRow_TempVent_Position",
        "AC_SecRowFootVentilaPosn": "SecondRow_FootVent_Position",
        "AC_ModeVentilaPosn": "ModeVent_Position",
        "AC_SecRowModeVentilaPosn": "SecondRow_ModeVent_Position",
        "AC_ThrdTempVentilaPosn": "ThirdRow_TempVent_Position",
        "AC_BLOW_FaceVentilaPosn": "Blow_FaceVent_Position",

        # AC_RHVAC (0x541) — 后排空调/风门电压
        "AC_RBlowFaceVentPosn": "Rear_BlowFaceVent_Position",
        "AC_RBlowFootVentPosn": "Rear_BlowFootVent_Position",
        "AC_ThrdRowModeVentPosn": "ThirdRow_ModeVent_Position",
        "AC_ThrdBlowVol": "ThirdRow_Blower_Voltage",
        "AC_12VOUT": "AC_12V_Output",
        "AC_INGVol": "AC_IGN_Voltage",
        "AC_FrantFootVentPosn": "Front_FootVent_Position",
        "AC_DefrostVentilaPosn": "Defrost_Vent_Position",

        # CAN-Send-2 0x100: GPS/温度/湿度
        "C________1": "Signal_C1",
        "GPS__": "GPS_Latitude",
        "GPS___1": "GPS_Longitude",
        "TA_FdHeadTempLe": "FrontHeadTemp_Left",
        "TA_FdHeadTempRi": "FrontHeadTemp_Right",
        "TA_FpHeadTempLe": "FrontPanelTemp_Left",
        "TA_FpHeadTempRi": "FrontPanelTemp_Right",
        "TS_FrntWidTemp": "FrontWindshield_Temp",
        "V_FrntHum": "Front_Humidity",
        "V_SecHum": "Secondary_Humidity",
    }
    
    # 需要提取数值的信号
    NUMERIC_SIGNALS = {
        "AC_FanLvReq_Frnt", "AC_FanLvReq_Rr", "AC_FanLvReq_Third",
        "AC_FanLvSts_Frnt", "AC_FanLvSts_Rr", "AC_FanLvSts_Third",
        "AC_TempReq_Drv", "AC_TempReq_Pass", "AC_TempReq_Rr", "AC_TempReq_Third",
        "AC_TempSts_Drv", "AC_TempSts_Pass", "AC_TempSts_Rr", "AC_TempSts_Third",
        "AC_Pres_HP", "AC_Pres_LPT_Chiller", "AC_Pres_LPT_EVAP",
        "AC_BlowerVolt", "AC_RBlowerVolt",
        "AC_FilterRemain", "AC_VentHoldTimeSts", "AC_VentTimeSts",
        "DriverSeat_SlidePosition", "DriverSeat_BackrestPosition",
        "DriverSeat_CushionHeight", "DriverSeat_CushionTilt",
        "DriverSeat_CushionExtension",
        "PassengerSeat_SlidePosition", "PassengerSeat_BackrestPosition",
        "PassengerSeat_CushionHeight", "PassengerSeat_CushionTilt",
        "PassengerSeat_LegSupport", "PassengerSeat_FootSupport",
        "PassengerSeat_SpindlePosition",
        # CAN-Send-2 0x100
        "Signal_C1", "GPS_Latitude", "GPS_Longitude",
        "FrontHeadTemp_Left", "FrontHeadTemp_Right",
        "FrontPanelTemp_Left", "FrontPanelTemp_Right",
        "FrontWindshield_Temp",
        "Front_Humidity", "Secondary_Humidity",
    }
    
    @classmethod
    def convert_to_rte(cls, can_id: int, parsed_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        将解析的 CAN 信号转换为 RTE 格式
        
        Args:
            can_id: CAN ID
            parsed_signals: 解析后的信号字典
        
        Returns:
            RTE 格式的信号字典
        """
        rte_data = {}
        
        for can_signal, value in parsed_signals.items():
            # 跳过 raw 信号
            if can_signal.endswith('_raw'):
                continue
            
            # 获取 RTE 信号名
            rte_signal = cls.SIGNAL_MAP.get(can_signal)
            if not rte_signal:
                continue
            
            # 转换值
            converted_value = cls._convert_value(can_signal, rte_signal, value)
            if converted_value is not None:
                rte_data[rte_signal] = converted_value
        
        return rte_data
    
    @classmethod
    def _convert_value(cls, can_signal: str, rte_signal: str, value: Any) -> Any:
        """转换信号值"""
        # 处理温度字符串（如 "24.5" -> 24.5）
        if isinstance(value, str):
            # 尝试转换为浮点数
            try:
                return float(value)
            except ValueError:
                pass
            
            # 枚举值映射
            enum_map = {
                "NO Request": 0, "OFF": 1, "ON": 2, "INVALID": 3,
                "INITIAL": 0, "Initial": 0, "initial": 0,
                "LO": 0, "HI": 31,
                "Level1": 1, "Level2": 2, "Level3": 3, "Level4": 4,
                "Level5": 5, "Level6": 6, "Level7": 7, "Level8": 8,
                "Level9": 9, "Level10": 10,
                "BlowFace": 1, "BlowFace&Foot": 2, "BlowFoot": 3,
                "BlowFoot&Windows": 4, "BlowWindows": 5,
                "BlowFace&Windows": 6, "BlowFace&Foot&Windows": 7,
                "FACE": 1, "FACE_AND_FOOT": 2, "FOOT": 3,
                "Internal": 1, "External": 2, "Auto": 3, "AUTO": 3,
                "Open": 1, "Close": 2, "Not Arrived": 1, "Arrived": 2,
                "Normal": 0, "Error": 1, "Fault": 1,
                "Valid": 1, "Invalid": 0,
                "Auto": 0, "Not Auto": 1,
                "NORMAL": 0, "Voltage Fault": 1, "Sensor Fault": 2, "CCU Fault": 3,
            }
            return enum_map.get(value, value)
        
        # 数值直接返回
        if isinstance(value, (int, float)):
            return value
        
        return value




# ============================================================
# HTTP 服务
# ============================================================

def backend_ac_log(message: str, enabled: bool = True) -> None:
    if ENABLE_BACKEND_AC_LOG and enabled:
        print(f"[BackendAC] {message}")


def backend_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class BackendACControlState:
    """Thread-safe state shared by HTTP and Socket-to-CAN paths."""

    HTTP_MODE_FROM_CAN_MODE = {
        3: 1,
        1: 2,
        2: 3,
        5: 4,
        4: 5,
        6: 6,
        7: 7,
    }
    CAN_MODE_FROM_HTTP_MODE = {
        http_mode: can_mode
        for can_mode, http_mode in HTTP_MODE_FROM_CAN_MODE.items()
    }
    AC_FIELD_NAMES = (
        "ac_mode",
        "ac_level",
        "driver_ac_temple",
        "passenger_ac_temple",
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._ai = "on"
        self._block_socket_0x387 = False
        self._backend_values: Dict[str, Any] = {
            "ac_mode": BACKEND_AC_DEFAULT_MODE,
            "ac_level": BACKEND_AC_DEFAULT_LEVEL,
            "driver_ac_temple": BACKEND_AC_DEFAULT_DRIVER_TEMP,
            "passenger_ac_temple": BACKEND_AC_DEFAULT_PASSENGER_TEMP,
        }
        self._client_feedback: Dict[str, Any] = {
            "ac_mode": BACKEND_AC_DEFAULT_MODE,
            "ac_level": BACKEND_AC_DEFAULT_LEVEL,
            "driver_ac_temple": BACKEND_AC_DEFAULT_DRIVER_TEMP,
            "passenger_ac_temple": BACKEND_AC_DEFAULT_PASSENGER_TEMP,
        }
        self._socket_feedback: Dict[str, Any] = {
            "ac_mode": BACKEND_AC_DEFAULT_MODE,
            "ac_level": BACKEND_AC_DEFAULT_LEVEL,
            "driver_ac_temple": BACKEND_AC_DEFAULT_DRIVER_TEMP,
            "passenger_ac_temple": BACKEND_AC_DEFAULT_PASSENGER_TEMP,
        }

    @staticmethod
    def _validate_int(name: str, value: Any, minimum: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"{name} must be an integer")
        result = int(number)
        if result < minimum or result > maximum:
            raise ValueError(f"{name} must be in range {minimum}..{maximum}")
        return result

    @staticmethod
    def _validate_temperature(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        result = float(value)
        if result < 16.0 or result > 31.0:
            raise ValueError(f"{name} must be in range 16..31")
        if abs(result * 2.0 - round(result * 2.0)) > 1e-9:
            raise ValueError(f"{name} step must be 0.5")
        return result

    @classmethod
    def parse_adjust_payload(cls, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")

        values: Dict[str, Any] = {}
        global_data = payload.get("global")
        if global_data is not None:
            if not isinstance(global_data, dict):
                raise ValueError("global must be an object")
            if "ai" in global_data:
                ai = global_data["ai"]
                if ai not in ("on", "off"):
                    raise ValueError("global.ai must be 'on' or 'off'")
                values["ai"] = ai
            if "ac_mode" in global_data:
                values["ac_mode"] = cls._validate_int(
                    "global.ac_mode", global_data["ac_mode"], 1, 7
                )
            if "ac_level" in global_data:
                values["ac_level"] = cls._validate_int(
                    "global.ac_level", global_data["ac_level"], 1, 10
                )

        for side_name, state_name in (
            ("driver_side", "driver_ac_temple"),
            ("passenger_side", "passenger_ac_temple"),
        ):
            side_data = payload.get(side_name)
            if side_data is None:
                continue
            if not isinstance(side_data, dict):
                raise ValueError(f"{side_name} must be an object")
            if "ac_temple" in side_data:
                values[state_name] = cls._validate_temperature(
                    f"{side_name}.ac_temple", side_data["ac_temple"]
                )

        return values

    def apply_adjust(self, payload: Any) -> Dict[str, Any]:
        values = self.parse_adjust_payload(payload)
        with self._lock:
            updated_fields = [
                name for name in ("ai",) + self.AC_FIELD_NAMES
                if name in values
            ]
            old_ai = self._ai
            self._ai = values.get("ai", self._ai)
            if old_ai != self._ai:
                # 仅在 ai 状态发生变化时打印一次，重复 POST 相同值不打印
                backend_ac_log(
                    f"ai state changed: {old_ai} -> {self._ai}",
                    ENABLE_BACKEND_AC_AI_STATE_LOG,
                )
            for name in self.AC_FIELD_NAMES:
                if name in values:
                    self._backend_values[name] = values[name]

            if self._ai == "off":
                self._socket_feedback.update(self._backend_values)
                effective_source = "backend"
            else:
                self._socket_feedback.update(self._client_feedback)
                effective_source = "socket"

            return {
                "ai": self._ai,
                "effective_source": effective_source,
                "updated_fields": updated_fields,
            }

    def get_ai(self) -> str:
        # _ai 为不可变 str，单次读取是原子操作；推送线程仅需该字段，无需与其他字段一致，故不加锁
        # 注意：返回真实 _ai，不受 ENABLE_BACKEND_AC_DEBUG_* 模拟开关影响
        return self._ai

    def is_socket_0x387_blocked(self) -> bool:
        return False

    @staticmethod
    def _debug_ac_values() -> Dict[str, Any]:
        return {
            "ac_mode": BACKEND_AC_DEBUG_MODE,
            "ac_level": BACKEND_AC_DEBUG_LEVEL,
            "driver_ac_temple": BACKEND_AC_DEBUG_DRIVER_TEMP,
            "passenger_ac_temple": BACKEND_AC_DEBUG_PASSENGER_TEMP,
        }

    @classmethod
    def _build_0x387_override(cls, values: Dict[str, Any],
                              source: str) -> Optional[Dict[str, Any]]:
        can_mode = cls.CAN_MODE_FROM_HTTP_MODE.get(int(values["ac_mode"]))
        if can_mode is None:
            return None

        driver_raw = int(round(
            (float(values["driver_ac_temple"]) - 16.0) * 2.0
        ))
        passenger_raw = int(round(
            (float(values["passenger_ac_temple"]) - 16.0) * 2.0
        ))

        return {
            "source": source,
            "mode_raw": can_mode,
            "level_raw": int(values["ac_level"]),
            "driver_raw": driver_raw,
            "passenger_raw": passenger_raw,
        }

    def backend_0x387_override(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if ENABLE_BACKEND_AC_DEBUG_CAN_OVERRIDE:
                debug_values = self._debug_ac_values()
                self._socket_feedback.update(debug_values)
                return self._build_0x387_override(debug_values, "debug")

            if self._ai != "off":
                return None

            self._socket_feedback.update(self._backend_values)
            return self._build_0x387_override(self._backend_values, "backend")

    def update_socket_feedback(self, decoded: Dict[str, Any]) -> None:
        updates: Dict[str, Any] = {}

        can_mode = decoded.get("CDC_FHvacModeSet")
        if isinstance(can_mode, (int, float)):
            http_mode = self.HTTP_MODE_FROM_CAN_MODE.get(int(can_mode))
            if http_mode is not None:
                updates["ac_mode"] = http_mode

        ac_level = decoded.get("CDC_FHvacBlowLvSet")
        if isinstance(ac_level, (int, float)) and 1 <= int(ac_level) <= 10:
            updates["ac_level"] = int(ac_level)

        for signal_name, state_name in (
            ("CDC_DriverTempCSet", "driver_ac_temple"),
            ("CDC_PassengerTempCSet", "passenger_ac_temple"),
        ):
            value = decoded.get(signal_name)
            if isinstance(value, (int, float)):
                temperature = float(value)
                if 16.0 <= temperature <= 31.0:
                    updates[state_name] = temperature

        if updates:
            with self._lock:
                self._client_feedback.update(updates)
                if self._ai == "on":
                    self._socket_feedback.update(updates)

    def feedback_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if ENABLE_BACKEND_AC_DEBUG_FEEDBACK:
                return {
                    "global": {
                        "ai": BACKEND_AC_DEBUG_AI,
                        "ac_mode": BACKEND_AC_DEBUG_MODE,
                        "ac_level": BACKEND_AC_DEBUG_LEVEL,
                    },
                    "driver_side": {
                        "ac_temple": BACKEND_AC_DEBUG_DRIVER_TEMP
                    },
                    "passenger_side": {
                        "ac_temple": BACKEND_AC_DEBUG_PASSENGER_TEMP
                    },
                }

            if ENABLE_BACKEND_AC_DEBUG_CAN_OVERRIDE:
                debug_values = self._debug_ac_values()
                return {
                    "global": {
                        "ai": self._ai,
                        "ac_mode": debug_values["ac_mode"],
                        "ac_level": debug_values["ac_level"],
                    },
                    "driver_side": {
                        "ac_temple": debug_values["driver_ac_temple"]
                    },
                    "passenger_side": {
                        "ac_temple": debug_values["passenger_ac_temple"]
                    },
                }

            if any(value is None for value in self._socket_feedback.values()):
                return None
            return {
                "global": {
                    "ai": self._ai,
                    "ac_mode": self._socket_feedback["ac_mode"],
                    "ac_level": self._socket_feedback["ac_level"],
                },
                "driver_side": {
                    "ac_temple": self._socket_feedback["driver_ac_temple"]
                },
                "passenger_side": {
                    "ac_temple": self._socket_feedback["passenger_ac_temple"]
                },
            }


class BackendAdjustHandler(BaseHTTPRequestHandler):
    """HTTP线程中接收后台/adjust请求."""

    state: BackendACControlState = None

    def setup(self):
        super().setup()
        self.connection.settimeout(BACKEND_HTTP_TIMEOUT_SECONDS)

    def _send_json(self, status_code: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        # 只处理后台空调调节接口，其他路径直接返回失败。
        if self.path.split("?", 1)[0] != "/api/airConditioningControl/adjust":
            self._send_json(404, {
                "timestamp": backend_timestamp(),
                "code": 500,
                "status": "failed",
                "msg": "endpoint not found",
            })
            return

        try:
            # 限制请求体大小，避免异常请求长时间占用监听线程。
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > BACKEND_MAX_BODY_BYTES:
                raise ValueError("invalid Content-Length")

            # 先保留后台发送的原始报文，再进行 UTF-8 和 JSON 解析。
            raw_body = self.rfile.read(content_length)
            raw_json = raw_body.decode("utf-8")
            backend_ac_log(
                f"收到后台原始 JSON: {raw_json}",
                ENABLE_BACKEND_AC_RAW_JSON_LOG,
            )
            payload = json.loads(raw_json)

            # 串行同步 AI 状态、后台控制值和 /updateInfo 最终回显值。
            result = self.state.apply_adjust(payload)
            backend_ac_log(
                f"adjust ai={result['ai']} "
                f"effective_source={result['effective_source']} "
                f"updated={result['updated_fields']}",
                ENABLE_BACKEND_AC_RX_LOG,
            )
            self._send_json(200, {
                "timestamp": backend_timestamp(),
                "code": 200,
                "status": "success",
                "msg": "",
            })
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            backend_ac_log(f"adjust rejected: {exc}", True)
            self._send_json(500, {
                "timestamp": backend_timestamp(),
                "code": 500,
                "status": "failed",
                "msg": str(exc),
            })
        except Exception as exc:
            backend_ac_log(f"adjust error: {exc}", True)
            self._send_json(500, {
                "timestamp": backend_timestamp(),
                "code": 500,
                "status": "failed",
                "msg": str(exc),
            })

    def log_message(self, format, *args):
        if ENABLE_BACKEND_AC_LOG and ENABLE_BACKEND_AC_RX_LOG:
            print(f"[BackendAC][HTTP] {format % args}")


class BackendHTTPServer(HTTPServer):
    allow_reuse_address = True


# ============================================================
# 2. Unix Socket 服务器
# ============================================================

class UnixSocketServer:
    """支持客户端过滤的 Unix Socket 服务器"""

    def __init__(self, socket_path: str = SOCKET_PATH, enable_rte: bool = True):
        self.socket_path = socket_path
        self.server_socket = None
        self.running = False
        self.clients: Dict[socket.socket, Dict] = {}  # socket -> client_info
        self.lock = threading.Lock()
        self.enable_rte = enable_rte  # 是否启用 RTE 转换

        # ========== 心跳检测配置 ==========
        self.heartbeat_interval = 0.5  # 500ms 发送一次心跳
        self.heartbeat_timeout = 3     # 连续3次无响应则断开连接
        self.heartbeat_thread = None   # 心跳检测线程
        # CAN发送相关回调
        self.send_can_callback = None
        self.send_event_callback = None
        self.start_periodic_callback = None
        self.stop_periodic_callback = None
        self.client_disconnected_callback = None
        self.send_signal_callback = None
    
    # MODIFIED: 注入所有回调
    def set_send_callbacks(self, send_cb, send_event_cb,
                           start_periodic_cb, stop_periodic_cb,
                           client_disconnected_cb,
                           send_signal_cb=None):
        self.send_can_callback = send_cb
        self.send_event_callback = send_event_cb
        self.start_periodic_callback = start_periodic_cb
        self.stop_periodic_callback = stop_periodic_cb
        self.client_disconnected_callback = client_disconnected_cb
        self.send_signal_callback = send_signal_cb
    # 向指定客户端发送JSON消息
    def send_to_client(self, client_socket, message: dict):
        """向指定客户端发送一条JSON消息（带换行符）"""
        try:
            data = json.dumps(message, ensure_ascii=False) + "\n"
            with self.lock:
                if client_socket in self.clients:  # 确保客户端仍连接
                    client_socket.sendall(data.encode('utf-8'))
                    return True
        except Exception as e:
            print(f"[Socket] Failed to send message to client: {e}")
        return False
    def start(self):
        """启动服务器"""
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(0.5)
        self.running = True
        print(f"[Socket] Server started at {self.socket_path}")
        
        # ========== 启动心跳检测线程 ==========
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="HeartbeatMonitor"
        )
        self.heartbeat_thread.start()
        print(f"[Socket] Heartbeat monitor started (interval={self.heartbeat_interval}s, timeout={self.heartbeat_timeout})")
        # =====================================
    
	
    # ==================== 心跳检测相关方法 ====================
    def _heartbeat_loop(self):
        """心跳检测循环 - 每隔 heartbeat_interval 秒发送一次心跳，并检查超时"""
        while self.running:
            time.sleep(self.heartbeat_interval)
            self._check_clients_heartbeat()
    
    def _check_clients_heartbeat(self):
        """检查所有客户端的心跳状态，连续超时则断开连接"""
        dead_clients = []
        
        with self.lock:
            for client, info in list(self.clients.items()):
                # 初始化心跳计数器
                if 'heartbeat_miss' not in info:
                    info['heartbeat_miss'] = 0
                    info['last_heartbeat_time'] = time.time()
                
                # 发送心跳请求（非阻塞，避免因客户端 buffer 满而超时踢人）
                try:
                    heartbeat_msg = json.dumps({
                        "type": "heartbeat",
                        "timestamp": time.time()
                    })
                    data_bytes = heartbeat_msg.encode('utf-8') + b"\n"
                    # MSG_DONTWAIT: 非阻塞发送，buffer 满则跳过（不死踢客户端）
                    try:
                        client.send(data_bytes, socket.MSG_DONTWAIT)
                    except (BlockingIOError, OSError):
                        pass  # buffer 满，跳过本次心跳，等 pong 来判断死活
                    info['heartbeat_miss'] += 1

                except Exception as e:
                    # 严重错误（socket 已关闭等），标记死亡
                    print(f"[Socket] Heartbeat send failed for {info.get('id', 'unknown')}: {e}")
                    dead_clients.append(client)
                    continue

                # 检查是否达到超时阈值（仅靠 pong 响应判断死活）
                if info['heartbeat_miss'] >= self.heartbeat_timeout:
                    print(f"[Socket] Client {info.get('id', 'unknown')} no pong for {info['heartbeat_miss']} heartbeats, disconnecting...")
                    dead_clients.append(client)
        
        # 清理死亡客户端
        for client in dead_clients:
            self._remove_client(client)
    
    def _remove_client(self, client):
        """移除客户端连接并关闭socket"""
        with self.lock:
            if client in self.clients:
                client_info = self.clients[client]
                client_id = client_info.get('id', 'unknown')
                print(f"[Socket] Removing client {client_id} (missed {client_info.get('heartbeat_miss', 0)} heartbeats)")
                del self.clients[client]
        
        try:
            client.close()
        except:
            pass
        # 通知 CANReceiver 清理该客户端的所有周期发送任务
        if self.client_disconnected_callback:
            try:
                self.client_disconnected_callback(client)
            except Exception as e:
                print(f"[Socket] Error in client_disconnected_callback: {e}")
    
    @staticmethod
    def _parse_can_id(val):
        if isinstance(val, int): return val
        if isinstance(val, str): return int(val, 0)
        return None

    def _handle_client_message(self, client, data):
        try:
            msg = json.loads(data.strip())
            msg_type = msg.get("type", "")
            client_id = self.clients[client]["id"] if client in self.clients else "unknown"
            # 心跳响应
            if msg_type == "pong":
                with self.lock:
                    if client in self.clients:
                        self.clients[client]['heartbeat_miss'] = 0
                        self.clients[client]['last_pong_time'] = time.time()
                return
				
            # 单帧发送（原始bytes，需自行编码）
            elif msg_type == "send_can":
                if self.send_can_callback:
                    can_id = self._parse_can_id(msg.get("can_id"))
                    data = bytes(msg.get("data", []))
                    if can_id is not None and data:
                        success = self.send_can_callback(can_id, data)
                        if not success:
                            self.send_to_client(client, {
                                "type": "send_can_error",
                                "can_id": can_id,
                                "reason": "CAN bus unavailable or send failed"
                            })
                return
            # 按信号名+物理值发送（自动DBC编码+ID路由+通道选择）
            elif msg_type == "send_signal":
                if self.send_signal_callback:
                    signals = msg.get("signals", {})
                    count = msg.get("count", 1)
                    interval_ms = msg.get("interval_ms", 20)
                    if signals:
                        # 单次或多次发送
                        for i in range(count):
                            result = self.send_signal_callback(signals)
                            if result.get("status") == "error":
                                self.send_to_client(client, {
                                    "type": "send_signal_error",
                                    "reason": result.get("reason", "unknown")
                                })
                                break
                            if i < count - 1:
                                time.sleep(interval_ms / 1000.0)
                return
            # 事件周期发送
            elif msg_type == "send_event":
                if self.send_event_callback:
                    can_id = self._parse_can_id(msg.get("can_id"))
                    data = bytes(msg.get("data", []))
                    count = msg.get("count", 3)
                    interval_ms = msg.get("interval_ms", 100)
                    if can_id is not None and data:
                        self.send_event_callback(can_id, data, count, interval_ms)
                return
            # 启动周期发送
            elif msg_type == "start_periodic":
                if self.start_periodic_callback:
                    can_id = self._parse_can_id(msg.get("can_id"))
                    data = bytes(msg.get("data", []))
                    interval_ms = msg.get("interval_ms", 1000)
                    task_id = msg.get("task_id", str(time.time()))
                    if can_id is not None and data:
                        # 调用回调，现在会返回结果信息
                        result = self.start_periodic_callback(client, task_id, can_id, data, interval_ms)
						
                        # 如果回调返回了错误信息，发送给客户端
                        if isinstance(result, dict) and result.get("status") == "error":
                            self.send_to_client(client, result)
                return
				
			
            # 停止周期发送
            elif msg_type == "stop_periodic":
                if self.stop_periodic_callback:
                    task_id = msg.get("task_id")
                    if task_id:
                        self.stop_periodic_callback(client, task_id)
                return
        except json.JSONDecodeError:
            # 不是JSON消息，忽略
            pass
        except Exception as e:
            print(f"[Socket] Error handling client message: {e}")
    
    def _client_receive_loop(self, client):
        """客户端消息接收循环（专门处理心跳响应等控制消息）"""
        client_id = None
        with self.lock:
            if client in self.clients:
                client_id = self.clients[client].get("id", "unknown")
        
        try:
            while self.running:
                try:
                    client.settimeout(1.0)
                    data = client.recv(1024).decode('utf-8').strip()
                    if data:
                        self._handle_client_message(client, data)
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[Socket] Receive error from {client_id}: {e}")
                    break
        except:
            pass
        finally:
            # 客户端接收循环结束，清理连接
            with self.lock:
                if client in self.clients:
                    client_id = self.clients[client].get("id", "unknown")
                    print(f"[Socket] Client {client_id} receive loop ended, cleaning up...")
                    # 注意：这里不直接删除，让心跳检测或广播失败时处理
        
        # 如果客户端还存在，标记为需要清理
        self._remove_client(client)
    # ==========================================================
    
	
    def accept_clients(self):
        """接受新客户端并注册"""
        init_msg = ''
        try:
            client, _ = self.server_socket.accept()

            # 打印客户端连接信息
            client_addr = client.getpeername() if hasattr(client, 'getpeername') else 'unknown'
            print(f"\n[Socket] New client connecting from: {client_addr}")
            
            # 接收客户端标识（设超时防止死锁：客户端connect后未及时send会卡住accept）
            print(f"[Socket] Waiting for init from {client_addr} (timeout=1s)...")
            client.settimeout(1.0)
            try:
                init_msg = client.recv(1024).decode('utf-8').strip()
                client_info = json.loads(init_msg)
                client_id = client_info.get("client_id", f"unknown_{id(client)}")
                client_filters = client_info.get("filters", [])
                client_version = client_info.get("version", "unknown")
                
                # 打印接收到的客户端消息
                print(f"[Socket] <<< Received subscription from client [{client_id}]:")
                print(f"  Client ID: {client_id}")
                print(f"  Version: {client_version}")
                try:
                    print(f"  Filters: {[hex(f) if isinstance(f, int) else f for f in client_filters]}")
                except Exception:
                    print(f"  Filters: {client_filters}")
                print(f"  Raw message: {init_msg[:200]}{'...' if len(init_msg) > 200 else ''}")

                
            except socket.timeout:
                print(f"[Socket] Client init timeout, closing")
                client.close()
                return
            except Exception as e:
                client_id = f"anonymous_{id(client)}"
                client_filters = []
                client_version = "unknown"
                print(f"[Socket] <<< Received from client (parse error): {e}")
                print(f"  Raw message: {init_msg if init_msg else 'N/A'}")

            # 先发送 ACK，成功后再注册（防止瞬连客户端残留）
            ack_msg = json.dumps({
                "status": "ok",
                "message": "connected",
                "server_version": SOFTWARE_VERSION,
                "server_name": SOFTWARE_NAME,
                "timestamp": time.time(),
                "heartbeat_interval": self.heartbeat_interval
            })
            try:
                client.settimeout(0.5)
                client.sendall(ack_msg.encode('utf-8') + b"\n")
                client.settimeout(None)
            except (OSError, socket.timeout):
                client.close()
                return

            with self.lock:
                self.clients[client] = {
                    "id": client_id,
                    "filters": {int(f, 0) if isinstance(f, str) else int(f) for f in client_filters if f is not None},
                    "address": client_addr,
                    "version": client_version,
                    "connected_time": time.time(),
                    "heartbeat_miss": 0,
                    "last_heartbeat_time": time.time(),
                    "last_pong_time": time.time()
                }

            print(f"[Socket] Client registered: {client_id}, filters: {[hex(f) if isinstance(f, int) else f for f in client_filters]}, version: {client_version}")

            receive_thread = threading.Thread(
                target=self._client_receive_loop,
                args=(client,),
                daemon=True,
                name=f"ClientRecv_{client_id}"
            )
            receive_thread.start()
            
            # 打印发送到客户端的确认消息
            # print(f"[Socket] >>> Sent ack to client [{client_id}]:")
            # print(f"  {ack_msg}")
            # print(f"[Socket] Total connected clients: {len(self.clients)}")
            
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[Socket] Accept error: {e}")
    
    def broadcast_filtered(self, can_id: int, packet: dict):
        """根据客户端订阅的过滤器发送数据"""
        if not self.clients:
            return

        dead_clients = []
        
        # 构建输出数据包
        if self.enable_rte and "data" in packet:
            rte_data = RTESignalMapper.convert_to_rte(can_id, packet["data"])
            output_packet = {
                "version": packet.get("version", "1.0"),
                "timestamp": packet.get("timestamp", time.time()),
                "data": rte_data,
                "type": "data"  # 添加消息类型标识，便于客户端区分
            }
        else:
            output_packet = packet
        
        # 统计发送的客户端数量
        sent_count = 0
        
        with self.lock:
            for client, info in list(self.clients.items()):
                filters = info.get("filters", set())
                if not filters or can_id in filters:
                    try:
                        message = json.dumps(output_packet, ensure_ascii=False) + "\n"
                        data_bytes = message.encode('utf-8')
                        # send + MSG_DONTWAIT: 非阻塞，buf满跳过，不卡CAN接收
                        try:
                            n = client.send(data_bytes, socket.MSG_DONTWAIT)
                            if n == len(data_bytes):
                                sent_count += 1
                        except (BlockingIOError, OSError):
                            pass
                        
                        # 打印发送到客户端的消息（频率高时可能影响性能）
                        # 摘要信息
                        # client_id = info.get("id", "unknown")
                        # print(f"[Socket] >>> Sent to client [{client_id}] (CAN ID: 0x{can_id:03X}):")
                        # print(f"  timestamp: {output_packet.get('timestamp', 0):.3f}s")
                        # print(f"  data keys: {list(output_packet.get('data', {}).keys())[:5]}")

                        
                    except Exception as e:
                        print(f"[Socket] Failed to send to client {info.get('id', 'unknown')}: {e}")
                        dead_clients.append(client)
            
            for client in dead_clients:
                client_id = self.clients.get(client, {}).get("id", "unknown")
                print(f"[Socket] Client {client_id} disconnected, removing...")
                del self.clients[client]
                try:
                    client.close()
                except:
                    pass
            
            # if sent_count > 0:
                # print(f"[Socket] Data broadcast to {sent_count} client(s) for CAN ID 0x{can_id:03X}")

    
    def broadcast_to_all(self, data: dict):
        """广播给所有客户端"""
        if not self.clients:
            return
        
        dead_clients = []
        sent_count = 0
        
        with self.lock:
            for client in list(self.clients.keys()):
                try:
                    message = json.dumps(data, ensure_ascii=False) + "\n"
                    client.sendall(message.encode('utf-8'))
                    sent_count += 1
                    
                    client_id = self.clients.get(client, {}).get("id", "unknown")
                    # print(f"[Socket] >>> Sent to client [{client_id}]:")
                    # print(f"  {message[:150]}{'...' if len(message) > 150 else ''}")
                    
                except Exception as e:
                    print(f"[Socket] Failed to broadcast to client: {e}")
                    dead_clients.append(client)
            
            for client in dead_clients:
                client_id = self.clients.get(client, {}).get("id", "unknown")
                print(f"[Socket] Client {client_id} disconnected, removing...")
                del self.clients[client]
                try:
                    client.close()
                except:
                    pass
            
            if sent_count > 0:
                print(f"[Socket] Broadcast to {sent_count} client(s)")
    
    def get_client_count(self) -> int:
        """获取客户端数量"""
        with self.lock:
            return len(self.clients)
    
    def get_client_list(self) -> List[Dict]:
        """获取所有客户端信息列表"""
        with self.lock:
            return [
                {
                    "id": info.get("id"),
                    "filters": list(info.get("filters", [])),
                    "version": info.get("version"),
                    "connected_time": info.get("connected_time"),
                    # 心跳状态信息
                    "heartbeat_miss": info.get("heartbeat_miss", 0)
                }
                for info in self.clients.values()
            ]
    
    def stop(self):
        """停止服务器"""
        self.running = False
        
        # ========== 等待心跳线程结束 ==========
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2)
        # ====================================
        
        with self.lock:
            for client in list(self.clients.keys()):
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        
        if self.server_socket:
            self.server_socket.close()
        
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        print("[Socket] Server stopped")


# ============================================================
# 3. Motorola LSB
# ============================================================

class MotorolaLSBParser:
    """Motorola (@0) 格式信号解析器 - start_bit为MSB，逐位向LSB提取，跨字节换行"""
    @staticmethod
    def get_signal(data: bytes, start_bit: int, length: int) -> int:
        result = 0
        row = start_bit // 8
        col = start_bit % 8
        for i in range(length):
            if row >= len(data):
                break
            bit = (data[row] >> col) & 1
            result = (result << 1) | bit
            col -= 1
            if col < 0:
                col = 7
                row += 1
        return result


class MotorolaMSBParser:
    """
    Motorola MSB (大端) 信号解析器
    起始位：信号的最高有效位 (MSB) 位置
    跨字节时，低地址字节存放高位数据（大端字节序）
    """
    @staticmethod
    def get_signal(data: bytes, start_bit: int, length: int) -> int:
        """
        提取 Motorola 大端信号
        
        Args:
            data: CAN 帧数据字节数组
            start_bit: 起始位（信号的 MSB 所在位位置，位0为最低位）
            length: 信号长度（位）
        
        Returns:
            解析后的整数值（低地址字节的高位数据放在结果的高位）
        """
        result = 0
        # 确定起始字节和起始位偏移（位偏移从高到低）
        byte_idx = start_bit // 8
        bit_offset = start_bit % 8          # 大端：bit7是最高位，bit0是最低位
        bits_remaining = length
        
        while bits_remaining > 0 and byte_idx < len(data):
            # 当前字节可提取的位数：从 bit_offset 向下到 0
            bits_take = min(bits_remaining, bit_offset + 1)
            # 提取当前字节的高 bits_take 位
            mask = ((1 << bits_take) - 1) << (bit_offset - bits_take + 1)
            byte_val = (data[byte_idx] & mask) >> (bit_offset - bits_take + 1)
            # 将提取的字节片段拼接到结果末尾
            result = (result << bits_take) | byte_val
            # 更新剩余位数
            bits_remaining -= bits_take
            # 移动到下一个字节，起始位偏移重置为 7（下一个字节的最高位）
            byte_idx += 1
            bit_offset = 7
        
        return result





# ============================================================
# 4. 数据对象定义层
# ============================================================

@dataclass
class CANFrame:
    """CAN帧数据对象"""
    can_id: int
    dlc: int
    data: bytes
    timestamp_us: int
    bus_name: str = "can2"
    is_fd: bool = False
    
    def hex_dump(self) -> str:
        return ' '.join(f'{b:02X}' for b in self.data)
    
    def __repr__(self):
        return f"CANFrame(id=0x{self.can_id:03X}, dlc={self.dlc}, data={self.hex_dump()})"


@dataclass
class ParsedCANData:
    """解析后的CAN数据"""
    can_id: int
    timestamp_us: int
    parsed_signals: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default=None):
        return self.parsed_signals.get(key, default)
    
    def __repr__(self):
        signals = ', '.join([f"{k}={v}" for k, v in self.parsed_signals.items()])
        return f"ParsedData(id=0x{self.can_id:03X}, {signals})"




# ============================================================
# 5. CAN 报文解析器
# ============================================================

class CANParser:
    """CAN报文解析器 - 基于DBC定义"""
    
    # 获取温度字符串
    @classmethod
    def get_temp_string(cls, raw_value: int) -> str:
        """将原始温度值转换为温度字符串"""
        
        raw = raw_value & 0x1F  # 只取低5位
        
        # 根据 DBC 定义：温度 = 16.0 + raw × 0.5
        if raw == 0x00:
            return "LO"
        elif raw == 0x1F:
            return "HI"
        else:
            temp = 16.0 + raw * 0.5
            return f"{temp:.1f}"
    
    def __init__(self, parsed_ids: List[int]):
        self.parsed_ids = set(parsed_ids)
        self.parser = MotorolaLSBParser()
        print(f"[Parser] Will parse {len(self.parsed_ids)} CAN IDs")
    
    def should_parse(self, can_id: int) -> bool:
        return can_id in self.parsed_ids
    
    def parse(self, frame: CANFrame) -> Optional[ParsedCANData]:
        if not self.should_parse(frame.can_id):
            return None
        
        parsed_data = ParsedCANData(
            can_id=frame.can_id,
            timestamp_us=frame.timestamp_us
        )
        
        if frame.can_id == 0x387:
            self._parse_viu_ac_set1(frame, parsed_data)
        elif frame.can_id == 0x381:
            self._parse_viu_ac_set2(frame, parsed_data)
        elif frame.can_id == 0x33A:
            self._parse_ac_sts1(frame, parsed_data)
        elif frame.can_id == 0x33B:
            self._parse_ac_sts2(frame, parsed_data)
        elif frame.can_id == 0x33C:
            self._parse_ac_sts3(frame, parsed_data)
        elif frame.can_id == 0x35B:
            self._parse_viu_35b(frame, parsed_data)
        elif frame.can_id == 0x3B9:
            self._parse_viu_lin1(frame, parsed_data)
        elif frame.can_id == 0x33E:
            self._parse_ac_temp2(frame, parsed_data)
        elif frame.can_id == 0x33F:
            self._parse_ac_temp3(frame, parsed_data)
        elif frame.can_id == 0x338:
            self._parse_ac_temp1(frame, parsed_data)
        elif frame.can_id == 0x18F:
            self._parse_viu_cha_18f(frame, parsed_data)
        elif frame.can_id == 0x3BE:
            self._parse_viu_passive_sec_seat_3be(frame, parsed_data)
        elif frame.can_id == 0x371:
            self._parse_ac_temp5(frame, parsed_data)
        elif frame.can_id == 0x370:
            self._parse_ac_temp4(frame, parsed_data)
        elif frame.can_id == 0x541:
            self._parse_ac_rhvac(frame, parsed_data)
        elif frame.can_id == 0x552:
            self._parse_viu_dsm_552(frame, parsed_data)
        elif frame.can_id == 0x553:
            self._parse_viu_psm_553(frame, parsed_data)
        elif frame.can_id == 0x100:
            self._parse_c_100(frame, parsed_data)

        return parsed_data
    
    
    # ------------------- 0x387 解析 -------------------
    def _parse_viu_ac_set1(self, frame, result):
        data = frame.data
        if len(data) < 8: return
        byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7 = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]

        # CDC_ACSystemOnOffSet
        val = (byte0 >> 6) & 0x03
        result.parsed_signals["CDC_ACSystemOnOffSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_ACSystemOnOffSet_raw"] = val          # 新增

        # CDC_RACSystemOnOffSet
        val = (byte0 >> 4) & 0x03
        result.parsed_signals["CDC_RACSystemOnOffSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_RACSystemOnOffSet_raw"] = val

        # CDC_ACSet
        val = (byte0 >> 2) & 0x03
        result.parsed_signals["CDC_ACSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_ACSet_raw"] = val

        # CDC_FrontDefrostSet
        val = byte0 & 0x03
        result.parsed_signals["CDC_FrontDefrostSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_FrontDefrostSet_raw"] = val

        # CDC_FHvacBlowLvSet
        val_fan_front = (byte1 >> 4) & 0x0F
        fan_map = {0: "No Request", 1: "Level1", 2: "Level2", 3: "Level3", 4: "Level4", 5: "Level5", 6: "Level6", 7: "Level7", 8: "Level8", 9: "Level9", 10: "Level10"}
        result.parsed_signals["CDC_FHvacBlowLvSet"] = fan_map.get(val_fan_front, f"Reserved({val_fan_front})")
        result.parsed_signals["CDC_FHvacBlowLvSet_raw"] = val_fan_front

        # CDC_RHvacBlowLvSet
        val_fan_rear = byte1 & 0x0F
        result.parsed_signals["CDC_RHvacBlowLvSet"] = fan_map.get(val_fan_rear, f"Reserved({val_fan_rear})")
        result.parsed_signals["CDC_RHvacBlowLvSet_raw"] = val_fan_rear

        # CDC_FHvacModeSet
        val_mode_front = (byte2 >> 4) & 0x0F
        mode_map = {0: "NO Request", 1: "BlowFace", 2: "BlowFace&Foot", 3: "BlowFoot", 4: "BlowFoot&Windows", 5: "BlowWindows", 6: "BlowFace&Windows", 7: "BlowFace&Foot&Windows"}
        result.parsed_signals["CDC_FHvacModeSet"] = mode_map.get(val_mode_front, f"Reserved({val_mode_front})")
        result.parsed_signals["CDC_FHvacModeSet_raw"] = val_mode_front

        # CDC_RHvacModeSet
        val_mode_rear = (byte2 >> 1) & 0x07
        rear_mode_map = {0: "NO Request", 1: "BlowFace", 2: "BlowFace&Foot", 3: "BlowFoot"}
        result.parsed_signals["CDC_RHvacModeSet"] = rear_mode_map.get(val_mode_rear, f"Reserved({val_mode_rear})")
        result.parsed_signals["CDC_RHvacModeSet_raw"] = val_mode_rear

        # CDC_DriverTempCSet (已有 raw)
        high_bit = (byte2 >> 0) & 0x01
        low_4bits = (byte3 >> 4) & 0x0F
        raw_temp = (high_bit << 4) | low_4bits
        if raw_temp == 0x00:
            temp_value = "LO"
        elif raw_temp == 0x1F:
            temp_value = "HI"
        else:
            temp_calc = 16.0 + raw_temp * 0.5
            temp_value = f"{temp_calc:.1f}"
        result.parsed_signals["CDC_DriverTempCSet"] = temp_value
        result.parsed_signals["CDC_DriverTempCSet_raw"] = raw_temp

        # CDC_PassengerTempCSet
        high_4bits = byte3 & 0x0F
        low_bit = (byte4 >> 7) & 0x01
        raw_temp_pass = (high_4bits << 1) | low_bit
        if raw_temp_pass == 0x00:
            temp_value_pass = "LO"
        elif raw_temp_pass == 0x1F:
            temp_value_pass = "HI"
        else:
            temp_calc_pass = 16.0 + raw_temp_pass * 0.5
            temp_value_pass = f"{temp_calc_pass:.1f}"
        result.parsed_signals["CDC_PassengerTempCSet"] = temp_value_pass
        result.parsed_signals["CDC_PassengerTempCSet_raw"] = raw_temp_pass

        # CDC_RrTempCSet
        raw_temp_rear = (byte4 >> 2) & 0x1F
        if raw_temp_rear == 0x00:
            temp_value_rear = "LO"
        elif raw_temp_rear == 0x1F:
            temp_value_rear = "HI"
        else:
            temp_calc_rear = 16.0 + raw_temp_rear * 0.5
            temp_value_rear = f"{temp_calc_rear:.1f}"
        result.parsed_signals["CDC_RrTempCSet"] = temp_value_rear
        result.parsed_signals["CDC_RrTempCSet_raw"] = raw_temp_rear

        # TBOX_ACONOFFSet
        val = (byte5 >> 6) & 0x03
        result.parsed_signals["TBOX_ACONOFFSet"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["TBOX_ACONOFFSet_raw"] = val

        # TBOX_ACTempSet
        raw_temp = byte5 & 0x1F
        if raw_temp == 0x00:
            temp_value = "LO"
        elif raw_temp == 0x1E:
            temp_value = "HI"
        elif raw_temp == 0x1F:
            temp_value = "INVALID"
        else:
            temp_calc = 16.0 + raw_temp * 0.5
            temp_value = f"{temp_calc:.1f}"
        result.parsed_signals["TBOX_ACTempSet"] = temp_value
        result.parsed_signals["TBOX_ACTempSet_raw"] = raw_temp

        # TBOX_ACVentSet
        val = (byte6 >> 6) & 0x03
        result.parsed_signals["TBOX_ACVentSet"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["TBOX_ACVentSet_raw"] = val

        # TBOX_ACDefrostSet
        val = (byte6 >> 4) & 0x03
        result.parsed_signals["TBOX_ACDefrostSet"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["TBOX_ACDefrostSet_raw"] = val

        # TBOX_MaxACSet
        val = (byte6 >> 2) & 0x03
        result.parsed_signals["TBOX_MaxACSet"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["TBOX_MaxACSet_raw"] = val

        # TBOX_MaxHeatSet
        val = byte6 & 0x03
        result.parsed_signals["TBOX_MaxHeatSet"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["TBOX_MaxHeatSet_raw"] = val

        # TBOX_PurificationSet
        val = (byte7 >> 6) & 0x03
        result.parsed_signals["TBOX_PurificationSet"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["TBOX_PurificationSet_raw"] = val

        # TBOX_UVCSet
        val = (byte7 >> 4) & 0x03
        result.parsed_signals["TBOX_UVCSet"] = {0: "NO Request", 1: "OFF", 2: "Level1", 3: "Level2"}.get(val, "UNKNOWN")
        result.parsed_signals["TBOX_UVCSet_raw"] = val

        # CDC_ThirdRowACSystemOnOffSet
        val = (byte7 >> 2) & 0x03
        result.parsed_signals["CDC_ThirdRowACSystemOnOffSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_ThirdRowACSystemOnOffSet_raw"] = val

    
    # ------------------- 0x381 解析-------------------
    def _parse_viu_ac_set2(self, frame, result):
        data = frame.data
        if len(data) < 8: return
        byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7 = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
        
        # 为每一个信号添加 _raw，例如：
        val = (byte0 >> 6) & 0x03
        result.parsed_signals["CDC_ACMaxSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_ACMaxSet_raw"] = val

        val = (byte0 >> 4) & 0x03
        result.parsed_signals["CDC_HeatMaxSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_HeatMaxSet_raw"] = val

        val = (byte0 >> 2) & 0x03
        result.parsed_signals["CDC_HvacAutoModeSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_HvacAutoModeSet_raw"] = val

        val = byte0 & 0x03
        result.parsed_signals["CDC_RHvacAutoModeSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_RHvacAutoModeSet_raw"] = val

        val = (byte1 >> 6) & 0x03
        result.parsed_signals["CDC_AutoDefogSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_AutoDefogSet_raw"] = val

        val = (byte1 >> 4) & 0x03
        result.parsed_signals["CDC_ACFourZoneSync"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_ACFourZoneSync_raw"] = val

        val = (byte1 >> 2) & 0x03
        result.parsed_signals["CDC_TempExclusive"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_TempExclusive_raw"] = val

        val = byte1 & 0x03
        result.parsed_signals["CDC_CircModSet"] = {0: "NO Request", 1: "Internal", 2: "External", 3: "AUTO"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_CircModSet_raw"] = val

        val = (byte2 >> 6) & 0x03
        result.parsed_signals["CDC_IntelligentizePurSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_IntelligentizePurSet_raw"] = val

        val = (byte2 >> 4) & 0x03
        result.parsed_signals["CDC_ACParkAutoVentSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_ACParkAutoVentSet_raw"] = val

        val = (byte2 >> 2) & 0x03
        result.parsed_signals["CDC_IONset"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_IONset_raw"] = val

        val = byte2 & 0x03
        result.parsed_signals["CDC_UVCset"] = {0: "NO Request", 1: "OFF", 2: "Level1", 3: "Level2"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_UVCset_raw"] = val

        val = (byte3 >> 4) & 0x01
        result.parsed_signals["CDC_FilterRenewSet"] = {0: "NO request", 1: "Renew"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_FilterRenewSet_raw"] = val

        val = (byte3 >> 2) & 0x03
        result.parsed_signals["CDC_AutoDrySet"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_AutoDrySet_raw"] = val

        val = byte3 & 0x03
        result.parsed_signals["CDC_ACECOModReq"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_ACECOModReq_raw"] = val

        val = (byte4 >> 6) & 0x03
        result.parsed_signals["CDC_VentSet"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_VentSet_raw"] = val

        val = (byte4 >> 4) & 0x03
        result.parsed_signals["CDC_OdorRemoveSet"] = {0: "NO Request", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_OdorRemoveSet_raw"] = val

        val = (byte5 >> 4) & 0x0F
        fan_map = {0: "No Request", 1: "Level1", 2: "Level2", 3: "Level3", 4: "Level4", 5: "Level5", 6: "Level6", 7: "Level7"}
        result.parsed_signals["CDC_ThirdRowHvacBlowLvSet"] = fan_map.get(val, f"Reserved({val})")
        result.parsed_signals["CDC_ThirdRowHvacBlowLvSet_raw"] = val

        val = (byte5 >> 1) & 0x07
        mode_map = {0: "NO Request", 1: "FACE", 2: "FACE_AND_FOOT", 3: "FOOT"}
        result.parsed_signals["CDC_ThirdRowHvacModeSet"] = mode_map.get(val, f"Reserved({val})")
        result.parsed_signals["CDC_ThirdRowHvacModeSet_raw"] = val

        raw_temp = (byte6 >> 3) & 0x1F
        if raw_temp == 0x00:
            temp_value = "LO"
        elif raw_temp == 0x1F:
            temp_value = "HI"
        else:
            temp_calc = 16.0 + raw_temp * 0.5
            temp_value = f"{temp_calc:.1f}"
        result.parsed_signals["CDC_ThirdRowTempCSet"] = temp_value
        result.parsed_signals["CDC_ThirdRowTempCSet_raw"] = raw_temp

        val = (byte6 >> 1) & 0x03
        result.parsed_signals["CDC_ThirdRowHvacAutoModeSet"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["CDC_ThirdRowHvacAutoModeSet_raw"] = val

        val = (byte7 >> 6) & 0x03
        result.parsed_signals["TBOX_ACParkAutoVentSetSts"] = {0: "Initial", 1: "OFF", 2: "ON"}.get(val, "UNKNOWN")
        result.parsed_signals["TBOX_ACParkAutoVentSetSts_raw"] = val
    
    
    # ------------------- 0x33A 解析 -------------------
    def _parse_ac_sts1(self, frame, result):
        data = frame.data
        if len(data) < 8: return
        byte0, byte1, byte2, byte3, byte4, byte5, byte6 = data[0], data[1], data[2], data[3], data[4], data[5], data[6]

        val = (byte0 >> 6) & 0x03
        result.parsed_signals["AC_ACSystemOnOffSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_ACSystemOnOffSts_raw"] = val

        val = (byte0 >> 4) & 0x03
        result.parsed_signals["AC_RACSystemOnOffSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_RACSystemOnOffSts_raw"] = val

        val = (byte0 >> 2) & 0x03
        result.parsed_signals["AC_ACSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_ACSts_raw"] = val

        val = byte0 & 0x03
        result.parsed_signals["AC_FrontDefrostSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_FrontDefrostSts_raw"] = val

        val = (byte1 >> 4) & 0x0F
        if val == 0:
            result.parsed_signals["AC_FBlowSpeedLevel"] = "Initial"
        elif val == 0xF:
            result.parsed_signals["AC_FBlowSpeedLevel"] = "Invalid"
        else:
            result.parsed_signals["AC_FBlowSpeedLevel"] = f"Level{val}"
        result.parsed_signals["AC_FBlowSpeedLevel_raw"] = val

        val = byte1 & 0x0F
        if val == 0:
            result.parsed_signals["AC_RBlowSpeedLevel"] = "Initial"
        elif val == 0xF:
            result.parsed_signals["AC_RBlowSpeedLevel"] = "Invalid"
        else:
            result.parsed_signals["AC_RBlowSpeedLevel"] = f"Level{val}"
        result.parsed_signals["AC_RBlowSpeedLevel_raw"] = val

        val = (byte2 >> 4) & 0x0F
        mode_map = {0: "Initial", 1: "BlowFace", 2: "BlowFace&Foot", 3: "BlowFoot", 4: "BlowFoot&Windows", 5: "BlowWindows", 6: "BlowFace&Windows", 7: "BlowFace&Foot&Windows", 0xF: "Invalid"}
        result.parsed_signals["AC_FModeAdjustSts"] = mode_map.get(val, f"Reserved({val})")
        result.parsed_signals["AC_FModeAdjustSts_raw"] = val

        val = (byte2 >> 1) & 0x07
        rear_mode_map = {0: "Initial", 1: "BlowFace", 2: "BlowFace&Foot", 3: "BlowFoot", 7: "INVALID"}
        result.parsed_signals["AC_RModeAdjustSts"] = rear_mode_map.get(val, f"Reserved({val})")
        result.parsed_signals["AC_RModeAdjustSts_raw"] = val

        # ---------- 温度信号 ----------
        # AC_DriverTempC: MSB 在 byte2.0，其余位在 byte3.7-4
        raw_temp = ((byte2 & 0x01) << 4) | ((byte3 >> 4) & 0x0F)
        result.parsed_signals["AC_DriverTempC"] = self.get_temp_string(raw_temp)
        result.parsed_signals["AC_DriverTempC_raw"] = raw_temp

        # AC_RrTempC: 完全在 byte4.2-6
        raw_temp = (byte4 >> 2) & 0x1F
        result.parsed_signals["AC_RrTempC"] = self.get_temp_string(raw_temp)
        result.parsed_signals["AC_RrTempC_raw"] = raw_temp

        # AC_PassengerTempC: 高4位(bit4..bit1)在 byte3.3-0，最低位(bit0)在 byte4.7
        raw_temp_pass = ((byte3 & 0x0F) << 1) | ((byte4 >> 7) & 0x01)
        result.parsed_signals["AC_PassengerTempC"] = self.get_temp_string(raw_temp_pass)
        result.parsed_signals["AC_PassengerTempC_raw"] = raw_temp_pass

        # AC_CircleModeSts: DBC start=33 len=3
        val = MotorolaLSBParser.get_signal(data, 33, 3)
        circ_map = {0: "Initial", 1: "Internal", 2: "External", 3: "Auto", 7: "Invalid"}
        result.parsed_signals["AC_CircleModeSts"] = circ_map.get(val, f"Reserved({val})")
        result.parsed_signals["AC_CircleModeSts_raw"] = val

        val = (byte5 >> 5) & 0x03
        result.parsed_signals["AC_ParkAutoVentSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_ParkAutoVentSts_raw"] = val

        val = (byte5 >> 2) & 0x01
        result.parsed_signals["AC_FHvacBlowLvAutoSts"] = "Auto" if val == 0 else "Not Auto"
        result.parsed_signals["AC_FHvacBlowLvAutoSts_raw"] = val

        val = (byte5 >> 1) & 0x01
        result.parsed_signals["AC_RHvacBlowLvAutoSts"] = "Auto" if val == 0 else "Not Auto"
        result.parsed_signals["AC_RHvacBlowLvAutoSts_raw"] = val

        val = byte5 & 0x01
        result.parsed_signals["AC_FBlowModeAutoSts"] = "Auto" if val == 0 else "Not Auto"
        result.parsed_signals["AC_FBlowModeAutoSts_raw"] = val

        val = (byte6 >> 7) & 0x01
        result.parsed_signals["AC_RBlowModeAutoSts"] = "Auto" if val == 0 else "Not Auto"
        result.parsed_signals["AC_RBlowModeAutoSts_raw"] = val

        val = (byte6 >> 5) & 0x03
        result.parsed_signals["AC_ThirdRowACSystemOnOffSts"] = {0: "INITIAL", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_ThirdRowACSystemOnOffSts_raw"] = val

        val = (byte6 >> 4) & 0x01
        result.parsed_signals["AC_ThirdRowHvacBlowLvAutoSts"] = "Auto" if val == 0 else "Not Auto"
        result.parsed_signals["AC_ThirdRowHvacBlowLvAutoSts_raw"] = val

        val = (byte6 >> 3) & 0x01
        result.parsed_signals["AC_ThirdRowBlowModeAutoSts"] = "Auto" if val == 0 else "Not Auto"
        result.parsed_signals["AC_ThirdRowBlowModeAutoSts_raw"] = val

    # ------------------- 0x33B 解析 -------------------
    def _parse_ac_sts2(self, frame, result):
        data = frame.data
        if len(data) < 8: return
        byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7 = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]

        val = (byte0 >> 6) & 0x03
        result.parsed_signals["AC_ACMaxSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_ACMaxSts_raw"] = val

        val = (byte0 >> 4) & 0x03
        result.parsed_signals["AC_HeatMaxSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_HeatMaxSts_raw"] = val

        val = (byte0 >> 2) & 0x03
        result.parsed_signals["AC_AutoModeSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_AutoModeSts_raw"] = val

        val = byte0 & 0x03
        result.parsed_signals["AC_RAutoModeSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_RAutoModeSts_raw"] = val

        val = (byte1 >> 6) & 0x03
        result.parsed_signals["AC_AutoDefogSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_AutoDefogSts_raw"] = val

        val = (byte1 >> 4) & 0x03
        result.parsed_signals["AC_ACFourZoneSyncSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_ACFourZoneSyncSts_raw"] = val

        val = (byte1 >> 2) & 0x03
        result.parsed_signals["AC_TempExclusiveSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_TempExclusiveSts_raw"] = val

        val = byte1 & 0x03
        result.parsed_signals["AC_AutoDefogRunSts"] = {0: "initial", 1: "Running", 2: "STOP", 3: "invaild"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_AutoDefogRunSts_raw"] = val

        val = (byte2 >> 4) & 0x0F
        fault_map = {0: "NORMAL", 1: "Voltage Fault", 2: "Sensor Fault", 3: "CCU Fault", 0xF: "Invalid"}
        result.parsed_signals["AC_FaultREASON"] = fault_map.get(val, f"Reserved({val})")
        result.parsed_signals["AC_FaultREASON_raw"] = val

        # AC_Pres_HP 
        raw = MotorolaLSBParser.get_signal(data, 19, 9)
        if raw != 0x1FF:
            result.parsed_signals["AC_Pres_HP"] = round(raw * 0.1, 1)
            result.parsed_signals["AC_Pres_HP_unit"] = "Bar"
        else:
            result.parsed_signals["AC_Pres_HP"] = "Invalid"
        result.parsed_signals["AC_Pres_HP_raw"] = raw

        # AC_Pres_LPT_Chiller
        raw = MotorolaLSBParser.get_signal(data, 26, 7)
        if raw != 0x7F:
            result.parsed_signals["AC_Pres_LPT_Chiller"] = round(raw * 0.1, 1)
            result.parsed_signals["AC_Pres_LPT_Chiller_unit"] = "Bar"
        else:
            result.parsed_signals["AC_Pres_LPT_Chiller"] = "Invalid"
        result.parsed_signals["AC_Pres_LPT_Chiller_raw"] = raw

        # AC_Pres_LPT_EVAP
        raw = MotorolaLSBParser.get_signal(data, 35, 7)
        if raw != 0x7F:
            result.parsed_signals["AC_Pres_LPT_EVAP"] = round(raw * 0.1, 1)
            result.parsed_signals["AC_Pres_LPT_EVAP_unit"] = "Bar"
        else:
            result.parsed_signals["AC_Pres_LPT_EVAP"] = "Invalid"
        result.parsed_signals["AC_Pres_LPT_EVAP_raw"] = raw

        val = (byte6 >> 4) & 0x0F
        if val == 0:
            result.parsed_signals["AC_ThirdRowBlowSpeedLevel"] = "No Request"
        elif val <= 7:
            result.parsed_signals["AC_ThirdRowBlowSpeedLevel"] = f"Level{val}"
        else:
            result.parsed_signals["AC_ThirdRowBlowSpeedLevel"] = f"Reserved({val})"
        result.parsed_signals["AC_ThirdRowBlowSpeedLevel_raw"] = val

        val = (byte6 >> 1) & 0x07
        mode_map = {0: "Initial", 1: "BlowFace", 2: "BlowFace&Foot", 3: "BlowFoot"}
        result.parsed_signals["AC_ThirdRowModeAdjustSts"] = mode_map.get(val, f"Reserved({val})")
        result.parsed_signals["AC_ThirdRowModeAdjustSts_raw"] = val

        raw_temp = (byte7 >> 3) & 0x1F
        result.parsed_signals["AC_ThirdRowTempC"] = self.get_temp_string(raw_temp)
        result.parsed_signals["AC_ThirdRowTempC_raw"] = raw_temp

        val = (byte7 >> 1) & 0x03
        result.parsed_signals["AC_ThirdRowAutoModeSts"] = {0: "NO Request", 1: "OFF", 2: "ON", 3: "INVALID"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_ThirdRowAutoModeSts_raw"] = val

    # ------------------- 0x33C 解析 -------------------
    def _parse_ac_sts3(self, frame, result):
        data = frame.data
        if len(data) < 8: return
        byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7 = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]

        val = (byte0 >> 6) & 0x03
        result.parsed_signals["AC_TimeVentSt"] = {0: "Initial", 1: "Open", 2: "Close", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_TimeVentSt_raw"] = val

        val = (byte0 >> 4) & 0x03
        result.parsed_signals["AC_VentTimeFlag"] = {0: "Initial", 1: "未到期", 2: "已到期", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_VentTimeFlag_raw"] = val

        raw = MotorolaLSBParser.get_signal(data, 11, 9)
        if raw != 0x1FF:
            result.parsed_signals["AC_VentholdTime"] = raw
            result.parsed_signals["AC_VentholdTime_unit"] = "s"
        result.parsed_signals["AC_VentholdTime_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 16, 10)
        if raw <= 0x3FE:
            result.parsed_signals["AC_VentTime"] = raw
        result.parsed_signals["AC_VentTime_raw"] = raw

        val = byte3
        if val != 0:
            result.parsed_signals["AC_BlowerVolt"] = round(val * 0.1, 1)
            result.parsed_signals["AC_BlowerVolt_unit"] = "V"
        result.parsed_signals["AC_BlowerVolt_raw"] = val

        val = byte4
        if val != 0:
            result.parsed_signals["AC_RBlowerVolt"] = round(val * 0.1, 1)
            result.parsed_signals["AC_RBlowerVolt_unit"] = "V"
        result.parsed_signals["AC_RBlowerVolt_raw"] = val

        val = (byte5 >> 6) & 0x03
        result.parsed_signals["AC_UVCSts"] = {0: "Initial", 1: "OFF", 2: "Level1", 3: "Level2"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_UVCSts_raw"] = val

        val = (byte5 >> 5) & 0x01
        result.parsed_signals["AC_UVC_Err"] = "Error" if val == 1 else "Normal"
        result.parsed_signals["AC_UVC_Err_raw"] = val

        val = (byte5 >> 4) & 0x01
        result.parsed_signals["AC_UVC_ErrValid"] = "Valid" if val == 1 else "Invalid"
        result.parsed_signals["AC_UVC_ErrValid_raw"] = val

        val = (byte5 >> 2) & 0x03
        result.parsed_signals["AC_AutoDrySts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_AutoDrySts_raw"] = val

        val = byte5 & 0x03
        result.parsed_signals["AC_ACECOModSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Invalid"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_ACECOModSts_raw"] = val

        val = (byte6 >> 6) & 0x03
        result.parsed_signals["AC_VentSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Reserved"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_VentSts_raw"] = val

        val = (byte6 >> 4) & 0x03
        result.parsed_signals["AC_RemFaultSts"] = {0: "Normal", 1: "Fault", 2: "Reserved", 3: "Reserved"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_RemFaultSts_raw"] = val

        val = (byte6 >> 2) & 0x03
        result.parsed_signals["AC_OdorRemoveSts"] = {0: "Initial", 1: "OFF", 2: "ON", 3: "Reserved"}.get(val, "UNKNOWN")
        result.parsed_signals["AC_OdorRemoveSts_raw"] = val

        val = byte7 & 0x7F
        if val == 0x7E:
            result.parsed_signals["AC_FilterRemain"] = "Initial"
        elif val == 0x7F:
            result.parsed_signals["AC_FilterRemain"] = "Invalid"
        else:
            result.parsed_signals["AC_FilterRemain"] = f"{val}%"
        result.parsed_signals["AC_FilterRemain_raw"] = val


    # ==================== 0x35B VIU_35B 解析 ====================
    def _parse_viu_35b(self, frame: CANFrame, result: ParsedCANData):
        """解析 VIU_35B_35B (环境温度/车窗状态)"""
        data = frame.data
        if len(data) < 8:
            return
        
        byte0 = data[0]
        byte1 = data[1]
        byte2 = data[2]
        byte3 = data[3]
        byte4 = data[4]
        byte5 = data[5]
        byte6 = data[6]
        byte7 = data[7]

        # VIU_AmbTVld: 起始位1, len=1 -> byte0 的 bit1
        val = MotorolaLSBParser.get_signal(data, 9, 1)
        result.parsed_signals["VIU_AmbTVld"] = {0: "Invalid", 1: "Valid"}.get(val, "Unknown")
        result.parsed_signals["VIU_AmbTVld_raw"] = val

        # VIU_AmbT: 起始位3, len=11 (Motorola LSB)
        raw_temp = MotorolaLSBParser.get_signal(data, 23, 11)
        temp_value = raw_temp * 0.1 - 40
        result.parsed_signals["VIU_AmbT"] = round(temp_value, 1)
        result.parsed_signals["VIU_AmbT_raw"] = raw_temp

        # VIU_VehAntithftSt: 起始位26, len=3 -> byte3 的 bit2-4
        val = (byte3 >> 2) & 0x07
        antithft_map = {
            0: "unset",
            1: "set",
            2: "part_set",
            3: "alarm",
            4: "pre_set",
            5: "reserved",
            6: "reserved",
            7: "invalid"
        }
        result.parsed_signals["VIU_VehAntithftSt"] = antithft_map.get(val, "reserved")
        result.parsed_signals["VIU_VehAntithftSt_raw"] = val

        # CDC_SceneModePowerOnSts: 起始位24, len=2 -> byte3 的 bit0-1
        val = byte3 & 0x03
        scene_map = {
            0: "INVALID",
            1: "ON",
            2: "ON_REMOTE",
            3: "OFF"
        }
        result.parsed_signals["CDC_SceneModePowerOnSts"] = scene_map.get(val, "UNKNOWN")
        result.parsed_signals["CDC_SceneModePowerOnSts_raw"] = val

        # VIU_FLWinOpenDeg: 起始位32, len=7 -> byte4 bit0-6
        val = byte4 & 0x7F
        if val == 0x00:
            result.parsed_signals["VIU_FLWinOpenDeg"] = "Initial"
        elif val == 0x01:
            result.parsed_signals["VIU_FLWinOpenDeg"] = "0%"
        elif val == 0x65:
            result.parsed_signals["VIU_FLWinOpenDeg"] = "100%"
        elif val == 0x7F:
            result.parsed_signals["VIU_FLWinOpenDeg"] = "Invalid"
        elif 0x66 <= val <= 0x7E:
            result.parsed_signals["VIU_FLWinOpenDeg"] = "Reserved"
        else:
            # val 范围 2~0x64 (2~100) 对应 1%~99%
            result.parsed_signals["VIU_FLWinOpenDeg"] = f"{val - 1}%"
        result.parsed_signals["VIU_FLWinOpenDeg_raw"] = val

        # VIU_FRWinOpenDeg: 起始位40, len=7 -> byte5 bit0-6
        val = byte5 & 0x7F
        if val == 0x00:
            result.parsed_signals["VIU_FRWinOpenDeg"] = "Initial"
        elif val == 0x01:
            result.parsed_signals["VIU_FRWinOpenDeg"] = "0%"
        elif val == 0x65:
            result.parsed_signals["VIU_FRWinOpenDeg"] = "100%"
        elif val == 0x7F:
            result.parsed_signals["VIU_FRWinOpenDeg"] = "Invalid"
        elif 0x66 <= val <= 0x7E:
            result.parsed_signals["VIU_FRWinOpenDeg"] = "Reserved"
        else:
            result.parsed_signals["VIU_FRWinOpenDeg"] = f"{val - 1}%"
        result.parsed_signals["VIU_FRWinOpenDeg_raw"] = val

        # VIU_RLWinOpenDeg: 起始位48, len=7 -> byte6 bit0-6
        val = byte6 & 0x7F
        if val == 0x00:
            result.parsed_signals["VIU_RLWinOpenDeg"] = "Initial"
        elif val == 0x01:
            result.parsed_signals["VIU_RLWinOpenDeg"] = "0%"
        elif val == 0x65:
            result.parsed_signals["VIU_RLWinOpenDeg"] = "100%"
        elif val == 0x7F:
            result.parsed_signals["VIU_RLWinOpenDeg"] = "Invalid"
        elif 0x66 <= val <= 0x7E:
            result.parsed_signals["VIU_RLWinOpenDeg"] = "Reserved"
        else:
            result.parsed_signals["VIU_RLWinOpenDeg"] = f"{val - 1}%"
        result.parsed_signals["VIU_RLWinOpenDeg_raw"] = val

        # VIU_RRWinOpenDeg: 起始位56, len=7 -> byte7 bit0-6
        val = byte7 & 0x7F
        if val == 0x00:
            result.parsed_signals["VIU_RRWinOpenDeg"] = "Initial"
        elif val == 0x01:
            result.parsed_signals["VIU_RRWinOpenDeg"] = "0%"
        elif val == 0x65:
            result.parsed_signals["VIU_RRWinOpenDeg"] = "100%"
        elif val == 0x7F:
            result.parsed_signals["VIU_RRWinOpenDeg"] = "Invalid"
        elif 0x66 <= val <= 0x7E:
            result.parsed_signals["VIU_RRWinOpenDeg"] = "Reserved"
        else:
            result.parsed_signals["VIU_RRWinOpenDeg"] = f"{val - 1}%"
        result.parsed_signals["VIU_RRWinOpenDeg_raw"] = val

    # ==================== 0x3B9 VIU_LIN1 解析 ====================
    def _parse_viu_lin1(self, frame: CANFrame, result: ParsedCANData):
        """解析 VIU_LIN1_3B9 (LIN信号：续航SOC/遮阳帘/雨量/温湿度/光照强度)"""
        data = frame.data
        if len(data) < 8:
            return
        
        byte0 = data[0]
        byte1 = data[1]
        byte2 = data[2]
        byte3 = data[3]
        byte4 = data[4]
        byte5 = data[5]
        byte6 = data[6]
        byte7 = data[7]

        # BMS_PackSOCRange: DBC start=7 len=10 (Motorola LSB)
        raw_soc = MotorolaLSBParser.get_signal(data, 7, 10)
        if raw_soc == 0x3FE:
            result.parsed_signals["BMS_PackSOCRange"] = "Initial"
        elif raw_soc == 0x3FF:
            result.parsed_signals["BMS_PackSOCRange"] = "Invalid"
        else:
            soc_value = raw_soc * 0.1
            result.parsed_signals["BMS_PackSOCRange"] = round(soc_value, 1)
        result.parsed_signals["BMS_PackSOCRange_raw"] = raw_soc

        # SSM_F_Position: DBC start=13 len=3 (Motorola LSB)
        raw_pos = MotorolaLSBParser.get_signal(data, 13, 3)
        pos_map = {
            0: "Reserved",
            1: "First half of slide",
            2: "Second half of slide",
            3: "Fully close",
            4: "Half open",
            5: "Reversed",
            6: "Fully open",
            7: "Uninitialized"
        }
        result.parsed_signals["SSM_F_Position"] = pos_map.get(raw_pos, "Unknown")
        result.parsed_signals["SSM_F_Position_raw"] = raw_pos

        # RSM_DewPointT: DBC start=10 len=11 (Motorola LSB)
        raw_dew = MotorolaLSBParser.get_signal(data, 10, 11)
        if raw_dew == 0x7FF:
            result.parsed_signals["RSM_DewPointT"] = "Invalid"
        else:
            dew_value = raw_dew * 0.1 - 39.6
            result.parsed_signals["RSM_DewPointT"] = round(dew_value, 1)
        result.parsed_signals["RSM_DewPointT_raw"] = raw_dew

        # RSM_RainFalLev: DBC start=30 len=4 (Motorola LSB)
        raw_rain = MotorolaLSBParser.get_signal(data, 30, 4)
        rain_map = {
            0: "No Rain",
            1: "Level1",
            2: "Level2",
            3: "Level3",
            4: "Level4",
            5: "Level5",
            6: "Level6",
            7: "Level7",
            8: "Level8",
            9: "Level9",
            10: "Level10",
            11: "Level11",
            12: "Level12",
            13: "Level13",
            14: "Reversed",
            15: "Invalid"
        }
        result.parsed_signals["RSM_RainFalLev"] = rain_map.get(raw_rain, "Unknown")
        result.parsed_signals["RSM_RainFalLev_raw"] = raw_rain

        # RSM_WinT: DBC start=26 len=11 (Motorola LSB)
        raw_win_temp = MotorolaLSBParser.get_signal(data, 26, 11)
        if raw_win_temp == 0x7FF:
            result.parsed_signals["RSM_WinT"] = "Invalid"
        else:
            win_temp_value = raw_win_temp * 0.1 - 39.6
            result.parsed_signals["RSM_WinT"] = round(win_temp_value, 1)
        result.parsed_signals["RSM_WinT_raw"] = raw_win_temp


        # RSM_RelHum: DBC start=47 len=8 (Motorola LSB)
        raw_hum = MotorolaLSBParser.get_signal(data, 47, 8)
        if raw_hum == 0xFF:
            result.parsed_signals["RSM_RelHum"] = "Invalid"
        else:
            hum_value = raw_hum * 0.5 - 0.5
            result.parsed_signals["RSM_RelHum"] = round(hum_value, 1)
        result.parsed_signals["RSM_RelHum_raw"] = raw_hum

        # RSM_LeSolarInten: DBC start=55 len=8 (Motorola LSB)
        raw_left_solar = MotorolaLSBParser.get_signal(data, 55, 8)
        if raw_left_solar == 0xFE:
            result.parsed_signals["RSM_LeSolarInten"] = "Initial"
        elif raw_left_solar == 0xFF:
            result.parsed_signals["RSM_LeSolarInten"] = "Invalid"
        else:
            left_solar_value = raw_left_solar * 5
            result.parsed_signals["RSM_LeSolarInten"] = left_solar_value
        result.parsed_signals["RSM_LeSolarInten_raw"] = raw_left_solar

        # RSM_RiSolarInten: DBC start=63 len=8 (Motorola LSB)
        raw_right_solar = MotorolaLSBParser.get_signal(data, 63, 8)
        if raw_right_solar == 0xFE:
            result.parsed_signals["RSM_RiSolarInten"] = "Initial"
        elif raw_right_solar == 0xFF:
            result.parsed_signals["RSM_RiSolarInten"] = "Invalid"
        else:
            right_solar_value = raw_right_solar * 5
            result.parsed_signals["RSM_RiSolarInten"] = right_solar_value
        result.parsed_signals["RSM_RiSolarInten_raw"] = raw_right_solar


    # ==================== 0x33E AC_Temp2 解析 ====================
    def _parse_ac_temp2(self, frame: CANFrame, result: ParsedCANData):
        """解析 AC_Temp2_33E (蒸发器温度/温湿度/制冷采暖等级)"""
        data = frame.data
        if len(data) < 8:
            return
        
        byte0 = data[0]
        byte1 = data[1]
        byte2 = data[2]
        byte3 = data[3]
        byte4 = data[4]
        byte5 = data[5]
        byte6 = data[6]
        byte7 = data[7]

        # AC_FEvapTargetTemp: DBC start=7 len=6 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 7, 6)
        value = raw * 0.5
        result.parsed_signals["AC_FEvapTargetTemp"] = round(value, 1)
        result.parsed_signals["AC_FEvapTargetTemp_raw"] = raw

        # AC_FEvapCurrentTemp: DBC start=1 len=10 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 1, 10)
        value = raw * 0.1 - 40
        result.parsed_signals["AC_FEvapCurrentTemp"] = round(value, 1)
        result.parsed_signals["AC_FEvapCurrentTemp_raw"] = raw

        # AC_REvapTargetTemp: DBC start=23 len=6 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 23, 6)
        value = raw * 0.5
        result.parsed_signals["AC_REvapTargetTemp"] = round(value, 1)
        result.parsed_signals["AC_REvapTargetTemp_raw"] = raw

        # AC_REvapCurrentTemp: DBC start=17 len=10 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 17, 10)
        value = raw * 0.1 - 40
        result.parsed_signals["AC_REvapCurrentTemp"] = round(value, 1)
        result.parsed_signals["AC_REvapCurrentTemp_raw"] = raw

        # AC_THS_RelHum: 起始位32, len=8 (Motorola LSB)
        # DBC start=39 len=8 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 39, 8)
        value = raw * 0.5
        result.parsed_signals["AC_THS_RelHum"] = round(value, 1)
        result.parsed_signals["AC_THS_RelHum_raw"] = raw

        # AC_THS_DewPointT: DBC start=47 len=8 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 47, 8)
        value = raw * 0.5 - 40
        result.parsed_signals["AC_THS_DewPointT"] = round(value, 1)
        result.parsed_signals["AC_THS_DewPointT_raw"] = raw

        # AC_THS_T: DBC start=55 len=8 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 55, 8)
        value = raw * 0.5 - 40
        result.parsed_signals["AC_THS_T"] = round(value, 1)
        result.parsed_signals["AC_THS_T_raw"] = raw

        # AC_cabinCoolingLevel: DBC start=58 len=3 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 58, 3)
        cooling_map = {
            0: "STOP",
            1: "冷却等级1",
            2: "冷却等级2",
            3: "冷却等级3",
            4: "冷却等级4",
            5: "冷却等级5",
            6: "冷却等级6",
            7: "Reserved"
        }
        result.parsed_signals["AC_cabinCoolingLevel"] = cooling_map.get(raw, "Unknown")
        result.parsed_signals["AC_cabinCoolingLevel_raw"] = raw

        # AC_cabinheatingLevel: DBC start=61 len=3 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 61, 3)
        heating_map = {
            0: "STOP",
            1: "采暖等级1",
            2: "采暖等级2",
            3: "采暖等级3",
            4: "采暖等级4",
            5: "采暖等级5",
            6: "采暖等级6",
            7: "Reserved"
        }
        result.parsed_signals["AC_cabinheatingLevel"] = heating_map.get(raw, "Unknown")
        result.parsed_signals["AC_cabinheatingLevel_raw"] = raw

    # ==================== 0x33F AC_Temp3 解析 ====================
    def _parse_ac_temp3(self, frame: CANFrame, result: ParsedCANData):
        """解析 AC_Temp3_33F (吹面出风温度)"""
        data = frame.data
        if len(data) < 8:
            return
        
        byte0 = data[0]
        byte1 = data[1]
        byte2 = data[2]
        byte3 = data[3]
        byte4 = data[4]
        byte5 = data[5]
        byte6 = data[6]
        byte7 = data[7]

        # AC_DrvrFaceVentTargetT: 起始位0, len=8 (Motorola LSB)
        raw = byte0 & 0xFF
        value = raw * 0.5 - 40
        result.parsed_signals["AC_DrvrFaceVentTargetT"] = round(value, 1)
        result.parsed_signals["AC_DrvrFaceVentTargetT_raw"] = raw

        # AC_PassFaceVentTargetT: 起始位8, len=8 (Motorola LSB)
        raw = byte1 & 0xFF
        value = raw * 0.5 - 40
        result.parsed_signals["AC_PassFaceVentTargetT"] = round(value, 1)
        result.parsed_signals["AC_PassFaceVentTargetT_raw"] = raw

        # AC_SecRowFaceVentTargetT: 起始位16, len=8 (Motorola LSB)
        raw = byte2 & 0xFF
        value = raw * 0.5 - 40
        result.parsed_signals["AC_SecRowFaceVentTargetT"] = round(value, 1)
        result.parsed_signals["AC_SecRowFaceVentTargetT_raw"] = raw

        # AC_DrvrFaceVentActT: 起始位24, len=8 (Motorola LSB)
        raw = byte3 & 0xFF
        value = raw * 0.5 - 40
        result.parsed_signals["AC_DrvrFaceVentActT"] = round(value, 1)
        result.parsed_signals["AC_DrvrFaceVentActT_raw"] = raw

        # AC_PassFaceVentActT: 起始位32, len=8 (Motorola LSB)
        raw = byte4 & 0xFF
        value = raw * 0.5 - 40
        result.parsed_signals["AC_PassFaceVentActT"] = round(value, 1)
        result.parsed_signals["AC_PassFaceVentActT_raw"] = raw

        # AC_SecRowFaceVentActT: 起始位40, len=8 (Motorola LSB)
        raw = byte5 & 0xFF
        value = raw * 0.5 - 40
        result.parsed_signals["AC_SecRowFaceVentActT"] = round(value, 1)
        result.parsed_signals["AC_SecRowFaceVentActT_raw"] = raw

        # AC_ThrdRowVentTargetT: 起始位48, len=8 (Motorola LSB)
        raw = byte6 & 0xFF
        value = raw * 0.5 - 40
        result.parsed_signals["AC_ThrdRowVentTargetT"] = round(value, 1)
        result.parsed_signals["AC_ThrdRowVentTargetT_raw"] = raw

        # AC_ThrdRowVentActT: 起始位56, len=8 (Motorola LSB)
        raw = byte7 & 0xFF
        value = raw * 0.5 - 40
        result.parsed_signals["AC_ThrdRowVentActT"] = round(value, 1)
        result.parsed_signals["AC_ThrdRowVentActT_raw"] = raw


    # ==================== 0x338 AC_Temp1 瑙ｆ瀽锛堜慨姝ｇ増锛?====================
    def _parse_ac_temp1(self, frame: CANFrame, result: ParsedCANData):
        """ 解析 AC_Temp1_338 (前排/后排车内温度、过热温度等) """
        data = frame.data
        if len(data) < 8:
            return

        # 1. AC_FrntInCarT: DBC start=7 len=12 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 7, 12)
        if raw == 0xFFF:
            result.parsed_signals["AC_FrntInCarT"] = "Invalid"
        else:
            value = raw * 0.1 - 48
            result.parsed_signals["AC_FrntInCarT"] = round(value, 1)
        result.parsed_signals["AC_FrntInCarT_raw"] = raw

        # 2. AC_ReInCarT: DBC start=11 len=12 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 11, 12)
        if raw == 0xFFF:
            result.parsed_signals["AC_ReInCarT"] = "Invalid"
        else:
            value = raw * 0.1 - 48
            result.parsed_signals["AC_ReInCarT"] = round(value, 1)
        result.parsed_signals["AC_ReInCarT_raw"] = raw

        # 3. AC_FrntInCarTValid: byte3 bit7
        raw = (data[3] >> 7) & 0x01
        result.parsed_signals["AC_FrntInCarTValid"] = "Valid" if raw == 1 else "Invalid"
        result.parsed_signals["AC_FrntInCarTValid_raw"] = raw

        # 4. AC_ReInCarTValid: byte3 bit6
        raw = (data[3] >> 6) & 0x01
        result.parsed_signals["AC_ReInCarTValid"] = "Valid" if raw == 1 else "Invalid"
        result.parsed_signals["AC_ReInCarTValid_raw"] = raw

        # 5. AC_OverheatdT: DBC start=29 len=9 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 29, 9)
        if raw == 0x1FF:
            result.parsed_signals["AC_OverheatdT"] = "Invalid"
        else:
            result.parsed_signals["AC_OverheatdT"] = round(raw * 0.1 - 10, 1)
        result.parsed_signals["AC_OverheatdT_raw"] = raw

        # 6. AC_Forward_BlwPwmOut: DBC start=47 len=8 (Motorola LSB)
        raw = MotorolaLSBParser.get_signal(data, 47, 8)
        if raw == 0xFF:
            result.parsed_signals["AC_Forward_BlwPwmOut"] = "Invalid"
        else:
            value = raw * 0.5 - 40
            result.parsed_signals["AC_Forward_BlwPwmOut"] = round(value, 1)
        result.parsed_signals["AC_Forward_BlwPwmOut_raw"] = raw

        # 7. AC_LPTSnsrT: DBC start=55 len=8
        raw = MotorolaLSBParser.get_signal(data, 55, 8)
        if raw == 0xFF:
            result.parsed_signals["AC_LPTSnsrT"] = "Invalid"
        else:
            value = raw * 0.5 - 40
            result.parsed_signals["AC_LPTSnsrT"] = round(value, 1)
        result.parsed_signals["AC_LPTSnsrT_raw"] = raw

        # 8. AC_Second_BlwPwmOut: DBC start=63 len=7
        raw = MotorolaLSBParser.get_signal(data, 63, 7)
        if raw == 0x7F:
            result.parsed_signals["AC_Second_BlwPwmOut"] = "Invalid"
        else:
            result.parsed_signals["AC_Second_BlwPwmOut"] = raw
        result.parsed_signals["AC_Second_BlwPwmOut_raw"] = raw


    # ==================== 0x18F VIU_CHA_18F 解析 ====================
    def _parse_viu_cha_18f(self, frame: CANFrame, result: ParsedCANData):
        """ VIU_CHA_18F (高压上电状态、车速等) """
        data = frame.data
        if len(data) < 8:
            return

        # 1. VDC_HV_State: 起始位13, len=3 (Motorola LSB)
        #    位13-15 完全位于 byte1 的 bit5-7
        raw = (data[1] >> 5) & 0x07
        hv_map = {
            0: "HIGH_VOLT_DOWN",
            1: "HIGH_VOLT_UP",
            2: "RESERVED",
            3: "RESERVED",
            4: "RESERVED",
            5: "RESERVED",
            6: "RESERVED",
            7: "RESERVED",
        }
        result.parsed_signals["VDC_HV_State"] = hv_map.get(raw, "UNKNOWN")
        result.parsed_signals["VDC_HV_State_raw"] = raw

        # 2. IPB_VehicleSpeedValid: 起始位24, len=1 (Motorola LSB)
        raw = (data[3] >> 0) & 0x01
        result.parsed_signals["IPB_VehicleSpeedValid"] = "Valid" if raw == 1 else "Invalid"
        result.parsed_signals["IPB_VehicleSpeedValid_raw"] = raw

        # 3. IPB_VehicleSpeed: 起始位43, len=13 (特殊布局，根据实际数据反推)
        #    原始值 = (byte5 << 5) | (byte6 >> 3)
        raw = (data[4] << 5) | (data[5] >> 3)
        if raw == 0x1FFF:
            result.parsed_signals["IPB_VehicleSpeed"] = "Invalid"
        else:
            value = raw * 0.05625
            result.parsed_signals["IPB_VehicleSpeed"] = round(value, 2)
        result.parsed_signals["IPB_VehicleSpeed_raw"] = raw

    # ==================== 0x3BE VIU_PassiveSecSeat_3BE  ====================
    def _parse_viu_passive_sec_seat_3be(self, frame: CANFrame, result: ParsedCANData):
        """
        解析VIU_PassiveSecSeat_3BE
        """
        data = frame.data
        if len(data) < 8:
            return

        # 1. VIU_DrvrDoorSt: 起始位6, 长度2
        raw = (data[0] >> 6) & 0x03
        drv_door_map = {0: "Initial", 1: "Closed", 2: "Opened", 3: "Invalid"}
        result.parsed_signals["VIU_DrvrDoorSt"] = drv_door_map.get(raw, "Unknown")
        result.parsed_signals["VIU_DrvrDoorSt_raw"] = raw

        # 2. VIU_PassDoorSt: 起始位4, 长度2
        raw = (data[0] >> 4) & 0x03
        pass_door_map = {0: "Initial", 1: "Closed", 2: "Opened", 3: "Invalid"}
        result.parsed_signals["VIU_PassDoorSt"] = pass_door_map.get(raw, "Unknown")
        result.parsed_signals["VIU_PassDoorSt_raw"] = raw

        # 3. VIU_RLDoorSt: 起始位2, 长度2
        raw = (data[0] >> 2) & 0x03
        rl_door_map = {0: "Initial", 1: "Closed", 2: "Opened", 3: "Invalid"}
        result.parsed_signals["VIU_RLDoorSt"] = rl_door_map.get(raw, "Unknown")
        result.parsed_signals["VIU_RLDoorSt_raw"] = raw

        # 4. VIU_RRDoorSt: 起始位0, 长度2
        raw = data[0] & 0x03
        rr_door_map = {0: "Initial", 1: "Closed", 2: "Opened", 3: "Invalid"}
        result.parsed_signals["VIU_RRDoorSt"] = rr_door_map.get(raw, "Unknown")
        result.parsed_signals["VIU_RRDoorSt_raw"] = raw

        # 5. VIU_TailgateSt: 起始位12, 长度4
        raw = (data[1] >> 4) & 0x0F
        tailgate_map = {
            0: "Initial", 1: "closed", 2: "opened",
            3: "closeing", 4: "opening",
            5: "Reserved", 6: "Reserved", 7: "Reserved",
            8: "Reserved", 9: "Reserved", 10: "Reserved",
            11: "Reserved", 12: "Reserved", 13: "Reserved",
            14: "Reserved", 15: "Invalid"
        }
        result.parsed_signals["VIU_TailgateSt"] = tailgate_map.get(raw, "Unknown")
        result.parsed_signals["VIU_TailgateSt_raw"] = raw

        # 6. VIU_HoodSts: 起始位10, 长度2
        raw = (data[1] >> 2) & 0x03
        hood_map = {0: "Initial", 1: "Closed", 2: "Opened", 3: "Invalid"}
        result.parsed_signals["VIU_HoodSts"] = hood_map.get(raw, "Unknown")
        result.parsed_signals["VIU_HoodSts_raw"] = raw

        # 7. VIU_DriverSeatOccptSt: 起始位46, 长度2
        raw = (data[5] >> 6) & 0x03
        seat_occ_map = {0: "Initial", 1: "Free", 2: "Occupied", 3: "Invalid"}
        result.parsed_signals["VIU_DriverSeatOccptSt"] = seat_occ_map.get(raw, "Unknown")
        result.parsed_signals["VIU_DriverSeatOccptSt_raw"] = raw

        # 8. VIU_PassSeatOccptSt: 起始位44, 长度2
        raw = (data[5] >> 4) & 0x03
        result.parsed_signals["VIU_PassSeatOccptSt"] = seat_occ_map.get(raw, "Unknown")
        result.parsed_signals["VIU_PassSeatOccptSt_raw"] = raw

        if SEAT_PARSE_LOGGING_ENABLED:
            print(
                f"[Parse][0x3BE] DriverSeatOccptSt={result.parsed_signals['VIU_DriverSeatOccptSt']} "
                f"(raw={result.parsed_signals['VIU_DriverSeatOccptSt_raw']}), "
                f"PassSeatOccptSt={result.parsed_signals['VIU_PassSeatOccptSt']} "
                f"(raw={result.parsed_signals['VIU_PassSeatOccptSt_raw']})"
            )

        # 9. VIU_RLSecRowSeatOccptSt: 起始位42, 长度2
        raw = (data[5] >> 2) & 0x03
        result.parsed_signals["VIU_RLSecRowSeatOccptSt"] = seat_occ_map.get(raw, "Unknown")
        result.parsed_signals["VIU_RLSecRowSeatOccptSt_raw"] = raw

        # 10. VIU_RMSecRowSeatOccptSt: 起始位40, 长度2
        raw = data[5] & 0x03
        result.parsed_signals["VIU_RMSecRowSeatOccptSt"] = seat_occ_map.get(raw, "Unknown")
        result.parsed_signals["VIU_RMSecRowSeatOccptSt_raw"] = raw

        # 11. VIU_RRSecRowSeatOccptSt: 起始位54, 长度2
        raw = (data[6] >> 6) & 0x03
        result.parsed_signals["VIU_RRSecRowSeatOccptSt"] = seat_occ_map.get(raw, "Unknown")
        result.parsed_signals["VIU_RRSecRowSeatOccptSt_raw"] = raw

        # 12. VIU_RLThrdRowSeatOccptSt: 起始位52, 长度2
        raw = (data[6] >> 4) & 0x03
        result.parsed_signals["VIU_RLThrdRowSeatOccptSt"] = seat_occ_map.get(raw, "Unknown")
        result.parsed_signals["VIU_RLThrdRowSeatOccptSt_raw"] = raw

        # 13. VIU_RRThrdRowSeatOccptSt: 起始位50, 长度2
        raw = (data[6] >> 2) & 0x03
        result.parsed_signals["VIU_RRThrdRowSeatOccptSt"] = seat_occ_map.get(raw, "Unknown")
        result.parsed_signals["VIU_RRThrdRowSeatOccptSt_raw"] = raw

    # ==================== 0x552 VIU_DSM_552 解析 ====================
    def _parse_viu_dsm_552(self, frame: CANFrame, result: ParsedCANData):
        """解析 VIU_DSM_552 (主驾座椅电机位置)"""
        data = frame.data
        if len(data) < 8:
            return

        raw = MotorolaLSBParser.get_signal(data, 2, 7)
        result.parsed_signals["DSM_SLCUotPosn"] = raw
        result.parsed_signals["DSM_SLCUotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 11, 7)
        result.parsed_signals["DSM_BackrestMotPosn"] = raw
        result.parsed_signals["DSM_BackrestMotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 20, 7)
        result.parsed_signals["DSM_CushHeiMotPosn"] = raw
        result.parsed_signals["DSM_CushHeiMotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 29, 7)
        result.parsed_signals["DSM_CushTiltMotPosn"] = raw
        result.parsed_signals["DSM_CushTiltMotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 38, 7)
        result.parsed_signals["DSM_CushExtnMotPosn"] = raw
        result.parsed_signals["DSM_CushExtnMotPosn_raw"] = raw

    # ==================== 0x553 VIU_PSM_553 解析 ====================
    def _parse_viu_psm_553(self, frame: CANFrame, result: ParsedCANData):
        """解析 VIU_PSM_553 (副驾座椅电机位置)"""
        data = frame.data
        if len(data) < 8:
            return

        raw = MotorolaLSBParser.get_signal(data, 2, 7)
        result.parsed_signals["PSM_SLCUotPosn"] = raw
        result.parsed_signals["PSM_SLCUotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 11, 7)
        result.parsed_signals["PSM_BackrestMotPosn"] = raw
        result.parsed_signals["PSM_BackrestMotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 20, 7)
        result.parsed_signals["PSM_CushHeiMotPosn"] = raw
        result.parsed_signals["PSM_CushHeiMotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 29, 7)
        result.parsed_signals["PSM_CushTiltMotPosn"] = raw
        result.parsed_signals["PSM_CushTiltMotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 38, 7)
        result.parsed_signals["PSM_LegSupportMotPosn"] = raw
        result.parsed_signals["PSM_LegSupportMotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 47, 7)
        result.parsed_signals["PSM_FootSupportMotPosn"] = raw
        result.parsed_signals["PSM_FootSupportMotPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 40, 7)
        result.parsed_signals["PSM_SpindleMotPosn"] = raw
        result.parsed_signals["PSM_SpindleMotPosn_raw"] = raw


    # ==================== 0x100 C_100 解析 (CAN-Send-2.dbc) ====================
    def _parse_c_100(self, frame: CANFrame, result: ParsedCANData):
        """解析 C_100 (GPS/温度/湿度 — CAN FD 25字节帧, Intel @1 格式)"""
        data = frame.data
        if len(data) < 25:
            return

        # C________1: start=0, len=8, unsigned (@1+) → byte0
        raw = data[0]
        result.parsed_signals["C________1"] = raw
        result.parsed_signals["C________1_raw"] = raw

        # DBC @1 means Intel/little-endian byte order.
        # GPS__ (纬度): start=24, len=32, signed (@1-) → bytes3-6 小端
        raw = int.from_bytes(data[3:7], byteorder="little", signed=True)
        result.parsed_signals["GPS__"] = raw
        result.parsed_signals["GPS___raw"] = raw

        # GPS___1 (经度): start=56, len=32, signed (@1-) → bytes7-10 小端
        raw = int.from_bytes(data[7:11], byteorder="little", signed=True)
        result.parsed_signals["GPS___1"] = raw
        result.parsed_signals["GPS___1_raw"] = raw

        # 温度 scale/offset (5个温度信号共用)
        TEMP_SCALE = 0.0102235446707866
        TEMP_OFFSET = -270

        # TA_FdHeadTempLe: start=88, len=16, unsigned (@1+) → bytes11-12 小端
        raw = int.from_bytes(data[11:13], byteorder="little", signed=False)
        result.parsed_signals["TA_FdHeadTempLe"] = round(raw * TEMP_SCALE + TEMP_OFFSET, 1)
        result.parsed_signals["TA_FdHeadTempLe_raw"] = raw

        # TA_FdHeadTempRi: start=104, len=16, unsigned (@1+) → bytes13-14
        raw = int.from_bytes(data[13:15], byteorder="little", signed=False)
        result.parsed_signals["TA_FdHeadTempRi"] = round(raw * TEMP_SCALE + TEMP_OFFSET, 1)
        result.parsed_signals["TA_FdHeadTempRi_raw"] = raw

        # TA_FpHeadTempLe: start=120, len=16, unsigned (@1+) → bytes15-16
        raw = int.from_bytes(data[15:17], byteorder="little", signed=False)
        result.parsed_signals["TA_FpHeadTempLe"] = round(raw * TEMP_SCALE + TEMP_OFFSET, 1)
        result.parsed_signals["TA_FpHeadTempLe_raw"] = raw

        # TA_FpHeadTempRi: start=136, len=16, unsigned (@1+) → bytes17-18
        raw = int.from_bytes(data[17:19], byteorder="little", signed=False)
        result.parsed_signals["TA_FpHeadTempRi"] = round(raw * TEMP_SCALE + TEMP_OFFSET, 1)
        result.parsed_signals["TA_FpHeadTempRi_raw"] = raw

        # TS_FrntWidTemp: start=152, len=16, unsigned (@1+) → bytes19-20
        raw = int.from_bytes(data[19:21], byteorder="little", signed=False)
        result.parsed_signals["TS_FrntWidTemp"] = round(raw * TEMP_SCALE + TEMP_OFFSET, 1)
        result.parsed_signals["TS_FrntWidTemp_raw"] = raw

        # 湿度 scale/offset
        HUM_SCALE = 0.00305180437933928
        HUM_OFFSET = -100

        # V_FrntHum: start=168, len=16, unsigned (@1+) → bytes21-22
        raw = int.from_bytes(data[21:23], byteorder="little", signed=False)
        result.parsed_signals["V_FrntHum"] = round(raw * HUM_SCALE + HUM_OFFSET, 1)
        result.parsed_signals["V_FrntHum_raw"] = raw

        # V_SecHum: start=184, len=16, unsigned (@1+) → bytes23-24
        raw = int.from_bytes(data[23:25], byteorder="little", signed=False)
        result.parsed_signals["V_SecHum"] = round(raw * HUM_SCALE + HUM_OFFSET, 1)
        result.parsed_signals["V_SecHum_raw"] = raw

    # ==================== 0x371 AC_Temp5 解析 ====================
    def _parse_ac_temp5(self, frame: CANFrame, result: ParsedCANData):
        """解析 AC_Temp5_371 (主副驾吹脚出风实际温度等)"""
        data = frame.data
        if len(data) < 8:
            return

        raw = MotorolaLSBParser.get_signal(data, 7, 8)
        value = raw * 0.5 - 40
        result.parsed_signals["AC_DrvrFootVentTargetT"] = round(value, 1)
        result.parsed_signals["AC_DrvrFootVentTargetT_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 15, 8)
        value = raw * 0.5 - 40
        result.parsed_signals["AC_PassFootVentTargetT"] = round(value, 1)
        result.parsed_signals["AC_PassFootVentTargetT_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 23, 8)
        value = raw * 0.5 - 40
        result.parsed_signals["AC_ThrdRowFootVentTargetT"] = round(value, 1)
        result.parsed_signals["AC_ThrdRowFootVentTargetT_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 31, 8)
        value = raw * 0.5 - 40
        result.parsed_signals["AC_DrvrFootVentActT"] = round(value, 1)
        result.parsed_signals["AC_DrvrFootVentActT_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 39, 8)
        value = raw * 0.5 - 40
        result.parsed_signals["AC_PassFootVentActT"] = round(value, 1)
        result.parsed_signals["AC_PassFootVentActT_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 47, 8)
        value = raw * 0.5 - 40
        result.parsed_signals["AC_ThrdRowFootVentActT"] = round(value, 1)
        result.parsed_signals["AC_ThrdRowFootVentActT_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 55, 8)
        result.parsed_signals["AC_Forward_AirFlowTarget"] = raw * 2
        result.parsed_signals["AC_Forward_AirFlowTarget_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 63, 8)
        result.parsed_signals["AC_Second_AirFlowTarget"] = raw * 2
        result.parsed_signals["AC_Second_AirFlowTarget_raw"] = raw

    # ==================== 0x370 AC_Temp4 (风门位置) ====================
    def _parse_ac_temp4(self, frame: CANFrame, result: ParsedCANData):
        """解析 AC_Temp4_370 (风门位置信号)"""
        data = frame.data
        if len(data) < 8:
            return

        raw = MotorolaLSBParser.get_signal(data, 0, 7)
        result.parsed_signals["AC_PassTempVentilaPosn"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_PassTempVentilaPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 7, 7)
        result.parsed_signals["AC_DrvrTempVentilaPosn"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_DrvrTempVentilaPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 9, 8)
        result.parsed_signals["AC_CircleModeVentilaPosn"] = raw
        result.parsed_signals["AC_CircleModeVentilaPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 25, 7)
        result.parsed_signals["AC_SecRowTempVentilaPosn"] = raw
        result.parsed_signals["AC_SecRowTempVentilaPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 27, 2)
        result.parsed_signals["AC_SecRowFootVentilaPosn"] = raw
        result.parsed_signals["AC_SecRowFootVentilaPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 31, 4)
        result.parsed_signals["AC_ModeVentilaPosn"] = raw
        result.parsed_signals["AC_ModeVentilaPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 34, 4)
        result.parsed_signals["AC_SecRowModeVentilaPosn"] = raw
        result.parsed_signals["AC_SecRowModeVentilaPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 54, 7)
        result.parsed_signals["AC_ThrdTempVentilaPosn"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_ThrdTempVentilaPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 62, 7)
        result.parsed_signals["AC_BLOW_FaceVentilaPosn"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_BLOW_FaceVentilaPosn_raw"] = raw

    # ==================== 0x541 AC_RHVAC (后排空调/风门电压) ====================
    def _parse_ac_rhvac(self, frame: CANFrame, result: ParsedCANData):
        """解析 AC_RHVAC_541 (后排空调控制/风门电压信号)"""
        data = frame.data
        if len(data) < 8:
            return

        raw = MotorolaLSBParser.get_signal(data, 6, 7)
        result.parsed_signals["AC_RBlowFaceVentPosn"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_RBlowFaceVentPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 14, 7)
        result.parsed_signals["AC_RBlowFootVentPosn"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_RBlowFootVentPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 22, 7)
        result.parsed_signals["AC_ThrdRowModeVentPosn"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_ThrdRowModeVentPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 31, 8)
        result.parsed_signals["AC_ThrdBlowVol"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_ThrdBlowVol_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 39, 8)
        result.parsed_signals["AC_12VOUT"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_12VOUT_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 47, 8)
        result.parsed_signals["AC_INGVol"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_INGVol_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 54, 7)
        result.parsed_signals["AC_FrantFootVentPosn"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_FrantFootVentPosn_raw"] = raw

        raw = MotorolaLSBParser.get_signal(data, 62, 7)
        result.parsed_signals["AC_DefrostVentilaPosn"] = round(raw * 0.1, 1)
        result.parsed_signals["AC_DefrostVentilaPosn_raw"] = raw



# ============================================================
# 6. SocketCAN 接口
# ============================================================

class SocketCANInterface:
    """SocketCAN"""
    
    def __init__(self, interface: str = 'can2'):
        self.interface = interface
        self.bus = None
        self.is_connected = False
        
    def open(self) -> bool:
        try:
            # can5 carries the 0x100 telemetry as CAN FD (25-byte payload,
            # encoded as a 32-byte FD frame on the wire).  python-can defaults
            # to a Classical CAN socket unless fd=True is supplied, in which
            # case the kernel silently withholds those FD frames.
            is_fd_interface = self.interface == "can5"
            self.bus = can.interface.Bus(
                channel=self.interface,
                interface='socketcan',
                fd=is_fd_interface,
            )
            self.is_connected = True
            print(f"[SocketCAN] Opened {self.interface}")
            return True
        except Exception as e:
            print(f"[SocketCAN] Failed to open {self.interface}: {e}")
            return False
    
    def receive(self, timeout: float = 0.1) -> Optional[can.Message]:
        if not self.bus:
            raise RuntimeError("CAN bus not opened")
        return self.bus.recv(timeout=timeout)
    
    def close(self):
        if self.bus:
            self.bus.shutdown()
            print(f"[SocketCAN] Closed {self.interface}")




# ============================================================
# 7. CAN 接收器类
# ============================================================

class CANReceiver:
    """ CAN总线接收器（支持多接口） """

    def __init__(self, interfaces: List[str] = None, parsed_ids: List[int] = None,
                 print_interval: int = PRINT_INTERVAL, enable_socket: bool = True,
                 socket_path: str = SOCKET_PATH, enable_rte: bool = False):
        if interfaces is None:
            interfaces = ["can2"]
        self.interfaces = interfaces
        self.can_drivers = [SocketCANInterface(iface) for iface in interfaces]
        self.can_driver = self.can_drivers[0]  # 主接口（写操作默认用）

        if parsed_ids is None:
            parsed_ids = PARSED_CAN_IDS
        self.parser = CANParser(parsed_ids)

        # 加载DBC用于发送编码 + 构建信号名→CAN ID反向索引
        self.dbc_db = None
        self._signal_to_can_id: Dict[str, int] = {}
        self._dbc_msgs: Dict[int, object] = {}
        self._default_raw: Dict[int, Dict[str, int]] = {}  # CAN ID → 默认raw值
        try:
            import cantools
            # 加载主 DBC + CAN-Send-2.dbc，合并消息
            DBC_DIR = os.path.dirname(os.path.abspath(__file__))
            DBC_PATH = os.path.join(DBC_DIR, "CAN.dbc")
            DBC2_PATH = os.path.join(DBC_DIR, "CAN-Send-2.dbc")
            self.dbc_db = cantools.database.load_file(DBC_PATH)
            if os.path.exists(DBC2_PATH):
                dbc2 = cantools.database.load_file(DBC2_PATH)
                for msg in dbc2.messages:
                    self.dbc_db.messages.append(msg)
                print(f"[Receiver] Loaded CAN-Send-2.dbc, {len(dbc2.messages)} additional messages")
            for cid in parsed_ids:
                try:
                    msg = self.dbc_db.get_message_by_frame_id(cid)
                    self._dbc_msgs[cid] = msg
                    defaults = {}
                    for sig in msg.signals:
                        self._signal_to_can_id[sig.name] = cid
                        defaults[sig.name] = 0
                    self._default_raw[cid] = defaults
                except Exception:
                    pass
            # 0x387 温度信号默认 0x1F (HI)，其余保持 0
            if 0x387 in self._default_raw:
                for name in ("CDC_DriverTempCSet", "CDC_PassengerTempCSet",
                             "CDC_RrTempCSet", "TBOX_ACTempSet"):
                    if name in self._default_raw[0x387]:
                        self._default_raw[0x387][name] = 0x1F
            print(f"[Receiver] DBC loaded, {len(self._signal_to_can_id)} signals indexed")
        except ImportError:
            print("[Receiver] cantools not available, send_signal disabled")
        except FileNotFoundError:
            print("[Receiver] DBC file not found, send_signal disabled")
        except Exception as e:
            print(f"[Receiver] DBC load failed: {e}")

        self.callbacks: List[Callable[[ParsedCANData], None]] = []
        self.running = False

        self.frame_counters = defaultdict(int)
        self.total_received_frames = 0
        self.print_interval = print_interval

        self.enable_socket = enable_socket
        self.enable_rte = enable_rte
        self.socket_server = None if not enable_socket else UnixSocketServer(socket_path, enable_rte=False)
        self.socket_thread = None

        self.stats = {
            'total_received': 0,
            'parsed_count': 0,
            'error_count': 0,
            'start_time': time.time()
        }
        self._stats_lock = threading.Lock()

        self.send_lock = threading.Lock()
        self.periodic_tasks = {}
        self.periodic_threads = {}
        self.id_ownership = {}
        self.send_stats = {
            'total_sent': 0,
            'send_errors': 0,
            'per_id_sent': defaultdict(int)
        }

        self._seat_states = {
            "driver": SeatCycleState(),
            "passenger": SeatCycleState(),
        }
        self._script_exec_lock = threading.Lock()
        self.seat_script_path = "/media/test/nvme0n1p1/workspace/yzy/Image_Recognition/analyze_once.sh"

        # CAN ID → 接口 自动路由表（接收时自动学习）
        self._id_route: Dict[int, str] = {}

        self.backend_ac_state = BackendACControlState()
        self.backend_stop_event = threading.Event()
        self.backend_http_server = None
        self.backend_listener_thread = None
        self.backend_sender_thread = None
        self.ai_state_publish_thread = None
        self._last_backend_gate_log = 0.0

    # ---------- CAN 发送方法 ----------
    def _phys_to_raw(self, sig, value):
        """将物理值/枚举数字转为raw值."""
        if isinstance(value, str):
            if sig.choices:
                for rv, choice in sig.choices.items():
                    if choice == value:
                        return rv
            raise ValueError(f"Unknown choice '{value}' for {sig.name}")
        # 温度等有scale/offset的信号: 物理值→raw
        if sig.scale != 1 or sig.offset != 0:
            return int(round((float(value) - sig.offset) / sig.scale))
        # 枚举量直接用数字
        return int(value)

    def send_signal_values(self, signals: Dict[str, float], source: str = "internal") -> Dict:
        """
        按信号名+值自动编码并发送到正确的CAN总线上。
        自动填充该CAN ID所有信号的默认值（温度0x1F，其余0），
        然后用调用者提供的值覆盖，DBC编码后路由到对应接口。

        Args:
            signals: {信号名: 值}, 枚举用数字, 温度用°C
                     如 {"CDC_DriverTempCSet": 22.0, "CDC_FHvacBlowLvSet": 5}

        Returns:
            {"status": "ok", "sent": [(can_id, iface), ...]}
            或 {"status": "error", "reason": ...}
        """
        if not self.dbc_db:
            return {"status": "error", "reason": "DBC not loaded, cantools not available"}

        # 按 CAN ID 分组
        grouped: Dict[int, Dict[str, object]] = {}
        unknown = []
        for name, value in signals.items():
            cid = self._signal_to_can_id.get(name)
            if cid is None:
                unknown.append(name)
                continue
            if cid not in grouped:
                grouped[cid] = {}
            grouped[cid][name] = value

        if unknown:
            return {"status": "error",
                    "reason": f"Unknown signals: {', '.join(unknown[:5])}"}

        if not grouped:
            return {"status": "error", "reason": "No valid signals provided"}

        sent_list = []
        errors = []
        for cid, caller_signals in grouped.items():
            msg = self._dbc_msgs.get(cid)
            if msg is None:
                errors.append(f"No DBC message for 0x{cid:03X}")
                continue

            # 从默认raw值开始，覆盖调用者提供的信号（仅对传入值限幅）
            defaults = self._default_raw.get(cid, {})
            raw_signals = dict(defaults)
            for name, value in caller_signals.items():
                sig = None
                for s in msg.signals:
                    if s.name == name:
                        sig = s
                        break
                if sig is None:
                    continue
                try:
                    raw_val = self._phys_to_raw(sig, value)
                    # 限幅: 按DBC位宽 [0, 2^len-1]，温度排除0x1F(INVALID)
                    raw_max = (1 << sig.length) - 1
                    if name in ("CDC_DriverTempCSet", "CDC_PassengerTempCSet",
                                "CDC_RrTempCSet", "TBOX_ACTempSet"):
                        raw_max = 0x1E
                    raw_signals[name] = max(0, min(raw_max, raw_val))
                except Exception as e:
                    errors.append(f"Convert {name}={value}: {e}")

            try:
                data = msg.encode(raw_signals, scaling=False)
            except Exception as e:
                errors.append(f"Encode 0x{cid:03X}: {e}")
                continue

            success = self.send_can_message(cid, data, source=source)
            if success:
                target = self._id_route.get(cid, self.interfaces[0])
                sent_list.append((cid, target))
            else:
                errors.append(f"Send 0x{cid:03X} failed")

        if errors:
            return {"status": "partial" if sent_list else "error",
                    "sent": sent_list, "errors": errors}
        return {"status": "ok", "sent": sent_list}

    def send_signal_values_repeated(self, signals: Dict[str, float],
                                     count: int, interval_ms: int):
        """重复发送（用于可靠性），在后台线程中执行"""
        def _run():
            for i in range(count):
                self.send_signal_values(signals)
                if i < count - 1:
                    time.sleep(interval_ms / 1000.0)
        threading.Thread(target=_run, daemon=True).start()

    # ---------- 原始CAN发送 ----------
    def _capture_socket_0x387_feedback(self, data: bytes) -> None:
        if len(data) < 8:
            return

        decoded = None
        message = self._dbc_msgs.get(0x387)
        if message is not None:
            try:
                decoded = message.decode(data, decode_choices=False)
            except Exception as exc:
                backend_ac_log(
                    f"0x387 DBC decode failed: {exc}",
                    ENABLE_BACKEND_AC_TX_LOG,
                )

        if decoded is None:
            mode_raw = (data[2] >> 4) & 0x0F
            level_raw = (data[1] >> 4) & 0x0F
            driver_raw = ((data[2] & 0x01) << 4) | ((data[3] >> 4) & 0x0F)
            passenger_raw = ((data[3] & 0x0F) << 1) | ((data[4] >> 7) & 0x01)
            decoded = {
                "CDC_FHvacModeSet": mode_raw,
                "CDC_FHvacBlowLvSet": level_raw,
            }
            if driver_raw != 0x1F:
                decoded["CDC_DriverTempCSet"] = 16.0 + driver_raw * 0.5
            if passenger_raw != 0x1F:
                decoded["CDC_PassengerTempCSet"] = 16.0 + passenger_raw * 0.5

        self.backend_ac_state.update_socket_feedback(decoded)

    def _log_outgoing_0x387_raw_frame(self, data: bytes,
                                      source: str = "unknown") -> None:
        if not ENABLE_BACKEND_AC_SOCKET_0X387_RAW_LOG:
            return
        if len(data) < 8:
            backend_ac_log(
                f"outgoing 0x387 raw before rewrite path_source={source} "
                f"invalid_len={len(data)} data={data.hex()}",
                True,
            )
            return

        mode_raw = (data[2] >> 4) & 0x0F
        level_raw = (data[1] >> 4) & 0x0F
        driver_raw = ((data[2] & 0x01) << 4) | ((data[3] >> 4) & 0x0F)
        passenger_raw = ((data[3] & 0x0F) << 1) | ((data[4] >> 7) & 0x01)

        driver_desc = "HI" if driver_raw == 0x1F else f"{16.0 + driver_raw * 0.5:.1f}"
        passenger_desc = "HI" if passenger_raw == 0x1F else f"{16.0 + passenger_raw * 0.5:.1f}"
        backend_ac_log(
            f"outgoing 0x387 raw before rewrite path_source={source} "
            f"mode_raw={mode_raw} level={level_raw} "
            f"driver_raw={driver_raw} driver_temp={driver_desc} "
            f"passenger_raw={passenger_raw} passenger_temp={passenger_desc} "
            f"data={data.hex()}",
            True,
        )

    def _rewrite_socket_0x387_from_backend(self, data: bytes,
                                           source: str = "unknown") -> bytes:
        override = self.backend_ac_state.backend_0x387_override()
        if override is None or len(data) < 8:
            return data

        payload = bytearray(data)
        override_source = override.get("source", "backend")
        mode_raw = override["mode_raw"] & 0x0F
        level_raw = override["level_raw"] & 0x0F
        driver_raw = override["driver_raw"] & 0x1F
        passenger_raw = override["passenger_raw"] & 0x1F

        payload[1] = (payload[1] & 0x0F) | (level_raw << 4)
        payload[2] = (payload[2] & 0x0F) | (mode_raw << 4)
        payload[2] = (payload[2] & 0xFE) | ((driver_raw >> 4) & 0x01)
        payload[3] = (payload[3] & 0x0F) | ((driver_raw & 0x0F) << 4)
        payload[3] = (payload[3] & 0xF0) | ((passenger_raw >> 1) & 0x0F)
        payload[4] = (payload[4] & 0x7F) | ((passenger_raw & 0x01) << 7)

        now = time.monotonic()
        if now - self._last_backend_gate_log >= 1.0:
            driver_temp = 16.0 + driver_raw * 0.5
            passenger_temp = 16.0 + passenger_raw * 0.5
            backend_ac_log(
                f"rewrote 0x387 from {override_source} values "
                f"path_source={source} mode_raw={mode_raw} "
                f"level={level_raw} driver_temp={driver_temp:.1f} "
                f"passenger_temp={passenger_temp:.1f} data={bytes(payload).hex()}",
                ENABLE_BACKEND_AC_GATE_LOG or ENABLE_BACKEND_AC_DEBUG_CAN_OVERRIDE,
            )
            self._last_backend_gate_log = now
        return bytes(payload)

    def send_can_message(self, can_id: int, data: bytes,
                         source: str = "internal") -> bool:
        """根据 CAN ID 自动路由到正确的接口发送"""
        if ENABLE_BACKEND_AC_CONTROL and can_id == 0x387:
            self._log_outgoing_0x387_raw_frame(data, source=source)
            if source == "socket":
                self._capture_socket_0x387_feedback(data)
            if source == "socket" or ENABLE_BACKEND_AC_DEBUG_CAN_OVERRIDE:
                data = self._rewrite_socket_0x387_from_backend(data, source=source)

        target_iface = self._id_route.get(can_id, self.interfaces[0])
        driver = None
        for d in self.can_drivers:
            if d.interface == target_iface and d.bus:
                driver = d
                break
        if driver is None:
            for d in self.can_drivers:
                if d.bus:
                    driver = d
                    break
        if driver is None:
            self.send_stats['send_errors'] += 1
            print("[CAN] No bus available")
            return False
        msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
        try:
            with self.send_lock:
                driver.bus.send(msg)
                self.send_stats['total_sent'] += 1
                self.send_stats['per_id_sent'][can_id] += 1
            print(f"[CAN] Sent ID=0x{can_id:X} → {driver.interface}, data={data.hex()}")
            return True
        except Exception as e:
            self.send_stats['send_errors'] += 1
            print(f"[CAN] Send failed: {e}")
            return False

    def send_event_frames(self, can_id: int, data: bytes, count: int,
                          interval_ms: int, source: str = "internal"):
        """ 事件周期发送，不对客户端反馈错误 """
        def _run():
            for i in range(count):
                self.send_can_message(can_id, data, source=source)
                if i < count - 1:
                    time.sleep(interval_ms / 1000.0)
            print(f"[Event] Sent {count} frames of ID=0x{can_id:X}")
        threading.Thread(target=_run, daemon=True).start()

    def start_periodic_send(self, client_socket, task_id: str,
                            can_id: int, data: bytes, interval_ms: int) -> dict:
        
        """
        启动周期发送，返回状态字典
        成功：{"status": "ok", "task_id": ...}
        失败：{"status": "error", "task_id": ..., "reason": ...}
        """
        key = (client_socket, task_id)

        # 检查是否已存在相同的任务
        if key in self.periodic_tasks:
            reason = f"Task {task_id} already exists"
            print(f"[Periodic] {reason}")
            return {"status": "error", "task_id": task_id, "reason": reason}

        # 冲突检测：同一 CAN ID 只能由一个客户端占用
        current_owner = self.id_ownership.get(can_id)
        if current_owner is not None and current_owner[0] != client_socket:
            reason = f"ID 0x{can_id:X} already owned by {current_owner[1]}"
            print(f"[Periodic] {reason}")
            return {"status": "error", "task_id": task_id, "reason": reason}

        # 如果同一客户端已经占用该ID，但task_id不同，则先停止旧任务
        if current_owner and current_owner[0] == client_socket and current_owner[1] != task_id:
            old_key = (client_socket, current_owner[1])
            old_event = self.periodic_tasks.get(old_key)
            if old_event:
                old_event.set()

        # 注册占用
        self.id_ownership[can_id] = (client_socket, task_id)

        stop_event = threading.Event()
        self.periodic_tasks[key] = stop_event

        def _sender():
            print(f"[Periodic] Start {task_id}, ID=0x{can_id:X}, interval={interval_ms}ms")
            try:
                while not stop_event.is_set():
                    self.send_can_message(can_id, data, source="socket")
                    stop_event.wait(timeout=interval_ms / 1000.0)
            finally:
                if self.id_ownership.get(can_id) == (client_socket, task_id):
                    del self.id_ownership[can_id]
                self.periodic_tasks.pop(key, None)
                self.periodic_threads.pop(key, None)
                print(f"[Periodic] Stopped {task_id}")

        t = threading.Thread(target=_sender, daemon=True, name=f"Periodic_{task_id}")
        self.periodic_threads[key] = t
        t.start()
        return {"status": "ok", "task_id": task_id, "can_id": f"0x{can_id:X}"}

    def stop_periodic_send(self, client_socket, task_id: str):
        """ 停止指定的周期发送任务 """
        key = (client_socket, task_id)
        event = self.periodic_tasks.get(key)
        if event:
            event.set()
        else:
            print(f"[Periodic] Task {task_id} not found")

    def _cleanup_client_tasks(self, client_socket):
        """ 客户端断开时，停止其所有周期任务并释放ID占用 """
        keys_to_stop = [k for k in self.periodic_tasks if k[0] == client_socket]
        for key in keys_to_stop:
            event = self.periodic_tasks[key]
            event.set()
        owned_ids = [cid for cid, (cs, _) in self.id_ownership.items() if cs == client_socket]
        for cid in owned_ids:
            del self.id_ownership[cid]

    def _setup_socket_callbacks(self):
        if self.socket_server:
            self.socket_server.set_send_callbacks(
                send_cb=lambda can_id, data: self.send_can_message(
                    can_id, data, source="socket"
                ),
                send_event_cb=lambda can_id, data, count, interval_ms:
                    self.send_event_frames(
                        can_id, data, count, interval_ms, source="socket"
                    ),
                start_periodic_cb=self.start_periodic_send,
                stop_periodic_cb=self.stop_periodic_send,
                client_disconnected_cb=self._cleanup_client_tasks,
                send_signal_cb=lambda signals: self.send_signal_values(
                    signals, source="socket"
                )
            )

    def _backend_listener_loop(self):
        """后台调节接口监听线程，不执行 CAN 收发和周期回显。"""
        try:
            # HTTP 处理器和 CANReceiver 共用同一个线程安全状态对象。
            BackendAdjustHandler.state = self.backend_ac_state
            self.backend_http_server = BackendHTTPServer(
                (BACKEND_LISTEN_HOST, BACKEND_LISTEN_PORT),
                BackendAdjustHandler,
            )
            # 使用短超时轮询停止事件，避免退出时长期阻塞在监听调用中。
            self.backend_http_server.timeout = 0.2
            backend_ac_log(
                f"adjust listener started at "
                f"{BACKEND_LISTEN_HOST}:{BACKEND_LISTEN_PORT}",
                True,
            )
            # HTTPServer 为单线程处理，保证后台 POST 按接收顺序串行生效。
            while not self.backend_stop_event.is_set():
                self.backend_http_server.handle_request()
        except Exception as exc:
            backend_ac_log(f"adjust listener stopped by error: {exc}", True)
        finally:
            # 仅清理新增的后台 HTTP 监听资源，不影响原有 Socket/CAN 服务。
            if self.backend_http_server is not None:
                self.backend_http_server.server_close()
                self.backend_http_server = None
            backend_ac_log("adjust listener stopped", True)

    def _backend_sender_loop(self):
        if not BACKEND_UPDATE_URL:
            backend_ac_log(
                "BACKEND_UPDATE_URL is empty; update sender is disabled",
                True,
            )
            return

        next_send = time.monotonic() + BACKEND_UPDATE_INTERVAL_SECONDS
        waiting_logged = False
        last_error_log = 0.0
        backend_ac_log(f"update sender started: {BACKEND_UPDATE_URL}", True)

        while not self.backend_stop_event.is_set():
            wait_seconds = max(0.0, next_send - time.monotonic())
            if self.backend_stop_event.wait(wait_seconds):
                break

            snapshot = self.backend_ac_state.feedback_snapshot()
            if snapshot is None:
                if not waiting_logged:
                    backend_ac_log(
                        "waiting for complete Socket 0x387 feedback",
                        ENABLE_BACKEND_AC_TX_LOG,
                    )
                    waiting_logged = True
            else:
                waiting_logged = False
                body = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
                http_request = urllib_request.Request(
                    BACKEND_UPDATE_URL,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib_request.urlopen(
                        http_request,
                        timeout=BACKEND_HTTP_TIMEOUT_SECONDS,
                    ) as response:
                        response_body = response.read(BACKEND_MAX_BODY_BYTES)
                        response_data = json.loads(response_body.decode("utf-8"))
                        if response.status != 200 or response_data.get("code") != 200:
                            raise ValueError(
                                f"status={response.status}, body={response_data}"
                            )
                    backend_ac_log(
                        f"updateInfo sent: {snapshot}",
                        ENABLE_BACKEND_AC_TX_LOG,
                    )
                except (urllib_error.URLError, TimeoutError, ValueError,
                        UnicodeDecodeError, json.JSONDecodeError) as exc:
                    now = time.monotonic()
                    if now - last_error_log >= 10.0:
                        backend_ac_log(f"updateInfo failed: {exc}", True)
                        last_error_log = now
                except Exception as exc:
                    now = time.monotonic()
                    if now - last_error_log >= 10.0:
                        backend_ac_log(f"updateInfo error: {exc}", True)
                        last_error_log = now

            next_send += BACKEND_UPDATE_INTERVAL_SECONDS
            now = time.monotonic()
            if next_send <= now:
                next_send = now + BACKEND_UPDATE_INTERVAL_SECONDS

        backend_ac_log("update sender stopped", True)

    def _ai_state_publish_loop(self):
        """AI 状态推送线程：每秒向订阅了 VIRTUAL_ID_AI_STATE 的客户端推送当前 ai 状态。
        推送本身不打印，仅在线程启动/退出时各打印一次，异常限频打印。"""
        last_error_log = 0.0
        backend_ac_log(
            f"ai state publisher started: id=0x{VIRTUAL_ID_AI_STATE:X}, "
            f"interval={AI_STATE_PUBLISH_INTERVAL_SECONDS}s, "
            f"initial ai={self.backend_ac_state.get_ai()}",
            ENABLE_BACKEND_AC_AI_STATE_LOG,
        )

        while self.running and not self.backend_stop_event.is_set():
            try:
                if self.socket_server and self.socket_server.running:
                    packet = {
                        "version": "1.0",
                        "type": "ai_state",
                        "timestamp": time.time(),
                        "data": {"ai": self.backend_ac_state.get_ai()},
                    }
                    # 仅 filters 中包含 VIRTUAL_ID_AI_STATE 的客户端会收到
                    self.socket_server.broadcast_filtered(VIRTUAL_ID_AI_STATE, packet)
            except Exception as exc:
                now = time.monotonic()
                if now - last_error_log >= 10.0:
                    backend_ac_log(f"ai state publish error: {exc}", True)
                    last_error_log = now

            # 使用 event 等待，服务停止时可立即退出
            if self.backend_stop_event.wait(AI_STATE_PUBLISH_INTERVAL_SECONDS):
                break

        backend_ac_log("ai state publisher stopped", ENABLE_BACKEND_AC_AI_STATE_LOG)

    def _start_ai_state_publisher(self):
        # 仅在 Socket 启用且后台空调集成总开关打开时启动（总开关关闭时 _ai 无真实来源）
        if not ENABLE_BACKEND_AC_CONTROL or not self.socket_server:
            return
        self.ai_state_publish_thread = threading.Thread(
            target=self._ai_state_publish_loop,
            daemon=True,
            name="AIStatePublisher",
        )
        self.ai_state_publish_thread.start()

    def _start_backend_ac_threads(self):
        if not ENABLE_BACKEND_AC_CONTROL:
            return
        self.backend_stop_event.clear()
        self.backend_listener_thread = threading.Thread(
            target=self._backend_listener_loop,
            daemon=True,
            name="BackendAdjustListener",
        )
        self.backend_sender_thread = threading.Thread(
            target=self._backend_sender_loop,
            daemon=True,
            name="BackendUpdateSender",
        )
        self.backend_listener_thread.start()
        self.backend_sender_thread.start()

    def _stop_backend_ac_threads(self):
        if not ENABLE_BACKEND_AC_CONTROL:
            return
        self.backend_stop_event.set()
        for thread in (self.backend_listener_thread, self.backend_sender_thread):
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

    def register_callback(self, callback: Callable[[ParsedCANData], None]):
        self.callbacks.append(callback)
        print(f"[Receiver] Registered callback")
    
    def _convert_to_canframe(self, message: can.Message, interface: str = "") -> CANFrame:
        timestamp_us = int(message.timestamp * 1_000_000)
        return CANFrame(
            can_id=message.arbitration_id,
            dlc=message.dlc,
            data=message.data,
            timestamp_us=timestamp_us,
            bus_name=interface,
            is_fd=message.is_fd,
        )
    
    


    def _display_parsed_data(self, data: ParsedCANData):
        """ 显示解析后的空调数据 - 每个信号独立一行显示（过滤 raw 信号） """
        id_names = {
            0x387: "VIU_ACSet1",
            0x381: "VIU_ACSet2",
            0x33A: "AC_Sts1",
            0x33B: "AC_Sts2",
            0x33C: "AC_Sts3",
            0x35B: "VIU_35B",
            0x3B9: "VIU_LIN1",
            0x33E: "AC_Temp2",
            0x33F: "AC_Temp3",
            0x338: "AC_Temp1",
            0x18F: "VIU_CHA_18F",
            0x3BE: "VIU_PassiveSecSeat_3BE",
            0x371: "AC_Temp5",
            0x370: "AC_Temp4",
            0x541: "AC_RHVAC",
            0x552: "VIU_DSM_552",
            0x553: "VIU_PSM_553",
            0x100: "C_100",
        }
        name = id_names.get(data.can_id, f"0x{data.can_id:03X}")

        self.frame_counters[data.can_id] += 1
        count = self.frame_counters[data.can_id]
        
        if count % self.print_interval == 0:
            print(f"\n[ DATA ] {name} [Frame #{count}] - {len(data.parsed_signals)} signals:")
            print("-" * 80)
            for key, value in data.parsed_signals.items():
                if key.endswith('_raw'):
                    continue
                print(f"  {key:40s} = {value}")
            print("-" * 80)

    def _send_to_socket(self, parsed_data: ParsedCANData):
        """ 发送数据到 Socket 客户端（输出物理值，附带_raw原始值） """
        if not self.socket_server or not self.socket_server.running:
            return

        # 物理值（排除_unit和_raw后缀）
        phys_data = {k: v for k, v in parsed_data.parsed_signals.items()
                     if not k.endswith('_raw') and not k.endswith('_unit')}
        # 原始值（_raw后缀 → 去掉后缀）
        raw_data = {k[:-4]: v for k, v in parsed_data.parsed_signals.items() if k.endswith('_raw')}

        # 枚举字符串值 → 替换为原始数字（仅非数字字符串，保护温度值如"22.0"）
        for k in list(phys_data.keys()):
            if isinstance(phys_data[k], str) and k in raw_data:
                try:
                    float(phys_data[k])
                except ValueError:
                    phys_data[k] = raw_data[k]

        packet = {
            "version": "1.0",
            "timestamp": parsed_data.timestamp_us / 1_000_000.0,
            "data": phys_data,
            "raw": raw_data,
            "type": "data"
        }

        self.socket_server.broadcast_filtered(parsed_data.can_id, packet)

    def _socket_accept_loop(self):
        """ Socket 接受客户端 """
        current_thread = threading.current_thread()
        print(f"[Socket] Accept thread started: {current_thread.name}")
        
        wait_count = 0
        while wait_count < 50 and (not self.socket_server or not self.socket_server.running):
            time.sleep(0.1)
            wait_count += 1
        
        print(f"[Socket] Loop conditions: running={self.running}, socket_server={self.socket_server is not None}, server_running={self.socket_server.running if self.socket_server else False}")
        
        while self.running and self.socket_server and self.socket_server.running:
            try:
                self.socket_server.accept_clients()
            except Exception as e:
                print(f"[Socket] Accept error: {e}")
            time.sleep(0.1)
        
        print(f"[Socket] Accept thread stopped: {current_thread.name}")

    def _show_periodic_stats(self):
        if self.stats['parsed_count'] > 0:
            elapsed = time.time() - self.stats['start_time']
            total_parsed_frames = sum(self.frame_counters.values())
            id_names = {
                0x387: "VIU_ACSet1",
                0x381: "VIU_ACSet2", 
                0x33A: "AC_Sts1",
                0x33B: "AC_Sts2",
                0x33C: "AC_Sts3",
                0x35B: "VIU_35B",
                0x3B9: "VIU_LIN1",
                0x33E: "AC_Temp2",
                0x33F: "AC_Temp3",
                0x338: "AC_Temp1",
                0x18F: "VIU_CHA_18F",
                0x3BE: "VIU_PassiveSecSeat_3BE",
            0x371: "AC_Temp5",
            0x370: "AC_Temp4",
            0x541: "AC_RHVAC",
            0x552: "VIU_DSM_552",
            0x553: "VIU_PSM_553",
            0x100: "C_100",
            }
            print(f"\n[STATS] Time: {elapsed:.1f}s | Total parsed: {self.stats['parsed_count']} | Rate: {self.stats['parsed_count']/elapsed:.1f} fps")
            print(f"[STATS] Frame counts by ID:")
            for can_id, cnt in sorted(self.frame_counters.items()):
                id_name = id_names.get(can_id, f"0x{can_id:03X}")
                percentage = (cnt / total_parsed_frames * 100) if total_parsed_frames > 0 else 0
                print(f"  {id_name} (0x{can_id:03X}): {cnt} frames ({percentage:.1f}%)")
            print(f"  Total: {total_parsed_frames} frames")
            if self.socket_server:
                count = self.socket_server.get_client_count()
                clients = self.socket_server.get_client_list()
                print(f"  Socket clients: {count}")
                if clients:
                    print(f"  Client list:")
                    for c in clients:
                        cid = c.get('id', '?')
                        filters = c.get('filters', [])
                        missed = c.get('heartbeat_miss', 0)
                        dur = time.time() - c.get('connected_time', time.time())
                        filter_str = ', '.join(f'0x{f:03X}' if isinstance(f, int) else str(f) for f in filters[:5])
                        if len(filters) > 5:
                            filter_str += f' ...(+{len(filters)-5})'
                        flags = ''
                        if missed > 0:
                            flags += f' [miss:{missed}]'
                        print(f"    [{cid}] {dur:.0f}s  filters=[{filter_str}]{flags}")
            print()
        # 发送统计输出
        if SHOW_ALL_SEND:
            elapsed = time.time() - self.stats['start_time']
            print(f"\n[SEND STATS] Time: {elapsed:.1f}s")
            print(f"  Total sent: {self.send_stats['total_sent']}")
            print(f"  Send errors: {self.send_stats['send_errors']}")
            if self.send_stats['per_id_sent']:
                print(f"  Per-ID sent frames:")
                for cid, cnt in sorted(self.send_stats['per_id_sent'].items()):
                    print(f"    0x{cid:03X}: {cnt} frames")

    def _trigger_callbacks(self, data: ParsedCANData):
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"[Callback Error] {e}")

    # ====================  ====================
    def _seat_log(self, message: str) -> None:
        if SEAT_TRIGGER_LOGGING_ENABLED:
            print(message)

    def _reset_seat_state(self, seat_key: str) -> None:
        self._seat_states[seat_key] = SeatCycleState()

    def _enqueue_seat_script(self, seat_label: str, event_type: str) -> None:
        def _runner():
            self._seat_log(f"[SeatTrigger] {seat_label} {event_type} waiting for script lock")
            with self._script_exec_lock:
                self._seat_log(f"[SeatTrigger] {seat_label} {event_type} acquired script lock")
                self._run_seat_script(seat_label, event_type)

        threading.Thread(target=_runner, daemon=True).start()

    def _run_seat_script(self, seat_label: str, event_type: str) -> None:
        """执行外部脚本，全局锁保护，同一时刻只跑一个。"""
        script_dir = os.path.dirname(self.seat_script_path)
        script_name = os.path.basename(self.seat_script_path)
        try:
            subprocess.run(
                ["bash", script_name],
                check=True,
                timeout=SEAT_SCRIPT_TIMEOUT_SECONDS,
                cwd=script_dir
            )
            self._seat_log(f"[SeatTrigger] {seat_label} {event_type} script executed successfully.")
        except subprocess.CalledProcessError as e:
            self._seat_log(f"[SeatTrigger] {seat_label} {event_type} script failed with code {e.returncode}")
        except subprocess.TimeoutExpired:
            self._seat_log(f"[SeatTrigger] {seat_label} {event_type} script timeout after {SEAT_SCRIPT_TIMEOUT_SECONDS}s")
        except Exception as e:
            self._seat_log(f"[SeatTrigger] {seat_label} {event_type} script error: {e}")

    def _process_seat_cycle(
        self,
        seat_key: str,
        seat_label: str,
        occupied: bool,
        current_time: float
    ) -> None:
        state = self._seat_states[seat_key]

        if state.phase == "IDLE":
            if occupied:
                state.phase = "OCCUPY_PENDING"
                state.occupied_start = current_time
                state.leave_start = None
                self._seat_log(
                    f"[SeatTrigger] {seat_label} occupied detected, start occupy timer "
                    f"(t={current_time:.2f}, phase=IDLE)"
                )
            else:
                self._seat_log(
                    f"[SeatTrigger] {seat_label} idle snapshot "
                    f"(t={current_time:.2f}, occupied={occupied})"
                )
            return

        if state.phase == "OCCUPY_PENDING":
            if occupied:
                elapsed = current_time - state.occupied_start
                self._seat_log(
                    f"[SeatTrigger] {seat_label} occupy pending "
                    f"(elapsed={elapsed:.2f}s/{SEAT_TRIGGER_STABLE_SECONDS:.2f}s, "
                    f"start={state.occupied_start:.2f}, now={current_time:.2f})"
                )
                if elapsed >= SEAT_TRIGGER_STABLE_SECONDS:
                    self._seat_log(
                        f"[SeatTrigger] {seat_label} occupied stable for {elapsed:.2f}s, trigger occupy event"
                    )
                    self._enqueue_seat_script(seat_label, "OCCUPY")
                    state.phase = "OCCUPIED_TRIGGERED"
                    state.occupied_start = None
                    state.leave_start = None
            else:
                self._seat_log(
                    f"[SeatTrigger] {seat_label} occupy timer cancelled "
                    f"(occupied became False at t={current_time:.2f})"
                )
                self._reset_seat_state(seat_key)
            return

        if state.phase == "OCCUPIED_TRIGGERED":
            if not occupied:
                state.phase = "LEAVE_PENDING"
                state.leave_start = current_time
                self._seat_log(
                    f"[SeatTrigger] {seat_label} leave detected, start leave timer "
                    f"(t={current_time:.2f}, phase=OCCUPIED_TRIGGERED)"
                )
            else:
                self._seat_log(
                    f"[SeatTrigger] {seat_label} occupied already triggered, keep waiting "
                    f"(t={current_time:.2f})"
                )
            return

        if state.phase == "LEAVE_PENDING":
            if not occupied:
                elapsed = current_time - state.leave_start
                self._seat_log(
                    f"[SeatTrigger] {seat_label} leave pending "
                    f"(elapsed={elapsed:.2f}s/{SEAT_TRIGGER_STABLE_SECONDS:.2f}s, "
                    f"start={state.leave_start:.2f}, now={current_time:.2f})"
                )
                if elapsed >= SEAT_TRIGGER_STABLE_SECONDS:
                    self._seat_log(
                        f"[SeatTrigger] {seat_label} leave stable for {elapsed:.2f}s, trigger leave event"
                    )
                    self._enqueue_seat_script(seat_label, "LEAVE")
                    self._reset_seat_state(seat_key)
            else:
                self._seat_log(
                    f"[SeatTrigger] {seat_label} leave timer cancelled "
                    f"(occupied became True at t={current_time:.2f})"
                )
                state.phase = "OCCUPIED_TRIGGERED"
                state.leave_start = None

    def _check_seat_occupancy_trigger(self, parsed_data: ParsedCANData):
        """检查主驾/副驾座椅状态并驱动各自的占座/离座状态机。"""
        current_time = time.time()
        driver_raw = parsed_data.parsed_signals.get("VIU_DriverSeatOccptSt")
        pass_raw = parsed_data.parsed_signals.get("VIU_PassSeatOccptSt")
        driver_occ = driver_raw == "Occupied"
        pass_occ = pass_raw == "Occupied"

        self._seat_log(
            f"[SeatTrigger] snapshot t={current_time:.2f} "
            f"driver_raw={driver_raw!r}, driver_phase={self._seat_states['driver'].phase}, "
            f"driver_occupied_start={self._seat_states['driver'].occupied_start}, "
            f"driver_leave_start={self._seat_states['driver'].leave_start}; "
            f"pass_raw={pass_raw!r}, pass_phase={self._seat_states['passenger'].phase}, "
            f"pass_occupied_start={self._seat_states['passenger'].occupied_start}, "
            f"pass_leave_start={self._seat_states['passenger'].leave_start}"
        )

        self._process_seat_cycle("driver", "Driver", driver_occ, current_time)
        self._process_seat_cycle("passenger", "Passenger", pass_occ, current_time)


    def start(self):
        """启动接收器（每个 CAN 接口一个接收线程）"""
        current_thread = threading.current_thread()
        current_thread.name = "CAN0ReceiverService"
        print(f"[Receiver] Main thread: {current_thread.name}")

        # 打开所有 CAN 接口
        opened = []
        for drv in self.can_drivers:
            if drv.open():
                opened.append(drv)
            else:
                print(f"[Receiver] Failed to open {drv.interface}, skipping")
        if not opened:
            print("[Receiver] No CAN interfaces available")
            return

        self.running = True
        self._start_backend_ac_threads()

        if self.enable_socket:
            self.socket_server.start()
            self._setup_socket_callbacks()
            self.socket_thread = threading.Thread(
                target=self._socket_accept_loop,
                daemon=True,
                name="CAN0SocketService"
            )
            self.socket_thread.start()
            print(f"[Receiver] Socket accept thread started: {self.socket_thread.name}")
            self._start_ai_state_publisher()

        print(f"[Receiver] Active interfaces: {[d.interface for d in opened]}")
        print(f"[Receiver] Parsing {len(self.parser.parsed_ids)} CAN IDs")
        if self.enable_socket:
            print(f"[Receiver] Socket: {self.socket_server.socket_path}")
        print("-" * 60)

        # 每接口一个接收线程
        recv_threads = []
        for drv in opened:
            t = threading.Thread(
                target=self._recv_loop, args=(drv,),
                daemon=True, name=f"Recv-{drv.interface}"
            )
            t.start()
            recv_threads.append(t)

        # 定期统计
        while self.running:
            time.sleep(5)
            if self.running:
                self._show_periodic_stats()

        # 等待接收线程结束
        for t in recv_threads:
            t.join(timeout=2)

        self._stop_backend_ac_threads()
        if self.ai_state_publish_thread and self.ai_state_publish_thread.is_alive():
            self.ai_state_publish_thread.join(timeout=2.0)
        self._show_final_stats()
        if self.socket_server:
            self.socket_server.stop()
        for drv in self.can_drivers:
            drv.close()

    def _recv_loop(self, can_driver: SocketCANInterface):
        """单个 CAN 接口的接收循环"""
        iface = can_driver.interface
        print(f"[{iface}] Receive thread started")

        while self.running:
            try:
                message = can_driver.receive(timeout=0.001)
                if message is None:
                    continue

                with self._stats_lock:
                    self.stats['total_received'] += 1

                frame = self._convert_to_canframe(message, iface)

                if not self.parser.should_parse(frame.can_id):
                    continue

                parsed_data = self.parser.parse(frame)
                if parsed_data:
                    # 记录 CAN ID → 接口 映射（用于写操作自动路由）
                    self._id_route[frame.can_id] = iface

                    with self._stats_lock:
                        self.stats['parsed_count'] += 1

                    self.frame_counters[frame.can_id] += 1
                    if SHOW_PARSED_ONLY:
                        self._display_parsed_data(parsed_data)
                    self._trigger_callbacks(parsed_data)
                    self._send_to_socket(parsed_data)

                    if frame.can_id == 0x3BE:
                        self._check_seat_occupancy_trigger(parsed_data)

            except Exception as e:
                with self._stats_lock:
                    self.stats['error_count'] += 1
                if self.running:
                    print(f"[{iface}] Error: {e}")

        print(f"[{iface}] Receive thread stopped")

    def _show_final_stats(self):
        elapsed = time.time() - self.stats['start_time']
        total_parsed_frames = sum(self.frame_counters.values())
        
        print("-" * 60)
        print(f"[Receiver] Final Statistics:")
        print(f"  Total received (all CAN IDs): {self.stats['total_received']}")
        print(f"  Total parsed frames: {self.stats['parsed_count']}")
        print(f"  Errors: {self.stats['error_count']}")
        print(f"  Runtime: {elapsed:.1f}s")
        print(f"  Average rate: {self.stats['parsed_count']/elapsed:.1f} fps")
        print(f"\n  Per-ID frame counts (parsed):")
        
        id_names = {
            0x387: "VIU_ACSet1",
            0x381: "VIU_ACSet2",
            0x33A: "AC_Sts1",
            0x33B: "AC_Sts2",
            0x33C: "AC_Sts3",
            0x35B: "VIU_35B",
            0x3B9: "VIU_LIN1",
            0x33E: "AC_Temp2",
            0x33F: "AC_Temp3",
            0x338: "AC_Temp1",
            0x18F: "VIU_CHA_18F",
            0x3BE: "VIU_PassiveSecSeat_3BE",
            0x371: "AC_Temp5",
            0x370: "AC_Temp4",
            0x541: "AC_RHVAC",
            0x552: "VIU_DSM_552",
            0x553: "VIU_PSM_553",
            0x100: "C_100",
        }
        for can_id, count in sorted(self.frame_counters.items()):
            id_name = id_names.get(can_id, f"0x{can_id:03X}")
            percentage = (count / total_parsed_frames * 100) if total_parsed_frames > 0 else 0
            print(f"    {id_name} (0x{can_id:03X}): {count} frames ({percentage:.1f}%)")
        print(f"    Total: {total_parsed_frames} frames")
        # 发送统计
        if SHOW_ALL_SEND:
            print(f"\n[SEND STATS] Total sent: {self.send_stats['total_sent']}, Send errors: {self.send_stats['send_errors']}")
            if self.send_stats['per_id_sent']:
                print(f"  Per-ID sent frames:")
                for cid, cnt in sorted(self.send_stats['per_id_sent'].items()):
                    print(f"    0x{cid:03X}: {cnt} frames")
    
    def stop(self):
        self.running = False
        self.backend_stop_event.set()


# ============================================================
# 8. 回调函数
# ============================================================
def on_ac_data_received(data):
    if data.can_id == 0x33B:
        fault = data.get("AC_FaultREASON", "Normal")
        if fault != "NORMAL" and fault != "Invalid":
            print(f"  [ALERT] AC Fault Detected: {fault}")


# ============================================================
# 9. 主函数
# ============================================================

def run_receiver_process(interfaces: List[str], parsed_ids: List[int], print_interval: int,
                         enable_socket: bool, socket_path: str, enable_rte: bool,
                         cpu_affinity: Optional[int], stop_event: mp.Event):
    """子进程：CANReceiver（多接口）"""
    if cpu_affinity is not None:
        os.sched_setaffinity(0, {cpu_affinity})
        print(f"[Process] CPU affinity set to core {cpu_affinity}")

    receiver = CANReceiver(interfaces=interfaces, parsed_ids=parsed_ids,
                          print_interval=print_interval, enable_socket=enable_socket,
                          socket_path=socket_path, enable_rte=enable_rte)
    receiver.register_callback(on_ac_data_received)
    
    def signal_handler(signum, frame):
        print("\n[Process] Received shutdown signal")
        receiver.stop()
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    receiver.start()


def main():
    print("=" * 60)
    print(f"{SOFTWARE_NAME}")
    print(f"Version: {SOFTWARE_VERSION} | Build: {BUILD_DATE}")
    print("=" * 60)

    print(f"\n[Config] CAN interfaces: {CAN_INTERFACES}")
    if CPU_AFFINITY is not None:
        print(f"[Config] CPU affinity: core {CPU_AFFINITY}")
    print(f"[Config] Parsing {len(PARSED_CAN_IDS)} CAN IDs across all interfaces:")
    id_names = {
        0x387: "VIU_ACSet1",
        0x381: "VIU_ACSet2",
        0x33A: "AC_Sts1",
        0x33B: "AC_Sts2",
        0x33C: "AC_Sts3",
        0x35B: "VIU_35B",
        0x3B9: "VIU_LIN1",
        0x33E: "AC_Temp2",
        0x33F: "AC_Temp3",
        0x338: "AC_Temp1",
        0x18F: "VIU_CHA_18F",
        0x3BE: "VIU_PassiveSecSeat_3BE",
            0x371: "AC_Temp5",
            0x370: "AC_Temp4",
            0x541: "AC_RHVAC",
            0x552: "VIU_DSM_552",
            0x553: "VIU_PSM_553",
            0x100: "C_100",
    }
    for can_id in PARSED_CAN_IDS:
        print(f"  - 0x{can_id:03X}: {id_names[can_id]}")

    print()
    print(f"[Config] Socket path: {SOCKET_PATH}")
    print(f"[Config] RTE conversion: ON (signals will be converted to RTE format)")
    print(f"[Config] To receive data, connect to socket using client script")
    print()
    print(f"[Process] Starting CAN receiver in separate process...")

    stop_event = mp.Event()

    receiver_process = mp.Process(
        target=run_receiver_process,
        args=(CAN_INTERFACES, PARSED_CAN_IDS, PRINT_INTERVAL, True, SOCKET_PATH, False, CPU_AFFINITY, stop_event),
        name='CANReceiverProcess'
    )
    receiver_process.start()
    
    print(f"[Process] Receiver process started with PID: {receiver_process.pid}")
    print(f"[Process] Press Ctrl+C to stop...")
    
    try:
        receiver_process.join()
    except KeyboardInterrupt:
        print("\n[Main] Stopping receiver process...")
        stop_event.set()
        receiver_process.terminate()
        receiver_process.join(timeout=2)
        if receiver_process.is_alive():
            receiver_process.kill()
        print("[Main] Receiver process stopped.")


if __name__ == '__main__':
    main()
