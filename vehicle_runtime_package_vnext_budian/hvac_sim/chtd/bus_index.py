"""CHTD bus index constants.

Canonical index mapping between Simulink Bus_CHTD_u / Bus_CHTD_x fields and
Python 0-based array indices.

IMPORTANT: Simulink uses 1-based indexing. Python uses 0-based.
    Simulink u(k)  ↔  Python u[k-1]
    Simulink x(k)  ↔  Python x[k-1]

Source authority:
    buses/CHTD_input_bus.json  (54-element input bus, confirmed from SLDD)
    buses/CHTD_state_bus.json  (28-element state bus, confirmed from SLDD)
    buses/CHTD_output_bus.json (28-element output bus, same layout as state bus)

CRITICAL: HeadTempSp is at Python index x[6] (Simulink x(7)).
    The old thermal.py implementation has N_STATES=27 and is missing HeadTempSp.
    All indices from HeadTempSd onward are wrong in thermal.py until T6 rewrite.
    See TODO in thermal.py.
"""

# ---------------------------------------------------------------------------
# Input bus: Bus_CHTD_u — 54 elements
# Order is STRICTLY the canonical Simulink bus order (Simulink 1-based → Python 0-based).
# ---------------------------------------------------------------------------
CHTD_U_NAMES = [
    # [0]  Simulink u(1)
    "ICT",
    # [1]  Simulink u(2)
    "RawAmbT",
    # [2]  Simulink u(3)  ← Phase 3.5 fix: AmbT is at index 2, NOT 0
    "AmbT",
    # [3]  Simulink u(4)  [SolarFdHoriz alias — see CHTD_defect_policy.md Policy 2]
    "SolarFd",
    # [4]  Simulink u(5)  [SolarFpHoriz alias — see CHTD_defect_policy.md Policy 3]
    "SolarFp",
    # [5]  Simulink u(6)
    "VehSpd",
    # [6]  Simulink u(7)
    "WinShdTEst",
    # [7]  Simulink u(8)  [pre-computed upstream; see CHTD_defect_policy.md Policy 5]
    "FrntDefTmaEst",
    # ── Front vent temperatures ─────────────────────────────────────────────
    # [8]  Simulink u(9)
    "FrntFdvTma",
    # [9]  Simulink u(10)
    "FrntFdfTma",
    # [10] Simulink u(11)
    "FrntFpvTma",
    # [11] Simulink u(12)
    "FrntFpfTma",
    # [12] Simulink u(13)
    "FrntSdvTma",
    # [13] Simulink u(14)
    "FrntSdfTma",
    # [14] Simulink u(15)
    "FrntSpvTma",
    # [15] Simulink u(16)
    "FrntSpfTma",
    # ── Rear vent temperatures ───────────────────────────────────────────────
    # [16] Simulink u(17)
    "RearSdvTma",
    # [17] Simulink u(18)
    "RearSdfTma",
    # [18] Simulink u(19)
    "RearSpvTma",
    # [19] Simulink u(20)
    "RearSpfTma",
    # [20] Simulink u(21)
    "RearTdvTma",
    # [21] Simulink u(22)
    "RearTdfTma",
    # [22] Simulink u(23)
    "RearTpvTma",
    # [23] Simulink u(24)
    "RearTpfTma",
    # ── Front volumetric flows ───────────────────────────────────────────────
    # [24] Simulink u(25)  ← Phase 3.5 fix: FrntFdDefFlow at index 24, NOT 31
    "FrntFdDefFlow",
    # [25] Simulink u(26)  ← Phase 3.5 fix: FrntFdvFlow at index 25, NOT 24
    "FrntFdvFlow",
    # [26] Simulink u(27)  ← Phase 3.5 fix: FrntFdfFlow at index 26, NOT 25
    "FrntFdfFlow",
    # [27] Simulink u(28)
    "FrntFpDefFlow",
    # [28] Simulink u(29)
    "FrntFpvFlow",
    # [29] Simulink u(30)
    "FrntFpfFlow",
    # [30] Simulink u(31)
    "FrntSdvFlow",
    # [31] Simulink u(32)
    "FrntSdfFlow",
    # [32] Simulink u(33)
    "FrntSpvFlow",
    # [33] Simulink u(34)
    "FrntSpfFlow",
    # ── Rear volumetric flows ────────────────────────────────────────────────
    # [34] Simulink u(35)
    "RearSdvFlow",
    # [35] Simulink u(36)
    "RearSdfFlow",
    # [36] Simulink u(37)
    "RearSpvFlow",
    # [37] Simulink u(38)
    "RearSpfFlow",
    # [38] Simulink u(39)
    "RearTdvFlow",
    # [39] Simulink u(40)
    "RearTdfFlow",
    # [40] Simulink u(41)
    "RearTpvFlow",
    # [41] Simulink u(42)
    "RearTpfFlow",
    # ── Human body heat to head zones ────────────────────────────────────────
    # [42] Simulink u(43)
    "HumHeadFdPower",
    # [43] Simulink u(44)
    "HumHeadFpPower",
    # [44] Simulink u(45)
    "HumHeadSdPower",
    # [45] Simulink u(46)
    "HumHeadSpPower",
    # [46] Simulink u(47)
    "HumHeadTdPower",
    # [47] Simulink u(48)
    "HumHeadTpPower",
    # ── Human body heat to feet/cabin zones ──────────────────────────────────
    # [48] Simulink u(49)
    "HumFeetFdPower",
    # [49] Simulink u(50)
    "HumFeetFpPower",
    # [50] Simulink u(51)
    "HumFeetSdPower",
    # [51] Simulink u(52)
    "HumFeetSpPower",
    # [52] Simulink u(53)
    "HumFeetTdPower",
    # [53] Simulink u(54)
    "HumFeetTpPower",
]

N_U = len(CHTD_U_NAMES)  # must be 54

# ---------------------------------------------------------------------------
# State bus: Bus_CHTD_x — 28 elements
# Order is STRICTLY the canonical Simulink bus order (Simulink 1-based → Python 0-based).
#
# CRITICAL: HeadTempSp is at Python index 6 (Simulink x(7)).
#   x[0]  HoodTemp
#   x[1]  CabinFrntTemp
#   x[2]  ConsoleTemp
#   x[3]  HeadTempFd
#   x[4]  HeadTempFp
#   x[5]  HeadTempSd
#   x[6]  HeadTempSp   ← must not be omitted; omission shifts ALL later indices by -1
#   x[7]  HeadTempTd
#   x[8]  HeadTempTp
#   x[9]  FeetTempFd
#   ...
#   x[27] RoofTemp
# ---------------------------------------------------------------------------
CHTD_X_NAMES = [
    # ── Structural zones ─────────────────────────────────────────────────────
    "HoodTemp",         # [0]  Simulink x(1)
    "CabinFrntTemp",    # [1]  Simulink x(2)
    "ConsoleTemp",      # [2]  Simulink x(3)
    # ── Head zones ───────────────────────────────────────────────────────────
    "HeadTempFd",       # [3]  Simulink x(4)
    "HeadTempFp",       # [4]  Simulink x(5)
    "HeadTempSd",       # [5]  Simulink x(6)
    "HeadTempSp",       # [6]  Simulink x(7) ← CRITICAL — do not omit
    "HeadTempTd",       # [7]  Simulink x(8)
    "HeadTempTp",       # [8]  Simulink x(9)
    # ── Feet zones ───────────────────────────────────────────────────────────
    "FeetTempFd",       # [9]  Simulink x(10)
    "FeetTempFp",       # [10] Simulink x(11)
    "FeetTempSd",       # [11] Simulink x(12)
    "FeetTempSp",       # [12] Simulink x(13)
    "FeetTempTd",       # [13] Simulink x(14)
    "FeetTempTp",       # [14] Simulink x(15)
    # ── Cabin air zones ──────────────────────────────────────────────────────
    "CabinTempFd",      # [15] Simulink x(16)
    "CabinTempFp",      # [16] Simulink x(17)
    "CabinTempSd",      # [17] Simulink x(18)
    "CabinTempSp",      # [18] Simulink x(19)
    "CabinTempTd",      # [19] Simulink x(20)
    "CabinTempTp",      # [20] Simulink x(21)
    # ── Window zones ─────────────────────────────────────────────────────────
    "WinTempFd",        # [21] Simulink x(22)
    "WinTempFp",        # [22] Simulink x(23)
    "WinTempSd",        # [23] Simulink x(24)
    "WinTempSp",        # [24] Simulink x(25)
    "WinTempTd",        # [25] Simulink x(26)
    "WinTempTp",        # [26] Simulink x(27)
    # ── Roof zone ────────────────────────────────────────────────────────────
    "RoofTemp",         # [27] Simulink x(28)
]

N_X_STATES = len(CHTD_X_NAMES)  # must be 28

# ---------------------------------------------------------------------------
# Output bus: xStep — same field layout as Bus_CHTD_x.
# Values are dT increments (degC/step), not absolute temperatures.
# Caller integrates: x_new = x_old + xStep.
# ---------------------------------------------------------------------------
CHTD_Y_NAMES = CHTD_X_NAMES  # same field names; alias for clarity
N_Y = N_X_STATES

# ---------------------------------------------------------------------------
# Index lookup dicts  (field name → 0-based Python index)
# ---------------------------------------------------------------------------
U_INDEX: dict = {name: i for i, name in enumerate(CHTD_U_NAMES)}
X_INDEX: dict = {name: i for i, name in enumerate(CHTD_X_NAMES)}
Y_INDEX: dict = X_INDEX  # alias

# ---------------------------------------------------------------------------
# Input bus — integer constants (U_*)
# Named in UPPER_SNAKE_CASE. Value = 0-based Python index.
# ---------------------------------------------------------------------------

# ── Ambient / vehicle / surface estimates ────────────────────────────────────
U_ICT               = 0   # Simulink u(1)  — Initial Cabin Temp (EKF output)
U_RAW_AMB_T         = 1   # Simulink u(2)  — unfiltered exterior ambient
U_AMB_T             = 2   # Simulink u(3)  — filtered ambient; WAS WRONG (u(1)) before Phase 3.5 fix
U_SOLAR_FD          = 3   # Simulink u(4)  — SolarFd (alias for SolarFdHoriz, Policy 2)
U_SOLAR_FP          = 4   # Simulink u(5)  — SolarFp (alias for SolarFpHoriz, Policy 3)
U_VEH_SPD           = 5   # Simulink u(6)  — CAN vehicle speed [km/h]
U_WIN_SHD_T_EST     = 6   # Simulink u(7)  — windshield temperature estimate
U_FRNT_DEF_TMA_EST  = 7   # Simulink u(8)  — front defrost mixed-air temp (upstream module)

# ── Front vent temperatures ──────────────────────────────────────────────────
U_FRNT_FDV_TMA  = 8    # Simulink u(9)
U_FRNT_FDF_TMA  = 9    # Simulink u(10)
U_FRNT_FPV_TMA  = 10   # Simulink u(11)
U_FRNT_FPF_TMA  = 11   # Simulink u(12)
U_FRNT_SDV_TMA  = 12   # Simulink u(13)
U_FRNT_SDF_TMA  = 13   # Simulink u(14)
U_FRNT_SPV_TMA  = 14   # Simulink u(15)
U_FRNT_SPF_TMA  = 15   # Simulink u(16)

# ── Rear vent temperatures ───────────────────────────────────────────────────
U_REAR_SDV_TMA  = 16   # Simulink u(17)
U_REAR_SDF_TMA  = 17   # Simulink u(18)
U_REAR_SPV_TMA  = 18   # Simulink u(19)
U_REAR_SPF_TMA  = 19   # Simulink u(20)
U_REAR_TDV_TMA  = 20   # Simulink u(21)
U_REAR_TDF_TMA  = 21   # Simulink u(22)
U_REAR_TPV_TMA  = 22   # Simulink u(23)
U_REAR_TPF_TMA  = 23   # Simulink u(24)

# ── Front volumetric flows ───────────────────────────────────────────────────
# NOTE: Phase 3.5 corrections — these indices were wrong in the original golden case script.
U_FRNT_FD_DEF_FLOW  = 24   # Simulink u(25)  FrntFdDefFlow  [was u(32) before fix]
U_FRNT_FDV_FLOW     = 25   # Simulink u(26)  FrntFdvFlow    [was u(25) before fix]
U_FRNT_FDF_FLOW     = 26   # Simulink u(27)  FrntFdfFlow    [was u(26) before fix]
U_FRNT_FP_DEF_FLOW  = 27   # Simulink u(28)  FrntFpDefFlow
U_FRNT_FPV_FLOW     = 28   # Simulink u(29)  FrntFpvFlow
U_FRNT_FPF_FLOW     = 29   # Simulink u(30)  FrntFpfFlow
U_FRNT_SDV_FLOW     = 30   # Simulink u(31)  FrntSdvFlow
U_FRNT_SDF_FLOW     = 31   # Simulink u(32)  FrntSdfFlow
U_FRNT_SPV_FLOW     = 32   # Simulink u(33)  FrntSpvFlow
U_FRNT_SPF_FLOW     = 33   # Simulink u(34)  FrntSpfFlow

# ── Rear volumetric flows ────────────────────────────────────────────────────
U_REAR_SDV_FLOW  = 34   # Simulink u(35)
U_REAR_SDF_FLOW  = 35   # Simulink u(36)
U_REAR_SPV_FLOW  = 36   # Simulink u(37)
U_REAR_SPF_FLOW  = 37   # Simulink u(38)
U_REAR_TDV_FLOW  = 38   # Simulink u(39)
U_REAR_TDF_FLOW  = 39   # Simulink u(40)
U_REAR_TPV_FLOW  = 40   # Simulink u(41)
U_REAR_TPF_FLOW  = 41   # Simulink u(42)

# ── Human body heat — head zones ─────────────────────────────────────────────
U_HUM_HEAD_FD_POWER = 42   # Simulink u(43)
U_HUM_HEAD_FP_POWER = 43   # Simulink u(44)
U_HUM_HEAD_SD_POWER = 44   # Simulink u(45)
U_HUM_HEAD_SP_POWER = 45   # Simulink u(46)
U_HUM_HEAD_TD_POWER = 46   # Simulink u(47)
U_HUM_HEAD_TP_POWER = 47   # Simulink u(48)

# ── Human body heat — feet/cabin zones ───────────────────────────────────────
U_HUM_FEET_FD_POWER = 48   # Simulink u(49)
U_HUM_FEET_FP_POWER = 49   # Simulink u(50)
U_HUM_FEET_SD_POWER = 50   # Simulink u(51)
U_HUM_FEET_SP_POWER = 51   # Simulink u(52)
U_HUM_FEET_TD_POWER = 52   # Simulink u(53)
U_HUM_FEET_TP_POWER = 53   # Simulink u(54)

# ── Convenience alias (user-specified) ──────────────────────────────────────
U_AMBT = U_AMB_T  # short alias

# ---------------------------------------------------------------------------
# State bus — integer constants (X_*)
# Named in UPPER_SNAKE_CASE. Value = 0-based Python index.
# ---------------------------------------------------------------------------

# ── Structural zones ─────────────────────────────────────────────────────────
X_HOOD_TEMP       = 0    # Simulink x(1)
X_CABIN_FRNT_TEMP = 1    # Simulink x(2)
X_CONSOLE_TEMP    = 2    # Simulink x(3)

# ── Head zones ───────────────────────────────────────────────────────────────
X_HEAD_TEMP_FD    = 3    # Simulink x(4)
X_HEAD_TEMP_FP    = 4    # Simulink x(5)
X_HEAD_TEMP_SD    = 5    # Simulink x(6)
X_HEAD_TEMP_SP    = 6    # Simulink x(7) ← CRITICAL: was missing in old thermal.py (N_STATES=27)
X_HEAD_TEMP_TD    = 7    # Simulink x(8) ← was index 6 in old thermal.py (off-by-one)
X_HEAD_TEMP_TP    = 8    # Simulink x(9) ← was index 7 in old thermal.py (off-by-one)

# ── Feet zones ───────────────────────────────────────────────────────────────
X_FEET_TEMP_FD    = 9    # Simulink x(10) ← was index 8 in old thermal.py (off-by-one)
X_FEET_TEMP_FP    = 10   # Simulink x(11)
X_FEET_TEMP_SD    = 11   # Simulink x(12)
X_FEET_TEMP_SP    = 12   # Simulink x(13)
X_FEET_TEMP_TD    = 13   # Simulink x(14)
X_FEET_TEMP_TP    = 14   # Simulink x(15)

# ── Cabin air zones ──────────────────────────────────────────────────────────
X_CABIN_TEMP_FD   = 15   # Simulink x(16)
X_CABIN_TEMP_FP   = 16   # Simulink x(17)
X_CABIN_TEMP_SD   = 17   # Simulink x(18)
X_CABIN_TEMP_SP   = 18   # Simulink x(19)
X_CABIN_TEMP_TD   = 19   # Simulink x(20)
X_CABIN_TEMP_TP   = 20   # Simulink x(21)

# ── Window zones ─────────────────────────────────────────────────────────────
X_WIN_TEMP_FD     = 21   # Simulink x(22)
X_WIN_TEMP_FP     = 22   # Simulink x(23)
X_WIN_TEMP_SD     = 23   # Simulink x(24)
X_WIN_TEMP_SP     = 24   # Simulink x(25)
X_WIN_TEMP_TD     = 25   # Simulink x(26)
X_WIN_TEMP_TP     = 26   # Simulink x(27)

# ── Roof zone ────────────────────────────────────────────────────────────────
X_ROOF_TEMP       = 27   # Simulink x(28)

# ---------------------------------------------------------------------------
# Output (xStep) — same indices as state bus
# ---------------------------------------------------------------------------
Y_HOOD_TEMP       = X_HOOD_TEMP
Y_HEAD_TEMP_FD    = X_HEAD_TEMP_FD
Y_HEAD_TEMP_SP    = X_HEAD_TEMP_SP
Y_ROOF_TEMP       = X_ROOF_TEMP
# (full alias set matches X_* — add as needed)

# ---------------------------------------------------------------------------
# Sanity check (executed at import time)
# ---------------------------------------------------------------------------
assert N_U == 54, f"CHTD_U_NAMES length must be 54, got {N_U}"
assert N_X_STATES == 28, f"CHTD_X_NAMES length must be 28, got {N_X_STATES}"
assert CHTD_X_NAMES[6] == "HeadTempSp", \
    f"HeadTempSp must be at index 6, found '{CHTD_X_NAMES[6]}'"
assert CHTD_X_NAMES[27] == "RoofTemp", \
    f"RoofTemp must be at index 27, found '{CHTD_X_NAMES[27]}'"
assert U_INDEX["AmbT"] == 2, \
    f"AmbT must be at U index 2 (Phase 3.5 fix), got {U_INDEX['AmbT']}"
assert U_INDEX["FrntFdvFlow"] == 25, \
    f"FrntFdvFlow must be at U index 25 (Phase 3.5 fix), got {U_INDEX['FrntFdvFlow']}"
