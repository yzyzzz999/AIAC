"""Air thermodynamic properties."""

import math

_R_AIR = 287.05  # J/(kg K), matches the Simulink Rho subsystem


def air_density(
    temp_celsius: float,
    pressure_pa: float = 101_325.0,
    rh_percent: float = 0.0,
) -> float:
    """Return humid-air density [kg/m^3].

    Mirrors Libraries/Rho.slx:
        rho = P / (287.05 * T_K) + 1.335e-5 * RH * e_s / T_K
    where e_s is the Magnus saturation vapor pressure in Pa.
    """
    T_k = temp_celsius + 273.15
    e_s = 610.78 * math.exp(17.27 * temp_celsius / (temp_celsius + 237.3))
    return pressure_pa / (_R_AIR * T_k) + 1.335e-5 * rh_percent * e_s / T_k


def air_density_standard() -> float:
    """Standard dry-air density at 20 C, 101325 Pa."""
    return air_density(20.0)
