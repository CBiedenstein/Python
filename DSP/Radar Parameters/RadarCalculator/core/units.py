"""
Unit conversion utilities for radar calculations.
Supports toggling between different unit systems.
"""

import math
from enum import Enum
from typing import Union
from .constants import NMI_TO_METERS, KM_TO_METERS, MI_TO_METERS


class RangeUnit(Enum):
    METERS = "m"
    KILOMETERS = "km"
    NAUTICAL_MILES = "nmi"
    MILES = "mi"


class FrequencyUnit(Enum):
    HERTZ = "Hz"
    KILOHERTZ = "kHz"
    MEGAHERTZ = "MHz"
    GIGAHERTZ = "GHz"


class PowerUnit(Enum):
    WATTS = "W"
    KILOWATTS = "kW"
    MEGAWATTS = "MW"
    DBM = "dBm"
    DBW = "dBW"


class RCSUnit(Enum):
    SQUARE_METERS = "m²"
    DBSM = "dBsm"


class UnitConverter:
    """Handles all unit conversions for radar parameters."""

    # Range conversions (to/from meters)
    @staticmethod
    def range_to_meters(value: float, unit: RangeUnit) -> float:
        """Convert range from specified unit to meters."""
        if unit == RangeUnit.METERS:
            return value
        elif unit == RangeUnit.KILOMETERS:
            return value * KM_TO_METERS
        elif unit == RangeUnit.NAUTICAL_MILES:
            return value * NMI_TO_METERS
        elif unit == RangeUnit.MILES:
            return value * MI_TO_METERS
        raise ValueError(f"Unknown range unit: {unit}")

    @staticmethod
    def range_from_meters(value: float, unit: RangeUnit) -> float:
        """Convert range from meters to specified unit."""
        if unit == RangeUnit.METERS:
            return value
        elif unit == RangeUnit.KILOMETERS:
            return value / KM_TO_METERS
        elif unit == RangeUnit.NAUTICAL_MILES:
            return value / NMI_TO_METERS
        elif unit == RangeUnit.MILES:
            return value / MI_TO_METERS
        raise ValueError(f"Unknown range unit: {unit}")

    # Frequency conversions (to/from Hz)
    @staticmethod
    def freq_to_hz(value: float, unit: FrequencyUnit) -> float:
        """Convert frequency from specified unit to Hz."""
        if unit == FrequencyUnit.HERTZ:
            return value
        elif unit == FrequencyUnit.KILOHERTZ:
            return value * 1e3
        elif unit == FrequencyUnit.MEGAHERTZ:
            return value * 1e6
        elif unit == FrequencyUnit.GIGAHERTZ:
            return value * 1e9
        raise ValueError(f"Unknown frequency unit: {unit}")

    @staticmethod
    def freq_from_hz(value: float, unit: FrequencyUnit) -> float:
        """Convert frequency from Hz to specified unit."""
        if unit == FrequencyUnit.HERTZ:
            return value
        elif unit == FrequencyUnit.KILOHERTZ:
            return value / 1e3
        elif unit == FrequencyUnit.MEGAHERTZ:
            return value / 1e6
        elif unit == FrequencyUnit.GIGAHERTZ:
            return value / 1e9
        raise ValueError(f"Unknown frequency unit: {unit}")

    # Power conversions (to/from Watts)
    @staticmethod
    def power_to_watts(value: float, unit: PowerUnit) -> float:
        """Convert power from specified unit to Watts."""
        if unit == PowerUnit.WATTS:
            return value
        elif unit == PowerUnit.KILOWATTS:
            return value * 1e3
        elif unit == PowerUnit.MEGAWATTS:
            return value * 1e6
        elif unit == PowerUnit.DBM:
            return 10 ** ((value - 30) / 10)
        elif unit == PowerUnit.DBW:
            return 10 ** (value / 10)
        raise ValueError(f"Unknown power unit: {unit}")

    @staticmethod
    def power_from_watts(value: float, unit: PowerUnit) -> float:
        """Convert power from Watts to specified unit."""
        if unit == PowerUnit.WATTS:
            return value
        elif unit == PowerUnit.KILOWATTS:
            return value / 1e3
        elif unit == PowerUnit.MEGAWATTS:
            return value / 1e6
        elif unit == PowerUnit.DBM:
            if value <= 0:
                return -999.0  # Avoid log of zero/negative
            return 10 * math.log10(value) + 30
        elif unit == PowerUnit.DBW:
            if value <= 0:
                return -999.0
            return 10 * math.log10(value)
        raise ValueError(f"Unknown power unit: {unit}")

    # RCS conversions
    @staticmethod
    def rcs_to_m2(value: float, unit: RCSUnit) -> float:
        """Convert RCS from specified unit to square meters."""
        if unit == RCSUnit.SQUARE_METERS:
            return value
        elif unit == RCSUnit.DBSM:
            return 10 ** (value / 10)
        raise ValueError(f"Unknown RCS unit: {unit}")

    @staticmethod
    def rcs_from_m2(value: float, unit: RCSUnit) -> float:
        """Convert RCS from square meters to specified unit."""
        if unit == RCSUnit.SQUARE_METERS:
            return value
        elif unit == RCSUnit.DBSM:
            if value <= 0:
                return -999.0
            return 10 * math.log10(value)
        raise ValueError(f"Unknown RCS unit: {unit}")

    # dB conversions
    @staticmethod
    def linear_to_db(value: float) -> float:
        """Convert linear value to dB."""
        if value <= 0:
            return -999.0
        return 10 * math.log10(value)

    @staticmethod
    def db_to_linear(value_db: float) -> float:
        """Convert dB value to linear."""
        return 10 ** (value_db / 10)

    @staticmethod
    def watts_to_dbm(watts: float) -> float:
        """Convert Watts to dBm."""
        if watts <= 0:
            return -999.0
        return 10 * math.log10(watts * 1000)

    @staticmethod
    def dbm_to_watts(dbm: float) -> float:
        """Convert dBm to Watts."""
        return 10 ** ((dbm - 30) / 10)
