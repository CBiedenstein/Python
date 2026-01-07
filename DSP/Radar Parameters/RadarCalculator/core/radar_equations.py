"""
Consolidated radar equation calculations.
All calculations use SI base units internally (Hz, m, W, etc.)

This module consolidates the logic from:
- radarRange.py
- Radar_Range_MDS.py
- radar_range_RCS.py
- Range_Calculator_matrix.py
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .constants import C, K_BOLTZMANN, T0, PI


@dataclass
class RadarParameters:
    """Radar system parameters for calculations."""
    frequency_hz: float
    peak_power_w: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    bandwidth_hz: float
    noise_figure_db: float
    system_loss_db: float
    required_snr_db: float = 13.0  # Default for high Pd

    @property
    def wavelength_m(self) -> float:
        """Calculate wavelength in meters."""
        return C / self.frequency_hz

    @property
    def total_gain_dbi(self) -> float:
        """Combined Tx + Rx gain in dBi."""
        return self.tx_gain_dbi + self.rx_gain_dbi


@dataclass
class TargetScenario:
    """Target parameters for a specific scenario."""
    rcs_m2: float
    range_m: float
    name: str = "Custom Target"


@dataclass
class CalculationResults:
    """Complete results from radar performance calculation."""
    # Input echo
    frequency_hz: float
    wavelength_m: float
    peak_power_w: float
    peak_power_dbm: float
    range_m: float
    rcs_m2: float
    rcs_dbsm: float

    # Calculated values
    path_loss_db: float
    received_power_dbm: float
    thermal_noise_dbm: float
    mds_dbm: float
    sensitivity_dbm: float
    snr_db: float

    # Detection status
    detected: bool
    detection_margin_db: float

    # Derived values
    max_range_m: float
    min_detectable_rcs_m2: float

    # Reference
    fspl_one_way_db: float


class RadarCalculator:
    """
    Core radar performance calculator.

    Implements the monostatic radar range equation and derived calculations.
    """

    def __init__(self, params: RadarParameters):
        """Initialize calculator with radar system parameters."""
        self.params = params
        self._cache_linear_values()

    def _cache_linear_values(self):
        """Pre-compute linear values from dB parameters for efficiency."""
        self._nf_linear = 10 ** (self.params.noise_figure_db / 10)
        self._snr_req_linear = 10 ** (self.params.required_snr_db / 10)
        self._loss_linear = 10 ** (self.params.system_loss_db / 10)
        self._tx_gain_linear = 10 ** (self.params.tx_gain_dbi / 10)
        self._rx_gain_linear = 10 ** (self.params.rx_gain_dbi / 10)

    def update_params(self, params: RadarParameters):
        """Update radar parameters and recache linear values."""
        self.params = params
        self._cache_linear_values()

    # =========================================================================
    # Core Calculations
    # =========================================================================

    def calculate_wavelength(self) -> float:
        """Calculate wavelength in meters."""
        return C / self.params.frequency_hz

    def calculate_path_loss_db(self, range_m: float) -> float:
        """
        Calculate radar path loss in dB.

        Radar path loss formula: 10 * log10( ((4*pi)^3 * R^4) / lambda^2 )
        This accounts for the two-way trip and spreading loss (1/R^4).
        """
        wavelength = self.calculate_wavelength()
        numerator = ((4 * PI) ** 3) * (range_m ** 4)
        denominator = wavelength ** 2
        path_loss_linear = numerator / denominator
        return 10 * math.log10(path_loss_linear)

    def calculate_fspl_one_way_db(self, range_m: float) -> float:
        """Calculate one-way free space path loss (for reference)."""
        wavelength = self.calculate_wavelength()
        return 20 * math.log10((4 * PI * range_m) / wavelength)

    def calculate_received_power_dbm(self, rcs_m2: float, range_m: float) -> float:
        """
        Calculate received power in dBm using radar range equation.

        Pr = Pt + Gt + Gr + RCS - Losses - PathLoss
        """
        # Convert to dB/dBm values
        pt_dbm = 10 * math.log10(self.params.peak_power_w * 1000)  # W to dBm
        rcs_dbsm = 10 * math.log10(rcs_m2) if rcs_m2 > 0 else -999
        path_loss_db = self.calculate_path_loss_db(range_m)

        # Link budget
        pr_dbm = (pt_dbm +
                  self.params.tx_gain_dbi +
                  self.params.rx_gain_dbi +
                  rcs_dbsm -
                  self.params.system_loss_db -
                  path_loss_db)

        return pr_dbm

    def calculate_thermal_noise_dbm(self) -> float:
        """
        Calculate thermal noise floor in dBm.

        N = k * T0 * B (in Watts)
        """
        noise_watts = K_BOLTZMANN * T0 * self.params.bandwidth_hz
        return 10 * math.log10(noise_watts * 1000)  # Convert to dBm

    def calculate_mds_dbm(self) -> float:
        """
        Calculate Minimum Detectable Signal in dBm.

        MDS = Thermal Noise + Noise Figure
        This is the power level where Signal = Noise (0 dB SNR).
        """
        thermal_noise_dbm = self.calculate_thermal_noise_dbm()
        return thermal_noise_dbm + self.params.noise_figure_db

    def calculate_sensitivity_dbm(self) -> float:
        """
        Calculate receiver sensitivity in dBm.

        Sensitivity = MDS + Required SNR
        This is the minimum power needed for reliable detection.
        """
        return self.calculate_mds_dbm() + self.params.required_snr_db

    def calculate_snr_db(self, rcs_m2: float, range_m: float) -> float:
        """Calculate actual SNR in dB for given target and range."""
        pr_dbm = self.calculate_received_power_dbm(rcs_m2, range_m)
        mds_dbm = self.calculate_mds_dbm()
        return pr_dbm - mds_dbm

    # =========================================================================
    # Inverse Calculations
    # =========================================================================

    def calculate_max_range_m(self, rcs_m2: float) -> float:
        """
        Calculate maximum detectable range for a given RCS.

        Solves radar range equation for R given all other parameters.
        """
        wavelength = self.calculate_wavelength()

        # Calculate sensitivity threshold in Watts
        thermal_noise_w = K_BOLTZMANN * T0 * self.params.bandwidth_hz
        p_min_w = thermal_noise_w * self._nf_linear * self._snr_req_linear

        # Solve for R^4
        # R^4 = (Pt * Gt * Gr * lambda^2 * sigma) / ((4pi)^3 * Pmin * Loss)
        numerator = (self.params.peak_power_w *
                     self._tx_gain_linear *
                     self._rx_gain_linear *
                     (wavelength ** 2) *
                     rcs_m2)
        denominator = ((4 * PI) ** 3) * p_min_w * self._loss_linear

        r4 = numerator / denominator
        return r4 ** 0.25

    def calculate_min_rcs_m2(self, range_m: float) -> float:
        """
        Calculate minimum detectable RCS at a given range.

        Solves radar range equation for sigma given all other parameters.
        """
        wavelength = self.calculate_wavelength()

        # Calculate sensitivity threshold in Watts
        thermal_noise_w = K_BOLTZMANN * T0 * self.params.bandwidth_hz
        p_min_w = thermal_noise_w * self._nf_linear * self._snr_req_linear

        # Solve for sigma
        # sigma = (Pmin * Loss * (4pi)^3 * R^4) / (Pt * Gt * Gr * lambda^2)
        numerator = (p_min_w *
                     self._loss_linear *
                     ((4 * PI) ** 3) *
                     (range_m ** 4))
        denominator = (self.params.peak_power_w *
                       self._tx_gain_linear *
                       self._rx_gain_linear *
                       (wavelength ** 2))

        return numerator / denominator

    # =========================================================================
    # Full Performance Calculation
    # =========================================================================

    def calculate_performance(self, target: TargetScenario) -> CalculationResults:
        """
        Perform complete radar performance calculation for a target scenario.

        Returns all relevant metrics in a CalculationResults dataclass.
        """
        wavelength = self.calculate_wavelength()
        pt_dbm = 10 * math.log10(self.params.peak_power_w * 1000)
        rcs_dbsm = 10 * math.log10(target.rcs_m2) if target.rcs_m2 > 0 else -999

        path_loss_db = self.calculate_path_loss_db(target.range_m)
        fspl_one_way_db = self.calculate_fspl_one_way_db(target.range_m)
        pr_dbm = self.calculate_received_power_dbm(target.rcs_m2, target.range_m)

        thermal_noise_dbm = self.calculate_thermal_noise_dbm()
        mds_dbm = self.calculate_mds_dbm()
        sensitivity_dbm = self.calculate_sensitivity_dbm()
        snr_db = pr_dbm - mds_dbm

        detected = snr_db >= self.params.required_snr_db
        detection_margin_db = snr_db - self.params.required_snr_db

        max_range_m = self.calculate_max_range_m(target.rcs_m2)
        min_rcs_m2 = self.calculate_min_rcs_m2(target.range_m)

        return CalculationResults(
            frequency_hz=self.params.frequency_hz,
            wavelength_m=wavelength,
            peak_power_w=self.params.peak_power_w,
            peak_power_dbm=pt_dbm,
            range_m=target.range_m,
            rcs_m2=target.rcs_m2,
            rcs_dbsm=rcs_dbsm,
            path_loss_db=path_loss_db,
            received_power_dbm=pr_dbm,
            thermal_noise_dbm=thermal_noise_dbm,
            mds_dbm=mds_dbm,
            sensitivity_dbm=sensitivity_dbm,
            snr_db=snr_db,
            detected=detected,
            detection_margin_db=detection_margin_db,
            max_range_m=max_range_m,
            min_detectable_rcs_m2=min_rcs_m2,
            fspl_one_way_db=fspl_one_way_db
        )

    # =========================================================================
    # Array/Sweep Calculations for Plotting
    # =========================================================================

    def snr_vs_range(self, rcs_m2: float,
                     range_min_m: float,
                     range_max_m: float,
                     num_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SNR vs range curve for a given RCS.

        Returns:
            Tuple of (range_array, snr_array) in meters and dB
        """
        ranges = np.linspace(range_min_m, range_max_m, num_points)
        snr_values = np.array([self.calculate_snr_db(rcs_m2, r) for r in ranges])
        return ranges, snr_values

    def received_power_vs_range(self, rcs_m2: float,
                                 range_min_m: float,
                                 range_max_m: float,
                                 num_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate received power vs range curve.

        Returns:
            Tuple of (range_array, pr_array) in meters and dBm
        """
        ranges = np.linspace(range_min_m, range_max_m, num_points)
        pr_values = np.array([self.calculate_received_power_dbm(rcs_m2, r) for r in ranges])
        return ranges, pr_values

    def min_rcs_vs_range(self,
                         range_min_m: float,
                         range_max_m: float,
                         num_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate minimum detectable RCS vs range curve.

        Returns:
            Tuple of (range_array, rcs_array) in meters and m²
        """
        ranges = np.linspace(range_min_m, range_max_m, num_points)
        rcs_values = np.array([self.calculate_min_rcs_m2(r) for r in ranges])
        return ranges, rcs_values

    def max_range_vs_rcs(self,
                         rcs_min_m2: float,
                         rcs_max_m2: float,
                         num_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate maximum detection range vs RCS curve.

        Returns:
            Tuple of (rcs_array, range_array) in m² and meters
        """
        # Use log spacing for RCS since it spans many orders of magnitude
        rcs_values = np.logspace(np.log10(rcs_min_m2), np.log10(rcs_max_m2), num_points)
        range_values = np.array([self.calculate_max_range_m(rcs) for rcs in rcs_values])
        return rcs_values, range_values

    def snr_matrix(self,
                   rcs_values_m2: np.ndarray,
                   range_values_m: np.ndarray) -> np.ndarray:
        """
        Calculate SNR matrix for multiple RCS and range combinations.

        Returns:
            2D array of SNR values [rcs_idx, range_idx]
        """
        snr_matrix = np.zeros((len(rcs_values_m2), len(range_values_m)))
        for i, rcs in enumerate(rcs_values_m2):
            for j, rng in enumerate(range_values_m):
                snr_matrix[i, j] = self.calculate_snr_db(rcs, rng)
        return snr_matrix


def print_link_budget(results: CalculationResults, params: RadarParameters):
    """Print a formatted link budget breakdown (for debugging/CLI use)."""
    print("=" * 50)
    print("RADAR SYSTEM PERFORMANCE ANALYSIS")
    print("=" * 50)
    print(f"Frequency:          {results.frequency_hz/1e9:.2f} GHz")
    print(f"Wavelength:         {results.wavelength_m:.4f} m")
    print(f"Target Range:       {results.range_m/1000:.2f} km")
    print(f"Target RCS:         {results.rcs_m2:.2f} m² ({results.rcs_dbsm:.2f} dBsm)")
    print("-" * 50)
    print(f"Transmit Power:     {results.peak_power_dbm:.2f} dBm")
    print(f"+ Tx Gain:          {params.tx_gain_dbi:.2f} dBi")
    print(f"+ Rx Gain:          {params.rx_gain_dbi:.2f} dBi")
    print(f"+ RCS:              {results.rcs_dbsm:.2f} dBsm")
    print(f"- System Loss:      {params.system_loss_db:.2f} dB")
    print(f"- Path Loss:        {results.path_loss_db:.2f} dB")
    print("-" * 50)
    print(f"RECEIVED POWER:     {results.received_power_dbm:.2f} dBm")
    print(f"NOISE FLOOR (MDS):  {results.mds_dbm:.2f} dBm")
    print(f"ACTUAL SNR:         {results.snr_db:.2f} dB")
    print("-" * 50)
    if results.detected:
        print(f"STATUS: DETECTED (Margin: +{results.detection_margin_db:.2f} dB)")
    else:
        print(f"STATUS: NOT DETECTED (Below threshold by {-results.detection_margin_db:.2f} dB)")
    print(f"Max Range (this RCS): {results.max_range_m/1000:.2f} km")
    print(f"Min RCS (this range): {results.min_detectable_rcs_m2:.4e} m²")
    print("=" * 50)
