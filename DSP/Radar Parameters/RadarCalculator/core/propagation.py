"""
RF propagation models for terrain-aware radar coverage analysis.

Provides:
- Line-of-sight (LOS) calculation with earth curvature
- Free-space path loss with terrain masking
- Longley-Rice (ITM) model (optional, requires SPLAT!)
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, TYPE_CHECKING

from .constants import C, PI, EARTH_RADIUS_M, EARTH_RADIUS_43_M

if TYPE_CHECKING:
    from .terrain import TerrainData


class PropagationModel(Enum):
    """Available propagation model types."""
    LINE_OF_SIGHT = auto()
    FREE_SPACE_TERRAIN = auto()
    LONGLEY_RICE = auto()


@dataclass
class PropagationParams:
    """
    Parameters for propagation calculations.

    All heights are Above Ground Level (AGL) unless noted.
    """
    radar_height_m: float = 30.0        # Radar antenna height AGL
    target_height_m: float = 100.0      # Target height AGL
    refractivity_n: float = 301.0       # Surface refractivity (N-units)
    ground_conductivity: float = 0.005  # Ground conductivity (S/m)
    ground_permittivity: float = 15.0   # Relative permittivity
    climate: int = 5                     # ITM climate code (5=continental temperate)
    polarization: int = 0                # 0=horizontal, 1=vertical
    use_43_earth: bool = True           # Use 4/3 effective earth radius for refraction


@dataclass
class LOSResult:
    """Result of line-of-sight calculation."""
    is_visible: bool
    clearance_m: float          # Minimum clearance above terrain (negative = blocked)
    clearance_fraction: float   # Clearance as fraction of first Fresnel zone (>1 = clear)
    blocking_distance_m: float  # Distance to blocking terrain (0 if clear)
    blocking_elevation_m: float # Elevation of blocking terrain


class PropagationCalculator(ABC):
    """
    Abstract base class for propagation models.

    All propagation calculators implement methods for:
    - Checking line-of-sight visibility
    - Calculating path loss in dB
    """

    @abstractmethod
    def is_visible(self, terrain: 'TerrainData',
                   radar_lat: float, radar_lon: float,
                   target_lat: float, target_lon: float,
                   frequency_hz: float,
                   params: PropagationParams) -> LOSResult:
        """
        Determine if target is visible from radar.

        Args:
            terrain: TerrainData object with elevation grid
            radar_lat, radar_lon: Radar position (degrees)
            target_lat, target_lon: Target position (degrees)
            frequency_hz: Operating frequency
            params: Propagation parameters

        Returns:
            LOSResult with visibility status and details
        """
        pass

    @abstractmethod
    def calculate_loss(self, terrain: 'TerrainData',
                       radar_lat: float, radar_lon: float,
                       target_lat: float, target_lon: float,
                       frequency_hz: float,
                       params: PropagationParams) -> float:
        """
        Calculate total path loss in dB.

        Args:
            terrain: TerrainData object with elevation grid
            radar_lat, radar_lon: Radar position (degrees)
            target_lat, target_lon: Target position (degrees)
            frequency_hz: Operating frequency
            params: Propagation parameters

        Returns:
            One-way path loss in dB (use this value twice for radar two-way)
        """
        pass


class LineOfSightCalculator(PropagationCalculator):
    """
    Simple geometric line-of-sight calculation with earth curvature.

    Uses 4/3 effective earth radius model to account for atmospheric
    refraction under standard conditions.
    """

    def __init__(self, num_profile_points: int = 100):
        """
        Initialize LOS calculator.

        Args:
            num_profile_points: Number of points to sample along path
        """
        self.num_profile_points = num_profile_points

    def is_visible(self, terrain: 'TerrainData',
                   radar_lat: float, radar_lon: float,
                   target_lat: float, target_lon: float,
                   frequency_hz: float,
                   params: PropagationParams) -> LOSResult:
        """Check if target is in line-of-sight."""
        from .terrain import haversine_distance

        # Get terrain profile
        distances, elevations = terrain.get_elevation_profile(
            radar_lat, radar_lon, target_lat, target_lon,
            self.num_profile_points
        )

        total_distance = distances[-1]
        if total_distance < 1:
            return LOSResult(True, params.target_height_m, float('inf'), 0, 0)

        # Get radar and target ground elevations
        radar_ground = terrain.get_elevation(radar_lat, radar_lon)
        target_ground = terrain.get_elevation(target_lat, target_lon)

        # Absolute heights
        radar_height = radar_ground + params.radar_height_m
        target_height = target_ground + params.target_height_m

        # Earth radius for curvature calculation
        earth_r = EARTH_RADIUS_43_M if params.use_43_earth else EARTH_RADIUS_M

        # Calculate LOS line height at each point, accounting for earth curvature
        min_clearance = float('inf')
        blocking_distance = 0
        blocking_elevation = 0

        for i, (d, elev) in enumerate(zip(distances, elevations)):
            if d == 0 or d == total_distance:
                continue

            # Earth curvature correction (bulge at midpoint)
            # h_bulge = d * (D - d) / (2 * R_earth)
            earth_bulge = d * (total_distance - d) / (2 * earth_r)

            # Linear interpolation of LOS line height
            t = d / total_distance
            los_height = radar_height + t * (target_height - radar_height)

            # Effective LOS height accounting for earth curvature
            # The ground appears to bulge up relative to the straight line
            los_height_corrected = los_height - earth_bulge

            # Clearance is LOS height minus terrain elevation
            clearance = los_height_corrected - elev

            if clearance < min_clearance:
                min_clearance = clearance
                if clearance < 0:
                    blocking_distance = d
                    blocking_elevation = elev

        # Calculate first Fresnel zone radius at point of minimum clearance
        wavelength = C / frequency_hz
        if blocking_distance > 0:
            d1 = blocking_distance
            d2 = total_distance - blocking_distance
            fresnel_radius = math.sqrt(wavelength * d1 * d2 / total_distance)
        else:
            # Use midpoint for Fresnel calculation if no blocking
            fresnel_radius = math.sqrt(wavelength * total_distance / 4)

        clearance_fraction = min_clearance / fresnel_radius if fresnel_radius > 0 else float('inf')

        return LOSResult(
            is_visible=min_clearance >= 0,
            clearance_m=min_clearance,
            clearance_fraction=clearance_fraction,
            blocking_distance_m=blocking_distance,
            blocking_elevation_m=blocking_elevation
        )

    def calculate_loss(self, terrain: 'TerrainData',
                       radar_lat: float, radar_lon: float,
                       target_lat: float, target_lon: float,
                       frequency_hz: float,
                       params: PropagationParams) -> float:
        """
        Calculate path loss for LOS model.

        Returns 0 dB if clear LOS (just geometry, no atmospheric loss)
        Returns infinity if blocked by terrain.
        """
        los_result = self.is_visible(
            terrain, radar_lat, radar_lon, target_lat, target_lon,
            frequency_hz, params
        )

        if los_result.is_visible:
            return 0.0  # LOS model: no additional loss beyond free space
        else:
            return float('inf')  # Blocked


class FreeSpaceTerrainCalculator(PropagationCalculator):
    """
    Free-space path loss with terrain masking.

    Combines free-space path loss (1/R^2 for one-way) with terrain
    obstruction checking. If terrain blocks LOS, returns infinite loss.
    """

    def __init__(self, num_profile_points: int = 100):
        """Initialize calculator."""
        self._los_calc = LineOfSightCalculator(num_profile_points)

    def is_visible(self, terrain: 'TerrainData',
                   radar_lat: float, radar_lon: float,
                   target_lat: float, target_lon: float,
                   frequency_hz: float,
                   params: PropagationParams) -> LOSResult:
        """Check LOS visibility (delegates to LOS calculator)."""
        return self._los_calc.is_visible(
            terrain, radar_lat, radar_lon, target_lat, target_lon,
            frequency_hz, params
        )

    def calculate_loss(self, terrain: 'TerrainData',
                       radar_lat: float, radar_lon: float,
                       target_lat: float, target_lon: float,
                       frequency_hz: float,
                       params: PropagationParams) -> float:
        """
        Calculate free-space path loss with terrain masking.

        Returns:
            One-way free-space path loss in dB, or infinity if blocked
        """
        # Check LOS first
        los_result = self._los_calc.is_visible(
            terrain, radar_lat, radar_lon, target_lat, target_lon,
            frequency_hz, params
        )

        if not los_result.is_visible:
            return float('inf')

        # Calculate distance
        from .terrain import haversine_distance
        distance_m = haversine_distance(radar_lat, radar_lon, target_lat, target_lon)

        if distance_m < 1:
            return 0.0

        # Free-space path loss (one-way)
        # FSPL = 20*log10(4*pi*d/lambda)
        wavelength = C / frequency_hz
        fspl_db = 20 * math.log10((4 * PI * distance_m) / wavelength)

        return fspl_db


class LongleyRiceCalculator(PropagationCalculator):
    """
    Longley-Rice Irregular Terrain Model (ITM).

    Provides accurate propagation predictions considering:
    - Terrain diffraction
    - Tropospheric scatter
    - Surface reflection
    - Atmospheric refraction

    Requires optional dependency (SPLAT! or pysplat).
    """

    def __init__(self):
        """Initialize and check for ITM availability."""
        self._splat_available = self._check_splat()
        self._fallback_calc = FreeSpaceTerrainCalculator()

    def _check_splat(self) -> bool:
        """Check if SPLAT!/ITM is available."""
        try:
            # Try importing pysplat or similar
            # This is a placeholder - actual implementation depends on
            # which ITM library is installed
            import subprocess
            result = subprocess.run(['splat', '-h'],
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        """Check if Longley-Rice model is available."""
        return self._splat_available

    def is_visible(self, terrain: 'TerrainData',
                   radar_lat: float, radar_lon: float,
                   target_lat: float, target_lon: float,
                   frequency_hz: float,
                   params: PropagationParams) -> LOSResult:
        """Check visibility using ITM or fallback."""
        if not self._splat_available:
            return self._fallback_calc.is_visible(
                terrain, radar_lat, radar_lon, target_lat, target_lon,
                frequency_hz, params
            )

        # ITM-based visibility check would go here
        # For now, use fallback
        return self._fallback_calc.is_visible(
            terrain, radar_lat, radar_lon, target_lat, target_lon,
            frequency_hz, params
        )

    def calculate_loss(self, terrain: 'TerrainData',
                       radar_lat: float, radar_lon: float,
                       target_lat: float, target_lon: float,
                       frequency_hz: float,
                       params: PropagationParams) -> float:
        """
        Calculate path loss using Longley-Rice ITM.

        Falls back to free-space + terrain if ITM not available.
        """
        if not self._splat_available:
            return self._fallback_calc.calculate_loss(
                terrain, radar_lat, radar_lon, target_lat, target_lon,
                frequency_hz, params
            )

        # Full ITM calculation would go here using SPLAT! or pysplat
        # This requires:
        # 1. Write terrain profile to SPLAT! format
        # 2. Run SPLAT! with appropriate parameters
        # 3. Parse output for path loss

        # Placeholder: use fallback for now
        return self._fallback_calc.calculate_loss(
            terrain, radar_lat, radar_lon, target_lat, target_lon,
            frequency_hz, params
        )


def create_propagation_calculator(model: PropagationModel,
                                  num_profile_points: int = 100) -> PropagationCalculator:
    """
    Factory function to create propagation calculator.

    Args:
        model: PropagationModel enum value
        num_profile_points: Number of points for terrain profiling

    Returns:
        PropagationCalculator instance
    """
    if model == PropagationModel.LINE_OF_SIGHT:
        return LineOfSightCalculator(num_profile_points)
    elif model == PropagationModel.FREE_SPACE_TERRAIN:
        return FreeSpaceTerrainCalculator(num_profile_points)
    elif model == PropagationModel.LONGLEY_RICE:
        return LongleyRiceCalculator()
    else:
        raise ValueError(f"Unknown propagation model: {model}")


def fresnel_zone_radius(wavelength_m: float, d1_m: float, d2_m: float,
                        zone_number: int = 1) -> float:
    """
    Calculate Fresnel zone radius at a point.

    Args:
        wavelength_m: Wavelength in meters
        d1_m: Distance from transmitter to point
        d2_m: Distance from point to receiver
        zone_number: Fresnel zone number (1 = first Fresnel zone)

    Returns:
        Fresnel zone radius in meters
    """
    total_d = d1_m + d2_m
    if total_d == 0:
        return 0.0
    return math.sqrt(zone_number * wavelength_m * d1_m * d2_m / total_d)


def knife_edge_diffraction_loss(clearance_m: float, fresnel_radius_m: float) -> float:
    """
    Estimate knife-edge diffraction loss.

    Uses simplified Fresnel-Kirchhoff formula.

    Args:
        clearance_m: Clearance above obstacle (negative if blocked)
        fresnel_radius_m: First Fresnel zone radius at obstacle

    Returns:
        Additional path loss in dB due to diffraction
    """
    if fresnel_radius_m == 0:
        return 0.0 if clearance_m >= 0 else float('inf')

    # Fresnel diffraction parameter
    v = -clearance_m * math.sqrt(2) / fresnel_radius_m

    if v < -0.78:
        # Well into LOS region - no additional loss
        return 0.0
    elif v > 2.4:
        # Deep shadow region - approximate formula
        return 20 * math.log10(v) + 9.0
    else:
        # Transition region - use polynomial approximation
        # Based on ITU-R P.526 approximation
        return 6.9 + 20 * math.log10(math.sqrt((v - 0.1)**2 + 1) + v - 0.1)
