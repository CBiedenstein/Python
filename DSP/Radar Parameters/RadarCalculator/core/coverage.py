"""
Radar coverage calculation engine.

Combines terrain data, propagation models, and radar equations
to compute coverage maps over geographic areas.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, TYPE_CHECKING

from .constants import C, PI, KM_TO_METERS
from .propagation import (
    PropagationModel, PropagationParams, PropagationCalculator,
    create_propagation_calculator
)
from .radar_equations import RadarCalculator, RadarParameters
from .terrain import TerrainData, TerrainBounds, haversine_distance

if TYPE_CHECKING:
    from ..profiles.radar_profile import RadarProfile


@dataclass
class CoverageResult:
    """
    Results of radar coverage calculation.

    Contains grids of various metrics over the coverage area.
    All grids have shape (num_lat, num_lon) where:
    - Rows correspond to latitude (decreasing with row index)
    - Columns correspond to longitude (increasing with column index)
    """
    # Grid coordinates
    lat_array: np.ndarray       # 1D array of latitudes (degrees)
    lon_array: np.ndarray       # 1D array of longitudes (degrees)

    # Terrain data
    elevation_m: np.ndarray     # 2D grid of terrain elevations (m)

    # Propagation results
    distance_m: np.ndarray      # 2D grid of distances from radar (m)
    path_loss_db: np.ndarray    # 2D grid of one-way path loss (dB)
    is_visible: np.ndarray      # 2D boolean grid of LOS visibility

    # Radar performance
    snr_db: np.ndarray          # 2D grid of SNR values (dB)
    detected: np.ndarray        # 2D boolean grid of detection status

    # Radar position
    radar_lat: float
    radar_lon: float
    radar_height_m: float
    radar_elevation_m: float    # Ground elevation at radar site

    # Parameters used
    target_rcs_m2: float
    target_height_m: float
    frequency_hz: float
    max_range_km: float
    propagation_model: PropagationModel

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (num_lat, num_lon) shape of grids."""
        return self.elevation_m.shape

    @property
    def bounds(self) -> TerrainBounds:
        """Return geographic bounds of coverage area."""
        return TerrainBounds(
            lat_min=self.lat_array[-1],
            lat_max=self.lat_array[0],
            lon_min=self.lon_array[0],
            lon_max=self.lon_array[-1]
        )

    @property
    def coverage_area_km2(self) -> float:
        """Calculate total detected area in km^2."""
        # Calculate cell area at center latitude
        center_lat = (self.lat_array[0] + self.lat_array[-1]) / 2
        lat_res = abs(self.lat_array[0] - self.lat_array[-1]) / (len(self.lat_array) - 1)
        lon_res = abs(self.lon_array[-1] - self.lon_array[0]) / (len(self.lon_array) - 1)

        # Cell dimensions in km
        lat_km = lat_res * 111.0
        lon_km = lon_res * 111.0 * math.cos(math.radians(center_lat))
        cell_area_km2 = lat_km * lon_km

        # Count detected cells
        num_detected = np.sum(self.detected)
        return num_detected * cell_area_km2

    @property
    def coverage_percentage(self) -> float:
        """Percentage of area with detection capability."""
        total_cells = self.detected.size
        detected_cells = np.sum(self.detected)
        return 100.0 * detected_cells / total_cells if total_cells > 0 else 0.0

    @property
    def max_detection_range_m(self) -> float:
        """Maximum distance at which target is detected."""
        detected_distances = self.distance_m[self.detected]
        return float(np.max(detected_distances)) if detected_distances.size > 0 else 0.0

    def get_radial_coverage(self, num_azimuths: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract radial coverage (max range vs azimuth).

        Args:
            num_azimuths: Number of azimuth angles to sample

        Returns:
            (azimuths_deg, ranges_m) tuple for polar plotting
        """
        azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)
        ranges = np.zeros(num_azimuths)

        # Create meshgrid of distances and azimuths for each cell
        lat_mesh, lon_mesh = np.meshgrid(self.lat_array, self.lon_array, indexing='ij')

        # Calculate azimuth to each cell from radar
        dlat = lat_mesh - self.radar_lat
        dlon = lon_mesh - self.radar_lon

        # Approximate azimuth (good enough for visualization)
        cell_azimuths = np.degrees(np.arctan2(
            dlon * math.cos(math.radians(self.radar_lat)),
            dlat
        ))
        cell_azimuths = (cell_azimuths + 360) % 360  # Normalize to 0-360

        # For each azimuth bin, find max detected range
        azimuth_bin_width = 360.0 / num_azimuths

        for i, az in enumerate(azimuths):
            # Find cells in this azimuth bin
            az_min = (az - azimuth_bin_width / 2) % 360
            az_max = (az + azimuth_bin_width / 2) % 360

            if az_min < az_max:
                in_bin = (cell_azimuths >= az_min) & (cell_azimuths < az_max)
            else:  # Wrap around 0/360
                in_bin = (cell_azimuths >= az_min) | (cell_azimuths < az_max)

            # Get max detected range in this bin
            detected_in_bin = self.detected & in_bin
            if np.any(detected_in_bin):
                ranges[i] = np.max(self.distance_m[detected_in_bin])

        return azimuths, ranges


class CoverageCalculator:
    """
    Calculates radar coverage over terrain.

    Combines:
    - Terrain elevation data
    - RF propagation models
    - Radar range equation calculations
    """

    def __init__(self, radar_params: RadarParameters,
                 terrain: TerrainData,
                 prop_model: PropagationModel = PropagationModel.FREE_SPACE_TERRAIN,
                 num_profile_points: int = 50):
        """
        Initialize coverage calculator.

        Args:
            radar_params: RadarParameters for radar system
            terrain: TerrainData with elevation grid
            prop_model: Propagation model to use
            num_profile_points: Points for terrain profiling (LOS checks)
        """
        self._radar_params = radar_params
        self._radar_calc = RadarCalculator(radar_params)
        self._terrain = terrain
        self._prop_model = prop_model
        self._propagator = create_propagation_calculator(prop_model, num_profile_points)

    @classmethod
    def from_profile(cls, profile: 'RadarProfile',
                     terrain: TerrainData,
                     prop_model: PropagationModel = PropagationModel.FREE_SPACE_TERRAIN,
                     num_profile_points: int = 50) -> 'CoverageCalculator':
        """
        Create calculator from a RadarProfile.

        Args:
            profile: RadarProfile object
            terrain: TerrainData with elevation grid
            prop_model: Propagation model to use
            num_profile_points: Points for terrain profiling
        """
        params = RadarParameters(
            frequency_hz=profile.frequency_hz,
            peak_power_w=profile.peak_power_w,
            tx_gain_dbi=profile.tx_gain_dbi,
            rx_gain_dbi=profile.rx_gain_dbi,
            bandwidth_hz=profile.bandwidth_hz,
            noise_figure_db=profile.noise_figure_db,
            system_loss_db=profile.system_loss_db,
            required_snr_db=getattr(profile, 'required_snr_db', 13.0)
        )
        return cls(params, terrain, prop_model, num_profile_points)

    def calculate_coverage(self,
                           radar_lat: float,
                           radar_lon: float,
                           radar_height_m: float,
                           target_rcs_m2: float,
                           target_height_m: float = 100.0,
                           max_range_km: float = 150.0,
                           grid_resolution_m: float = 1000.0,
                           progress_callback: Optional[Callable[[float], None]] = None
                           ) -> CoverageResult:
        """
        Calculate coverage grid.

        This is computationally intensive for large areas or fine resolution.

        Args:
            radar_lat: Radar latitude (degrees)
            radar_lon: Radar longitude (degrees)
            radar_height_m: Radar antenna height AGL (meters)
            target_rcs_m2: Target RCS in square meters
            target_height_m: Target height AGL (meters)
            max_range_km: Maximum range to calculate (km)
            grid_resolution_m: Approximate grid cell size (meters)
            progress_callback: Optional callback(progress_fraction) for UI updates

        Returns:
            CoverageResult with all calculated grids
        """
        # Create output grid based on terrain bounds
        bounds = TerrainBounds.from_center_and_range(radar_lat, radar_lon, max_range_km)

        # Calculate number of grid points
        center_lat = bounds.center[0]
        lat_deg_per_m = 1.0 / 111000.0
        lon_deg_per_m = 1.0 / (111000.0 * math.cos(math.radians(center_lat)))

        lat_res = grid_resolution_m * lat_deg_per_m
        lon_res = grid_resolution_m * lon_deg_per_m

        num_lat = int(bounds.lat_span / lat_res) + 1
        num_lon = int(bounds.lon_span / lon_res) + 1

        # Limit grid size for performance
        max_points = 500
        if num_lat > max_points:
            num_lat = max_points
            lat_res = bounds.lat_span / (num_lat - 1)
        if num_lon > max_points:
            num_lon = max_points
            lon_res = bounds.lon_span / (num_lon - 1)

        # Create coordinate arrays
        lat_array = np.linspace(bounds.lat_max, bounds.lat_min, num_lat)
        lon_array = np.linspace(bounds.lon_min, bounds.lon_max, num_lon)

        # Initialize output grids
        elevation_grid = np.zeros((num_lat, num_lon))
        distance_grid = np.zeros((num_lat, num_lon))
        path_loss_grid = np.full((num_lat, num_lon), np.inf)
        visible_grid = np.zeros((num_lat, num_lon), dtype=bool)
        snr_grid = np.full((num_lat, num_lon), -np.inf)
        detected_grid = np.zeros((num_lat, num_lon), dtype=bool)

        # Get radar ground elevation
        radar_elevation = self._terrain.get_elevation(radar_lat, radar_lon)

        # Propagation parameters
        prop_params = PropagationParams(
            radar_height_m=radar_height_m,
            target_height_m=target_height_m
        )

        # Calculate for each grid cell
        max_range_m = max_range_km * KM_TO_METERS
        total_cells = num_lat * num_lon
        cells_processed = 0

        for i, lat in enumerate(lat_array):
            for j, lon in enumerate(lon_array):
                # Get elevation
                elevation_grid[i, j] = self._terrain.get_elevation(lat, lon)

                # Calculate distance
                distance_m = haversine_distance(radar_lat, radar_lon, lat, lon)
                distance_grid[i, j] = distance_m

                # Skip if beyond max range
                if distance_m > max_range_m:
                    cells_processed += 1
                    continue

                # Skip if at radar location
                if distance_m < 10:
                    visible_grid[i, j] = True
                    snr_grid[i, j] = 100.0  # Very high SNR at radar
                    detected_grid[i, j] = True
                    cells_processed += 1
                    continue

                # Calculate propagation loss
                try:
                    one_way_loss = self._propagator.calculate_loss(
                        self._terrain, radar_lat, radar_lon, lat, lon,
                        self._radar_params.frequency_hz, prop_params
                    )
                    path_loss_grid[i, j] = one_way_loss

                    # Check visibility
                    los_result = self._propagator.is_visible(
                        self._terrain, radar_lat, radar_lon, lat, lon,
                        self._radar_params.frequency_hz, prop_params
                    )
                    visible_grid[i, j] = los_result.is_visible

                    if not np.isinf(one_way_loss):
                        # Calculate radar performance
                        # For radar, we need two-way path loss, but the radar
                        # equation already accounts for R^4, so we just use
                        # the standard calculation
                        snr = self._radar_calc.calculate_snr_db(target_rcs_m2, distance_m)

                        # Apply additional propagation loss beyond free space
                        # (if model provides it)
                        fspl_one_way = self._radar_calc.calculate_fspl_one_way_db(distance_m)
                        excess_loss = max(0, one_way_loss - fspl_one_way)

                        # Two-way excess loss for radar
                        snr -= 2 * excess_loss

                        snr_grid[i, j] = snr
                        detected_grid[i, j] = snr >= self._radar_params.required_snr_db

                except Exception:
                    # Handle any calculation errors gracefully
                    pass

                cells_processed += 1

                # Update progress
                if progress_callback and cells_processed % 100 == 0:
                    progress_callback(cells_processed / total_cells)

        # Final progress update
        if progress_callback:
            progress_callback(1.0)

        return CoverageResult(
            lat_array=lat_array,
            lon_array=lon_array,
            elevation_m=elevation_grid,
            distance_m=distance_grid,
            path_loss_db=path_loss_grid,
            is_visible=visible_grid,
            snr_db=snr_grid,
            detected=detected_grid,
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            radar_height_m=radar_height_m,
            radar_elevation_m=radar_elevation,
            target_rcs_m2=target_rcs_m2,
            target_height_m=target_height_m,
            frequency_hz=self._radar_params.frequency_hz,
            max_range_km=max_range_km,
            propagation_model=self._prop_model
        )


def export_coverage_geotiff(coverage: CoverageResult,
                            output_path: str,
                            layer: str = 'snr') -> None:
    """
    Export coverage result to GeoTIFF.

    Args:
        coverage: CoverageResult to export
        output_path: Output file path (.tif)
        layer: Which layer to export ('snr', 'detected', 'elevation', 'distance')

    Raises:
        ImportError: If rasterio not installed
        ValueError: If invalid layer specified
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as e:
        raise ImportError(
            "GeoTIFF export requires 'rasterio' package. "
            "Install with: pip install rasterio"
        ) from e

    # Select data layer
    layer_map = {
        'snr': coverage.snr_db,
        'detected': coverage.detected.astype(np.float32),
        'elevation': coverage.elevation_m,
        'distance': coverage.distance_m,
        'path_loss': coverage.path_loss_db,
        'visible': coverage.is_visible.astype(np.float32)
    }

    if layer not in layer_map:
        raise ValueError(f"Invalid layer '{layer}'. Choose from: {list(layer_map.keys())}")

    data = layer_map[layer].astype(np.float32)

    # Handle infinities for GeoTIFF
    nodata = -9999.0
    data = np.where(np.isinf(data), nodata, data)

    # Create transform
    bounds = coverage.bounds
    transform = from_bounds(
        bounds.lon_min, bounds.lat_min,
        bounds.lon_max, bounds.lat_max,
        data.shape[1], data.shape[0]
    )

    # Write GeoTIFF
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data, 1)

        # Add description
        dst.update_tags(
            LAYER=layer,
            RADAR_LAT=str(coverage.radar_lat),
            RADAR_LON=str(coverage.radar_lon),
            RADAR_HEIGHT_M=str(coverage.radar_height_m),
            TARGET_RCS_M2=str(coverage.target_rcs_m2),
            TARGET_HEIGHT_M=str(coverage.target_height_m),
            FREQUENCY_HZ=str(coverage.frequency_hz),
            PROPAGATION_MODEL=coverage.propagation_model.name
        )
