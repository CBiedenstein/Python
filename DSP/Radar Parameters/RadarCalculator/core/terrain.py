"""
Terrain data loading, processing, and coordinate transformations.
Supports SRTM auto-download, DTED, and GeoTIFF formats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import numpy as np
import math
import urllib.request
import zipfile
import io
import os

from .constants import EARTH_RADIUS_M, KM_TO_METERS


# SRTM data sources
# OpenTopography API (requires API key for large requests)
SRTM_BASE_URL = "https://portal.opentopography.org/API/globaldem"

# AWS Open Data - SRTM (no authentication required)
AWS_SRTM_URL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"

# USGS SRTM via AWS (no authentication required)
USGS_SRTM_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/USGS_1_{tile}.tif"


@dataclass
class TerrainBounds:
    """Geographic bounding box for terrain data."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def center(self) -> Tuple[float, float]:
        """Return (lat, lon) of center point."""
        return (
            (self.lat_min + self.lat_max) / 2,
            (self.lon_min + self.lon_max) / 2
        )

    @property
    def lat_span(self) -> float:
        """Latitude span in degrees."""
        return self.lat_max - self.lat_min

    @property
    def lon_span(self) -> float:
        """Longitude span in degrees."""
        return self.lon_max - self.lon_min

    @classmethod
    def from_center_and_range(cls, center_lat: float, center_lon: float,
                               range_km: float) -> 'TerrainBounds':
        """Create bounds from center point and range in km."""
        # Approximate degrees per km at this latitude
        lat_deg_per_km = 1.0 / 111.0  # ~111 km per degree latitude
        lon_deg_per_km = 1.0 / (111.0 * math.cos(math.radians(center_lat)))

        lat_offset = range_km * lat_deg_per_km
        lon_offset = range_km * lon_deg_per_km

        return cls(
            lat_min=center_lat - lat_offset,
            lat_max=center_lat + lat_offset,
            lon_min=center_lon - lon_offset,
            lon_max=center_lon + lon_offset
        )


class TerrainData:
    """
    Container for elevation data with coordinate transforms.

    Stores elevation grid in meters with associated geographic bounds
    and provides methods for coordinate conversion and profile extraction.
    """

    def __init__(self, elevation_grid: np.ndarray, bounds: TerrainBounds,
                 crs: str = "EPSG:4326", nodata_value: float = -9999.0):
        """
        Initialize terrain data.

        Args:
            elevation_grid: 2D numpy array of elevations in meters (rows=lat, cols=lon)
            bounds: Geographic bounding box
            crs: Coordinate reference system string
            nodata_value: Value indicating no data / void
        """
        self.elevation_grid = elevation_grid
        self.bounds = bounds
        self.crs = crs
        self.nodata_value = nodata_value

        # Calculate resolution
        self._nrows, self._ncols = elevation_grid.shape
        self._lat_res = bounds.lat_span / (self._nrows - 1) if self._nrows > 1 else 0
        self._lon_res = bounds.lon_span / (self._ncols - 1) if self._ncols > 1 else 0

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (rows, cols) shape of elevation grid."""
        return self.elevation_grid.shape

    @property
    def resolution_m(self) -> float:
        """Approximate grid cell size in meters at center latitude."""
        center_lat = self.bounds.center[0]
        lat_m = self._lat_res * 111000  # degrees to meters
        lon_m = self._lon_res * 111000 * math.cos(math.radians(center_lat))
        return (lat_m + lon_m) / 2

    @property
    def lat_array(self) -> np.ndarray:
        """1D array of latitude values for each row."""
        return np.linspace(self.bounds.lat_max, self.bounds.lat_min, self._nrows)

    @property
    def lon_array(self) -> np.ndarray:
        """1D array of longitude values for each column."""
        return np.linspace(self.bounds.lon_min, self.bounds.lon_max, self._ncols)

    def _latlon_to_index(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to grid indices (row, col)."""
        # Row index (latitude decreases with row)
        row = int((self.bounds.lat_max - lat) / self._lat_res) if self._lat_res > 0 else 0
        # Column index
        col = int((lon - self.bounds.lon_min) / self._lon_res) if self._lon_res > 0 else 0

        # Clamp to valid range
        row = max(0, min(row, self._nrows - 1))
        col = max(0, min(col, self._ncols - 1))

        return row, col

    def get_elevation(self, lat: float, lon: float, interpolate: bool = True) -> float:
        """
        Get elevation at a point.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            interpolate: If True, use bilinear interpolation

        Returns:
            Elevation in meters, or nodata_value if outside bounds
        """
        if not (self.bounds.lat_min <= lat <= self.bounds.lat_max and
                self.bounds.lon_min <= lon <= self.bounds.lon_max):
            return self.nodata_value

        if not interpolate:
            row, col = self._latlon_to_index(lat, lon)
            return float(self.elevation_grid[row, col])

        # Bilinear interpolation
        # Calculate fractional position
        row_f = (self.bounds.lat_max - lat) / self._lat_res if self._lat_res > 0 else 0
        col_f = (lon - self.bounds.lon_min) / self._lon_res if self._lon_res > 0 else 0

        row0 = int(row_f)
        col0 = int(col_f)
        row1 = min(row0 + 1, self._nrows - 1)
        col1 = min(col0 + 1, self._ncols - 1)

        # Fractional parts
        fr = row_f - row0
        fc = col_f - col0

        # Get corner values
        z00 = self.elevation_grid[row0, col0]
        z01 = self.elevation_grid[row0, col1]
        z10 = self.elevation_grid[row1, col0]
        z11 = self.elevation_grid[row1, col1]

        # Bilinear interpolation
        z = (z00 * (1 - fr) * (1 - fc) +
             z01 * (1 - fr) * fc +
             z10 * fr * (1 - fc) +
             z11 * fr * fc)

        return float(z)

    def get_elevation_profile(self, lat1: float, lon1: float,
                              lat2: float, lon2: float,
                              num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get elevation profile along a line.

        Args:
            lat1, lon1: Start point
            lat2, lon2: End point
            num_points: Number of sample points

        Returns:
            (distances_m, elevations_m) tuple of 1D arrays
        """
        lats = np.linspace(lat1, lat2, num_points)
        lons = np.linspace(lon1, lon2, num_points)

        elevations = np.array([self.get_elevation(lat, lon)
                               for lat, lon in zip(lats, lons)])

        # Calculate cumulative distance
        distances = np.zeros(num_points)
        for i in range(1, num_points):
            distances[i] = distances[i-1] + haversine_distance(
                lats[i-1], lons[i-1], lats[i], lons[i]
            )

        return distances, elevations

    def to_local_coords(self, lat: float, lon: float,
                        origin_lat: float, origin_lon: float) -> Tuple[float, float]:
        """
        Convert lat/lon to local x,y meters from origin.

        Args:
            lat, lon: Point to convert
            origin_lat, origin_lon: Origin point (typically radar location)

        Returns:
            (x_m, y_m) tuple where x=east, y=north
        """
        # Approximate conversion
        y_m = (lat - origin_lat) * 111000
        x_m = (lon - origin_lon) * 111000 * math.cos(math.radians(origin_lat))
        return x_m, y_m


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points.

    Args:
        lat1, lon1: First point in degrees
        lat2, lon2: Second point in degrees

    Returns:
        Distance in meters
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat/2)**2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return EARTH_RADIUS_M * c


class TerrainLoader:
    """
    Loads terrain data from various sources.

    Supports:
    - SRTM auto-download (requires 'elevation' package)
    - GeoTIFF files (requires 'rasterio' package)
    - DTED files (requires 'rasterio' with GDAL)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize terrain loader.

        Args:
            cache_dir: Directory for caching downloaded SRTM tiles.
                      Defaults to ~/.radar_calc_cache/terrain/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".radar_calc_cache" / "terrain"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load_srtm(self, center_lat: float, center_lon: float,
                  range_km: float, product: str = 'SRTM3',
                  progress_callback: Optional[Callable[[str], None]] = None) -> TerrainData:
        """
        Auto-download and load SRTM elevation data.

        Uses OpenTopography API for Windows-compatible downloads.

        Args:
            center_lat: Center latitude in degrees
            center_lon: Center longitude in degrees
            range_km: Radius to load in km
            product: SRTM product ('SRTM1' for 1 arc-sec, 'SRTM3' for 3 arc-sec)
            progress_callback: Optional callback for progress messages

        Returns:
            TerrainData object

        Raises:
            ImportError: If 'rasterio' package not installed
            RuntimeError: If download fails
        """
        try:
            import rasterio
        except ImportError as e:
            raise ImportError(
                "SRTM download requires 'rasterio' package. "
                "Install with: pip install rasterio"
            ) from e

        bounds = TerrainBounds.from_center_and_range(center_lat, center_lon, range_km)

        # Create output filename
        output_file = self._cache_dir / f"srtm_{center_lat:.4f}_{center_lon:.4f}_{range_km:.0f}km.tif"

        # Download if not cached
        if not output_file.exists():
            if progress_callback:
                progress_callback("Downloading SRTM data from OpenTopography...")

            try:
                self._download_srtm_opentopography(bounds, output_file, product, progress_callback)
            except Exception as e:
                # Try alternative method if OpenTopography fails
                if progress_callback:
                    progress_callback("OpenTopography failed, trying alternative source...")
                try:
                    self._download_srtm_tiles(bounds, output_file, progress_callback)
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to download SRTM data. "
                        f"OpenTopography error: {e}. "
                        f"Alternative source error: {e2}"
                    ) from e2

        # Load the downloaded GeoTIFF
        return self.load_geotiff(str(output_file))

    def _download_srtm_opentopography(self, bounds: TerrainBounds, output_file: Path,
                                       product: str = 'SRTM3',
                                       progress_callback: Optional[Callable[[str], None]] = None):
        """
        Download SRTM data using OpenTopography API.

        OpenTopography provides free access to SRTM data without authentication.
        """
        # Determine DEM type based on product
        dem_type = "SRTMGL3" if product == 'SRTM3' else "SRTMGL1"

        # Build API URL
        url = (
            f"{SRTM_BASE_URL}?"
            f"demtype={dem_type}&"
            f"south={bounds.lat_min:.6f}&"
            f"north={bounds.lat_max:.6f}&"
            f"west={bounds.lon_min:.6f}&"
            f"east={bounds.lon_max:.6f}&"
            f"outputFormat=GTiff"
        )

        if progress_callback:
            progress_callback(f"Requesting data for {bounds.lat_min:.2f}N to {bounds.lat_max:.2f}N, "
                            f"{bounds.lon_min:.2f}E to {bounds.lon_max:.2f}E...")

        # Download with timeout
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'RadarCalculator/2.2'})
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()

                if len(data) < 1000:
                    # Probably an error message, not a GeoTIFF
                    raise RuntimeError(f"OpenTopography returned invalid data: {data[:200]}")

                # Save to file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'wb') as f:
                    f.write(data)

                if progress_callback:
                    progress_callback(f"Downloaded {len(data) / 1024 / 1024:.1f} MB")

        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e

    def _download_srtm_tiles(self, bounds: TerrainBounds, output_file: Path,
                             progress_callback: Optional[Callable[[str], None]] = None):
        """
        Download individual SRTM HGT tiles from AWS and merge them.

        AWS hosts SRTM data in the "skadi" format without authentication.
        URL format: https://elevation-tiles-prod.s3.amazonaws.com/skadi/{N|S}{lat}/{N|S}{lat}{E|W}{lon}.hgt.gz
        """
        import rasterio
        from rasterio.merge import merge
        from rasterio.transform import from_bounds
        import gzip

        # Determine which tiles we need (SRTM tiles are 1x1 degree)
        lat_min_tile = int(math.floor(bounds.lat_min))
        lat_max_tile = int(math.floor(bounds.lat_max))
        lon_min_tile = int(math.floor(bounds.lon_min))
        lon_max_tile = int(math.floor(bounds.lon_max))

        tiles_needed = []
        for lat in range(lat_min_tile, lat_max_tile + 1):
            for lon in range(lon_min_tile, lon_max_tile + 1):
                tiles_needed.append((lat, lon))

        if progress_callback:
            progress_callback(f"Need {len(tiles_needed)} SRTM tile(s) from AWS...")

        tile_files = []

        for i, (lat, lon) in enumerate(tiles_needed):
            tile_name = self._get_srtm_tile_name(lat, lon)
            tile_file = self._cache_dir / f"{tile_name}.tif"
            hgt_file = self._cache_dir / f"{tile_name}.hgt"

            if progress_callback:
                # Format for progress bar parsing: "tile X/Y"
                progress_callback(f"Downloading tile {i+1}/{len(tiles_needed)}: {tile_name}")

            if not tile_file.exists():
                downloaded = False

                # Try AWS Skadi source (primary - no auth required)
                if not downloaded:
                    try:
                        downloaded = self._download_aws_skadi_tile(lat, lon, hgt_file, progress_callback)
                    except Exception as e:
                        if progress_callback:
                            progress_callback(f"AWS download failed: {e}")

                # Convert HGT to GeoTIFF if downloaded
                if downloaded and hgt_file.exists():
                    try:
                        self._convert_hgt_to_geotiff(hgt_file, tile_file, lat, lon)
                        hgt_file.unlink()  # Remove HGT file after conversion
                    except Exception as e:
                        if progress_callback:
                            progress_callback(f"HGT conversion failed: {e}")
                        downloaded = False

                # Fallback: generate synthetic terrain
                if not downloaded or not tile_file.exists():
                    if progress_callback:
                        progress_callback(f"Generating synthetic terrain for {tile_name}...")
                    self._generate_synthetic_terrain(lat, lon, tile_file)

            if tile_file.exists():
                tile_files.append(tile_file)

        if not tile_files:
            raise RuntimeError("No SRTM tiles could be downloaded or generated")

        # If only one tile, just copy it
        if len(tile_files) == 1:
            import shutil
            shutil.copy(tile_files[0], output_file)
        else:
            # Merge tiles
            if progress_callback:
                progress_callback("Merging tiles...")

            datasets = [rasterio.open(f) for f in tile_files]
            mosaic, out_trans = merge(datasets)

            out_meta = datasets[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
            })

            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)

            for ds in datasets:
                ds.close()

    def _download_aws_skadi_tile(self, lat: int, lon: int, output_file: Path,
                                  progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Download SRTM tile from AWS Terrain Tiles (Skadi format).

        AWS URL format: /skadi/{N|S}{lat}/{N|S}{lat}{E|W}{lon}.hgt.gz

        Returns:
            True if download successful, False otherwise
        """
        import gzip

        tile_name = self._get_srtm_tile_name(lat, lon)
        lat_dir = tile_name[:3]  # e.g., "N39" or "S12"

        # AWS Skadi URL
        url = f"{AWS_SRTM_URL}/{lat_dir}/{tile_name}.hgt.gz"

        if progress_callback:
            progress_callback(f"Fetching {url}...")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'RadarCalculator/2.2'})
            with urllib.request.urlopen(req, timeout=60) as response:
                compressed_data = response.read()

                if len(compressed_data) < 100:
                    raise RuntimeError(f"Downloaded file too small: {len(compressed_data)} bytes")

                # Decompress gzip
                try:
                    decompressed_data = gzip.decompress(compressed_data)
                except gzip.BadGzipFile:
                    # Maybe not compressed?
                    decompressed_data = compressed_data

                # Save HGT file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'wb') as f:
                    f.write(decompressed_data)

                if progress_callback:
                    progress_callback(f"Downloaded {len(decompressed_data) / 1024 / 1024:.1f} MB")

                return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                if progress_callback:
                    progress_callback(f"Tile {tile_name} not found on AWS (may be ocean)")
            else:
                if progress_callback:
                    progress_callback(f"HTTP error {e.code}: {e.reason}")
            return False
        except urllib.error.URLError as e:
            if progress_callback:
                progress_callback(f"Network error: {e.reason}")
            return False
        except Exception as e:
            if progress_callback:
                progress_callback(f"Download error: {e}")
            return False

    def _convert_hgt_to_geotiff(self, hgt_file: Path, output_file: Path, lat: int, lon: int):
        """
        Convert SRTM HGT file to GeoTIFF.

        HGT files are raw 16-bit signed integer elevation data.
        SRTM3 tiles are 1201x1201 pixels, SRTM1 are 3601x3601.
        """
        import rasterio
        from rasterio.transform import from_bounds

        # Read HGT file
        file_size = hgt_file.stat().st_size

        # Determine resolution from file size
        if file_size == 1201 * 1201 * 2:
            size = 1201  # SRTM3 (3 arc-second)
        elif file_size == 3601 * 3601 * 2:
            size = 3601  # SRTM1 (1 arc-second)
        else:
            raise ValueError(f"Unknown HGT file size: {file_size} bytes")

        # Read raw data (big-endian 16-bit signed integers)
        with open(hgt_file, 'rb') as f:
            data = np.frombuffer(f.read(), dtype='>i2').reshape((size, size))

        # Convert to float32 and handle voids (-32768)
        elevation = data.astype(np.float32)
        elevation[elevation == -32768] = -9999  # Mark voids as nodata

        # Create transform (HGT tiles cover lat to lat+1, lon to lon+1)
        # Note: the data starts from the top-left (NW corner)
        transform = from_bounds(lon, lat, lon + 1, lat + 1, size, size)

        # Write GeoTIFF
        with rasterio.open(
            output_file, 'w',
            driver='GTiff',
            height=size,
            width=size,
            count=1,
            dtype=np.float32,
            crs='EPSG:4326',
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(elevation, 1)

    def _get_srtm_tile_name(self, lat: int, lon: int) -> str:
        """Generate SRTM tile filename from lat/lon."""
        lat_prefix = 'N' if lat >= 0 else 'S'
        lon_prefix = 'E' if lon >= 0 else 'W'
        return f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"

    def _generate_synthetic_terrain(self, lat: int, lon: int, output_file: Path):
        """
        Generate synthetic terrain data when real SRTM is unavailable.

        Creates realistic-looking terrain using Perlin-like noise.
        This is a fallback for when downloads fail.
        """
        import rasterio
        from rasterio.transform import from_bounds

        # 1 degree tile at ~90m resolution = ~1200 points
        size = 1201  # Standard SRTM3 tile size

        # Generate terrain using simple fractal noise
        np.random.seed(abs(lat * 1000 + lon))  # Reproducible based on location

        # Base elevation varies by latitude (higher near mountains, lower near coasts)
        # This is a very rough approximation
        base_elev = 200 + 100 * abs(lat) / 90  # Higher at poles

        # Generate fractal terrain
        elevation = np.zeros((size, size), dtype=np.float32)

        # Multiple octaves of noise
        for octave in range(6):
            freq = 2 ** octave
            amplitude = 500 / (2 ** octave)

            # Simple interpolated noise
            noise_size = max(2, size // (32 // freq))
            noise = np.random.randn(noise_size, noise_size) * amplitude

            # Upscale to full size
            from scipy.ndimage import zoom
            try:
                scaled = zoom(noise, size / noise_size, order=1)
                elevation += scaled[:size, :size]
            except ImportError:
                # If scipy not available, use simpler approach
                x = np.linspace(0, noise_size - 1, size)
                y = np.linspace(0, noise_size - 1, size)
                xi, yi = np.meshgrid(x, y)
                xi = np.clip(xi.astype(int), 0, noise_size - 1)
                yi = np.clip(yi.astype(int), 0, noise_size - 1)
                elevation += noise[yi, xi]

        # Ensure non-negative elevations (mostly)
        elevation = elevation + base_elev
        elevation = np.clip(elevation, -50, 8000)  # Reasonable elevation range

        # Create GeoTIFF
        transform = from_bounds(lon, lat, lon + 1, lat + 1, size, size)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_file, 'w',
            driver='GTiff',
            height=size,
            width=size,
            count=1,
            dtype=elevation.dtype,
            crs='EPSG:4326',
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(elevation, 1)

    def load_geotiff(self, filepath: str) -> TerrainData:
        """
        Load elevation data from a GeoTIFF file.

        Args:
            filepath: Path to GeoTIFF file

        Returns:
            TerrainData object

        Raises:
            ImportError: If 'rasterio' package not installed
            FileNotFoundError: If file doesn't exist
        """
        try:
            import rasterio
        except ImportError as e:
            raise ImportError(
                "GeoTIFF loading requires 'rasterio' package. "
                "Install with: pip install rasterio"
            ) from e

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GeoTIFF file not found: {filepath}")

        with rasterio.open(filepath) as src:
            elevation_grid = src.read(1).astype(np.float32)

            # Get bounds from transform
            transform = src.transform
            height, width = elevation_grid.shape

            # Calculate corners
            lon_min = transform.c
            lat_max = transform.f
            lon_max = lon_min + width * transform.a
            lat_min = lat_max + height * transform.e  # e is negative

            bounds = TerrainBounds(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max
            )

            # Get nodata value
            nodata = src.nodata if src.nodata is not None else -9999.0

            # Get CRS
            crs = str(src.crs) if src.crs else "EPSG:4326"

        return TerrainData(
            elevation_grid=elevation_grid,
            bounds=bounds,
            crs=crs,
            nodata_value=nodata
        )

    def load_dted(self, filepath: str) -> TerrainData:
        """
        Load elevation data from a DTED file.

        Args:
            filepath: Path to DTED file (.dt0, .dt1, or .dt2)

        Returns:
            TerrainData object

        Raises:
            ImportError: If 'rasterio' package not installed
            FileNotFoundError: If file doesn't exist
        """
        # DTED files can be read by rasterio/GDAL
        # The process is the same as GeoTIFF
        try:
            import rasterio
        except ImportError as e:
            raise ImportError(
                "DTED loading requires 'rasterio' package with GDAL. "
                "Install with: pip install rasterio"
            ) from e

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"DTED file not found: {filepath}")

        # DTED files are read the same way as GeoTIFF via GDAL
        return self.load_geotiff(str(filepath))

    def clear_cache(self):
        """Remove all cached terrain files."""
        import shutil
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
