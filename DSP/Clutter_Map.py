import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import rasterio
from rasterio.merge import merge
import glob
import os
from rasterio.transform import from_origin

# ==========================================
# 1. RADAR & SCENARIO CONFIGURATION
# ==========================================
# Radar Parameters
FREQ_HZ = 2.9e9             # 2.9 GHz (S-Band)
TX_POWER_PEAK_WATTS = 1e9   # 1 GW
BEAMWIDTH_DEG = 1.0         # 1 degree nominal
BEAM_STACK_COUNT = 6        # 6 beams
FIRST_BEAM_EL_DEG = 0.8     # Bottom beam elevation
BEAM_SPACING_DEG = 1.0      # Assume tight stacking (1 deg spacing)

# Physics Constants
C = 3e8                     # Speed of light
WAVELENGTH = C / FREQ_HZ
EARTH_RADIUS = 6371e3       # Meters
K_FACTOR = 4/3              # Standard refraction model
EFFECTIVE_EARTH_RADIUS = EARTH_RADIUS * K_FACTOR

# Location: Nueva Loja, Ecuador (Amazon Region)
RADAR_LAT = 0.0075
RADAR_LON = -77.3186
RADAR_HEIGHT_AGL = 20.0     # Radar antenna height above ground (meters)

# Clutter Model: Constant Gamma (Vegetation/Jungle)
# Gamma ~ -10 to -15 dB is typical for S-band in heavy forest
GAMMA_DB = -12.0 
GAMMA_LINEAR = 10**(GAMMA_DB/10.0)

# Map Settings
MAX_RANGE_KM = 250          # 100 km radius
GRID_RES_M = 100            # 100m pixel resolution (approx SRTM1 resolution)

# Flag: Set to True if you have a real SRTM GeoTIFF file
USE_REAL_DATA = True
#REAL_DEM_PATH = "S01W078.hgt"

# ==========================================
# 2. TERRAIN GENERATION / LOADING
# ==========================================
# ==========================================
# FILE SETUP
# ==========================================
# Put all your .hgt or .tif files in one folder
# You likely need: N00W078, N00W077, S01W078, S01W077

# Get the directory where this script file (.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Look for the 'terrain_files' folder INSIDE that directory
TERRAIN_FOLDER = os.path.join(SCRIPT_DIR, "terrain_files")

# ALSO: Check if your files are .tif or .hgt
# If you downloaded GeoTIFFs, change this to "*.tif"
FILE_EXTENSION = "*.tif"

def get_terrain_grid(lat_center, lon_center, max_range_km, resolution_m):
    """
    Stitches multiple terrain files together and interpolates to the radar grid.
    """
    # 1. Create Simulation Grid (Meters)
    print(f"Generating Simulation Grid ({max_range_km} km radius)...")
    extent_m = max_range_km * 1000
    x = np.arange(-extent_m, extent_m, resolution_m)
    y = np.arange(-extent_m, extent_m, resolution_m)
    X, Y = np.meshgrid(x, y) 
    
    # Calculate Ground Distances & Geometry
    R_ground = np.sqrt(X**2 + Y**2)
    
    # Default Z (Sea Level)
    Z = np.zeros_like(X)
    radar_ground_alt = 296.0 

    # 2. Load and Stitch Files
    search_path = os.path.join(TERRAIN_FOLDER, FILE_EXTENSION) # Or *.tif
    terrain_files = glob.glob(search_path)
    
    if not terrain_files:
        print(f"WARNING: No terrain files found in {TERRAIN_FOLDER}. Using synthetic.")
        # ... Insert synthetic generation fallback here if needed ...
        return X, Y, np.zeros_like(X), R_ground, 0

    try:
        print(f"Found {len(terrain_files)} terrain files. Stitching...")
        
        # Open all files as rasterio datasets
        src_files_to_mosaic = []
        for fp in terrain_files:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)
        
        # --- THE MAGIC STEP: MERGE ---
        # mosaic is the data (numpy array), out_trans is the new coordinate transform
        mosaic, out_trans = merge(src_files_to_mosaic)
        
        # Close the file handles
        for src in src_files_to_mosaic:
            src.close()
            
        # mosaic shape is usually (1, height, width) - remove the band dim
        Z_mosaic = mosaic[0]
        
        print(f"Mosaic created. Shape: {Z_mosaic.shape}")

        # 3. Prepare Data for Interpolation
        # We need to construct the Lat/Lon axes for the NEW stitched mosaic
        # out_trans[0] = pixel width (deg), out_trans[2] = left longitude
        # out_trans[4] = pixel height (deg, usually negative), out_trans[5] = top latitude
        
        height, width = Z_mosaic.shape
        
        # Generate the Latitude axis (Rows)
        # Note: SRTM is top-down, so step is negative. 
        # We need ascending for RegularGridInterpolator, so we might flip.
        start_lat = out_trans[5]
        lat_step = out_trans[4]
        lats_mosaic = start_lat + np.arange(height) * lat_step
        
        # Generate Longitude axis (Cols)
        start_lon = out_trans[2]
        lon_step = out_trans[0]
        lons_mosaic = start_lon + np.arange(width) * lon_step
        
        # Check directions for Interpolator (needs strictly ascending)
        if lats_mosaic[1] < lats_mosaic[0]:
            lats_mosaic = lats_mosaic[::-1]
            Z_mosaic = Z_mosaic[::-1, :] # Flip image vertical
            
        # 4. Map Simulation Grid to Lat/Lon
        # Convert meters to lat/lon offsets
        deg_per_km_lat = 1 / 111.32
        deg_per_km_lon = 1 / (111.32 * np.cos(np.radians(lat_center)))
        
        target_lats = lat_center + (Y / 1000.0) * deg_per_km_lat
        target_lons = lon_center + (X / 1000.0) * deg_per_km_lon
        
        # 5. Interpolate
        print("Interpolating to radar grid...")
        interp = RegularGridInterpolator((lats_mosaic, lons_mosaic), Z_mosaic, 
                                         bounds_error=False, fill_value=0)
        
        points = np.stack((target_lats.ravel(), target_lons.ravel()), axis=1)
        Z = interp(points).reshape(X.shape)
        
        # Update radar altitude from the map center
        radar_ground_alt = Z[Z.shape[0]//2, Z.shape[1]//2]
        print(f"Terrain Ready! Radar Ground Alt: {radar_ground_alt:.1f} m")

    except Exception as e:
        print(f"CRITICAL ERROR stitching terrain: {e}")
        import traceback
        traceback.print_exc()

    return X, Y, Z, R_ground, radar_ground_alt

# ==========================================
# 3. BEAM PATTERN MODELING
# ==========================================
def calculate_antenna_gain(target_el_deg, target_az_deg):
    """
    Now includes SIDE LOBES.
    Without this, the gain drops to -Infinity dB outside the beam, 
    making the ground invisible.
    """
    PEAK_GAIN_LIN = 30000.0  # ~45 dBi
    SIDE_LOBE_LEVEL_DB = -35.0 # Average side lobe floor
    SIDE_LOBE_LIN = PEAK_GAIN_LIN * (10**(SIDE_LOBE_LEVEL_DB/10.0))
    
    beam_centers = [FIRST_BEAM_EL_DEG + i * BEAM_SPACING_DEG for i in range(BEAM_STACK_COUNT)]
    max_gain = np.zeros_like(target_el_deg)
    
    for center in beam_centers:
        delta_el = target_el_deg - center
        # Gaussian main beam
        main_lobe = PEAK_GAIN_LIN * np.exp(-2.77 * (delta_el / BEAMWIDTH_DEG)**2)
        
        # Combine Main Lobe + Side Lobe Floor
        # We take the maximum of the Gaussian curve OR the side lobe floor
        beam_pattern = np.maximum(main_lobe, SIDE_LOBE_LIN)
        
        max_gain = np.maximum(max_gain, beam_pattern)
        
    return max_gain

# ==========================================
# 4. MAIN SIMULATION LOOP
# ==========================================
def generate_clutter_map():
    print(f"Initializing Simulation at {RADAR_LAT}, {RADAR_LON}...")
    X, Y, Z, R_ground, radar_gnd_alt = get_terrain_grid(RADAR_LAT, RADAR_LON, MAX_RANGE_KM, GRID_RES_M)
    
    radar_alt_abs = radar_gnd_alt + RADAR_HEIGHT_AGL
    
    # --- 1. ATMOSPHERE FIX (Amazon Ducting) ---
    # Standard atmosphere is k=1.33. 
    # The Amazon often has "Super Refraction" (k=1.5 to 1.8) due to humidity.
    # This curves the beam down, letting it see further over flat ground.
    AMAZON_K_FACTOR = 1.6 
    eff_earth_radius = EARTH_RADIUS * AMAZON_K_FACTOR
    
    # Calculate Height Delta relative to radar, accounting for curvature
    earth_drop = (R_ground**2) / (2 * eff_earth_radius)
    delta_h = Z - radar_alt_abs - earth_drop
    
    # Slant Range & Elevation
    R_slant = np.sqrt(R_ground**2 + delta_h**2)
    el_rad = np.arctan2(delta_h, R_ground)
    el_deg = np.degrees(el_rad)
    
    # --- 2. SHADOWING FIX (Diffraction) ---
    # Instead of a hard "Cut to Black" at the horizon, we apply "Diffraction Loss".
    # This allows the radar to see slightly over the curve, fading out naturally.
    
    horizon_dist_km = 4.12 * (np.sqrt(RADAR_HEIGHT_AGL) + np.sqrt(np.maximum(0, Z - radar_gnd_alt)))
    
    # Calculate how far past the horizon we are (if at all)
    dist_past_horizon_km = (R_ground/1000.0) - horizon_dist_km
    dist_past_horizon_km[dist_past_horizon_km < 0] = 0 # Not shadowed
    
    # Simple Diffraction Penalty: ~1-2 dB per km past horizon is a good rule of thumb
    diffraction_loss_db = dist_past_horizon_km * 2.0 
    diffraction_loss_lin = 10**(-diffraction_loss_db / 10.0)

    # --- Radar Equation ---
    G_lin = calculate_antenna_gain(el_deg, 0)
    
    grazing_angle_rad = np.abs(el_rad) 
    grazing_angle_rad[grazing_angle_rad < 0.001] = 0.001
    
    # Constant Gamma Model
    sigma0 = GAMMA_LINEAR * np.sin(grazing_angle_rad)
    clutter_area = GRID_RES_M**2
    sigma = sigma0 * clutter_area
    
    # Power Calculation
    R_slant[R_slant < 100] = 100 
    numerator = TX_POWER_PEAK_WATTS * (G_lin**2) * (WAVELENGTH**2) * sigma
    denominator = ((4 * np.pi)**3) * (R_slant**4)
    
    Pr_watts = numerator / denominator
    
    # Apply Diffraction Loss (Soft Shadowing)
    Pr_watts = Pr_watts * diffraction_loss_lin
    
    # Noise floor clamp (prevent log(0))
    Pr_watts[Pr_watts < 1e-25] = 1e-25
    Pr_watts[diffraction_loss_db > 40] = 1e-25
    
    # Convert to dBm
    Pr_dBm = 10 * np.log10(Pr_watts * 1000)
    
    return X, Y, Pr_dBm, el_deg

# ==========================================
# 5. VISUALIZATION
# ==========================================
X, Y, Pr_dBm, El_map = generate_clutter_map()

# Filter for reasonable dynamic range
vmin = -120
vmax = 10 # High power!

plt.figure(figsize=(12, 10))
plt.title(f"S-Band Radar Clutter Map (1 GW, {BEAM_STACK_COUNT} Beams)\nLoc: {RADAR_LAT}, {RADAR_LON} (Ecuador)", fontsize=14)

# Create the heatmap
mesh = plt.pcolormesh(X/1000, Y/1000, Pr_dBm, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
cbar = plt.colorbar(mesh, label="Received Clutter Power (dBm)")

# Add range rings
circle1 = plt.Circle((0, 0), 25, color='white', fill=False, linestyle='--', alpha=0.5)
circle2 = plt.Circle((0, 0), 50, color='white', fill=False, linestyle='--', alpha=0.5)
circle3 = plt.Circle((0, 0), 75, color='white', fill=False, linestyle='--', alpha=0.5)
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.gca().add_patch(circle3)

plt.xlabel("East-West Range (km)")
plt.ylabel("North-South Range (km)")
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

# Visualize Elevation Coverage
plt.figure(figsize=(10, 4))
plt.title("Elevation Angle Map (Degrees)")
mesh_el = plt.pcolormesh(X/1000, Y/1000, El_map, cmap='plasma', shading='auto')
plt.colorbar(mesh_el, label="Elevation Angle (deg)")
plt.xlabel("Range (km)")
plt.axis('equal')
plt.show()