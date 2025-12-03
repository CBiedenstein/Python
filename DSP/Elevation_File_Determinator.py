import numpy as np

def list_required_srtm_tiles(lat, lon, range_km):
    # Approx degrees required (1 deg ~ 111km)
    deg_radius = (range_km / 111.0) + 0.5 # Add buffer
    
    min_lat = int(np.floor(lat - deg_radius))
    max_lat = int(np.floor(lat + deg_radius))
    min_lon = int(np.floor(lon - deg_radius))
    max_lon = int(np.floor(lon + deg_radius))
    
    print(f"--- Required SRTM Tiles for {range_km} km radius ---")
    print(f"Latitude: {min_lat} to {max_lat}")
    print(f"Longitude: {min_lon} to {max_lon}")
    print(f"Total Tiles: {(max_lat - min_lat + 1) * (max_lon - min_lon + 1)}")
    print("-" * 30)
    
    files = []
    for lat_i in range(min_lat, max_lat + 1):
        for lon_i in range(min_lon, max_lon + 1):
            ns = 'N' if lat_i >= 0 else 'S'
            ew = 'E' if lon_i >= 0 else 'W'
            # Format: S01W078 (Lat is bottom edge, Lon is left edge)
            tile_name = f"{ns}{abs(lat_i):02d}{ew}{abs(lon_i):03d}.hgt"
            files.append(tile_name)
            
    # Print in a block for easy reading
    for i in range(0, len(files), 4):
        print("  ".join(files[i:i+4]))

# Run this once to see what you need to download
list_required_srtm_tiles(0.0075, -77.3186, 440)