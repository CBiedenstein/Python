import pandas as pd
import struct
import math
import os

def float_to_asterix_azimuth(deg):
    """
    Encodes degrees (0-360) to ASTERIX 16-bit azimuth.
    LSB = 360 / 2^16 approx 0.0055 degrees.
    """
    val = int((deg % 360) / 360.0 * 65536)
    return max(0, min(65535, val))

def float_to_asterix_range(nmi):
    """
    Encodes nautical miles to ASTERIX 16-bit range.
    LSB = 1/256 NM. Max range approx 256 NM.
    """
    val = int(nmi * 256)
    return max(0, min(65535, val))

def float_to_flight_level(altitude_ft):
    """
    Encodes altitude in feet to ASTERIX Flight Level.
    LSB = 1/4 FL = 25 ft.
    """
    # Convert to Flight Level (1 FL = 100 ft), then div by 0.25
    # Simplification: val = feet / 25
    val = int(altitude_ft / 25.0)
    # ASTERIX 2-byte signed integer limits
    return max(-32768, min(32767, val))

def create_asterix_packet(row, sac=0, sic=1):
    """
    Constructs a single CAT-048 message for a DataFrame row.
    
    We will use the following Data Items (FSPEC):
    - I048/010: Data Source Identifier (SAC/SIC)
    - I048/040: Measured Position in Polar Coords (Range/Az)
    - I048/090: Flight Level (Calculated from Altitude)
    - I048/140: Time of Day (Calculated from Time_Sec)
    """
    
    # --- 1. DATA ITEM GENERATION ---
    
    # I048/010 Data Source Identifier (2 bytes)
    # Byte 1: SAC, Byte 2: SIC
    b_010 = struct.pack('>BB', sac, sic)
    
    # I048/040 Measured Position in Polar Coordinates (4 bytes)
    # Byte 1-2: Range (1/256 NM), Byte 3-4: Azimuth (360/2^16)
    rng_val = float_to_asterix_range(row['Range_nmi'])
    az_val = float_to_asterix_azimuth(row['Azimuth_deg'])
    b_040 = struct.pack('>HH', rng_val, az_val)
    
    # I048/090 Flight Level (2 bytes)
    # V (1 bit) + G (1 bit) + FL (14 bits)
    # We assume validated (V=0) and garbled (G=0) for simulation
    fl_val = float_to_flight_level(row['Altitude_ft'])
    b_090 = struct.pack('>h', fl_val) # Signed short
    
    # I048/140 Time of Day (3 bytes)
    # LSB = 1/128 seconds. Wraps at midnight.
    time_val = int(row['Time_Sec'] * 128) % (24 * 3600 * 128)
    b_140 = struct.pack('>I', time_val)[1:] # Take last 3 bytes of 4-byte int
    
    # --- 2. FSPEC CONSTRUCTION ---
    # We need to build the Field Specification byte(s).
    # Bit 7 is first field, Bit 1 is FX (extension).
    
    # CAT 048 Standard FSPEC order (partial list for our items):
    # FSPEC 1 (Byte 1):
    #   Bit 7: I048/010 (DataSourceID) -> YES
    #   Bit 6: I048/140 (TimeOfDay)    -> YES
    #   Bit 5: I048/020 (TargetReport) -> NO (Skipping for simplicity)
    #   Bit 4: I048/040 (PolarCoords)  -> YES
    #   Bit 3: I048/070 (Mode3A)       -> NO
    #   Bit 2: I048/090 (FlightLevel)  -> YES
    #   Bit 1: I048/130 (PlotChar)     -> NO
    #   Bit 0: FX (Extension)          -> NO (End of profile)
    
    # Construct Byte: 1101 0100 = 0xD4
    fspec_byte = 0xD4
    
    # --- 3. PACKET ASSEMBLY ---
    
    # Header: CAT (1 byte) + LEN (2 bytes)
    cat_byte = 48
    
    # Data Block = FSPEC + Fields (in order of FSPEC)
    data_block = struct.pack('B', fspec_byte)
    data_block += b_010  # Bit 7
    data_block += b_140  # Bit 6
    data_block += b_040  # Bit 4
    data_block += b_090  # Bit 2
    
    # Total Length = 1 (CAT) + 2 (LEN) + len(data_block)
    total_len = 1 + 2 + len(data_block)
    
    header = struct.pack('>BH', cat_byte, total_len)
    
    return header + data_block

def convert_csv_to_asterix(csv_path, output_path):
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: CSV file not found. Please run the simulation script first.")
        return

    print(f"Converting {len(df)} records to ASTERIX CAT-048...")
    
    with open(output_path, 'wb') as f:
        for index, row in df.iterrows():
            packet = create_asterix_packet(row, sac=10, sic=43) # ID: 10/43 (Example)
            f.write(packet)
            
    print(f"Success! Binary ASTERIX data written to {output_path}")
    print(f"Total file size: {os.path.getsize(output_path)} bytes")

# --- EXECUTE ---
if __name__ == "__main__":
    csv_file = 'radar_track_data.csv'
    ast_file = 'radar_tracks.ast'
    convert_csv_to_asterix(csv_file, ast_file)