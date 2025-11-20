import math

def solve_max_range(
    freq_hz,
    pt_watts,
    bandwidth_hz,
    noise_figure_db,
    loss_sys_db,
    required_snr_db,
    rcs_list,        # List of (Name, Square Meters) tuples
    gain_list_dbi    # List of Antenna Gains in dBi
):
    """
    Generates a matrix of Maximum Detectable Ranges for varying RCS and Antenna Gains.
    """
    
    # --- 1. Setup Constants & Receiver Sensitivity ---
    c = 299792458
    wavelength = c / freq_hz
    
    # Calculate Thermal Noise (kTB) in Watts
    k = 1.380649e-23
    t0 = 290
    noise_thermal_watts = k * t0 * bandwidth_hz
    
    # Calculate Sensitivity (P_min) in Watts
    # Sensitivity = ThermalNoise + NoiseFigure + RequiredSNR
    # We convert everything to linear Watts for the R^4 formula to be clean
    
    # Convert dB factors to linear ratios
    nf_linear = 10**(noise_figure_db / 10)
    snr_linear = 10**(required_snr_db / 10)
    loss_sys_linear = 10**(loss_sys_db / 10)
    
    # P_min (Watts)
    p_min_watts = noise_thermal_watts * nf_linear * snr_linear
    
    # Print System Baseline
    print(f"--- SYSTEM BASELINE ---")
    print(f"Frequency:      {freq_hz/1e9:.2f} GHz")
    print(f"Tx Power:       {pt_watts} Watts")
    print(f"Sensitivity:    {10*math.log10(p_min_watts*1000):.2f} dBm (MDS + {required_snr_db}dB SNR)")
    print("-" * 71)
    
    # --- 2. Table Header ---
    # Formatting: Target Name | RCS (m^2) | Ant Gain (dBi) | Max Range (km) | Max Range (nm)
    header = f"| {'TARGET TYPE':<15} | {'RCS (m^2)':<10} | {'GAIN (dBi)':<10} | {'RANGE (km)':<10} | {'RANGE (nm)':<10} |"
    print(header)
    print("-" * 71)
    
    # --- 3. Iterate and Solve ---
    for target_name, rcs_val in rcs_list:
        for gain_dbi in gain_list_dbi:
            
            # Convert Gain dB to linear
            g_linear = 10**(gain_dbi / 10)
            
            # We assume Tx Gain = Rx Gain (Monostatic Radar)
            # Numerator: Pt * G^2 * lambda^2 * RCS
            numerator = pt_watts * (g_linear**2) * (wavelength**2) * rcs_val
            
            # Denominator: (4pi)^3 * P_min * Losses
            denominator = ((4 * math.pi)**3) * p_min_watts * loss_sys_linear
            
            # Solve R^4
            r4 = numerator / denominator
            
            # Fourth root to get Range in meters
            range_meters = r4 ** 0.25
            
            # Conversions
            range_km = range_meters / 1000
            range_nm = range_meters / 1852  # Nautical Miles
            
            print(f"| {target_name:<15} | {rcs_val:<10.4f} | {gain_dbi:<10.1f} | {range_km:<10.2f} | {range_nm:<10.2f} |")
            
        print("-" * 71) # Separator between target types

# --- CONFIGURATION ---

# 1. Define your Target Scenarios (Name, RCS in m^2)
targets = [
    ("Insect/Bullet", 0.001),
    ("Small Drone",   0.01),
    ("Person",        0.5),
    ("Car/Fighter",   5.0),
    ("Large Ship",    100.0)
]

# 2. Define your Array sizes (Gains in dBi)
# Rule of thumb: 20dBi is a small patch, 30dBi is a dish/large array, 40dBi is massive
gain_tx = 36 # dBi gain of the transmit antenna
gain_rx = 39 # dBi gain of the receive antenna
antenna_gains = [gain_tx + gain_rx] # Mock array for [25.0, 30.0, 35.0] 

# 3. Run the Calculator
solve_max_range(
    freq_hz=3.0e9,            # 3.0 GHz
    pt_watts=2.0e6,           # 2 MW
    bandwidth_hz=1.6e6,       # 1.6 MHz BW
    noise_figure_db=2,        # 2 dB NF
    loss_sys_db=5.0,          # 5 dB System Loss
    required_snr_db=5.0,      # 5 dB SNR required to "see" it
    rcs_list=targets,
    gain_list_dbi=antenna_gains
)