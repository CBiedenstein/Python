import math

def solve_min_detectable_rcs(
    freq_hz,
    pt_watts,
    gain_dbi,       # Assumes Monostatic (Tx Gain = Rx Gain)
    bandwidth_hz,
    noise_figure_db,
    loss_sys_db,
    required_snr_db,
    range_list_nmi   # List of Ranges in Kilometers
):
    """
    Calculates the minimum RCS required to detect a target at specific ranges.
    """
    
    # --- 1. Setup Constants ---
    c = 299792458
    wavelength = c / freq_hz
    
    # --- 2. Calculate Receiver Sensitivity (P_min) ---
    # Thermal Noise (Watts) = k * T0 * B
    k = 1.380649e-23
    t0 = 290
    noise_thermal_watts = k * t0 * bandwidth_hz
    
    # Convert dB factors to linear
    nf_linear = 10**(noise_figure_db / 10)
    snr_linear = 10**(required_snr_db / 10)
    loss_sys_linear = 10**(loss_sys_db / 10)
    gain_linear = 10**(gain_dbi / 10)
    
    # P_min (Watts) - The absolute lowest power the generic receiver needs
    p_min_watts = noise_thermal_watts * nf_linear * snr_linear
    
    # --- 3. Print System Config ---
    sensitivity_dbm = 10 * math.log10(p_min_watts * 1000)
    print(f"--- RADAR SENSITIVITY CONFIGURATION ---")
    print(f"Freq: {freq_hz/1e9:.1f} GHz | Power: {pt_watts} W | Gain: {gain_dbi} dBi")
    print(f"Sensitivity Threshold: {sensitivity_dbm:.2f} dBm")
    print("-" * 75)
    
    # --- 4. Table Header ---
    # We will output RCS in both square meters (linear) and dBsm (log)
    header = f"| {'RANGE (nmi)':<12} | {'MIN RCS (m^2)':<18} | {'MIN RCS (dBsm)':<18} | {'EQUIV TARGET (Approx)'} |"
    print(header)
    print("-" * 75)
    
    # --- 5. Iterate over Ranges ---
    for range_nmi in range_list_nmi:
        
        range_meters = range_nmi * 1000 * 1.852 # Converting from nmi to meters for calc
        
        # Formula: sigma = (P_min * Loss * (4pi)^3 * R^4) / (Pt * G^2 * lambda^2)
        
        numerator = p_min_watts * loss_sys_linear * ((4 * math.pi)**3) * (range_meters**4)
        denominator = pt_watts * (gain_linear**2) * (wavelength**2)
        
        rcs_m2 = numerator / denominator
        
        # Avoid log error if rcs is basically 0 (unlikely here)
        rcs_dbsm = 10 * math.log10(rcs_m2) if rcs_m2 > 0 else -999
        
        # Helper for context
        context = get_target_context(rcs_dbsm)
        
        print(f"| {range_nmi:<12.1f} | {rcs_m2:<18.4e} | {rcs_dbsm:<18.2f} | {context} |")
        
    print("-" * 75)

def get_target_context(dbsm):
    """Returns a string description of what typically has this RCS."""
    if dbsm < -30: return "Insect"
    if dbsm < -20: return "Bird / Bullet"
    if dbsm < -10: return "Creeping Drone"
    if dbsm < 0:   return "Person / Missile"
    if dbsm < 10:  return "Small Fighter / Car"
    if dbsm < 20:  return "Large Fighter / Truck"
    if dbsm < 40:  return "Corner Reflector / Ship"
    return "Mountain / Building"

# --- CONFIGURATION ---

# Define the specific ranges you want to query (in nmi)
target_ranges = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
gain_tx = 36 # receive gain
gain_rx = 39 # receive gain
total_gain = gain_tx + gain_rx

# Run the Solver
solve_min_detectable_rcs(
    freq_hz=3.0e9,          # 3.0 GHz (S-band)
    pt_watts=2.0e6,         # 2 MW
    gain_dbi=total_gain,    # 32 dBi Gain
    bandwidth_hz=1.6e6,     # 1.6 MHz
    noise_figure_db=2.0,    # 2 dB NF
    loss_sys_db=5.0,        # 5 dB Losses
    required_snr_db=5.0,    # 5 dB SNR
    range_list_nmi=target_ranges # convert from nmi to km
)