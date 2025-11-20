import math

def calculate_radar_performance(
    freq_hz, 
    pt_watts, 
    gt_dbi, 
    gr_dbi, 
    range_meters, 
    rcs_m2, 
    loss_sys_db,
    bandwidth_hz,
    noise_figure_db,
    required_snr_db=13.0 
):
    """
    Calculates Radar Link Budget, MDS, and Signal-to-Noise Ratio (SNR).
    
    Parameters:
    freq_hz         (float): Frequency in Hertz
    pt_watts        (float): Transmit Power in Watts
    gt_dbi          (float): Transmit Antenna Gain in dBi
    gr_dbi          (float): Receive Antenna Gain in dBi
    range_meters    (float): Target Range in Meters
    rcs_m2          (float): Radar Cross Section in Square Meters
    loss_sys_db     (float): System Losses in dB
    bandwidth_hz    (float): Receiver Bandwidth in Hz
    noise_figure_db (float): Receiver Noise Figure in dB
    required_snr_db (float): Min SNR required for detection (default 13dB for high prob.)
    """
    
    # --- 1. Constants ---
    c = 299792458  # Speed of light (m/s)
    k = 1.380649e-23  # Boltzmann constant (J/K)
    t0 = 290  # Standard noise temperature (Kelvin)
    wavelength = c / freq_hz
    
    # --- 2. Transmitter Parameters ---
    pt_dbm = 10 * math.log10(pt_watts * 1000)  # Watts to dBm
    rcs_dbsm = 10 * math.log10(rcs_m2)
    
    # --- 3. Path Loss Calculation (Radar Loop) ---
    # Radar Equation Factor: (4 * pi)^3 * R^4 / lambda^2
    numerator = ((4 * math.pi)**3) * (range_meters**4)
    denominator = wavelength**2
    radar_path_loss_linear = numerator / denominator
    radar_path_loss_db = 10 * math.log10(radar_path_loss_linear)
    
    # --- 4. Received Power (Pr) ---
    pr_dbm = (pt_dbm + gt_dbi + gr_dbi + rcs_dbsm) - loss_sys_db - radar_path_loss_db
    
    # --- 5. Noise Calculations (MDS) ---
    # Thermal Noise Power in Watts: k * T0 * B
    noise_thermal_watts = k * t0 * bandwidth_hz
    # Convert to dBm
    noise_thermal_dbm = 10 * math.log10(noise_thermal_watts * 1000)
    
    # MDS = Thermal Noise Floor + Noise Figure
    # This is the power level where Signal = Noise (0 dB SNR)
    mds_dbm = noise_thermal_dbm + noise_figure_db
    
    # Sensitivity = MDS + Required SNR
    sensitivity_dbm = mds_dbm + required_snr_db
    
    # --- 6. Signal-to-Noise Ratio (SNR) ---
    actual_snr_db = pr_dbm - mds_dbm
    
    # --- 7. Print Results ---
    print("=" * 50)
    print("RADAR SYSTEM PERFORMANCE ANALYSIS")
    print("=" * 50)
    print(f"Target Range:       {range_meters/1000:.2f} km")
    print(f"Frequency:          {freq_hz/1e9:.2f} GHz")
    print("-" * 50)
    print(f"Transmit Power:     {pt_dbm:.2f} dBm")
    print(f"Total Gains (Tx+Rx):{gt_dbi + gr_dbi:.2f} dBi")
    print(f"Radar Path Loss:    {radar_path_loss_db:.2f} dB")
    print(f"System Losses:      {loss_sys_db:.2f} dB")
    print("-" * 50)
    print(f"RECEIVED SIGNAL (Pr):        {pr_dbm:.2f} dBm")
    print(f"THERMAL NOISE FLOOR (kTB):   {noise_thermal_dbm:.2f} dBm")
    print(f"NOISE FIGURE (NF):           {noise_figure_db:.2f} dB")
    print("-" * 50)
    print(f"MDS (Noise Floor + NF):      {mds_dbm:.2f} dBm")
    print(f"REQ. SENSITIVITY (For Detect):{sensitivity_dbm:.2f} dBm")
    print("-" * 50)
    print(f"ACTUAL SNR:                  {actual_snr_db:.2f} dB")
    
    if actual_snr_db >= required_snr_db:
        print(f"STATUS: DETECTED (Margin: {actual_snr_db - required_snr_db:.2f} dB)")
    else:
        print(f"STATUS: NOT DETECTED (Below threshold by {required_snr_db - actual_snr_db:.2f} dB)")
    print("=" * 50)

# --- Example Usage ---
calculate_radar_performance(
    freq_hz=3.0e9,              # 3.0 GHz (S-Band)
    pt_watts=2e6,               # 2 MW Watts
    gt_dbi=36.0,                # 36 dBi Tx Gain
    gr_dbi=39.0,                # 39 dBi Rx Gain
    range_meters=240e3*1.852,   # 170 nmi
    rcs_m2=6.0,                 # 1 m^2 Target
    loss_sys_db=5.0,            # System losses
    bandwidth_hz=1.6e6,         # 1.6 MHz Bandwidth (typical for 1us pulse)
    noise_figure_db=2.0,        # 2 dB Low Noise Amplifier
    required_snr_db=5           # 5 dB required for high probability of detection
)