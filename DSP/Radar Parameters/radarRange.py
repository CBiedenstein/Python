import math

def calculate_radar_link_budget(
    freq_hz, 
    pt_watts, 
    gt_dbi, 
    gr_dbi, 
    range_meters, 
    rcs_m2, 
    loss_sys_db
):
    """
    Calculates the Radar Received Power in dBm based on the Radar Range Equation.
    
    Parameters:
    freq_hz      (float): Frequency in Hertz
    pt_watts     (float): Transmit Power in Watts
    gt_dbi       (float): Transmit Antenna Gain in dBi
    gr_dbi       (float): Receive Antenna Gain in dBi
    range_meters (float): Target Range in Meters
    rcs_m2       (float): Radar Cross Section in Square Meters
    loss_sys_db  (float): System Losses in dB (cabling, processing, etc.)
    """
    
    # 1. Constants and Basic Conversions
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq_hz
    
    # Convert Transmit Power to dBm (decibel-milliwatts)
    # 1 Watt = 1000 mW
    pt_dbm = 10 * math.log10(pt_watts * 1000)
    
    # Convert RCS to dBsm (decibels relative to a square meter)
    rcs_dbsm = 10 * math.log10(rcs_m2)
    
    # 2. Calculate Path Loss
    # Note: Radar Path Loss is different from one-way comms FSPL.
    # It involves the two-way trip and the spreading loss (1/R^4).
    # Formula for Radar Loop Loss: 10 * log10( ((4*pi)^3 * R^4) / lambda^2 )
    
    numerator = ((4 * math.pi)**3) * (range_meters**4)
    denominator = wavelength**2
    radar_path_loss_linear = numerator / denominator
    radar_path_loss_db = 10 * math.log10(radar_path_loss_linear)

    # Optional: Calculate One-Way Free Space Path Loss (FSPL) for reference
    # This is what is typically used in Comms, but Radar effectively doubles this + scattering
    fspl_one_way_db = 20 * math.log10((4 * math.pi * range_meters) / wavelength)

    # 3. Calculate Final Received Power (Pr)
    # Pr = Pt + Gt + Gr + RCS - Losses - PathLoss
    pr_dbm = (pt_dbm + gt_dbi + gr_dbi + rcs_dbsm) - loss_sys_db - radar_path_loss_db
    
    # 4. Print the Link Budget
    print("-" * 40)
    print("RADAR LINK BUDGET CALCULATOR")
    print("-" * 40)
    print(f"Frequency:          {freq_hz/1e9:.2f} GHz")
    print(f"Wavelength:         {wavelength:.4f} m")
    print(f"Range:              {range_meters/1000:.2f} km")
    print("-" * 40)
    print(f"Transmit Power:     {pt_dbm:.2f} dBm")
    print(f"+ Tx Antenna Gain:  {gt_dbi:.2f} dBi")
    print(f"+ Rx Antenna Gain:  {gr_dbi:.2f} dBi")
    print(f"+ Target RCS:       {rcs_dbsm:.2f} dBsm")
    print(f"- System Losses:    {loss_sys_db:.2f} dB")
    print(f"- Radar Path Loss:  {radar_path_loss_db:.2f} dB")
    print("-" * 40)
    print(f"RECEIVED POWER:     {pr_dbm:.2f} dBm")
    print("-" * 40)
    print(f"*(Ref: One-Way FSPL would be {fspl_one_way_db:.2f} dB)*")
    
    return pr_dbm

# --- Example Usage ---

# Inputs
frequency = 3e9      # 9.4 GHz (X-Band)
tx_power = 480e3       # 100 Watts
tx_gain = 36.0         # 30 dBi
rx_gain = 39.0         # 30 dBi
target_range = 440e3  # 5 km
target_rcs = 1.0       # 1 m^2 (small fighter jet / car)
system_loss = 5.0      # 5 dB internal loss

# Run Calculation
calculate_radar_link_budget(
    freq_hz=frequency,
    pt_watts=tx_power,
    gt_dbi=tx_gain,
    gr_dbi=rx_gain,
    range_meters=target_range,
    rcs_m2=target_rcs,
    loss_sys_db=system_loss
)