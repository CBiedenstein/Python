import math

def calculate_tps43_beam_power_upgrade():
    """
    Calculates power distribution per Channel AND per Beam for a 
    TPS-43 Radar with a Solid State Amplifier Upgrade (320 kW).
    """
    
    # ---------------------------------------------------------
    # 1. PARAMETERS FROM TPS-43 TECHNICAL MANUAL
    # ---------------------------------------------------------
    
    # Total Upgrade Power
    P_PEAK_NEW_KW = 2.4e6 #kW
    
    # Transmit Power Division (Table 5-1, Theory of Operation)
    # Channel: Percentage of Total Power
    tx_power_pct = {
        1: 15.66,
        2: 17.02,
        3: 14.18,
        4: 10.72,
        5: 12.38,
        6: 9.36,
        7: 6.03,
        8: 4.55,
        9: 2.93,
        10: 2.21,
        11: 1.42,
        12: 2.10,
        13: 1.35  # Channel 13 drives Horns E13, E14, E15
    }

    # Beam-Horn Mapping (Table 5-2, Beam-Horn Data)
    # Lists the Transmit Channels (Horns) that contribute to each Beam.
    # Note: Channel 13 covers Horns E13, E14, and E15.
    beam_configuration = {
        "Beam 1": [1, 2],                     # Horns E1, E2
        "Beam 2": [2, 3],                     # Horns E2, E3
        "Beam 3": [3, 4, 5],                  # Horns E3-E5
        "Beam 4": [4, 5, 6],                  # Horns E4-E6
        "Beam 5": [6, 7, 8, 9, 10, 11, 12],   # Horns E6-E12
        "Beam 6": [9, 10, 11, 12, 13]         # Horns E9-E15 (Ch13 = E13-15)
    }

    # Beam Characteristics (Table 5-3 for Receive Gains)
    # Used for range estimation per beam sector
    beam_rx_gain_db = {
        "Beam 1": 39.0,
        "Beam 2": 38.0,
        "Beam 3": 37.0,
        "Beam 4": 34.0,
        "Beam 5": 32.0,
        "Beam 6": 32.0
    }

    # Radar Constants
    FREQ_MHZ = 3000.0
    LAMBDA = 300.0 / FREQ_MHZ
    GAIN_TX_DB = 34.5       # Nominal Transmit Gain (Main Beam)
    MDS_DBM = -105.0        # Sensitivity
    SIGMA_M2 = 1.7          # Target Cross Section

    # ---------------------------------------------------------
    # 2. CALCULATE POWER PER CHANNEL
    # ---------------------------------------------------------
    channel_power_kw = {}
    for ch, pct in tx_power_pct.items():
        channel_power_kw[ch] = P_PEAK_NEW_KW * (pct / 100.0)

    print(f"{'='*70}")
    print(f"TPS-43 SOLID STATE UPGRADE: POWER ANALYSIS")
    print(f"Total System Power: {P_PEAK_NEW_KW} kW")
    print(f"{'='*70}")
    
    print(f"\n--- PART A: TRANSMIT DISTRIBUTION PER CHANNEL (Table 5-1) ---")
    print(f"{'Channel':<8} | {'% Input':<10} | {'Power (kW)':<12} | {'Power (dBm)':<12} | {'Horns Driven'}")
    print(f"{'-'*70}")
    
    horn_map = {13: "E13, E14, E15"} # Special case
    
    for ch in range(1, 14):
        p_kw = channel_power_kw[ch]
        p_dbm = 10 * math.log10(p_kw * 1000 * 1000)
        horns = horn_map.get(ch, f"E{ch}")
        print(f"{ch:<8} | {tx_power_pct[ch]:<10} | {p_kw:<12.2f} | {p_dbm:<12.2f} | {horns}")

    # ---------------------------------------------------------
    # 3. CALCULATE POWER PER BEAM
    # ---------------------------------------------------------
    print(f"\n\n--- PART B: TOTAL POWER PER BEAM (Table 5-2 Config) ---")
    print(f"Note: Beams overlap; shared horns contribute power to multiple beams.")
    print(f"{'-'*70}")
    print(f"{'Beam':<8} | {'Channels Included':<20} | {'Total %':<10} | {'Beam Power (kW)'}")
    print(f"{'-'*70}")

    for beam_name, channels in beam_configuration.items():
        # Sum the power percentages of all channels in this beam
        beam_pct_sum = sum(tx_power_pct[ch] for ch in channels)
        
        # Calculate absolute power in kW for this beam configuration
        beam_power_kw = P_PEAK_NEW_KW * (beam_pct_sum / 100.0)
        
        # Format channel list string
        ch_str = ",".join(str(c) for c in channels)
        
        print(f"{beam_name:<8} | {ch_str:<20} | {beam_pct_sum:<9.2f}% | {beam_power_kw:.2f} kW")

    # ---------------------------------------------------------
    # 4. RANGE CALCULATION (Using Beam 1 for Max Range)
    # ---------------------------------------------------------
    print(f"\n\n--- PART C: THEORETICAL DETECTION RANGE (Beam 1) ---")
    
    # Constants conversion
    gain_tx_lin = 10 ** (GAIN_TX_DB / 10.0)
    gain_rx_lin = 10 ** (beam_rx_gain_db["Beam 1"] / 10.0) # Use Beam 1 Rx Gain (39.0 dB)
    mds_watts = 10 ** ((MDS_DBM - 30) / 10.0)
    p_peak_watts = P_PEAK_NEW_KW * 1000.0
    
    # Radar Range Equation
    # R = [ (Pt * Gtx * Grx * lambda^2 * sigma) / ((4pi)^3 * Pmin) ] ^ 0.25
    numerator = p_peak_watts * gain_tx_lin * gain_rx_lin * (LAMBDA ** 2) * SIGMA_M2
    denominator = ((4 * math.pi) ** 3) * mds_watts
    range_m = (numerator / denominator) ** 0.25
    range_nmi = range_m / 1852.0

    print(f"Target Sigma: {SIGMA_M2} m^2")
    print(f"Rx Gain (Beam 1): {beam_rx_gain_db['Beam 1']} dB")
    print(f"Tx Gain (System): {GAIN_TX_DB} dB")
    print(f"{'-'*40}")
    print(f"Calculated Max Range: {range_nmi:.2f} nmi")
    print(f"{'='*70}")

if __name__ == "__main__":
    calculate_tps43_beam_power_upgrade()