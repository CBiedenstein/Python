import numpy as np
import matplotlib.pyplot as plt


# --- Constants & Radar Parameters ---
C = 3e8                     # Speed of light (m/s)
FC = 2.9e9                  # Frequency: 2.9 GHz
LAMBDA = C / FC             # Wavelength
PT = 1e9                    # Transmit Power: 1 GW (10^9 Watts)
BEAMWIDTH_DEG = 1.0         # 1 degree pencil beam
BIT_RATE = 2e6              # 2 Mbps
PULSE_WIDTH = 13 / BIT_RATE # Duration of 13 bits

# Barker 13 Sequence
BARKER_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])

# --- Helper Functions ---

def calculate_gain(bw_deg):
    """
    Approximates Antenna Gain (G) for a pencil beam.
    Using approximation G ~ 30000 / (theta^2) for degrees, 
    converted to linear ratio.
    """
    # G (dBi) approx 10 * log10(41253 / (theta * phi))
    # Using linear approximation for circular aperture
    gain_linear = 41253 / (bw_deg * bw_deg)
    return gain_linear

def estimate_building_rcs():
    """
    Estimates RCS (sigma) for an 8-story building.
    Assumptions: 
    - 1 story ~ 3.5m -> 28m tall. 
    - Width ~ 30m.
    - Material: Concrete/Glass (High reflectivity).
    - Modeled as a flat plate for maximum specular reflection.
    Formula: sigma = 4 * pi * A^2 / lambda^2
    """
    h = 28.0 # meters
    w = 30.0 # meters
    area = h * w
    sigma = (4 * np.pi * (area**2)) / (LAMBDA**2)
    return sigma

def generate_msk_pulse(sequence, samp_rate, bit_rate):
    """
    Generates complex IQ samples for an MSK modulated Barker code.
    MSK is treated here as OQPSK with sinusoidal pulse shaping.
    """
    samps_per_bit = int(samp_rate / bit_rate)
    t = np.linspace(0, len(sequence)/bit_rate, len(sequence)*samps_per_bit)
    
    # MSK Phase generation
    phase = np.zeros_like(t)
    current_phase = 0
    
    for i, bit in enumerate(sequence):
        t_bit = t[i*samps_per_bit : (i+1)*samps_per_bit]
        # Map 0/1 to -1/+1 for phase slope
        # Barker is already -1/1, so we use it directly
        slope = bit * np.pi / (2 * (1/bit_rate)) 
        
        # Calculate phase segment relative to start of bit
        segment_t = t_bit - t_bit[0]
        phase[i*samps_per_bit : (i+1)*samps_per_bit] = current_phase + slope * segment_t
        
        # Update phase for next bit to ensure continuity
        current_phase += bit * (np.pi / 2)

    i_data = np.cos(phase)
    q_data = np.sin(phase)
    return t, i_data + 1j * q_data

# --- Main Simulation ---

def run_simulation():
    print(f"--- Radar Simulation Initiated ---")
    print(f"Freq: {FC/1e9} GHz | Power: {PT/1e9} GW | Beam: {BEAMWIDTH_DEG} deg")
    
    # 1. Setup Environment
    gain = calculate_gain(BEAMWIDTH_DEG)
    rcs_building = estimate_building_rcs()
    print(f"Antenna Gain: {10*np.log10(gain):.2f} dBi")
    print(f"Target RCS: {10*np.log10(rcs_building):.2f} dBsm")

    # 2. Generate Transmit Waveform (Baseband IQ)
    fs = 20e6 # 20 MHz sampling for simulation resolution
    t_pulse, tx_iq = generate_msk_pulse(BARKER_13, fs, BIT_RATE)

    # 3. Geometry & Propagation Loop
    max_range = 440e3 # 440 km
    step_range = 10e3 # 10 km
    ranges = np.arange(step_range, max_range + step_range, step_range)
    
    # Spiral logic: 8 deg clockwise per step
    azimuths = (np.arange(len(ranges)) * 8) % 360
    #print(azimuths)
    
    clutter_data = []

    ### TESTING ###
    rangekm = 400000**4
    print(rangekm)
    

    for r, az in zip(ranges, azimuths):
        # Time delay
        delay = 2 * r / C
        
        # Radar Equation
        # Pr = (Pt * G^2 * lambda^2 * sigma) / ((4*pi)^3 * R^4)
        numerator = PT * (gain**2) * (LAMBDA**2) * rcs_building
        denominator = ((4 * np.pi)**3) * r**4
        pr_watts = (numerator / denominator)
        print(denominator)
        #print(pr_watts)
        #print(np.power(r, 4))

        # Convert to Voltage (amplitude) assuming 50 ohm system (sqrt(P*R)) or normalized
        # We will use sqrt(Pr) for signal amplitude scaling
        rx_amp = np.sqrt(pr_watts)
        
        # Phase shift due to range
        # phi = -4 * pi * R / lambda
        phase_shift = np.exp(-1j * 4 * np.pi * r / LAMBDA)
        
        # Store Data
        clutter_data.append({
            'range_m': r,
            'azimuth_deg': az,
            'rx_power_dbm': 10*np.log10(np.abs(pr_watts)) + 30, # Convert to dBm
            'rx_amp': rx_amp,
            'phase': phase_shift,
            'delay_s': delay
        })

    # 4. Generate Received IQ (Example for the closest and furthest target)
    # We apply the amplitude and phase shift to the original TX waveform
    
    target_idx_close = 0 # 10km
    target_idx_far = -1  # 440km
    
    rx_iq_close = tx_iq * clutter_data[target_idx_close]['rx_amp'] * clutter_data[target_idx_close]['phase']
    rx_iq_far = tx_iq * clutter_data[target_idx_far]['rx_amp'] * clutter_data[target_idx_far]['phase']

    # --- UPDATED Plotting Section (Fixing Alignment & Labels) ---
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Spatial PPI View (Corrected Geometry)
    ax1 = fig.add_subplot(2, 2, 1, projection='polar')
    
    # Data Prep
    theta_rad = np.deg2rad([d['azimuth_deg'] for d in clutter_data])
    r_km = np.array([d['range_m'] for d in clutter_data]) / 1000
    colors = [d['rx_power_dbm'] for d in clutter_data]
    
    
    # 1. Plot Spiral Path (The "Flight Line")
    ax1.plot(theta_rad, r_km, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    # 2. Plot Shadow Beamwidth (FIX: align='center')
    # This ensures the 1-degree beam is centered on the target angle
    beam_width_rad = np.deg2rad(BEAMWIDTH_DEG)
    ax1.bar(theta_rad, r_km, width=beam_width_rad, bottom=0.0, 
            color='blue', alpha=0.1, align='center', zorder=2)

    # 3. Plot Targets (The Buildings)
    sc = ax1.scatter(theta_rad, r_km, c=colors, cmap='plasma', s=80, 
                     edgecolors='black', linewidth=0.8, zorder=3)
    #print(np.rad2deg(theta_rad), r_km)
    
    # --- GEOMETRY FIXES ---
    ax1.set_title("PPI View: Clutter Map\n(Spiral Path 0-440km)", va='bottom', weight='bold')
    ax1.set_theta_zero_location("N") # 0 degrees = North
    ax1.set_theta_direction(-1)      # Rotate Clockwise
    
    # FIX: Move radial labels (100km, 200km text) to 180 degrees (South) 
    # so they don't overlap with the dense data at 45 degrees.
    ax1.set_rlabel_position(180)  
    
    # Define Angle Ticks (every 45 deg)
    ax1.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    
    # Ensure Grid is BEHIND data
    ax1.set_axisbelow(True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.colorbar(sc, ax=ax1, label='Rx Power (dBm)')


    # Plot 2: Rx Power vs Range
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(r_km, colors, 'o-', color='darkblue', markersize=4)
    ax2.set_xlabel("Range (km)")
    ax2.set_ylabel("Received Power (dBm)")
    ax2.set_title("Clutter Return Strength over Range")
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)

    # Plot 3: IQ Constellation
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(np.real(tx_iq), np.imag(tx_iq), 'k-', alpha=0.2, label='TX Ref')
    
    # Normalize and Plot Close Target
    norm_factor = np.max(np.abs(rx_iq_close))
    if norm_factor > 0:
        norm_rx = rx_iq_close / norm_factor
        ax3.plot(np.real(norm_rx), np.imag(norm_rx), 'r--', linewidth=2, label=f'RX @ 10km')
        
    ax3.set_title("IQ Constellation (Normalized)")
    ax3.set_xlabel("I")
    ax3.set_ylabel("Q")
    ax3.axis('equal')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    # Plot 4: Time Domain
    ax4 = fig.add_subplot(2, 2, 4)
    # Plot only a segment if pulse is long, but Barker 13 is short enough to show all
    ax4.plot(t_pulse*1e6, np.real(tx_iq), label="I", color='blue', linewidth=1.5)
    ax4.plot(t_pulse*1e6, np.imag(tx_iq), label="Q", color='orange', linewidth=1.5, alpha=0.8)
    ax4.set_title("Baseband Pulse (13-bit Barker)")
    ax4.set_xlabel("Time (us)")
    ax4.legend(loc='upper right')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()
    

# Execute
data = run_simulation()