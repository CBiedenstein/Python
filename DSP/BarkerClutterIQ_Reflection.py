import numpy as np
import matplotlib.pyplot as plt

# --- Constants & Radar Parameters ---
C = 3e8                     
FC = 2.9e9                  
LAMBDA = C / FC             
PT = 1e9                    # 1 GW
BEAMWIDTH_DEG = 1.0         
BIT_RATE = 2e6              

# Barker 13 Sequence
BARKER_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])

# --- Helper Functions ---
def calculate_gain(bw_deg):
    return 41253 / (bw_deg * bw_deg)

def estimate_building_rcs():
    # 28m x 30m flat plate
    h, w = 28.0, 30.0
    area = h * w
    sigma = (4 * np.pi * (area**2)) / (LAMBDA**2)
    return sigma

def generate_msk_pulse(sequence, samp_rate, bit_rate):
    samps_per_bit = int(samp_rate / bit_rate)
    t = np.linspace(0, len(sequence)/bit_rate, len(sequence)*samps_per_bit)
    phase = np.zeros_like(t)
    current_phase = 0
    
    for i, bit in enumerate(sequence):
        t_bit = t[i*samps_per_bit : (i+1)*samps_per_bit]
        slope = bit * np.pi / (2 * (1/bit_rate)) 
        segment_t = t_bit - t_bit[0]
        phase[i*samps_per_bit : (i+1)*samps_per_bit] = current_phase + slope * segment_t
        current_phase += bit * (np.pi / 2)

    i_data = np.cos(phase)
    q_data = np.sin(phase)
    return t, i_data + 1j * q_data

# --- Main Simulation ---
def run_simulation():
    print(f"--- Radar Simulation Initiated ---")
    
    # 1. Setup Environment
    gain = calculate_gain(BEAMWIDTH_DEG)
    rcs_building = estimate_building_rcs()

    # 2. Generate Transmit Waveform (Baseband IQ)
    # 10 Samples per bit for smooth circles
    fs = BIT_RATE * 10 
    t_pulse, tx_iq = generate_msk_pulse(BARKER_13, fs, BIT_RATE)

    # 3. Geometry & Propagation Loop
    max_range = 440e3 
    step_range = 10e3 
    ranges = np.arange(step_range, max_range + step_range, step_range)
    azimuths = (np.arange(len(ranges)) * 8) % 360
    
    clutter_data = []

    # Store IQ for specific targets to plot later
    rx_iq_10km = None
    rx_iq_200km = None

    for r, az in zip(ranges, azimuths):
        # Calculate Received Power (Watts)
        numerator = PT * (gain**2) * (LAMBDA**2) * rcs_building
        denominator = ((4 * np.pi)**3) * (r**4)
        pr_watts = numerator / denominator
        
        # Calculate Received Voltage (Amplitude)
        # Assuming 50 ohm system, V = sqrt(P * R_imp)
        # We will keep it relative: Amplitude factor = sqrt(Pr) / sqrt(Pt)
        # But to be realistic "Data Received", we multiply the normalized IQ by sqrt(Pr)
        rx_amp = np.sqrt(pr_watts)
        
        # Calculate Phase Shift
        # Two-way path: 2 * R
        phase_shift = np.exp(-1j * 4 * np.pi * r / LAMBDA)
        
        # Generate the specific IQ packet for this reflection
        # Signal = Transmitted_Waveform * Amplitude_Scaling * Phase_Rotation
        current_rx_iq = tx_iq * rx_amp * phase_shift
        
        # Save specific slices for plotting
        if r == 10000:
            rx_iq_10km = current_rx_iq
        if r == 200000:
            rx_iq_200km = current_rx_iq

        clutter_data.append({
            'range_m': r,
            'azimuth_deg': az,
            'rx_power_dbm': 10*np.log10(pr_watts * 1000), 
            'rx_amp': rx_amp
        })

    # --- PLOTTING ---
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Spatial PPI View (Corrected)
    ax1 = fig.add_subplot(2, 2, 1, projection='polar')
    theta_rad = np.deg2rad([d['azimuth_deg'] for d in clutter_data])
    r_km = np.array([d['range_m'] for d in clutter_data]) / 1000
    colors = [d['rx_power_dbm'] for d in clutter_data]
    
    ax1.plot(theta_rad, r_km, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=5)
    beam_width_rad = np.deg2rad(BEAMWIDTH_DEG)
    ax1.bar(theta_rad, r_km, width=beam_width_rad, bottom=0.0, color='blue', alpha=0.1, align='center', zorder=5)
    sc = ax1.scatter(theta_rad, r_km, c=colors, cmap='turbo', s=80, edgecolors='black', linewidth=0.8, zorder=10)
    
    ax1.set_title("PPI View: Clutter Map", weight='bold')
    ax1.set_theta_zero_location("N") 
    ax1.set_theta_direction(-1)      
    ax1.set_rlabel_position(180)  
    ax1.set_axisbelow(True)
    plt.colorbar(sc, ax=ax1, label='Rx Power (dBm)')

    # Plot 2: Rx Power vs Range
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(r_km, colors, 'o-', color='darkblue', markersize=4)
    ax2.set_xlabel("Range (km)")
    ax2.set_ylabel("Received Power (dBm)")
    ax2.grid(True, linestyle='--')

    # Plot 3: IQ Constellation (ACTUAL REFLECTION)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # 1. Plot Reference (Transmit) - Scaled down to be visible? 
    # No, let's normalize the PLOT axes, but plot the real shape data.
    # Since 10km return is HUGE voltage and 200km is small, we plot them separately.
    
    # Plotting 10km Return
    # We remove the mean phase to center the rotation for visualization, 
    # OR we plot it raw to show the phase shift. Let's plot RAW.
    
    # To make them visible on the same chart, we normalize them to Unity 
    # but keep the Phase Rotation (which is the critical data component)
    
    # Normalize magnitudes for comparison of Phase
    norm_10km = rx_iq_10km / np.abs(rx_iq_10km)
    norm_200km = rx_iq_200km / np.abs(rx_iq_200km)
    
    ax3.plot(np.real(norm_10km), np.imag(norm_10km), 'r-', linewidth=2, label='Rx @ 10km (Rotated)')
    ax3.plot(np.real(norm_200km), np.imag(norm_200km), 'g--', linewidth=2, label='Rx @ 200km (Rotated)')
    
    # Plot the original reference for comparison (Black)
    ax3.plot(np.real(tx_iq), np.imag(tx_iq), 'k:', alpha=0.5, label='Tx Reference (0 deg)')

    ax3.set_title("IQ Constellation: Received Reflections\n(Amplitude Normalized, Phase Preserved)")
    ax3.set_xlabel("In-Phase (I)")
    ax3.set_ylabel("Quadrature (Q)")
    ax3.axis('equal')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    
    # Add text annotation about the phase
    phase_10km = np.angle(rx_iq_10km[0])
    ax3.text(0, 0, f"Phase Shift 10km: {np.degrees(phase_10km):.1f}°", ha='center', va='center', fontsize=9)

    # Plot 4: Time Domain (Real Received Voltage at 10km)
    ax4 = fig.add_subplot(2, 2, 4)
    # Here we plot the ACTUAL voltage amplitude to show the raw data
    ax4.plot(t_pulse*1e6, np.real(rx_iq_10km), color='red', label="Rx I-Channel (Volts)")
    ax4.plot(t_pulse*1e6, np.imag(rx_iq_10km), color='darkred', alpha=0.5, label="Rx Q-Channel (Volts)")
    
    ax4.set_title(f"Received Pulse @ 10km (Raw Voltage)")
    ax4.set_xlabel("Time (us)")
    ax4.set_ylabel("Amplitude (Volts)")
    ax4.legend(loc='upper right')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

run_simulation()