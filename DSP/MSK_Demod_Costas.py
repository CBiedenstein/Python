import numpy as np
import matplotlib.pyplot as plt

def generate_barker13():
    # 13-bit Barker Sequence
    return np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])

def run_msk_simulation():
    # --- 1. CONFIGURATION ---
    Fs = 1000             # Sampling Rate (Hz)
    baud_rate = 50        # Symbol rate
    sps = Fs // baud_rate # Samples per symbol
    T = 1.0 / baud_rate   # Symbol duration
    
    freq_offset = 0.5     # Carrier Frequency offset (Hz)
    phase_offset = np.pi / 3 
    noise_level = 0.2

    # --- 2. MSK SIGNAL GENERATION ---
    # MSK encodes bits into frequency shifts, resulting in a continuous phase ramp.
    # +1 bit = phase slope up, -1 bit = phase slope down.
    
    barker = generate_barker13()
    random_data = np.sign(np.random.randn(50))
    # Prepend zeros to let the loop settle before the barker arrives
    bits = np.concatenate((np.ones(10), barker, random_data))
    
    # Upsample bits to sample rate
    upsampled_bits = np.repeat(bits, sps)
    
    # Calculate Phase Trajectory
    # In MSK, phase changes by +/- pi/2 over one symbol period (T)
    # Slope = (pi/2) / T. Per sample change = (pi/2) / sps
    phase_step = (bits * (np.pi / 2)) / sps
    phase_step_upsampled = np.repeat(phase_step, sps)
    
    # Cumulative sum creates the continuous phase ramp
    tx_phase = np.cumsum(phase_step_upsampled)
    
    # Generate Complex Baseband Signal
    tx_signal = np.exp(1j * tx_phase)

    # --- 3. CHANNEL EFFECTS ---
    t = np.arange(len(tx_signal)) / Fs
    channel_spin = np.exp(1j * (2 * np.pi * freq_offset * t + phase_offset))
    rx_signal = tx_signal * channel_spin
    
    # Add Noise
    rx_signal += noise_level * (np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))

    # --- 4. MSK COSTAS LOOP ---
    # To lock to MSK, we process the I and Q arms with the MSK "shaping functions"
    # This effectively "matched filters" the MSK pulses inside the loop.
    
    phase_est = 0.0
    freq_est = 0.0
    
    # Gains
    alpha = 0.05 
    beta = 0.001
    
    corrected_signal = np.zeros(len(rx_signal), dtype=complex)
    soft_bits = np.zeros(len(rx_signal)) # The "demodulated" data
    
    phase_log = []
    error_log = []

    # Pre-calculate the MSK subcarrier references (Symbol Clock)
    # Ideally, this requires a separate Symbol Timing Recovery loop.
    # For this simulation, we assume we are roughly aligned to the grid.
    subcarrier_arg = (np.pi * t) / (2 * T)
    msk_ref_i = np.cos(subcarrier_arg)
    msk_ref_q = np.sin(subcarrier_arg)

    for i in range(len(rx_signal)):
        # 1. Derotate by Carrier Estimate
        sample = rx_signal[i] * np.exp(-1j * phase_est)
        corrected_signal[i] = sample
        
        # 2. MSK Processing (The Modification)
        # We multiply the I and Q arms by the MSK subcarrier shapes
        # This converts the MSK phase transitions into amplitude information we can lock to.
        # Note: This is a simplified version of the standard MSK/OQPSK loop.
        
        # Branch I: Weighted by Cos
        val_i = np.real(sample) * msk_ref_i[i]
        
        # Branch Q: Weighted by Sin
        val_q = np.imag(sample) * msk_ref_q[i]
        
        # 3. Error Detector
        # In a locked MSK loop, one branch holds data, the other is zero (at optimal sampling).
        # The product drives the phase error.
        error = val_i * val_q 
        
        # 4. Loop Filter
        freq_est += (beta * error)
        phase_est += (freq_est + (alpha * error))
        
        # Capture the "Soft Bit" for correlation
        # In MSK demodulation, valid data is essentially the sum of the weighted branches
        soft_bits[i] = val_i + val_q

        phase_log.append(phase_est)
        error_log.append(error)

    # --- 5. CORRELATION (MATCHED FILTER) ---
    # We correlate the RECOVERED bits (soft_bits) against the Barker template.
    # Since we upsampled the Tx, we upsample the template.
    barker_template = np.repeat(barker, sps)
    
    # We correlate using the 'soft_bits' output from the MSK demodulator
    correlation = np.correlate(soft_bits, barker_template, mode='valid')
    peak_idx = np.argmax(np.abs(correlation))

    # --- 6. VISUALIZATION ---
    plt.figure(figsize=(12, 12))

    # Plot 1: Constellation
    plt.subplot(4, 1, 1)
    plt.plot(np.real(rx_signal), np.imag(rx_signal), '.', alpha=0.3, label='Raw Input (Circle)')
    # For MSK, the "Corrected" signal is still a circle (Constant Amplitude), 
    # but the phase is now stabilized relative to the carrier.
    plt.plot(np.real(corrected_signal), np.imag(corrected_signal), '.', color='red', alpha=0.3, label='Derotated')
    plt.title("Constellation (MSK is constant envelope, so it looks like a circle)")
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.grid()

    # Plot 2: Demodulated Soft Bits
    plt.subplot(4, 1, 2)
    plt.plot(soft_bits[0:1500])
    plt.title("Demodulated Soft Bits (Output of MSK Weighting)")
    plt.grid()

    # Plot 3: Loop Error
    plt.subplot(4, 1, 3)
    plt.plot(error_log)
    plt.title("Costas Loop Error Signal")
    plt.grid()

    # Plot 4: Correlation
    plt.subplot(4, 1, 4)
    norm_corr = np.abs(correlation) / np.max(np.abs(correlation))
    plt.plot(norm_corr)
    plt.axvline(x=peak_idx, color='r', linestyle='--', label='Barker Detection')
    plt.title("Correlation Peak (Finding the Barker Code)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_msk_simulation()