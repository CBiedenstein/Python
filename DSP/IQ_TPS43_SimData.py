import numpy as np
import matplotlib.pyplot as plt

def generate_tps75_pulse():
    # --- AN/TPS-75 Parameters ---
    # The TPS-75 uses a 6.8 microsecond pulse with Barker coding.
    pulse_duration = 6.8e-6  # 6.8 microseconds
    barker_code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    code_len = len(barker_code)
    
    # --- Simulation Config ---
    fs = 20e6  # 20 MHz sampling rate (High enough to see transition shapes)
    
    # Calculate timing
    total_samples = int(pulse_duration * fs)
    samples_per_chip = total_samples // code_len
    
    # 1. Generate the Baseband Signal (BPSK)
    # Upsample the Barker sequence to the sample rate
    # We repeat each 'chip' of the code for the calculated duration
    baseband_signal = np.repeat(barker_code, samples_per_chip)
    
    # Trim or Pad to match exact calculated time axis if rounding occurred
    t_axis = np.linspace(0, pulse_duration, len(baseband_signal))
    
    # 2. Generate IQ Data
    # For BPSK:
    # In-Phase (I) = Amplitude * Code (+1 or -1)
    # Quadrature (Q) = 0 (Ideal BPSK has no Q component relative to carrier)
    
    # However, to make the plot realistic, we can add a slight phase offset
    # or keep it purely Real for the theoretical waveform.
    # Let's assume a normalized amplitude of 1.0 (0 dB)
    
    i_data = baseband_signal * 1.0
    q_data = np.zeros_like(baseband_signal)
    
    # Combine into Complex IQ
    iq_complex = i_data + 1j * q_data

    return t_axis, iq_complex, barker_code

# --- Generate and Plot ---
t, iq, code = generate_tps75_pulse()

plt.figure(figsize=(10, 8))

# Plot 1: In-Phase (I) Component
plt.subplot(3, 1, 1)
plt.plot(t * 1e6, iq.real, color='blue', linewidth=1.5)
plt.title('AN/TPS-75 Waveform: In-Phase (I) Data')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.ylim(-1.2, 1.2)
# Annotate the chips
chip_duration = 6.8 / 13
for k in range(13):
    plt.axvline(x=k*chip_duration, color='gray', linestyle=':', alpha=0.5)

# Plot 2: Quadrature (Q) Component
plt.subplot(3, 1, 2)
plt.plot(t * 1e6, iq.imag, color='orange', linewidth=1.5)
plt.title('AN/TPS-75 Waveform: Quadrature (Q) Data (Ideal BPSK)')
plt.ylabel('Amplitude')
plt.ylim(-1.2, 1.2)
plt.grid(True, alpha=0.3)
plt.text(3.4, 0.5, "Zero Q component in ideal BPSK", ha='center', color='orange')

# Plot 3: Phase
plt.subplot(3, 1, 3)
# Calculate phase and convert to degrees
phase_deg = np.angle(iq) * (180/np.pi)
plt.step(t * 1e6, phase_deg, color='green', where='post')
plt.title('Phase Modulation (BPSK)')
plt.xlabel('Time (microseconds)')
plt.ylabel('Phase (Degrees)')
plt.yticks([-180, 0, 180])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()