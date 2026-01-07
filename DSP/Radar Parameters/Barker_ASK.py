import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore

# --- 1. DSP & Signal Simulation ---
def generate_qam_signal(num_symbols=1024, snr_db=20, oversample=8):
    """
    Generates a 16-QAM signal with pulse shaping and AWGN.
    """
    # 1. Generate random 16-QAM symbols
    # M=16, 4 bits per symbol. Map to -3, -1, 1, 3 grid.
    bits_i = np.random.choice([-3, -1, 1, 3], num_symbols)
    bits_q = np.random.choice([-3, -1, 1, 3], num_symbols)
    symbols = bits_i + 1j * bits_q

    # 2. Upsample (Zero stuffing)
    signal_up = np.zeros(num_symbols * oversample, dtype=complex)
    signal_up[::oversample] = symbols

    # 3. Pulse Shaping (Simulate Bandwidth Limit)
    # Using a Hamming window as a simple low-pass filter proxy for RRC
    # This gives the signal a realistic "hump" in the frequency domain
    t_filter = np.linspace(-4, 4, 8 * oversample) # Filter width
    # Sinc function * Window
    pulse = np.sinc(t_filter) * np.hamming(len(t_filter))
    
    # Convolve symbols with pulse
    tx_signal = np.convolve(signal_up, pulse, mode='same')

    # 4. Add AWGN (Noise)
    sig_power = np.mean(np.abs(tx_signal)**2)
    noise_power = sig_power / (10**(snr_db/10))
    noise = (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal))) \
            * np.sqrt(noise_power/2)
    
    rx_signal = tx_signal + noise

    # Extract symbols back for constellation plot (Downsample + simple phase match)
    # Ideally we would use a matched filter here, but simple decimation works for viz
    # We take the middle samples where the eye is open
    rx_symbols = rx_signal[::oversample] 
    
    # Normalize for plotting
    rx_symbols = rx_symbols / np.mean(np.abs(rx_symbols)) 

    return rx_signal, rx_symbols

# --- 2. Calculation Functions ---
def compute_fft(signal, fs=1.0):
    """Computes PSD for plotting."""
    fft_data = np.fft.fftshift(np.fft.fft(signal))
    fft_mag = 10 * np.log10(np.abs(fft_data)**2 + 1e-12) # Power in dB
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1/fs))
    return freqs, fft_mag

# --- 3. PyQtGraph Setup ---

app = QApplication(sys.argv)
pg.setConfigOptions(antialias=True)

# Main Window Layout
win = pg.GraphicsLayoutWidget(show=True, title="RX Signal Analysis")
win.resize(1200, 600)
win.setWindowTitle('Digital Signal Processing: 16-QAM')

# Generate initial data
fs = 10e6 # 10 MHz Sample Rate
rx_signal, rx_symbols = generate_qam_signal(num_symbols=2048, snr_db=25)
freqs, fft_mag = compute_fft(rx_signal, fs=fs)

# --- Plot 1: Frequency Domain (Left) ---
p_freq = win.addPlot(title="Received Spectrum (Frequency Domain)")
p_freq.setLabel('left', "Power Spectral Density", units='dB')
p_freq.setLabel('bottom', "Frequency", units='Hz')
p_freq.showGrid(x=True, y=True, alpha=0.3)
# Set fillLevel to a low number (your noise floor, e.g., -150 dB)
# The brush argument defines the fill color
curve_fft = p_freq.plot(freqs, fft_mag, 
                        pen=pg.mkPen("#00FDFDFF", width=1.5),
                        fillLevel=-150, 
                        brush=(0, 255, 255, 50))


# --- Plot 2: Constellation Diagram (Right) ---
p_const = win.addPlot(title="Constellation (IQ Plane)")
p_const.setLabel('left', "Quadrature (Q)")
p_const.setLabel('bottom', "In-Phase (I)")
p_const.showGrid(x=True, y=True, alpha=0.5)
p_const.setAspectLocked(True) # Crucial for Constellations to look square

# Use ScatterPlotItem for dots
scatter = pg.ScatterPlotItem(
    size=5, 
    pen=pg.mkPen(None), 
    brush=pg.mkBrush(255, 255, 0, 150) # Yellow with alpha
)
scatter.setData(x=rx_symbols.real, y=rx_symbols.imag)
p_const.addItem(scatter)


# --- Optional: Interactive Update ---
# This timer regenerates the noise every 100ms to make it look "live"
def update():
    # Simulate new burst
    rx_sig, rx_sym = generate_qam_signal(num_symbols=2048, snr_db=20) 
    
    # Update FFT
    f, mag = compute_fft(rx_sig, fs=fs)
    curve_fft.setData(f, mag)
    
    # Update fill (hacky way to update fill is often to remove/add, 
    # but strictly setting curve data often updates fill automatically in newer PG versions.
    # If fill doesn't update, we just update the curve)
    
    # Update Constellation
    scatter.setData(x=rx_sym.real, y=rx_sym.imag)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(100) # Update every 100ms

# --- Execution ---
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())