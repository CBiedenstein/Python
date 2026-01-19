"""
FM Up Chirp Radar Signal Generator

Generates raw IQ radar data using FM (Linear Frequency Modulation) up chirp waveforms.
Supports configurable chirp bandwidth, sampling frequency, and pulse parameters.

Author: Generated for radar signal processing applications
"""

import os
import sys

# Suppress Qt DPI warning on Windows (must be set before importing Qt)
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
if sys.platform == 'win32':
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QGroupBox, QLabel, QDoubleSpinBox,
                              QSpinBox, QPushButton, QComboBox, QFileDialog,
                              QCheckBox, QGridLayout)
from PyQt6 import QtCore
from dataclasses import dataclass
from typing import Tuple, Optional
import struct


@dataclass
class ChirpParameters:
    """Parameters defining an FM chirp waveform."""
    chirp_bandwidth_hz: float      # Chirp sweep bandwidth (Hz)
    sampling_freq_hz: float        # Sampling frequency (Hz)
    pulse_width_s: float           # Pulse duration (seconds)
    center_freq_hz: float = 0.0    # Center/IF frequency (Hz) - 0 for baseband
    start_freq_hz: Optional[float] = None  # Start frequency (if None, calculated from center)

    def __post_init__(self):
        """Calculate derived parameters."""
        if self.start_freq_hz is None:
            # Default: symmetric chirp around center frequency
            self.start_freq_hz = self.center_freq_hz - self.chirp_bandwidth_hz / 2

    @property
    def end_freq_hz(self) -> float:
        """End frequency of chirp."""
        return self.start_freq_hz + self.chirp_bandwidth_hz

    @property
    def chirp_rate_hz_per_s(self) -> float:
        """Chirp rate (Hz/s)."""
        return self.chirp_bandwidth_hz / self.pulse_width_s

    @property
    def num_samples(self) -> int:
        """Number of samples in one pulse."""
        return int(self.pulse_width_s * self.sampling_freq_hz)

    @property
    def range_resolution_m(self) -> float:
        """Range resolution in meters."""
        c = 299792458.0  # Speed of light
        return c / (2 * self.chirp_bandwidth_hz)

    @property
    def time_bandwidth_product(self) -> float:
        """Time-bandwidth product."""
        return self.chirp_bandwidth_hz * self.pulse_width_s

    @property
    def processing_gain_db(self) -> float:
        """Processing gain in dB."""
        return 10 * np.log10(self.time_bandwidth_product)


def generate_fm_up_chirp(params: ChirpParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an FM up chirp (Linear Frequency Modulation) signal.

    The chirp sweeps from start_freq to end_freq over the pulse duration.

    Args:
        params: ChirpParameters defining the waveform

    Returns:
        t: Time vector (seconds)
        signal: Complex IQ signal
    """
    # Time vector
    num_samples = params.num_samples
    t = np.arange(num_samples) / params.sampling_freq_hz

    # Chirp rate (frequency change per second)
    k = params.chirp_rate_hz_per_s

    # Instantaneous phase: phi(t) = 2*pi*(f0*t + k*t^2/2)
    # This gives instantaneous frequency: f(t) = f0 + k*t
    phase = 2 * np.pi * (params.start_freq_hz * t + k * t**2 / 2)

    # Complex signal (IQ)
    signal = np.exp(1j * phase)

    return t, signal


def generate_fm_down_chirp(params: ChirpParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an FM down chirp signal (frequency decreases over time).

    Args:
        params: ChirpParameters defining the waveform

    Returns:
        t: Time vector (seconds)
        signal: Complex IQ signal
    """
    # Time vector
    num_samples = params.num_samples
    t = np.arange(num_samples) / params.sampling_freq_hz

    # For down chirp, start at high frequency and sweep down
    start_freq = params.end_freq_hz
    k = -params.chirp_rate_hz_per_s  # Negative chirp rate

    phase = 2 * np.pi * (start_freq * t + k * t**2 / 2)
    signal = np.exp(1j * phase)

    return t, signal


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add AWGN to a signal at specified SNR.

    Args:
        signal: Input complex signal
        snr_db: Desired SNR in dB

    Returns:
        Noisy signal
    """
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10**(snr_db/10))
    noise = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) * np.sqrt(noise_power/2)
    return signal + noise


def apply_window(signal: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
    """
    Apply a window function to the signal for sidelobe reduction.

    Args:
        signal: Input signal
        window_type: 'hamming', 'hanning', 'blackman', 'kaiser', or 'none'

    Returns:
        Windowed signal
    """
    n = len(signal)
    if window_type == 'hamming':
        window = np.hamming(n)
    elif window_type == 'hanning':
        window = np.hanning(n)
    elif window_type == 'blackman':
        window = np.blackman(n)
    elif window_type == 'kaiser':
        window = np.kaiser(n, beta=6)
    else:
        window = np.ones(n)

    return signal * window


def compute_spectrum(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum of signal.

    Args:
        signal: Input signal
        fs: Sampling frequency

    Returns:
        freqs: Frequency vector (Hz)
        psd: Power spectral density (dB)
    """
    fft_data = np.fft.fftshift(np.fft.fft(signal))
    psd = 10 * np.log10(np.abs(fft_data)**2 + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1/fs))
    return freqs, psd


def matched_filter(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Apply matched filter (pulse compression) to signal.

    Args:
        signal: Received signal
        reference: Reference chirp (transmitted waveform)

    Returns:
        Compressed pulse output
    """
    # Matched filter = correlation with conjugate of reference
    # In frequency domain: FFT(signal) * conj(FFT(reference))
    ref_conj = np.conj(reference[::-1])  # Time-reversed conjugate
    compressed = np.correlate(signal, reference, mode='full')
    return compressed


def save_iq_binary(filename: str, signal: np.ndarray, dtype: str = 'complex64'):
    """
    Save IQ data to binary file.

    Args:
        filename: Output filename
        signal: Complex IQ signal
        dtype: Data type ('complex64', 'complex128', 'int16')
    """
    if dtype == 'complex64':
        signal.astype(np.complex64).tofile(filename)
    elif dtype == 'complex128':
        signal.astype(np.complex128).tofile(filename)
    elif dtype == 'int16':
        # Normalize and scale to int16 range
        max_val = np.max(np.abs(signal))
        scale = 32767 / max_val if max_val > 0 else 1
        i_data = (signal.real * scale).astype(np.int16)
        q_data = (signal.imag * scale).astype(np.int16)
        # Interleaved I/Q
        interleaved = np.empty(2 * len(signal), dtype=np.int16)
        interleaved[0::2] = i_data
        interleaved[1::2] = q_data
        interleaved.tofile(filename)


def load_iq_binary(filename: str, dtype: str = 'complex64') -> np.ndarray:
    """
    Load IQ data from binary file.

    Args:
        filename: Input filename
        dtype: Data type used when saving

    Returns:
        Complex IQ signal
    """
    if dtype == 'complex64':
        return np.fromfile(filename, dtype=np.complex64)
    elif dtype == 'complex128':
        return np.fromfile(filename, dtype=np.complex128)
    elif dtype == 'int16':
        interleaved = np.fromfile(filename, dtype=np.int16)
        i_data = interleaved[0::2].astype(np.float32)
        q_data = interleaved[1::2].astype(np.float32)
        return i_data + 1j * q_data


class ChirpGeneratorGUI(QMainWindow):
    """GUI for FM chirp radar signal generation and visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FM Chirp Radar Signal Generator")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize data
        self.signal = None
        self.t = None
        self.params = None

        self.setup_ui()
        self.generate_signal()

    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Right panel - Plots
        self.create_plot_widget()
        main_layout.addWidget(self.plot_widget)
        main_layout.setStretch(1, 2)  # Give plot widget more space

    def create_control_panel(self) -> QWidget:
        """Create the parameter control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Chirp Parameters Group
        chirp_group = QGroupBox("Chirp Parameters")
        chirp_layout = QGridLayout(chirp_group)

        # Chirp Bandwidth
        chirp_layout.addWidget(QLabel("Chirp Bandwidth:"), 0, 0)
        self.chirp_bw_spin = QDoubleSpinBox()
        self.chirp_bw_spin.setRange(0.1, 1000)
        self.chirp_bw_spin.setValue(8.0)
        self.chirp_bw_spin.setSuffix(" MHz")
        self.chirp_bw_spin.setDecimals(2)
        chirp_layout.addWidget(self.chirp_bw_spin, 0, 1)

        # Sampling Frequency
        chirp_layout.addWidget(QLabel("Sampling Frequency:"), 1, 0)
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(1, 10000)
        self.fs_spin.setValue(32.0)
        self.fs_spin.setSuffix(" MHz")
        self.fs_spin.setDecimals(2)
        chirp_layout.addWidget(self.fs_spin, 1, 1)

        # Pulse Width
        chirp_layout.addWidget(QLabel("Pulse Width:"), 2, 0)
        self.pulse_width_spin = QDoubleSpinBox()
        self.pulse_width_spin.setRange(0.1, 10000)
        self.pulse_width_spin.setValue(10.0)
        self.pulse_width_spin.setSuffix(" µs")
        self.pulse_width_spin.setDecimals(2)
        chirp_layout.addWidget(self.pulse_width_spin, 2, 1)

        # Center/IF Frequency
        chirp_layout.addWidget(QLabel("Center/IF Frequency:"), 3, 0)
        self.center_freq_spin = QDoubleSpinBox()
        self.center_freq_spin.setRange(0, 1000)
        self.center_freq_spin.setValue(0.0)
        self.center_freq_spin.setSuffix(" MHz")
        self.center_freq_spin.setDecimals(2)
        chirp_layout.addWidget(self.center_freq_spin, 3, 1)

        layout.addWidget(chirp_group)

        # Signal Options Group
        options_group = QGroupBox("Signal Options")
        options_layout = QGridLayout(options_group)

        # SNR
        options_layout.addWidget(QLabel("SNR:"), 0, 0)
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-20, 100)
        self.snr_spin.setValue(30.0)
        self.snr_spin.setSuffix(" dB")
        options_layout.addWidget(self.snr_spin, 0, 1)

        # Add Noise Checkbox
        self.add_noise_check = QCheckBox("Add Noise")
        self.add_noise_check.setChecked(False)
        options_layout.addWidget(self.add_noise_check, 1, 0, 1, 2)

        # Window Function
        options_layout.addWidget(QLabel("Window:"), 2, 0)
        self.window_combo = QComboBox()
        self.window_combo.addItems(['none', 'hamming', 'hanning', 'blackman', 'kaiser'])
        options_layout.addWidget(self.window_combo, 2, 1)

        # Chirp Direction
        options_layout.addWidget(QLabel("Chirp Direction:"), 3, 0)
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(['Up Chirp', 'Down Chirp'])
        options_layout.addWidget(self.direction_combo, 3, 1)

        layout.addWidget(options_group)

        # Calculated Parameters (Read-only display)
        calc_group = QGroupBox("Calculated Parameters")
        calc_layout = QGridLayout(calc_group)

        self.range_res_label = QLabel("Range Resolution: --")
        self.tbp_label = QLabel("Time-Bandwidth Product: --")
        self.proc_gain_label = QLabel("Processing Gain: --")
        self.num_samples_label = QLabel("Samples per Pulse: --")
        self.chirp_rate_label = QLabel("Chirp Rate: --")

        calc_layout.addWidget(self.range_res_label, 0, 0)
        calc_layout.addWidget(self.tbp_label, 1, 0)
        calc_layout.addWidget(self.proc_gain_label, 2, 0)
        calc_layout.addWidget(self.num_samples_label, 3, 0)
        calc_layout.addWidget(self.chirp_rate_label, 4, 0)

        layout.addWidget(calc_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.generate_signal)
        btn_layout.addWidget(self.generate_btn)

        self.save_btn = QPushButton("Save IQ Data")
        self.save_btn.clicked.connect(self.save_data)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

        # Data Format Selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Save Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['complex64', 'complex128', 'int16'])
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)

        layout.addStretch()

        return panel

    def create_plot_widget(self):
        """Create the plot widget with multiple plots."""
        pg.setConfigOptions(antialias=True)

        # Create as a QWidget subclass explicitly
        self.plot_widget = pg.GraphicsLayoutWidget(parent=self)

        # Time Domain - Real Part
        self.p_time_real = self.plot_widget.addPlot(title="Time Domain - In-Phase (I)")
        self.p_time_real.setLabel('left', "Amplitude")
        self.p_time_real.setLabel('bottom', "Time", units='s')
        self.p_time_real.showGrid(x=True, y=True, alpha=0.3)
        self.curve_time_real = self.p_time_real.plot(pen=pg.mkPen('#00FF00', width=1))

        # Time Domain - Imaginary Part
        self.p_time_imag = self.plot_widget.addPlot(title="Time Domain - Quadrature (Q)")
        self.p_time_imag.setLabel('left', "Amplitude")
        self.p_time_imag.setLabel('bottom', "Time", units='s')
        self.p_time_imag.showGrid(x=True, y=True, alpha=0.3)
        self.curve_time_imag = self.p_time_imag.plot(pen=pg.mkPen('#FF6600', width=1))

        self.plot_widget.nextRow()

        # Frequency Domain
        self.p_freq = self.plot_widget.addPlot(title="Power Spectrum")
        self.p_freq.setLabel('left', "Power", units='dB')
        self.p_freq.setLabel('bottom', "Frequency", units='Hz')
        self.p_freq.showGrid(x=True, y=True, alpha=0.3)
        self.curve_freq = self.p_freq.plot(
            pen=pg.mkPen('#00FDFD', width=1.5),
            fillLevel=-150,
            brush=(0, 255, 255, 50)
        )

        # Instantaneous Frequency
        self.p_inst_freq = self.plot_widget.addPlot(title="Instantaneous Frequency")
        self.p_inst_freq.setLabel('left', "Frequency", units='Hz')
        self.p_inst_freq.setLabel('bottom', "Time", units='s')
        self.p_inst_freq.showGrid(x=True, y=True, alpha=0.3)
        self.curve_inst_freq = self.p_inst_freq.plot(pen=pg.mkPen('#FF00FF', width=2))

        self.plot_widget.nextRow()

        # IQ Constellation
        self.p_iq = self.plot_widget.addPlot(title="IQ Constellation")
        self.p_iq.setLabel('left', "Quadrature (Q)")
        self.p_iq.setLabel('bottom', "In-Phase (I)")
        self.p_iq.showGrid(x=True, y=True, alpha=0.5)
        self.p_iq.setAspectLocked(True)
        self.scatter_iq = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None),
                                              brush=pg.mkBrush(255, 255, 0, 100))
        self.p_iq.addItem(self.scatter_iq)

        # Pulse Compression (Matched Filter Output)
        self.p_compressed = self.plot_widget.addPlot(title="Matched Filter Output (Pulse Compression)")
        self.p_compressed.setLabel('left', "Magnitude", units='dB')
        self.p_compressed.setLabel('bottom', "Sample")
        self.p_compressed.showGrid(x=True, y=True, alpha=0.3)
        self.curve_compressed = self.p_compressed.plot(pen=pg.mkPen('#00FF00', width=1.5))

    def get_parameters(self) -> ChirpParameters:
        """Get current parameters from UI."""
        return ChirpParameters(
            chirp_bandwidth_hz=self.chirp_bw_spin.value() * 1e6,
            sampling_freq_hz=self.fs_spin.value() * 1e6,
            pulse_width_s=self.pulse_width_spin.value() * 1e-6,
            center_freq_hz=self.center_freq_spin.value() * 1e6
        )

    def update_calculated_labels(self):
        """Update the calculated parameter labels."""
        if self.params is None:
            return

        range_res = self.params.range_resolution_m
        if range_res >= 1:
            self.range_res_label.setText(f"Range Resolution: {range_res:.2f} m")
        else:
            self.range_res_label.setText(f"Range Resolution: {range_res*100:.2f} cm")

        self.tbp_label.setText(f"Time-Bandwidth Product: {self.params.time_bandwidth_product:.1f}")
        self.proc_gain_label.setText(f"Processing Gain: {self.params.processing_gain_db:.1f} dB")
        self.num_samples_label.setText(f"Samples per Pulse: {self.params.num_samples:,}")

        chirp_rate = self.params.chirp_rate_hz_per_s
        if chirp_rate >= 1e12:
            self.chirp_rate_label.setText(f"Chirp Rate: {chirp_rate/1e12:.2f} THz/s")
        elif chirp_rate >= 1e9:
            self.chirp_rate_label.setText(f"Chirp Rate: {chirp_rate/1e9:.2f} GHz/s")
        else:
            self.chirp_rate_label.setText(f"Chirp Rate: {chirp_rate/1e6:.2f} MHz/s")

    def generate_signal(self):
        """Generate the chirp signal and update plots."""
        self.params = self.get_parameters()

        # Generate chirp
        if self.direction_combo.currentText() == 'Up Chirp':
            self.t, self.signal = generate_fm_up_chirp(self.params)
        else:
            self.t, self.signal = generate_fm_down_chirp(self.params)

        # Store clean reference for matched filter
        reference_signal = self.signal.copy()

        # Apply window if selected
        window_type = self.window_combo.currentText()
        if window_type != 'none':
            self.signal = apply_window(self.signal, window_type)

        # Add noise if selected
        if self.add_noise_check.isChecked():
            self.signal = add_noise(self.signal, self.snr_spin.value())

        # Update plots
        self.update_plots(reference_signal)
        self.update_calculated_labels()

    def update_plots(self, reference_signal: np.ndarray):
        """Update all plots with current signal."""
        if self.signal is None:
            return

        # Downsample for plotting if too many points
        max_plot_points = 10000
        if len(self.signal) > max_plot_points:
            step = len(self.signal) // max_plot_points
            t_plot = self.t[::step]
            sig_plot = self.signal[::step]
        else:
            t_plot = self.t
            sig_plot = self.signal

        # Time domain - Real
        self.curve_time_real.setData(t_plot, sig_plot.real)

        # Time domain - Imaginary
        self.curve_time_imag.setData(t_plot, sig_plot.imag)

        # Frequency domain
        freqs, psd = compute_spectrum(self.signal, self.params.sampling_freq_hz)
        self.curve_freq.setData(freqs, psd)

        # Instantaneous frequency (derivative of phase)
        phase = np.unwrap(np.angle(self.signal))
        inst_freq = np.diff(phase) * self.params.sampling_freq_hz / (2 * np.pi)
        t_inst = self.t[:-1]

        if len(inst_freq) > max_plot_points:
            step = len(inst_freq) // max_plot_points
            t_inst = t_inst[::step]
            inst_freq = inst_freq[::step]

        self.curve_inst_freq.setData(t_inst, inst_freq)

        # IQ Constellation
        if len(sig_plot) > 5000:
            # Random subsample for constellation
            indices = np.random.choice(len(sig_plot), 5000, replace=False)
            self.scatter_iq.setData(x=sig_plot.real[indices], y=sig_plot.imag[indices])
        else:
            self.scatter_iq.setData(x=sig_plot.real, y=sig_plot.imag)

        # Matched filter output
        compressed = matched_filter(self.signal, reference_signal)
        compressed_mag = 20 * np.log10(np.abs(compressed) + 1e-12)
        compressed_mag = compressed_mag - np.max(compressed_mag)  # Normalize to 0 dB peak

        if len(compressed_mag) > max_plot_points:
            step = len(compressed_mag) // max_plot_points
            compressed_mag = compressed_mag[::step]

        self.curve_compressed.setData(compressed_mag)

    def save_data(self):
        """Save the IQ data to file."""
        if self.signal is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save IQ Data", "",
            "Binary Files (*.bin);;NumPy Files (*.npy);;All Files (*)"
        )

        if filename:
            if filename.endswith('.npy'):
                np.save(filename, self.signal)
            else:
                dtype = self.format_combo.currentText()
                save_iq_binary(filename, self.signal, dtype)

            print(f"Saved {len(self.signal)} samples to {filename}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle('Fusion')

    window = ChirpGeneratorGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
