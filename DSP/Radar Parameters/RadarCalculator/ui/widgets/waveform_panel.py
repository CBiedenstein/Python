"""
Waveform selection and configuration panel.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QGroupBox, QComboBox
)
from PyQt6.QtCore import pyqtSignal

from core.waveforms import (
    WaveformType, WaveformParameters, calculate_waveform_performance,
    get_waveform_description, get_waveform_types
)


class WaveformPanel(QWidget):
    """
    Panel for configuring radar waveform and pulse compression.
    """

    waveform_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._update_display()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Waveform / Pulse Compression")
        group_layout = QVBoxLayout(group)

        # Waveform selector
        wf_layout = QGridLayout()

        wf_layout.addWidget(QLabel("Waveform:"), 0, 0)
        self.waveform_combo = QComboBox()
        for wf_type in get_waveform_types():
            self.waveform_combo.addItem(wf_type.value, wf_type)
        wf_layout.addWidget(self.waveform_combo, 0, 1, 1, 2)

        # Description
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #888; font-style: italic; font-size: 9pt;")
        wf_layout.addWidget(self.description_label, 1, 0, 1, 3)

        group_layout.addLayout(wf_layout)

        # Pulse parameters
        pulse_layout = QGridLayout()

        pulse_layout.addWidget(QLabel("Pulse Width:"), 0, 0)
        self.pulse_width_spin = QDoubleSpinBox()
        self.pulse_width_spin.setRange(0.1, 10000.0)
        self.pulse_width_spin.setDecimals(2)
        self.pulse_width_spin.setValue(10.0)
        pulse_layout.addWidget(self.pulse_width_spin, 0, 1)

        self.pulse_width_unit = QComboBox()
        self.pulse_width_unit.addItems(["µs", "ms"])
        pulse_layout.addWidget(self.pulse_width_unit, 0, 2)

        # Chirp bandwidth (for FM chirp waveforms)
        pulse_layout.addWidget(QLabel("Chirp BW:"), 1, 0)
        self.chirp_bw_spin = QDoubleSpinBox()
        self.chirp_bw_spin.setRange(0.1, 1000.0)
        self.chirp_bw_spin.setDecimals(2)
        self.chirp_bw_spin.setValue(10.0)
        pulse_layout.addWidget(self.chirp_bw_spin, 1, 1)

        self.chirp_bw_unit = QComboBox()
        self.chirp_bw_unit.addItems(["MHz", "kHz"])
        pulse_layout.addWidget(self.chirp_bw_unit, 1, 2)

        group_layout.addLayout(pulse_layout)

        # Results display
        results_layout = QGridLayout()

        results_layout.addWidget(QLabel("Compression Ratio:"), 0, 0)
        self.compression_label = QLabel("--")
        self.compression_label.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(self.compression_label, 0, 1)

        results_layout.addWidget(QLabel("Processing Gain:"), 1, 0)
        self.gain_label = QLabel("--")
        self.gain_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        results_layout.addWidget(self.gain_label, 1, 1)

        results_layout.addWidget(QLabel("Range Resolution:"), 2, 0)
        self.resolution_label = QLabel("--")
        self.resolution_label.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(self.resolution_label, 2, 1)

        results_layout.addWidget(QLabel("Peak Sidelobes:"), 3, 0)
        self.sidelobe_label = QLabel("--")
        results_layout.addWidget(self.sidelobe_label, 3, 1)

        group_layout.addLayout(results_layout)

        layout.addWidget(group)

    def _connect_signals(self):
        self.waveform_combo.currentIndexChanged.connect(self._on_waveform_changed)
        self.pulse_width_spin.valueChanged.connect(self._on_value_changed)
        self.pulse_width_unit.currentIndexChanged.connect(self._on_value_changed)
        self.chirp_bw_spin.valueChanged.connect(self._on_value_changed)
        self.chirp_bw_unit.currentIndexChanged.connect(self._on_value_changed)

    def _on_waveform_changed(self, index):
        """Handle waveform type change."""
        wf_type = self.waveform_combo.currentData()

        # Enable/disable chirp bandwidth based on waveform type
        is_chirp = wf_type in [WaveformType.FM_CHIRP, WaveformType.NLFM_CHIRP]
        self.chirp_bw_spin.setEnabled(is_chirp)
        self.chirp_bw_unit.setEnabled(is_chirp)

        # Update description
        self.description_label.setText(get_waveform_description(wf_type))

        self._update_display()
        self.waveform_changed.emit()

    def _on_value_changed(self):
        self._update_display()
        self.waveform_changed.emit()

    def _update_display(self):
        """Update the calculated results display."""
        params = self.get_waveform_parameters()
        perf = calculate_waveform_performance(params)

        self.compression_label.setText(f"{perf.compression_ratio:.1f}x")
        self.gain_label.setText(f"+{perf.processing_gain_db:.1f} dB")

        # Format range resolution
        if perf.range_resolution_m >= 1000:
            self.resolution_label.setText(f"{perf.range_resolution_m/1000:.2f} km")
        elif perf.range_resolution_m >= 1:
            self.resolution_label.setText(f"{perf.range_resolution_m:.1f} m")
        else:
            self.resolution_label.setText(f"{perf.range_resolution_m*100:.1f} cm")

        self.sidelobe_label.setText(f"{perf.peak_sidelobe_level_db:.1f} dB")

    def get_pulse_width_s(self) -> float:
        """Get pulse width in seconds."""
        value = self.pulse_width_spin.value()
        unit = self.pulse_width_unit.currentText()
        if unit == "µs":
            return value * 1e-6
        elif unit == "ms":
            return value * 1e-3
        return value

    def get_chirp_bandwidth_hz(self) -> float:
        """Get chirp bandwidth in Hz."""
        value = self.chirp_bw_spin.value()
        unit = self.chirp_bw_unit.currentText()
        if unit == "MHz":
            return value * 1e6
        elif unit == "kHz":
            return value * 1e3
        return value

    def get_waveform_type(self) -> WaveformType:
        """Get the selected waveform type."""
        return self.waveform_combo.currentData()

    def get_waveform_parameters(self) -> WaveformParameters:
        """Get complete waveform parameters."""
        wf_type = self.get_waveform_type()

        # For chirp, use chirp bandwidth; otherwise use receiver bandwidth
        chirp_bw = None
        if wf_type in [WaveformType.FM_CHIRP, WaveformType.NLFM_CHIRP]:
            chirp_bw = self.get_chirp_bandwidth_hz()

        return WaveformParameters(
            waveform_type=wf_type,
            pulse_width_s=self.get_pulse_width_s(),
            bandwidth_hz=self.get_chirp_bandwidth_hz(),  # Default bandwidth
            chirp_bandwidth_hz=chirp_bw
        )

    def get_processing_gain_db(self) -> float:
        """Get the processing gain in dB."""
        params = self.get_waveform_parameters()
        perf = calculate_waveform_performance(params)
        return perf.processing_gain_db

    def get_compression_ratio(self) -> float:
        """Get the compression ratio."""
        params = self.get_waveform_parameters()
        perf = calculate_waveform_performance(params)
        return perf.compression_ratio
