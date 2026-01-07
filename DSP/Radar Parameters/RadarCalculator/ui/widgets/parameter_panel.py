"""
Parameter input panel for radar system parameters.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QGroupBox, QComboBox
)
from PyQt6.QtCore import pyqtSignal
from core.units import FrequencyUnit, PowerUnit


class ParameterPanel(QWidget):
    """
    Panel for inputting radar system parameters.
    Emits parameter_changed signal whenever any value changes.
    """

    parameter_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Transmitter Group
        tx_group = QGroupBox("Transmitter")
        tx_layout = QGridLayout(tx_group)

        # Frequency
        tx_layout.addWidget(QLabel("Frequency:"), 0, 0)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.001, 100.0)
        self.freq_spin.setDecimals(3)
        self.freq_spin.setValue(3.0)
        self.freq_spin.setSuffix("")
        tx_layout.addWidget(self.freq_spin, 0, 1)

        self.freq_unit = QComboBox()
        self.freq_unit.addItems(["GHz", "MHz"])
        tx_layout.addWidget(self.freq_unit, 0, 2)

        # Peak Power
        tx_layout.addWidget(QLabel("Peak Power:"), 1, 0)
        self.power_spin = QDoubleSpinBox()
        self.power_spin.setRange(0.001, 10000.0)
        self.power_spin.setDecimals(3)
        self.power_spin.setValue(2.4)
        tx_layout.addWidget(self.power_spin, 1, 1)

        self.power_unit = QComboBox()
        self.power_unit.addItems(["MW", "kW", "W"])
        tx_layout.addWidget(self.power_unit, 1, 2)

        layout.addWidget(tx_group)

        # Antenna Group
        ant_group = QGroupBox("Antenna")
        ant_layout = QGridLayout(ant_group)

        # Tx Gain
        ant_layout.addWidget(QLabel("Tx Gain:"), 0, 0)
        self.tx_gain_spin = QDoubleSpinBox()
        self.tx_gain_spin.setRange(0.0, 60.0)
        self.tx_gain_spin.setDecimals(1)
        self.tx_gain_spin.setValue(36.0)
        self.tx_gain_spin.setSuffix(" dBi")
        ant_layout.addWidget(self.tx_gain_spin, 0, 1)

        # Rx Gain
        ant_layout.addWidget(QLabel("Rx Gain:"), 1, 0)
        self.rx_gain_spin = QDoubleSpinBox()
        self.rx_gain_spin.setRange(0.0, 60.0)
        self.rx_gain_spin.setDecimals(1)
        self.rx_gain_spin.setValue(39.0)
        self.rx_gain_spin.setSuffix(" dBi")
        ant_layout.addWidget(self.rx_gain_spin, 1, 1)

        # Total Gain (read-only display)
        ant_layout.addWidget(QLabel("Total Gain:"), 2, 0)
        self.total_gain_label = QLabel("75.0 dBi")
        self.total_gain_label.setStyleSheet("font-weight: bold;")
        ant_layout.addWidget(self.total_gain_label, 2, 1)

        layout.addWidget(ant_group)

        # Receiver Group
        rx_group = QGroupBox("Receiver")
        rx_layout = QGridLayout(rx_group)

        # Bandwidth
        rx_layout.addWidget(QLabel("Bandwidth:"), 0, 0)
        self.bw_spin = QDoubleSpinBox()
        self.bw_spin.setRange(0.001, 1000.0)
        self.bw_spin.setDecimals(3)
        self.bw_spin.setValue(1.6)
        rx_layout.addWidget(self.bw_spin, 0, 1)

        self.bw_unit = QComboBox()
        self.bw_unit.addItems(["MHz", "kHz"])
        rx_layout.addWidget(self.bw_unit, 0, 2)

        # Noise Figure
        rx_layout.addWidget(QLabel("Noise Figure:"), 1, 0)
        self.nf_spin = QDoubleSpinBox()
        self.nf_spin.setRange(0.0, 20.0)
        self.nf_spin.setDecimals(1)
        self.nf_spin.setValue(2.0)
        self.nf_spin.setSuffix(" dB")
        rx_layout.addWidget(self.nf_spin, 1, 1)

        layout.addWidget(rx_group)

        # System Group
        sys_group = QGroupBox("System")
        sys_layout = QGridLayout(sys_group)

        # System Losses
        sys_layout.addWidget(QLabel("System Loss:"), 0, 0)
        self.loss_spin = QDoubleSpinBox()
        self.loss_spin.setRange(0.0, 30.0)
        self.loss_spin.setDecimals(1)
        self.loss_spin.setValue(5.0)
        self.loss_spin.setSuffix(" dB")
        sys_layout.addWidget(self.loss_spin, 0, 1)

        # Required SNR
        sys_layout.addWidget(QLabel("Required SNR:"), 1, 0)
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(0.0, 30.0)
        self.snr_spin.setDecimals(1)
        self.snr_spin.setValue(13.0)
        self.snr_spin.setSuffix(" dB")
        sys_layout.addWidget(self.snr_spin, 1, 1)

        layout.addWidget(sys_group)

        # Add stretch at bottom
        layout.addStretch()

    def _connect_signals(self):
        """Connect all spinbox value changes to emit parameter_changed."""
        self.freq_spin.valueChanged.connect(self._on_value_changed)
        self.freq_unit.currentIndexChanged.connect(self._on_value_changed)
        self.power_spin.valueChanged.connect(self._on_value_changed)
        self.power_unit.currentIndexChanged.connect(self._on_value_changed)
        self.tx_gain_spin.valueChanged.connect(self._on_gain_changed)
        self.rx_gain_spin.valueChanged.connect(self._on_gain_changed)
        self.bw_spin.valueChanged.connect(self._on_value_changed)
        self.bw_unit.currentIndexChanged.connect(self._on_value_changed)
        self.nf_spin.valueChanged.connect(self._on_value_changed)
        self.loss_spin.valueChanged.connect(self._on_value_changed)
        self.snr_spin.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        """Called when any value changes."""
        self.parameter_changed.emit()

    def _on_gain_changed(self):
        """Update total gain display when gains change."""
        total = self.tx_gain_spin.value() + self.rx_gain_spin.value()
        self.total_gain_label.setText(f"{total:.1f} dBi")
        self.parameter_changed.emit()

    def get_frequency_hz(self) -> float:
        """Get frequency in Hz."""
        value = self.freq_spin.value()
        unit = self.freq_unit.currentText()
        if unit == "GHz":
            return value * 1e9
        elif unit == "MHz":
            return value * 1e6
        return value

    def get_power_watts(self) -> float:
        """Get peak power in Watts."""
        value = self.power_spin.value()
        unit = self.power_unit.currentText()
        if unit == "MW":
            return value * 1e6
        elif unit == "kW":
            return value * 1e3
        return value

    def get_bandwidth_hz(self) -> float:
        """Get bandwidth in Hz."""
        value = self.bw_spin.value()
        unit = self.bw_unit.currentText()
        if unit == "MHz":
            return value * 1e6
        elif unit == "kHz":
            return value * 1e3
        return value

    def get_tx_gain_dbi(self) -> float:
        return self.tx_gain_spin.value()

    def get_rx_gain_dbi(self) -> float:
        return self.rx_gain_spin.value()

    def get_noise_figure_db(self) -> float:
        return self.nf_spin.value()

    def get_system_loss_db(self) -> float:
        return self.loss_spin.value()

    def get_required_snr_db(self) -> float:
        return self.snr_spin.value()

    def set_from_profile(self, profile):
        """Set all values from a RadarProfile object."""
        # Block signals to prevent multiple updates
        self.blockSignals(True)

        # Frequency - convert to appropriate unit
        freq_ghz = profile.frequency_hz / 1e9
        if freq_ghz >= 1.0:
            self.freq_spin.setValue(freq_ghz)
            self.freq_unit.setCurrentText("GHz")
        else:
            self.freq_spin.setValue(profile.frequency_hz / 1e6)
            self.freq_unit.setCurrentText("MHz")

        # Power - convert to appropriate unit
        power_mw = profile.peak_power_w / 1e6
        if power_mw >= 1.0:
            self.power_spin.setValue(power_mw)
            self.power_unit.setCurrentText("MW")
        elif profile.peak_power_w >= 1000:
            self.power_spin.setValue(profile.peak_power_w / 1e3)
            self.power_unit.setCurrentText("kW")
        else:
            self.power_spin.setValue(profile.peak_power_w)
            self.power_unit.setCurrentText("W")

        # Gains
        self.tx_gain_spin.setValue(profile.tx_gain_dbi)
        self.rx_gain_spin.setValue(profile.rx_gain_dbi)

        # Bandwidth
        bw_mhz = profile.bandwidth_hz / 1e6
        if bw_mhz >= 0.1:
            self.bw_spin.setValue(bw_mhz)
            self.bw_unit.setCurrentText("MHz")
        else:
            self.bw_spin.setValue(profile.bandwidth_hz / 1e3)
            self.bw_unit.setCurrentText("kHz")

        # Other parameters
        self.nf_spin.setValue(profile.noise_figure_db)
        self.loss_spin.setValue(profile.system_loss_db)
        self.snr_spin.setValue(profile.required_snr_db)

        # Update total gain display
        total = profile.tx_gain_dbi + profile.rx_gain_dbi
        self.total_gain_label.setText(f"{total:.1f} dBi")

        self.blockSignals(False)
        self.parameter_changed.emit()
