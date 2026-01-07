"""
Unit toggle bar widget for switching between unit systems.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QRadioButton, QButtonGroup, QFrame
)
from PyQt6.QtCore import pyqtSignal


class UnitToggleBar(QWidget):
    """
    Bottom toolbar with radio buttons for unit preferences.
    """

    units_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            UnitToggleBar {
                background-color: #313244;
                border-top: 1px solid #45475a;
            }
            QLabel {
                color: #a6adc8;
                font-size: 9pt;
            }
            QRadioButton {
                color: #cdd6f4;
                font-size: 9pt;
                spacing: 4px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(24)

        # Range units
        layout.addWidget(QLabel("Range:"))
        self.range_group = QButtonGroup(self)

        self.range_nmi = QRadioButton("nmi")
        self.range_nmi.setChecked(True)
        self.range_group.addButton(self.range_nmi)
        layout.addWidget(self.range_nmi)

        self.range_km = QRadioButton("km")
        self.range_group.addButton(self.range_km)
        layout.addWidget(self.range_km)

        self.range_mi = QRadioButton("mi")
        self.range_group.addButton(self.range_mi)
        layout.addWidget(self.range_mi)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("background-color: #45475a;")
        layout.addWidget(sep1)

        # Power units
        layout.addWidget(QLabel("Power:"))
        self.power_group = QButtonGroup(self)

        self.power_mw = QRadioButton("MW")
        self.power_mw.setChecked(True)
        self.power_group.addButton(self.power_mw)
        layout.addWidget(self.power_mw)

        self.power_kw = QRadioButton("kW")
        self.power_group.addButton(self.power_kw)
        layout.addWidget(self.power_kw)

        self.power_w = QRadioButton("W")
        self.power_group.addButton(self.power_w)
        layout.addWidget(self.power_w)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("background-color: #45475a;")
        layout.addWidget(sep2)

        # Frequency units
        layout.addWidget(QLabel("Freq:"))
        self.freq_group = QButtonGroup(self)

        self.freq_ghz = QRadioButton("GHz")
        self.freq_ghz.setChecked(True)
        self.freq_group.addButton(self.freq_ghz)
        layout.addWidget(self.freq_ghz)

        self.freq_mhz = QRadioButton("MHz")
        self.freq_group.addButton(self.freq_mhz)
        layout.addWidget(self.freq_mhz)

        # Stretch to push everything left
        layout.addStretch()

        # Connect signals
        self.range_group.buttonClicked.connect(self._on_units_changed)
        self.power_group.buttonClicked.connect(self._on_units_changed)
        self.freq_group.buttonClicked.connect(self._on_units_changed)

    def _on_units_changed(self, button):
        self.units_changed.emit()

    def get_range_unit(self) -> str:
        """Get the currently selected range unit."""
        if self.range_nmi.isChecked():
            return "nmi"
        elif self.range_km.isChecked():
            return "km"
        elif self.range_mi.isChecked():
            return "mi"
        return "nmi"

    def get_power_unit(self) -> str:
        """Get the currently selected power unit."""
        if self.power_mw.isChecked():
            return "MW"
        elif self.power_kw.isChecked():
            return "kW"
        elif self.power_w.isChecked():
            return "W"
        return "MW"

    def get_freq_unit(self) -> str:
        """Get the currently selected frequency unit."""
        if self.freq_ghz.isChecked():
            return "GHz"
        elif self.freq_mhz.isChecked():
            return "MHz"
        return "GHz"

    def set_range_unit(self, unit: str):
        """Set the range unit."""
        if unit == "nmi":
            self.range_nmi.setChecked(True)
        elif unit == "km":
            self.range_km.setChecked(True)
        elif unit == "mi":
            self.range_mi.setChecked(True)

    def set_power_unit(self, unit: str):
        """Set the power unit."""
        if unit == "MW":
            self.power_mw.setChecked(True)
        elif unit == "kW":
            self.power_kw.setChecked(True)
        elif unit == "W":
            self.power_w.setChecked(True)

    def set_freq_unit(self, unit: str):
        """Set the frequency unit."""
        if unit == "GHz":
            self.freq_ghz.setChecked(True)
        elif unit == "MHz":
            self.freq_mhz.setChecked(True)
