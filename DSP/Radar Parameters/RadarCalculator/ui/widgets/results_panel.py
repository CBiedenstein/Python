"""
Results display panel showing calculated radar performance metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt
from core.radar_equations import CalculationResults, RadarParameters


class ResultCard(QFrame):
    """A card widget for displaying a single result value."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            ResultCard {
                background-color: #313244;
                border-radius: 6px;
                padding: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #a6adc8; font-size: 9pt;")
        layout.addWidget(self.title_label)

        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(self.value_label)

    def set_value(self, value: str, color: str = None):
        """Set the displayed value."""
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {color};")
        else:
            self.value_label.setStyleSheet("font-size: 14pt; font-weight: bold;")


class ResultsPanel(QWidget):
    """
    Panel displaying calculated radar performance results.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Primary Results Group
        primary_group = QGroupBox("Results")
        primary_layout = QVBoxLayout(primary_group)

        # Detection status
        self.status_label = QLabel("CALCULATING...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            background-color: #585b70;
            color: #cdd6f4;
            border-radius: 4px;
            padding: 8px;
            font-weight: bold;
            font-size: 11pt;
        """)
        primary_layout.addWidget(self.status_label)

        # Result cards in 2x2 grid
        cards_grid = QGridLayout()
        cards_grid.setSpacing(8)

        self.pr_card = ResultCard("Received Power")
        cards_grid.addWidget(self.pr_card, 0, 0)

        self.snr_card = ResultCard("SNR (Raw)")
        cards_grid.addWidget(self.snr_card, 0, 1)

        self.snr_eff_card = ResultCard("SNR (w/ Compression)")
        cards_grid.addWidget(self.snr_eff_card, 1, 0)

        self.max_range_card = ResultCard("Max Range")
        cards_grid.addWidget(self.max_range_card, 1, 1)

        self.mds_card = ResultCard("MDS")
        cards_grid.addWidget(self.mds_card, 2, 0)

        self.proc_gain_card = ResultCard("Processing Gain")
        cards_grid.addWidget(self.proc_gain_card, 2, 1)

        primary_layout.addLayout(cards_grid)

        layout.addWidget(primary_group)

        # Link Budget Group
        budget_group = QGroupBox("Link Budget")
        budget_layout = QVBoxLayout(budget_group)

        self.budget_grid = QGridLayout()
        self.budget_grid.setSpacing(4)

        # Create budget rows
        self._budget_rows = {}
        budget_items = [
            ("pt", "Tx Power", "+"),
            ("gt", "+ Tx Gain", "+"),
            ("gr", "+ Rx Gain", "+"),
            ("rcs", "+ RCS", "+"),
            ("loss", "- System Loss", "-"),
            ("path", "- Path Loss", "-"),
        ]

        for row, (key, label, sign) in enumerate(budget_items):
            name_label = QLabel(label)
            if sign == "-":
                name_label.setStyleSheet("color: #f38ba8;")
            else:
                name_label.setStyleSheet("color: #a6e3a1;")
            self.budget_grid.addWidget(name_label, row, 0)

            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            value_label.setStyleSheet("font-family: monospace;")
            self.budget_grid.addWidget(value_label, row, 1)

            self._budget_rows[key] = value_label

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #45475a;")
        self.budget_grid.addWidget(sep, len(budget_items), 0, 1, 2)

        # Result row
        pr_name = QLabel("= Received Power")
        pr_name.setStyleSheet("font-weight: bold; color: #89b4fa;")
        self.budget_grid.addWidget(pr_name, len(budget_items) + 1, 0)

        self.budget_pr_value = QLabel("--")
        self.budget_pr_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.budget_pr_value.setStyleSheet("font-family: monospace; font-weight: bold; color: #89b4fa;")
        self.budget_grid.addWidget(self.budget_pr_value, len(budget_items) + 1, 1)

        budget_layout.addLayout(self.budget_grid)

        layout.addWidget(budget_group)

        # Additional Info Group
        info_group = QGroupBox("Additional Info")
        info_layout = QGridLayout(info_group)

        self.wavelength_label = QLabel("Wavelength: --")
        info_layout.addWidget(self.wavelength_label, 0, 0)

        self.min_rcs_label = QLabel("Min RCS: --")
        info_layout.addWidget(self.min_rcs_label, 0, 1)

        self.margin_label = QLabel("Margin: --")
        info_layout.addWidget(self.margin_label, 1, 0)

        self.noise_floor_label = QLabel("Noise Floor: --")
        info_layout.addWidget(self.noise_floor_label, 1, 1)

        layout.addWidget(info_group)

        # Add stretch
        layout.addStretch()

    def update_results(self, results: CalculationResults, params: RadarParameters,
                       range_unit: str = "nmi", processing_gain_db: float = 0.0):
        """Update all displayed results."""

        # Calculate effective SNR with pulse compression
        effective_snr_db = results.snr_db + processing_gain_db
        effective_margin_db = effective_snr_db - params.required_snr_db
        is_detected = effective_snr_db >= params.required_snr_db

        # Detection status (based on effective SNR with compression)
        if is_detected:
            self.status_label.setText(f"DETECTED (+{effective_margin_db:.1f} dB margin)")
            self.status_label.setStyleSheet("""
                background-color: #9ece6a;
                color: #1a1b26;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
                font-size: 11pt;
            """)
        else:
            self.status_label.setText(f"NOT DETECTED ({effective_margin_db:.1f} dB below)")
            self.status_label.setStyleSheet("""
                background-color: #f7768e;
                color: #1a1b26;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
                font-size: 11pt;
            """)

        # Result cards
        self.pr_card.set_value(f"{results.received_power_dbm:.1f} dBm")
        self.snr_card.set_value(
            f"{results.snr_db:.1f} dB",
            "#9ece6a" if results.snr_db >= params.required_snr_db else "#f7768e"
        )
        self.snr_eff_card.set_value(
            f"{effective_snr_db:.1f} dB",
            "#9ece6a" if is_detected else "#f7768e"
        )
        self.mds_card.set_value(f"{results.mds_dbm:.1f} dBm")

        # Processing gain
        if processing_gain_db > 0:
            self.proc_gain_card.set_value(f"+{processing_gain_db:.1f} dB", "#7aa2f7")
        else:
            self.proc_gain_card.set_value("None (0 dB)")

        # Max range with unit conversion
        max_range_display = self._convert_range(results.max_range_m, range_unit)
        self.max_range_card.set_value(f"{max_range_display:.1f} {range_unit}")

        # Link budget breakdown
        self._budget_rows["pt"].setText(f"{results.peak_power_dbm:.1f} dBm")
        self._budget_rows["gt"].setText(f"{params.tx_gain_dbi:.1f} dBi")
        self._budget_rows["gr"].setText(f"{params.rx_gain_dbi:.1f} dBi")
        self._budget_rows["rcs"].setText(f"{results.rcs_dbsm:.1f} dBsm")
        self._budget_rows["loss"].setText(f"{params.system_loss_db:.1f} dB")
        self._budget_rows["path"].setText(f"{results.path_loss_db:.1f} dB")
        self.budget_pr_value.setText(f"{results.received_power_dbm:.1f} dBm")

        # Additional info
        self.wavelength_label.setText(f"Wavelength: {results.wavelength_m:.4f} m")
        self.min_rcs_label.setText(f"Min RCS: {results.min_detectable_rcs_m2:.2e} m²")
        self.margin_label.setText(f"Margin: {results.detection_margin_db:+.1f} dB")
        self.noise_floor_label.setText(f"Noise Floor: {results.thermal_noise_dbm:.1f} dBm")

    def _convert_range(self, range_m: float, unit: str) -> float:
        """Convert range from meters to specified unit."""
        if unit == "nmi":
            return range_m / 1852.0
        elif unit == "km":
            return range_m / 1000.0
        elif unit == "mi":
            return range_m / 1609.344
        return range_m

    def clear(self):
        """Clear all displayed values."""
        self.status_label.setText("CALCULATING...")
        self.status_label.setStyleSheet("""
            background-color: #565f89;
            color: #c0caf5;
            border-radius: 4px;
            padding: 8px;
            font-weight: bold;
            font-size: 11pt;
        """)

        self.pr_card.set_value("--")
        self.snr_card.set_value("--")
        self.snr_eff_card.set_value("--")
        self.mds_card.set_value("--")
        self.max_range_card.set_value("--")
        self.proc_gain_card.set_value("--")

        for label in self._budget_rows.values():
            label.setText("--")
        self.budget_pr_value.setText("--")

        self.wavelength_label.setText("Wavelength: --")
        self.min_rcs_label.setText("Min RCS: --")
        self.margin_label.setText("Margin: --")
        self.noise_floor_label.setText("Noise Floor: --")
