"""
Target selection panel for radar scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QGroupBox, QComboBox
)
from PyQt6.QtCore import pyqtSignal
from core.targets import TARGET_LIBRARY, get_all_categories, get_targets_by_category


class TargetPanel(QWidget):
    """
    Panel for selecting target parameters (RCS and range).
    """

    target_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._populate_targets()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Target")
        group_layout = QVBoxLayout(group)

        # Target selector
        target_grid = QGridLayout()

        target_grid.addWidget(QLabel("Category:"), 0, 0)
        self.category_combo = QComboBox()
        target_grid.addWidget(self.category_combo, 0, 1, 1, 2)

        target_grid.addWidget(QLabel("Target:"), 1, 0)
        self.target_combo = QComboBox()
        target_grid.addWidget(self.target_combo, 1, 1, 1, 2)

        group_layout.addLayout(target_grid)

        # RCS input
        rcs_layout = QGridLayout()
        rcs_layout.addWidget(QLabel("RCS:"), 0, 0)
        self.rcs_spin = QDoubleSpinBox()
        self.rcs_spin.setRange(1e-6, 1e6)
        self.rcs_spin.setDecimals(4)
        self.rcs_spin.setValue(1.0)
        rcs_layout.addWidget(self.rcs_spin, 0, 1)

        self.rcs_unit = QComboBox()
        self.rcs_unit.addItems(["m²", "dBsm"])
        rcs_layout.addWidget(self.rcs_unit, 0, 2)

        group_layout.addLayout(rcs_layout)

        # Range input
        range_layout = QGridLayout()
        range_layout.addWidget(QLabel("Range:"), 0, 0)
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(0.1, 10000.0)
        self.range_spin.setDecimals(1)
        self.range_spin.setValue(240.0)
        range_layout.addWidget(self.range_spin, 0, 1)

        self.range_unit = QComboBox()
        self.range_unit.addItems(["nmi", "km", "mi"])
        range_layout.addWidget(self.range_unit, 0, 2)

        group_layout.addLayout(range_layout)

        layout.addWidget(group)

    def _connect_signals(self):
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        self.target_combo.currentIndexChanged.connect(self._on_target_selected)
        self.rcs_spin.valueChanged.connect(self._on_value_changed)
        self.rcs_unit.currentIndexChanged.connect(self._on_rcs_unit_changed)
        self.range_spin.valueChanged.connect(self._on_value_changed)
        self.range_unit.currentIndexChanged.connect(self._on_value_changed)

    def _populate_targets(self):
        """Populate category and target combo boxes."""
        self.category_combo.blockSignals(True)
        self.target_combo.blockSignals(True)

        # Add categories
        categories = sorted(get_all_categories())
        self.category_combo.addItems(categories)

        # Add "Custom" option at the top
        self.category_combo.insertItem(0, "Custom")

        self.category_combo.blockSignals(False)
        self.target_combo.blockSignals(False)

        # Trigger population of targets for first category
        self._on_category_changed(0)

    def _on_category_changed(self, index):
        """Update target list when category changes."""
        category = self.category_combo.currentText()

        self.target_combo.blockSignals(True)
        self.target_combo.clear()

        if category == "Custom":
            self.target_combo.addItem("Custom")
            self.rcs_spin.setEnabled(True)
        else:
            targets = get_targets_by_category(category)
            for name in sorted(targets.keys()):
                self.target_combo.addItem(name)

            # Also allow custom within category
            self.target_combo.addItem("Custom")

        self.target_combo.blockSignals(False)

        # Select first target
        if self.target_combo.count() > 0:
            self._on_target_selected(0)

    def _on_target_selected(self, index):
        """Update RCS value when target is selected."""
        target_name = self.target_combo.currentText()

        if target_name == "Custom":
            self.rcs_spin.setEnabled(True)
        elif target_name in TARGET_LIBRARY:
            self.rcs_spin.setEnabled(False)
            target = TARGET_LIBRARY[target_name]

            # Set RCS value based on current unit
            if self.rcs_unit.currentText() == "m²":
                self.rcs_spin.setValue(target.rcs_m2)
            else:
                import math
                self.rcs_spin.setValue(10 * math.log10(target.rcs_m2))

        self.target_changed.emit()

    def _on_rcs_unit_changed(self, index):
        """Convert RCS value when unit changes."""
        import math
        current_value = self.rcs_spin.value()
        new_unit = self.rcs_unit.currentText()

        self.rcs_spin.blockSignals(True)

        if new_unit == "dBsm":
            # Convert m² to dBsm
            if current_value > 0:
                self.rcs_spin.setRange(-60.0, 60.0)
                self.rcs_spin.setValue(10 * math.log10(current_value))
            else:
                self.rcs_spin.setRange(-60.0, 60.0)
                self.rcs_spin.setValue(-60.0)
        else:
            # Convert dBsm to m²
            self.rcs_spin.setRange(1e-6, 1e6)
            self.rcs_spin.setValue(10 ** (current_value / 10))

        self.rcs_spin.blockSignals(False)
        self.target_changed.emit()

    def _on_value_changed(self):
        self.target_changed.emit()

    def get_rcs_m2(self) -> float:
        """Get RCS value in square meters."""
        import math
        value = self.rcs_spin.value()
        if self.rcs_unit.currentText() == "dBsm":
            return 10 ** (value / 10)
        return value

    def get_range_m(self) -> float:
        """Get range value in meters."""
        value = self.range_spin.value()
        unit = self.range_unit.currentText()
        if unit == "nmi":
            return value * 1852.0
        elif unit == "km":
            return value * 1000.0
        elif unit == "mi":
            return value * 1609.344
        return value

    def get_target_name(self) -> str:
        """Get the name of the selected target."""
        return self.target_combo.currentText()

    def set_range(self, value: float, unit: str = "nmi"):
        """Set the range value and unit."""
        self.range_unit.setCurrentText(unit)
        self.range_spin.setValue(value)

    def set_rcs(self, value_m2: float):
        """Set the RCS value in square meters."""
        if self.rcs_unit.currentText() == "m²":
            self.rcs_spin.setValue(value_m2)
        else:
            import math
            self.rcs_spin.setValue(10 * math.log10(value_m2))
