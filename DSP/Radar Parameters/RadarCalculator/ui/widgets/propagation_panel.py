"""
Propagation model selection panel for Site Survey tab.
Provides RF propagation model selection and parameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QGroupBox, QComboBox, QCheckBox
)
from PyQt6.QtCore import pyqtSignal

from core.propagation import PropagationModel


class PropagationPanel(QWidget):
    """
    Panel for selecting and configuring RF propagation model.

    Provides:
    - Model selection (LOS, Free-space + terrain, Longley-Rice)
    - Target height configuration
    - Advanced propagation parameters
    """

    model_changed = pyqtSignal(object)  # PropagationModel
    parameters_changed = pyqtSignal()

    # Model descriptions for tooltip/label
    MODEL_DESCRIPTIONS = {
        PropagationModel.LINE_OF_SIGHT: (
            "Line-of-Sight (LOS)",
            "Simple geometric visibility check with 4/3 earth curvature model. "
            "Fast calculation but does not account for diffraction or atmospheric effects. "
            "Best for initial coverage estimates."
        ),
        PropagationModel.FREE_SPACE_TERRAIN: (
            "Free-Space + Terrain Masking",
            "Combines free-space path loss (1/R²) with terrain obstruction checking. "
            "If terrain blocks LOS, signal is considered blocked. "
            "Good balance of accuracy and speed."
        ),
        PropagationModel.LONGLEY_RICE: (
            "Longley-Rice (ITM)",
            "Irregular Terrain Model considering diffraction, tropospheric scatter, "
            "and surface reflections. Most accurate but requires SPLAT! installation. "
            "Best for detailed site surveys."
        )
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._update_description()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Model selection group
        model_group = QGroupBox("Propagation Model")
        model_layout = QVBoxLayout(model_group)

        # Model combo box
        self.model_combo = QComboBox()
        for model in PropagationModel:
            name, _ = self.MODEL_DESCRIPTIONS[model]
            self.model_combo.addItem(name, model)
        model_layout.addWidget(self.model_combo)

        # Model description
        self.model_desc = QLabel("")
        self.model_desc.setWordWrap(True)
        self.model_desc.setStyleSheet("color: #888; font-size: 9pt; padding: 4px;")
        model_layout.addWidget(self.model_desc)

        layout.addWidget(model_group)

        # Target parameters group
        target_group = QGroupBox("Target Parameters")
        target_layout = QGridLayout(target_group)

        # Target height AGL
        target_layout.addWidget(QLabel("Target Height:"), 0, 0)
        self.target_height_spin = QDoubleSpinBox()
        self.target_height_spin.setRange(0.0, 50000.0)
        self.target_height_spin.setDecimals(1)
        self.target_height_spin.setValue(100.0)
        self.target_height_spin.setSuffix(" m AGL")
        self.target_height_spin.setToolTip(
            "Target altitude Above Ground Level.\n"
            "Aircraft: 1000-12000m, Drones: 50-500m, Ground vehicles: 2-5m"
        )
        target_layout.addWidget(self.target_height_spin, 0, 1)

        # Target RCS
        target_layout.addWidget(QLabel("Target RCS:"), 1, 0)
        self.rcs_spin = QDoubleSpinBox()
        self.rcs_spin.setRange(0.0001, 10000.0)
        self.rcs_spin.setDecimals(4)
        self.rcs_spin.setValue(1.0)
        self.rcs_spin.setSuffix(" m²")
        self.rcs_spin.setToolTip(
            "Radar Cross Section of target.\n"
            "Stealth aircraft: 0.001-0.1 m², Fighter: 1-5 m², "
            "Airliner: 10-100 m², Ship: 100-10000 m²"
        )
        target_layout.addWidget(self.rcs_spin, 1, 1)

        layout.addWidget(target_group)

        # Advanced parameters group (collapsible via checkbox)
        self.advanced_group = QGroupBox("Advanced Parameters")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        adv_layout = QGridLayout(self.advanced_group)

        # Refractivity
        adv_layout.addWidget(QLabel("Refractivity:"), 0, 0)
        self.refractivity_spin = QDoubleSpinBox()
        self.refractivity_spin.setRange(200.0, 400.0)
        self.refractivity_spin.setDecimals(1)
        self.refractivity_spin.setValue(301.0)
        self.refractivity_spin.setSuffix(" N-units")
        self.refractivity_spin.setToolTip(
            "Surface refractivity for atmospheric effects.\n"
            "Standard: 301 N-units (continental temperate)"
        )
        adv_layout.addWidget(self.refractivity_spin, 0, 1)

        # Ground conductivity
        adv_layout.addWidget(QLabel("Ground Cond.:"), 1, 0)
        self.conductivity_spin = QDoubleSpinBox()
        self.conductivity_spin.setRange(0.0001, 5.0)
        self.conductivity_spin.setDecimals(4)
        self.conductivity_spin.setValue(0.005)
        self.conductivity_spin.setSuffix(" S/m")
        self.conductivity_spin.setToolTip(
            "Ground conductivity for surface reflections.\n"
            "Seawater: 5 S/m, Wet ground: 0.03 S/m, "
            "Average land: 0.005 S/m, Dry/rocky: 0.001 S/m"
        )
        adv_layout.addWidget(self.conductivity_spin, 1, 1)

        # Ground permittivity
        adv_layout.addWidget(QLabel("Permittivity:"), 2, 0)
        self.permittivity_spin = QDoubleSpinBox()
        self.permittivity_spin.setRange(1.0, 100.0)
        self.permittivity_spin.setDecimals(1)
        self.permittivity_spin.setValue(15.0)
        self.permittivity_spin.setToolTip(
            "Relative ground permittivity.\n"
            "Seawater: 81, Wet ground: 25, Average land: 15, Desert: 3"
        )
        adv_layout.addWidget(self.permittivity_spin, 2, 1)

        # 4/3 Earth checkbox
        self.use_43_earth = QCheckBox("Use 4/3 effective earth radius")
        self.use_43_earth.setChecked(True)
        self.use_43_earth.setToolTip(
            "Account for standard atmospheric refraction by using\n"
            "4/3 effective earth radius (8495 km vs 6371 km)"
        )
        adv_layout.addWidget(self.use_43_earth, 3, 0, 1, 2)

        layout.addWidget(self.advanced_group)

        # Grid resolution group
        resolution_group = QGroupBox("Calculation Grid")
        res_layout = QGridLayout(resolution_group)

        res_layout.addWidget(QLabel("Resolution:"), 0, 0)
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(100.0, 10000.0)
        self.resolution_spin.setDecimals(0)
        self.resolution_spin.setValue(1000.0)
        self.resolution_spin.setSuffix(" m")
        self.resolution_spin.setToolTip(
            "Grid cell size for coverage calculation.\n"
            "Smaller values = more accurate but slower.\n"
            "Recommended: 500-2000m for area surveys"
        )
        res_layout.addWidget(self.resolution_spin, 0, 1)

        layout.addWidget(resolution_group)

    def _connect_signals(self):
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.target_height_spin.valueChanged.connect(self._on_parameters_changed)
        self.rcs_spin.valueChanged.connect(self._on_parameters_changed)
        self.refractivity_spin.valueChanged.connect(self._on_parameters_changed)
        self.conductivity_spin.valueChanged.connect(self._on_parameters_changed)
        self.permittivity_spin.valueChanged.connect(self._on_parameters_changed)
        self.use_43_earth.stateChanged.connect(self._on_parameters_changed)
        self.resolution_spin.valueChanged.connect(self._on_parameters_changed)

    def _update_description(self):
        """Update model description label."""
        model = self.model_combo.currentData()
        if model in self.MODEL_DESCRIPTIONS:
            _, desc = self.MODEL_DESCRIPTIONS[model]
            self.model_desc.setText(desc)

    def _on_model_changed(self, index):
        self._update_description()
        model = self.model_combo.currentData()
        self.model_changed.emit(model)

    def _on_parameters_changed(self):
        self.parameters_changed.emit()

    # Public API

    def get_propagation_model(self) -> PropagationModel:
        """Get selected propagation model."""
        return self.model_combo.currentData()

    def get_target_height_m(self) -> float:
        """Get target height AGL in meters."""
        return self.target_height_spin.value()

    def get_target_rcs_m2(self) -> float:
        """Get target RCS in square meters."""
        return self.rcs_spin.value()

    def get_refractivity(self) -> float:
        """Get surface refractivity in N-units."""
        return self.refractivity_spin.value()

    def get_ground_conductivity(self) -> float:
        """Get ground conductivity in S/m."""
        return self.conductivity_spin.value()

    def get_ground_permittivity(self) -> float:
        """Get relative ground permittivity."""
        return self.permittivity_spin.value()

    def get_use_43_earth(self) -> bool:
        """Get whether to use 4/3 effective earth radius."""
        return self.use_43_earth.isChecked()

    def get_resolution_m(self) -> float:
        """Get grid resolution in meters."""
        return self.resolution_spin.value()

    def get_propagation_params(self):
        """Get PropagationParams object with current settings."""
        from core.propagation import PropagationParams
        return PropagationParams(
            radar_height_m=0,  # Set by caller
            target_height_m=self.get_target_height_m(),
            refractivity_n=self.get_refractivity(),
            ground_conductivity=self.get_ground_conductivity(),
            ground_permittivity=self.get_ground_permittivity(),
            use_43_earth=self.get_use_43_earth()
        )

    def set_propagation_model(self, model: PropagationModel):
        """Set the propagation model."""
        index = self.model_combo.findData(model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def set_target_height_m(self, height_m: float):
        """Set target height in meters."""
        self.target_height_spin.setValue(height_m)

    def set_target_rcs_m2(self, rcs_m2: float):
        """Set target RCS in square meters."""
        self.rcs_spin.setValue(rcs_m2)

    def set_resolution_m(self, resolution_m: float):
        """Set grid resolution in meters."""
        self.resolution_spin.setValue(resolution_m)
