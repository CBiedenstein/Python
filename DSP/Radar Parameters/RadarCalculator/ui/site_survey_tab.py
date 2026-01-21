"""
Site Survey tab for terrain-aware radar coverage analysis.

Combines elevation data with radar profiles to visualize RF coverage
using multiple propagation models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea,
    QTabWidget, QGroupBox, QPushButton, QProgressBar, QLabel,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot

from ui.widgets.radar_selector import RadarSelector
from ui.widgets.location_panel import LocationPanel
from ui.widgets.propagation_panel import PropagationPanel
from ui.plots.coverage_display import CoverageDisplay2D
from ui.plots.coverage_3d import CoverageDisplay3D


class CoverageCalculationThread(QThread):
    """Background thread for coverage calculation."""

    finished = pyqtSignal(object)  # CoverageResult or None
    progress = pyqtSignal(float)   # Progress 0.0 to 1.0
    error = pyqtSignal(str)

    def __init__(self, calculator, radar_lat, radar_lon, radar_height_m,
                 target_rcs_m2, target_height_m, max_range_km, resolution_m):
        super().__init__()
        self._calculator = calculator
        self._radar_lat = radar_lat
        self._radar_lon = radar_lon
        self._radar_height_m = radar_height_m
        self._target_rcs_m2 = target_rcs_m2
        self._target_height_m = target_height_m
        self._max_range_km = max_range_km
        self._resolution_m = resolution_m

    def run(self):
        try:
            result = self._calculator.calculate_coverage(
                radar_lat=self._radar_lat,
                radar_lon=self._radar_lon,
                radar_height_m=self._radar_height_m,
                target_rcs_m2=self._target_rcs_m2,
                target_height_m=self._target_height_m,
                max_range_km=self._max_range_km,
                grid_resolution_m=self._resolution_m,
                progress_callback=self._on_progress
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(None)

    def _on_progress(self, fraction):
        self.progress.emit(fraction)


class SiteSurveyTab(QWidget):
    """
    Site survey interface with terrain visualization and coverage analysis.

    Provides:
    - Radar profile selection
    - Location and terrain data loading
    - Propagation model configuration
    - 2D and 3D coverage visualization
    - GeoTIFF export
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._splitter_initialized = False
        self._terrain = None
        self._coverage_result = None
        self._calc_thread = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(0)

        # Main splitter (horizontal)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # Left panel - Controls (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(280)

        control_widget = self._create_control_panel()
        left_scroll.setWidget(control_widget)
        self.splitter.addWidget(left_scroll)

        # Center - Visualization (tabbed: 2D heatmap, 3D surface)
        self.viz_tabs = QTabWidget()
        self.coverage_2d = CoverageDisplay2D()
        self.coverage_3d = CoverageDisplay3D()
        self.viz_tabs.addTab(self.coverage_2d, "2D Coverage")
        self.viz_tabs.addTab(self.coverage_3d, "3D Surface")
        self.viz_tabs.setMinimumWidth(400)
        self.splitter.addWidget(self.viz_tabs)

        # Right panel - Results summary (scrollable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(250)

        results_widget = self._create_results_panel()
        right_scroll.setWidget(results_widget)
        self.splitter.addWidget(right_scroll)

        # Stretch factors: left fixed, center stretches, right fixed
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        layout.addWidget(self.splitter)

    def _create_control_panel(self) -> QWidget:
        """Create left control panel with all input groups."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Radar selector (reuse existing widget)
        self.radar_selector = RadarSelector()
        layout.addWidget(self.radar_selector)

        # Location panel
        self.location_panel = LocationPanel()
        layout.addWidget(self.location_panel)

        # Propagation panel
        self.propagation_panel = PropagationPanel()
        layout.addWidget(self.propagation_panel)

        # Calculate button
        self.calc_button = QPushButton("Calculate Coverage")
        self.calc_button.setObjectName("primaryButton")
        self.calc_button.setMinimumHeight(40)
        self.calc_button.setEnabled(False)
        layout.addWidget(self.calc_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Export group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_snr_btn = QPushButton("Export SNR Map...")
        self.export_snr_btn.setObjectName("secondaryButton")
        self.export_snr_btn.setEnabled(False)
        export_layout.addWidget(self.export_snr_btn)

        self.export_det_btn = QPushButton("Export Detection Map...")
        self.export_det_btn.setObjectName("secondaryButton")
        self.export_det_btn.setEnabled(False)
        export_layout.addWidget(self.export_det_btn)

        self.export_elev_btn = QPushButton("Export Elevation...")
        self.export_elev_btn.setObjectName("secondaryButton")
        self.export_elev_btn.setEnabled(False)
        export_layout.addWidget(self.export_elev_btn)

        layout.addWidget(export_group)

        layout.addStretch()
        return panel

    def _create_results_panel(self) -> QWidget:
        """Create right results summary panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Coverage results group
        results_group = QGroupBox("Coverage Results")
        results_layout = QVBoxLayout(results_group)

        # Status label
        self.status_label = QLabel("No coverage calculated")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888;")
        results_layout.addWidget(self.status_label)

        # Results display (will be populated after calculation)
        self.results_labels = {}

        result_items = [
            ("coverage_area", "Coverage Area:"),
            ("coverage_pct", "Coverage %:"),
            ("max_range", "Max Detection Range:"),
            ("radar_elev", "Radar Ground Elev:"),
            ("grid_size", "Grid Size:"),
            ("prop_model", "Propagation Model:"),
        ]

        for key, label_text in result_items:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setStyleSheet("color: #8b949e;")
            value = QLabel("---")
            value.setStyleSheet("color: #00ff88; font-weight: bold;")
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            results_layout.addLayout(row)
            self.results_labels[key] = value

        layout.addWidget(results_group)

        # Radar parameters group
        radar_group = QGroupBox("Radar Parameters")
        radar_layout = QVBoxLayout(radar_group)

        self.radar_labels = {}
        radar_items = [
            ("frequency", "Frequency:"),
            ("power", "Peak Power:"),
            ("gain", "Antenna Gain:"),
            ("bandwidth", "Bandwidth:"),
        ]

        for key, label_text in radar_items:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setStyleSheet("color: #8b949e;")
            value = QLabel("---")
            value.setStyleSheet("color: #58a6ff; font-weight: bold;")
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            radar_layout.addLayout(row)
            self.radar_labels[key] = value

        layout.addWidget(radar_group)

        # Target parameters group
        target_group = QGroupBox("Target Parameters")
        target_layout = QVBoxLayout(target_group)

        self.target_labels = {}
        target_items = [
            ("rcs", "Target RCS:"),
            ("height", "Target Height:"),
        ]

        for key, label_text in target_items:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setStyleSheet("color: #8b949e;")
            value = QLabel("---")
            value.setStyleSheet("color: #f0a000; font-weight: bold;")
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            target_layout.addLayout(row)
            self.target_labels[key] = value

        layout.addWidget(target_group)

        layout.addStretch()
        return panel

    def showEvent(self, event):
        """Handle widget show event to initialize splitter sizes."""
        super().showEvent(event)
        if not self._splitter_initialized:
            QTimer.singleShot(0, self._initialize_splitter_sizes)

    def _initialize_splitter_sizes(self):
        """Set initial splitter sizes based on widget width."""
        if self._splitter_initialized:
            return
        self._splitter_initialized = True

        total_width = self.width()
        if total_width < 100:
            total_width = 1200

        left_width = 300
        right_width = 280
        center_width = max(400, total_width - left_width - right_width - 20)

        self.splitter.setSizes([left_width, center_width, right_width])

    def _connect_signals(self):
        """Connect widget signals to slots."""
        # Terrain loaded enables calculation
        self.location_panel.terrain_loaded.connect(self._on_terrain_loaded)
        self.location_panel.terrain_load_error.connect(self._on_terrain_error)

        # Radar profile changed
        self.radar_selector.profile_changed.connect(self._on_profile_changed)

        # Calculate button
        self.calc_button.clicked.connect(self._on_calculate_clicked)

        # Export buttons
        self.export_snr_btn.clicked.connect(lambda: self._export_geotiff('snr'))
        self.export_det_btn.clicked.connect(lambda: self._export_geotiff('detected'))
        self.export_elev_btn.clicked.connect(lambda: self._export_geotiff('elevation'))

    def _on_terrain_loaded(self, terrain):
        """Handle terrain data loaded."""
        self._terrain = terrain
        self.calc_button.setEnabled(True)
        self.status_label.setText("Terrain loaded. Ready to calculate.")
        self.status_label.setStyleSheet("color: #00ff88;")

    def _on_terrain_error(self, error_msg):
        """Handle terrain loading error."""
        self.calc_button.setEnabled(False)
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #f85149;")

    def _on_profile_changed(self, profile):
        """Handle radar profile selection change."""
        if profile is None:
            return

        # Update radar parameter labels
        self.radar_labels["frequency"].setText(f"{profile.frequency_hz / 1e9:.2f} GHz")
        self.radar_labels["power"].setText(f"{profile.peak_power_w / 1000:.1f} kW")
        self.radar_labels["gain"].setText(f"{profile.tx_gain_dbi:.1f} dBi")
        self.radar_labels["bandwidth"].setText(f"{profile.bandwidth_hz / 1e6:.2f} MHz")

    def _on_calculate_clicked(self):
        """Handle calculate coverage button click."""
        if self._terrain is None:
            QMessageBox.warning(
                self, "No Terrain",
                "Please load terrain data first."
            )
            return

        profile = self.radar_selector.get_current_profile()
        if profile is None:
            QMessageBox.warning(
                self, "No Radar Profile",
                "Please select a radar profile."
            )
            return

        # Get parameters
        radar_lat = self.location_panel.get_latitude()
        radar_lon = self.location_panel.get_longitude()
        radar_height_m = self.location_panel.get_height_m()
        max_range_km = self.location_panel.get_range_km()

        target_rcs_m2 = self.propagation_panel.get_target_rcs_m2()
        target_height_m = self.propagation_panel.get_target_height_m()
        resolution_m = self.propagation_panel.get_resolution_m()
        prop_model = self.propagation_panel.get_propagation_model()

        # Update target labels
        self.target_labels["rcs"].setText(f"{target_rcs_m2:.2f} m²")
        self.target_labels["height"].setText(f"{target_height_m:.0f} m AGL")

        # Create calculator
        from core.coverage import CoverageCalculator
        from core.radar_equations import RadarParameters

        radar_params = RadarParameters(
            frequency_hz=profile.frequency_hz,
            peak_power_w=profile.peak_power_w,
            tx_gain_dbi=profile.tx_gain_dbi,
            rx_gain_dbi=profile.rx_gain_dbi,
            bandwidth_hz=profile.bandwidth_hz,
            noise_figure_db=profile.noise_figure_db,
            system_loss_db=profile.system_loss_db,
            required_snr_db=getattr(profile, 'required_snr_db', 13.0)
        )

        calculator = CoverageCalculator(
            radar_params=radar_params,
            terrain=self._terrain,
            prop_model=prop_model
        )

        # Disable UI during calculation
        self.calc_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Calculating coverage...")
        self.status_label.setStyleSheet("color: #58a6ff;")

        # Start calculation thread
        self._calc_thread = CoverageCalculationThread(
            calculator, radar_lat, radar_lon, radar_height_m,
            target_rcs_m2, target_height_m, max_range_km, resolution_m
        )
        self._calc_thread.progress.connect(self._on_calc_progress)
        self._calc_thread.finished.connect(self._on_calc_finished)
        self._calc_thread.error.connect(self._on_calc_error)
        self._calc_thread.start()

    @pyqtSlot(float)
    def _on_calc_progress(self, fraction):
        """Update progress bar."""
        self.progress_bar.setValue(int(fraction * 100))

    @pyqtSlot(object)
    def _on_calc_finished(self, result):
        """Handle calculation completion."""
        self.calc_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if result is None:
            return

        self._coverage_result = result

        # Update displays
        self.coverage_2d.update_coverage(result)
        self.coverage_3d.update_coverage(result)

        # Update results labels
        self.results_labels["coverage_area"].setText(f"{result.coverage_area_km2:.1f} km²")
        self.results_labels["coverage_pct"].setText(f"{result.coverage_percentage:.1f}%")
        self.results_labels["max_range"].setText(f"{result.max_detection_range_m / 1000:.1f} km")
        self.results_labels["radar_elev"].setText(f"{result.radar_elevation_m:.0f} m")
        self.results_labels["grid_size"].setText(f"{result.shape[0]}×{result.shape[1]}")
        self.results_labels["prop_model"].setText(result.propagation_model.name.replace('_', ' '))

        self.status_label.setText("Coverage calculation complete.")
        self.status_label.setStyleSheet("color: #00ff88;")

        # Enable export buttons
        self.export_snr_btn.setEnabled(True)
        self.export_det_btn.setEnabled(True)
        self.export_elev_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_calc_error(self, error_msg):
        """Handle calculation error."""
        self.calc_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #f85149;")

        QMessageBox.critical(
            self, "Calculation Error",
            f"Coverage calculation failed:\n{error_msg}"
        )

    def _export_geotiff(self, layer: str):
        """Export coverage layer to GeoTIFF."""
        if self._coverage_result is None:
            return

        # Determine default filename
        layer_names = {
            'snr': 'snr_coverage',
            'detected': 'detection_coverage',
            'elevation': 'elevation'
        }
        default_name = layer_names.get(layer, 'coverage')

        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export {layer.title()} GeoTIFF",
            f"{default_name}.tif",
            "GeoTIFF Files (*.tif *.tiff)"
        )

        if not filepath:
            return

        try:
            from core.coverage import export_coverage_geotiff
            export_coverage_geotiff(self._coverage_result, filepath, layer)

            QMessageBox.information(
                self, "Export Complete",
                f"Successfully exported {layer} map to:\n{filepath}"
            )
        except ImportError as e:
            QMessageBox.critical(
                self, "Export Error",
                f"GeoTIFF export requires rasterio package:\n{e}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export GeoTIFF:\n{e}"
            )

    def initial_setup(self):
        """Called after window is shown - perform initial setup."""
        # Load default profile if available
        profile = self.radar_selector.get_current_profile()
        if profile:
            self._on_profile_changed(profile)
