"""
2D Coverage visualization display for Site Survey tab.
Provides heatmap overlay on elevation contour using pyqtgraph.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QGroupBox, QCheckBox
)
from PyQt6.QtCore import pyqtSignal, Qt

from ui.styles import PLOT_COLORS


class CoverageDisplay2D(QWidget):
    """
    2D coverage heatmap display with elevation contour.

    Shows:
    - Elevation contours as background
    - Coverage/SNR/detection heatmap overlay
    - Radar site marker
    - Range rings
    - Coordinate display on hover
    """

    coordinate_hover = pyqtSignal(float, float, float, float)  # lat, lon, elevation, snr

    def __init__(self, theme: str = 'dark', parent=None):
        super().__init__(parent)
        self._theme = theme
        self._colors = PLOT_COLORS[theme]
        self._coverage_result = None
        self._terrain_data = None
        self._radar_lat = None
        self._radar_lon = None
        self._current_colormap = 'viridis'
        self._snr_min = -20.0
        self._snr_max = 50.0
        self._auto_range = True
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Control bar - first row
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(4, 4, 4, 4)

        ctrl_layout.addWidget(QLabel("Display:"))
        self.layer_combo = QComboBox()
        self.layer_combo.addItems([
            "SNR (dB)",
            "Detection",
            "Elevation (m)",
            "Path Loss (dB)",
            "LOS Visibility"
        ])
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        ctrl_layout.addWidget(self.layer_combo)

        ctrl_layout.addSpacing(15)

        # Colormap selection
        ctrl_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "Viridis",
            "Plasma",
            "Radar Green",
            "Inferno",
            "Turbo",
            "Grayscale"
        ])
        self.colormap_combo.setToolTip("Select color scheme for the coverage display")
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        ctrl_layout.addWidget(self.colormap_combo)

        ctrl_layout.addStretch()

        # Coordinate display
        self.coord_label = QLabel("Lat: ---, Lon: ---")
        self.coord_label.setStyleSheet(f"color: {self._colors['foreground']}; font-family: monospace;")
        ctrl_layout.addWidget(self.coord_label)

        layout.addLayout(ctrl_layout)

        # Control bar - second row (SNR range controls)
        range_layout = QHBoxLayout()
        range_layout.setContentsMargins(4, 0, 4, 4)

        # Auto range checkbox
        self.auto_range_check = QCheckBox("Auto Range")
        self.auto_range_check.setChecked(True)
        self.auto_range_check.setToolTip("Automatically determine color range from data")
        self.auto_range_check.stateChanged.connect(self._on_auto_range_changed)
        range_layout.addWidget(self.auto_range_check)

        range_layout.addSpacing(10)

        # SNR Min
        range_layout.addWidget(QLabel("SNR Min:"))
        self.snr_min_spin = QDoubleSpinBox()
        self.snr_min_spin.setRange(-100, 100)
        self.snr_min_spin.setValue(-20)
        self.snr_min_spin.setSuffix(" dB")
        self.snr_min_spin.setToolTip("Minimum SNR for color mapping (values below shown as min color)")
        self.snr_min_spin.setEnabled(False)  # Disabled when auto range is on
        self.snr_min_spin.valueChanged.connect(self._on_snr_range_changed)
        range_layout.addWidget(self.snr_min_spin)

        range_layout.addSpacing(10)

        # SNR Max
        range_layout.addWidget(QLabel("SNR Max:"))
        self.snr_max_spin = QDoubleSpinBox()
        self.snr_max_spin.setRange(-100, 100)
        self.snr_max_spin.setValue(50)
        self.snr_max_spin.setSuffix(" dB")
        self.snr_max_spin.setToolTip("Maximum SNR for color mapping (values above shown as max color)")
        self.snr_max_spin.setEnabled(False)  # Disabled when auto range is on
        self.snr_max_spin.valueChanged.connect(self._on_snr_range_changed)
        range_layout.addWidget(self.snr_max_spin)

        range_layout.addStretch()

        layout.addLayout(range_layout)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self._colors['background'])
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel('left', 'Latitude', units='°')
        self.plot_widget.setLabel('bottom', 'Longitude', units='°')

        # Configure axes
        self.plot_widget.getAxis('left').setTextPen(self._colors['foreground'])
        self.plot_widget.getAxis('bottom').setTextPen(self._colors['foreground'])
        self.plot_widget.getAxis('left').setPen(self._colors['grid'])
        self.plot_widget.getAxis('bottom').setPen(self._colors['grid'])

        # Enable grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Coverage heatmap (main visualization)
        self.coverage_image = pg.ImageItem()
        self.plot_widget.addItem(self.coverage_image)

        # Radar marker
        self.radar_marker = pg.ScatterPlotItem(
            pen=pg.mkPen(self._colors['danger'], width=2),
            brush=pg.mkBrush(self._colors['danger']),
            size=15,
            symbol='star'
        )
        self.plot_widget.addItem(self.radar_marker)

        # Range rings (will be added dynamically)
        self.range_rings = []

        # Connect mouse movement for coordinate display
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        layout.addWidget(self.plot_widget, 1)

        # Colorbar
        self._setup_colorbar()

    def _setup_colorbar(self):
        """Create colorbar for the heatmap."""
        # Create a gradient widget as colorbar
        self.colorbar_widget = pg.GradientWidget(orientation='right')
        self.colorbar_widget.setFixedWidth(30)
        # Default colormap
        self._set_colormap('viridis')

    def _set_colormap(self, name: str):
        """Set the colormap for the heatmap."""
        name_lower = name.lower().replace(' ', '_')
        self._current_colormap = name_lower

        if name_lower == 'viridis':
            colors = [
                (0.0, (68, 1, 84)),
                (0.25, (59, 82, 139)),
                (0.5, (33, 145, 140)),
                (0.75, (94, 201, 98)),
                (1.0, (253, 231, 37))
            ]
        elif name_lower == 'plasma':
            colors = [
                (0.0, (13, 8, 135)),
                (0.25, (126, 3, 168)),
                (0.5, (204, 71, 120)),
                (0.75, (248, 149, 64)),
                (1.0, (240, 249, 33))
            ]
        elif name_lower == 'radar_green':
            # Custom radar-style green colormap
            colors = [
                (0.0, (0, 20, 0)),
                (0.25, (0, 60, 20)),
                (0.5, (0, 120, 40)),
                (0.75, (0, 200, 80)),
                (1.0, (0, 255, 136))
            ]
        elif name_lower == 'inferno':
            # Hot colormap - black to yellow through red
            colors = [
                (0.0, (0, 0, 4)),
                (0.25, (87, 16, 110)),
                (0.5, (188, 55, 84)),
                (0.75, (249, 142, 9)),
                (1.0, (252, 255, 164))
            ]
        elif name_lower == 'turbo':
            # Rainbow-like colormap (good for distinguishing values)
            colors = [
                (0.0, (48, 18, 59)),
                (0.2, (68, 124, 223)),
                (0.4, (54, 201, 163)),
                (0.5, (130, 226, 91)),
                (0.6, (212, 225, 64)),
                (0.8, (250, 141, 42)),
                (1.0, (122, 4, 3))
            ]
        elif name_lower == 'grayscale':
            # Grayscale
            colors = [
                (0.0, (0, 0, 0)),
                (1.0, (255, 255, 255))
            ]
        elif name_lower == 'detection':
            # Binary detection colormap (red/green)
            colors = [
                (0.0, (218, 54, 51)),   # Not detected - red
                (0.5, (218, 54, 51)),
                (0.5001, (35, 134, 54)),  # Detected - green
                (1.0, (35, 134, 54))
            ]
        else:
            # Default to viridis
            colors = [
                (0.0, (68, 1, 84)),
                (0.25, (59, 82, 139)),
                (0.5, (33, 145, 140)),
                (0.75, (94, 201, 98)),
                (1.0, (253, 231, 37))
            ]

        # Create lookup table
        self._lut = self._create_lut(colors)

    def _create_lut(self, colors, n_colors=256):
        """Create a lookup table from color stops."""
        lut = np.zeros((n_colors, 4), dtype=np.ubyte)

        for i in range(n_colors):
            pos = i / (n_colors - 1)

            # Find surrounding color stops
            for j in range(len(colors) - 1):
                if colors[j][0] <= pos <= colors[j + 1][0]:
                    # Interpolate between stops
                    t = (pos - colors[j][0]) / (colors[j + 1][0] - colors[j][0])
                    c1 = colors[j][1]
                    c2 = colors[j + 1][1]
                    lut[i, 0] = int(c1[0] + t * (c2[0] - c1[0]))
                    lut[i, 1] = int(c1[1] + t * (c2[1] - c1[1]))
                    lut[i, 2] = int(c1[2] + t * (c2[2] - c1[2]))
                    lut[i, 3] = 200  # Semi-transparent
                    break

        return lut

    def _on_layer_changed(self, index):
        """Handle layer selection change."""
        self._update_display()

    def _on_colormap_changed(self, colormap_name: str):
        """Handle colormap selection change."""
        self._set_colormap(colormap_name)
        self._update_display()

    def _on_auto_range_changed(self, state):
        """Handle auto range checkbox change."""
        self._auto_range = (state == Qt.CheckState.Checked.value)
        self.snr_min_spin.setEnabled(not self._auto_range)
        self.snr_max_spin.setEnabled(not self._auto_range)
        self._update_display()

    def _on_snr_range_changed(self):
        """Handle manual SNR range change."""
        if not self._auto_range:
            self._snr_min = self.snr_min_spin.value()
            self._snr_max = self.snr_max_spin.value()
            self._update_display()

    def _on_mouse_moved(self, pos):
        """Handle mouse movement for coordinate display."""
        if self._coverage_result is None:
            return

        # Convert to plot coordinates
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        lon = mouse_point.x()
        lat = mouse_point.y()

        # Check if within bounds
        bounds = self._coverage_result.bounds
        if not (bounds.lat_min <= lat <= bounds.lat_max and
                bounds.lon_min <= lon <= bounds.lon_max):
            return

        # Find grid indices
        lat_idx = int((bounds.lat_max - lat) /
                      (bounds.lat_max - bounds.lat_min) *
                      (len(self._coverage_result.lat_array) - 1))
        lon_idx = int((lon - bounds.lon_min) /
                      (bounds.lon_max - bounds.lon_min) *
                      (len(self._coverage_result.lon_array) - 1))

        lat_idx = max(0, min(lat_idx, len(self._coverage_result.lat_array) - 1))
        lon_idx = max(0, min(lon_idx, len(self._coverage_result.lon_array) - 1))

        elev = self._coverage_result.elevation_m[lat_idx, lon_idx]
        snr = self._coverage_result.snr_db[lat_idx, lon_idx]

        snr_str = f"{snr:.1f}" if not np.isinf(snr) else "N/A"
        self.coord_label.setText(
            f"Lat: {lat:.4f}°  Lon: {lon:.4f}°  Elev: {elev:.0f}m  SNR: {snr_str} dB"
        )

        self.coordinate_hover.emit(lat, lon, elev, snr)

    def update_coverage(self, coverage_result, radar_lat: float = None, radar_lon: float = None):
        """
        Update display with new coverage results.

        Args:
            coverage_result: CoverageResult object
            radar_lat: Radar latitude (uses coverage_result if not provided)
            radar_lon: Radar longitude (uses coverage_result if not provided)
        """
        self._coverage_result = coverage_result
        self._radar_lat = radar_lat or coverage_result.radar_lat
        self._radar_lon = radar_lon or coverage_result.radar_lon

        self._update_display()

    def _update_display(self):
        """Update the visualization based on current settings."""
        if self._coverage_result is None:
            return

        result = self._coverage_result
        layer_text = self.layer_combo.currentText()

        # Get the currently selected colormap from dropdown
        selected_cmap = self.colormap_combo.currentText()
        use_selected_colormap = True  # Use user-selected colormap for continuous data

        # Select data layer and colormap
        if "SNR" in layer_text:
            data = result.snr_db.copy()
            data = np.where(np.isinf(data), np.nan, data)

            # Apply SNR range (either auto or manual)
            if self._auto_range:
                valid = data[~np.isnan(data)]
                if len(valid) > 0:
                    vmin, vmax = np.nanpercentile(valid, [2, 98])  # Use percentiles to avoid outliers
                else:
                    vmin, vmax = -20, 50
            else:
                vmin = self._snr_min
                vmax = self._snr_max

            # Update spinbox display values when auto-range
            if self._auto_range:
                self.snr_min_spin.blockSignals(True)
                self.snr_max_spin.blockSignals(True)
                self.snr_min_spin.setValue(vmin)
                self.snr_max_spin.setValue(vmax)
                self.snr_min_spin.blockSignals(False)
                self.snr_max_spin.blockSignals(False)

            data = np.clip(data, vmin, vmax)
            self._set_colormap(selected_cmap)

        elif "Detection" in layer_text:
            data = result.detected.astype(float)
            self._set_colormap('detection')
            use_selected_colormap = False
            vmin, vmax = 0, 1

        elif "Elevation" in layer_text:
            data = result.elevation_m.copy()
            self._set_colormap(selected_cmap)
            vmin, vmax = np.nanmin(data), np.nanmax(data)

        elif "Path Loss" in layer_text:
            data = result.path_loss_db.copy()
            data = np.clip(data, 0, 200)
            data = np.where(np.isinf(data), np.nan, data)
            self._set_colormap(selected_cmap)
            vmin, vmax = np.nanmin(data), np.nanmax(data)

        elif "LOS" in layer_text or "Visibility" in layer_text:
            data = result.is_visible.astype(float)
            self._set_colormap('detection')
            use_selected_colormap = False
            vmin, vmax = 0, 1

        else:
            data = result.snr_db.copy()
            self._set_colormap(selected_cmap)
            vmin, vmax = np.nanmin(data), np.nanmax(data)

        # Normalize data for colormap
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0 and vmax > vmin:
            norm_data = (data - vmin) / (vmax - vmin)
        else:
            norm_data = np.zeros_like(data)

        # Handle NaN values (make transparent)
        norm_data = np.nan_to_num(norm_data, nan=0)

        # Set image data
        # Note: ImageItem expects (width, height) order, so transpose
        # Also flip vertically because image origin is bottom-left
        display_data = np.flipud(norm_data)

        self.coverage_image.setImage(display_data.T)
        self.coverage_image.setLookupTable(self._lut)

        # Set image position and scale
        bounds = result.bounds
        self.coverage_image.setRect(
            bounds.lon_min,
            bounds.lat_min,
            bounds.lon_span,
            bounds.lat_span
        )

        # Update radar marker
        self.radar_marker.setData([self._radar_lon], [self._radar_lat])

        # Update range rings
        self._update_range_rings()

        # Auto-range to show full coverage
        self.plot_widget.setXRange(bounds.lon_min, bounds.lon_max, padding=0.02)
        self.plot_widget.setYRange(bounds.lat_min, bounds.lat_max, padding=0.02)

    def _update_range_rings(self):
        """Draw range rings around radar site."""
        # Remove existing rings
        for ring in self.range_rings:
            self.plot_widget.removeItem(ring)
        self.range_rings.clear()

        if self._coverage_result is None or self._radar_lat is None:
            return

        # Draw rings at 25%, 50%, 75%, 100% of max range
        max_range_km = self._coverage_result.max_range_km
        ring_distances = [max_range_km * f for f in [0.25, 0.5, 0.75, 1.0]]

        import math
        for dist_km in ring_distances:
            # Convert km to approximate degrees
            lat_deg = dist_km / 111.0
            lon_deg = dist_km / (111.0 * math.cos(math.radians(self._radar_lat)))

            # Create circle points
            angles = np.linspace(0, 2 * np.pi, 100)
            x = self._radar_lon + lon_deg * np.cos(angles)
            y = self._radar_lat + lat_deg * np.sin(angles)

            ring = pg.PlotDataItem(
                x, y,
                pen=pg.mkPen(self._colors['foreground'], width=1, style=Qt.PenStyle.DashLine)
            )
            self.plot_widget.addItem(ring)
            self.range_rings.append(ring)

    def set_theme(self, theme: str):
        """Update display theme."""
        self._theme = theme
        self._colors = PLOT_COLORS[theme]

        self.plot_widget.setBackground(self._colors['background'])
        self.plot_widget.getAxis('left').setTextPen(self._colors['foreground'])
        self.plot_widget.getAxis('bottom').setTextPen(self._colors['foreground'])

        self.radar_marker.setPen(pg.mkPen(self._colors['danger'], width=2))
        self.radar_marker.setBrush(pg.mkBrush(self._colors['danger']))

        self._update_display()

    def clear(self):
        """Clear the display."""
        self._coverage_result = None
        self._terrain_data = None
        self._radar_lat = None
        self._radar_lon = None

        self.coverage_image.clear()
        self.radar_marker.setData([], [])

        for ring in self.range_rings:
            self.plot_widget.removeItem(ring)
        self.range_rings.clear()

        self.coord_label.setText("Lat: ---, Lon: ---")
