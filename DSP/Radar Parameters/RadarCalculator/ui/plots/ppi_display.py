"""
Plan Position Indicator (PPI) radar display widget.

Displays targets on a polar coordinate radar scope with
range rings, azimuth lines, and track history.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient
import pyqtgraph as pg

from core.track_simulator import Track, TargetCategory


# Color mapping for target categories
CATEGORY_COLORS = {
    TargetCategory.COMMERCIAL_AIRCRAFT: '#00BFFF',     # Deep sky blue
    TargetCategory.GENERAL_AVIATION: '#87CEEB',        # Light sky blue
    TargetCategory.MILITARY_FIGHTER: '#FF6347',        # Tomato red
    TargetCategory.MILITARY_STEALTH: '#9400D3',        # Dark violet
    TargetCategory.MILITARY_HEAVY: '#FFA500',          # Orange
    TargetCategory.MILITARY_HELICOPTER: '#FFD700',     # Gold
    TargetCategory.CIVILIAN_HELICOPTER: '#98FB98',     # Pale green
    TargetCategory.UAV: '#FF69B4',                     # Hot pink
    TargetCategory.CARGO_SHIP: '#4682B4',              # Steel blue
    TargetCategory.TANKER: '#8B4513',                  # Saddle brown
    TargetCategory.CONTAINER_SHIP: '#2E8B57',          # Sea green
    TargetCategory.CRUISE_SHIP: '#FFFFFF',             # White
    TargetCategory.NAVAL_COMBATANT: '#DC143C',         # Crimson
    TargetCategory.NAVAL_CARRIER: '#FF0000',           # Red
    TargetCategory.PATROL_BOAT: '#00CED1',             # Dark turquoise
    TargetCategory.FISHING_VESSEL: '#808080',          # Gray
    TargetCategory.YACHT: '#E6E6FA',                   # Lavender
    TargetCategory.SMALL_BOAT: '#A9A9A9',              # Dark gray
}


class PPIDisplay(QWidget):
    """
    Plan Position Indicator display using pyqtgraph.
    """

    track_selected = pyqtSignal(int)  # Emits track_id when clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._max_range_nmi = 240
        self._range_rings = [60, 120, 180, 240]
        self._show_labels = True
        self._show_history = True
        self._tracks_data = []
        self._selected_track_id = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0a0a12')
        self.plot_widget.setAspectLocked(True)

        # Configure axes
        self.plot_widget.setLabel('left', 'North', units='nmi')
        self.plot_widget.setLabel('bottom', 'East', units='nmi')
        self.plot_widget.showGrid(x=False, y=False)

        # Set range
        self._update_range()

        layout.addWidget(self.plot_widget)

        # Create plot items
        self._create_static_elements()
        self._create_track_items()

    def _update_range(self):
        """Update plot range based on max range setting."""
        r = self._max_range_nmi * 1.05
        self.plot_widget.setXRange(-r, r, padding=0)
        self.plot_widget.setYRange(-r, r, padding=0)

    def _create_static_elements(self):
        """Create static display elements (range rings, cardinal lines)."""
        # Range rings
        self._range_ring_items = []
        theta = np.linspace(0, 2 * np.pi, 361)

        for range_nmi in self._range_rings:
            x = range_nmi * np.cos(theta)
            y = range_nmi * np.sin(theta)
            pen = pg.mkPen(color='#1a4a1a', width=1)
            item = self.plot_widget.plot(x, y, pen=pen)
            self._range_ring_items.append(item)

            # Range label
            text = pg.TextItem(f"{range_nmi} nmi", color='#2a6a2a', anchor=(0, 0.5))
            text.setPos(range_nmi + 2, 0)
            self.plot_widget.addItem(text)

        # Cardinal direction lines
        cardinal_pen = pg.mkPen(color='#1a3a1a', width=1, style=Qt.PenStyle.DashLine)

        # N-S line
        self.plot_widget.plot([0, 0], [-self._max_range_nmi, self._max_range_nmi],
                             pen=cardinal_pen)
        # E-W line
        self.plot_widget.plot([-self._max_range_nmi, self._max_range_nmi], [0, 0],
                             pen=cardinal_pen)

        # Azimuth lines every 30 degrees
        az_pen = pg.mkPen(color='#0a2a0a', width=1, style=Qt.PenStyle.DotLine)
        for az in range(0, 360, 30):
            if az % 90 != 0:  # Skip cardinals
                az_rad = math.radians(az)
                x_end = self._max_range_nmi * math.sin(az_rad)
                y_end = self._max_range_nmi * math.cos(az_rad)
                self.plot_widget.plot([0, x_end], [0, y_end], pen=az_pen)

        # Radar origin marker
        self._radar_marker = pg.ScatterPlotItem(
            [0], [0],
            pen=pg.mkPen(color='#00ff00', width=2),
            brush=pg.mkBrush(color='#003300'),
            size=12, symbol='o'
        )
        self.plot_widget.addItem(self._radar_marker)

        # Cardinal labels
        label_offset = self._max_range_nmi + 10
        for text, pos in [('N', (0, label_offset)), ('S', (0, -label_offset)),
                          ('E', (label_offset, 0)), ('W', (-label_offset, 0))]:
            label = pg.TextItem(text, color='#4a8a4a', anchor=(0.5, 0.5))
            label.setPos(pos[0], pos[1])
            label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
            self.plot_widget.addItem(label)

    def _create_track_items(self):
        """Create scatter plot items for tracks."""
        # Track position markers
        self._track_scatter = pg.ScatterPlotItem(size=10, pxMode=True)
        self.plot_widget.addItem(self._track_scatter)

        # Track history lines - will be dynamically managed
        self._history_plots = {}

        # Track labels
        self._track_labels = {}

        # Velocity vectors
        self._velocity_lines = {}

    def set_max_range(self, range_nmi: float):
        """Set the maximum display range."""
        self._max_range_nmi = range_nmi
        self._update_range()
        # Update range rings
        self._range_rings = [range_nmi * 0.25, range_nmi * 0.5,
                            range_nmi * 0.75, range_nmi]
        self._create_static_elements()

    def set_show_labels(self, show: bool):
        """Toggle track labels."""
        self._show_labels = show
        self._update_display()

    def set_show_history(self, show: bool):
        """Toggle track history trails."""
        self._show_history = show
        self._update_display()

    def update_tracks(self, tracks_data: list):
        """
        Update display with new track data.

        tracks_data: List of dicts with track info from TrackSimulator.get_track_data()
        """
        self._tracks_data = tracks_data
        self._update_display()

    def _update_display(self):
        """Refresh the display with current track data."""
        if not self._tracks_data:
            self._track_scatter.setData([], [])
            return

        # Prepare scatter plot data
        positions = []
        colors = []
        symbols = []
        sizes = []

        for track in self._tracks_data:
            # Convert to display coordinates (x=East, y=North)
            x_nmi = track['x_m'] / 1852.0
            y_nmi = track['y_m'] / 1852.0
            positions.append((x_nmi, y_nmi))

            # Color by category
            category = None
            for cat in TargetCategory:
                if cat.value == track['category']:
                    category = cat
                    break

            color = CATEGORY_COLORS.get(category, '#FFFFFF')
            colors.append(color)

            # Symbol by type (aircraft vs vessel)
            is_vessel = category in [
                TargetCategory.CARGO_SHIP, TargetCategory.TANKER,
                TargetCategory.CONTAINER_SHIP, TargetCategory.CRUISE_SHIP,
                TargetCategory.NAVAL_COMBATANT, TargetCategory.NAVAL_CARRIER,
                TargetCategory.PATROL_BOAT, TargetCategory.FISHING_VESSEL,
                TargetCategory.YACHT, TargetCategory.SMALL_BOAT
            ]
            symbols.append('s' if is_vessel else 't1')  # Square for ships, triangle for aircraft

            # Size based on RCS (logarithmic scale)
            rcs_dbsm = track['rcs_dbsm']
            size = max(6, min(20, 10 + rcs_dbsm / 5))
            sizes.append(size)

        # Update scatter plot
        x_vals = [p[0] for p in positions]
        y_vals = [p[1] for p in positions]

        brushes = [pg.mkBrush(c) for c in colors]
        pens = [pg.mkPen(c, width=1) for c in colors]

        self._track_scatter.setData(
            x=x_vals, y=y_vals,
            brush=brushes, pen=pens,
            symbol=symbols, size=sizes
        )

        # Update history trails
        self._update_history_trails()

        # Update labels
        self._update_labels()

        # Update velocity vectors
        self._update_velocity_vectors()

    def _update_history_trails(self):
        """Update track history trail lines."""
        # Remove old history plots
        for plot_item in self._history_plots.values():
            self.plot_widget.removeItem(plot_item)
        self._history_plots.clear()

        if not self._show_history:
            return

        for track in self._tracks_data:
            track_id = track['id']
            history = track['history']

            if len(history) > 1:
                # Convert history to display coordinates
                x_hist = [p[0] / 1852.0 for p in history]
                y_hist = [p[1] / 1852.0 for p in history]

                # Get color
                category = None
                for cat in TargetCategory:
                    if cat.value == track['category']:
                        category = cat
                        break
                color = CATEGORY_COLORS.get(category, '#FFFFFF')

                # Create fading trail
                pen = pg.mkPen(color=color, width=1)
                plot_item = self.plot_widget.plot(x_hist, y_hist, pen=pen)
                self._history_plots[track_id] = plot_item

    def _update_labels(self):
        """Update track labels."""
        # Remove old labels
        for label in self._track_labels.values():
            self.plot_widget.removeItem(label)
        self._track_labels.clear()

        if not self._show_labels:
            return

        for track in self._tracks_data:
            track_id = track['id']
            x_nmi = track['x_m'] / 1852.0
            y_nmi = track['y_m'] / 1852.0

            # Create label
            label_text = f"{track['name']}\n{track['speed_kts']:.0f}kts"
            label = pg.TextItem(label_text, color='#aaaaaa', anchor=(0, 1))
            label.setPos(x_nmi + 2, y_nmi + 2)
            label.setFont(QFont('Arial', 8))
            self.plot_widget.addItem(label)
            self._track_labels[track_id] = label

    def _update_velocity_vectors(self):
        """Update velocity vector lines."""
        # Remove old vectors
        for line in self._velocity_lines.values():
            self.plot_widget.removeItem(line)
        self._velocity_lines.clear()

        for track in self._tracks_data:
            track_id = track['id']
            x_nmi = track['x_m'] / 1852.0
            y_nmi = track['y_m'] / 1852.0

            # Calculate velocity vector endpoint (30 seconds of travel)
            heading_rad = math.radians(track['heading_deg'])
            speed_nmi_per_sec = track['speed_kts'] / 3600.0
            vector_length = speed_nmi_per_sec * 60  # 1 minute of travel

            x_end = x_nmi + vector_length * math.sin(heading_rad)
            y_end = y_nmi + vector_length * math.cos(heading_rad)

            # Get color
            category = None
            for cat in TargetCategory:
                if cat.value == track['category']:
                    category = cat
                    break
            color = CATEGORY_COLORS.get(category, '#FFFFFF')

            pen = pg.mkPen(color=color, width=1, style=Qt.PenStyle.DashLine)
            line = self.plot_widget.plot([x_nmi, x_end], [y_nmi, y_end], pen=pen)
            self._velocity_lines[track_id] = line

    def clear_display(self):
        """Clear all track data from display."""
        self._tracks_data = []
        self._track_scatter.setData([], [])

        for plot_item in self._history_plots.values():
            self.plot_widget.removeItem(plot_item)
        self._history_plots.clear()

        for label in self._track_labels.values():
            self.plot_widget.removeItem(label)
        self._track_labels.clear()

        for line in self._velocity_lines.values():
            self.plot_widget.removeItem(line)
        self._velocity_lines.clear()
