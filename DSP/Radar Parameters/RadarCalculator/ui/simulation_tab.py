"""
Simulation tab combining PPI display with track controls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QScrollArea, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer

from ui.plots.ppi_display import PPIDisplay
from ui.widgets.simulation_panel import SimulationControlPanel, TrackTablePanel


class SimulationTab(QWidget):
    """
    Complete simulation interface with PPI display and controls.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._splitter_initialized = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Main splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # Left panel - Controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(260)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        self.control_panel = SimulationControlPanel()
        left_layout.addWidget(self.control_panel)

        left_scroll.setWidget(left_widget)
        self.splitter.addWidget(left_scroll)

        # Center - PPI Display
        self.ppi_display = PPIDisplay()
        self.ppi_display.setMinimumWidth(400)
        self.splitter.addWidget(self.ppi_display)

        # Right panel - Track table
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(280)

        self.track_table = TrackTablePanel()
        right_scroll.setWidget(self.track_table)
        self.splitter.addWidget(right_scroll)

        # Set splitter proportions - center stretches, sides stay proportional
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        layout.addWidget(self.splitter)

    def showEvent(self, event):
        """Handle widget show event to initialize splitter sizes."""
        super().showEvent(event)
        if not self._splitter_initialized:
            # Defer splitter sizing to after the event loop processes the show
            QTimer.singleShot(0, self._initialize_splitter_sizes)

    def _initialize_splitter_sizes(self):
        """Set initial splitter sizes based on widget width."""
        if self._splitter_initialized:
            return
        self._splitter_initialized = True

        # Calculate sizes based on available width
        total_width = self.width()
        if total_width < 100:
            # Widget not properly sized yet, use defaults
            total_width = 1200

        # Left panel: fixed ~280px, Right panel: fixed ~320px, Center: remainder
        left_width = 280
        right_width = 320
        center_width = max(400, total_width - left_width - right_width - 20)

        self.splitter.setSizes([left_width, center_width, right_width])

    def _connect_signals(self):
        # Control panel signals
        self.control_panel.simulation_updated.connect(self._on_simulation_updated)
        self.control_panel.tracks_changed.connect(self._on_tracks_changed)

        # Display option signals
        self.control_panel.show_labels_check.stateChanged.connect(
            lambda: self.ppi_display.set_show_labels(
                self.control_panel.get_show_labels()
            )
        )
        self.control_panel.show_history_check.stateChanged.connect(
            lambda: self.ppi_display.set_show_history(
                self.control_panel.get_show_history()
            )
        )
        self.control_panel.max_range_spin.valueChanged.connect(
            lambda v: self.ppi_display.set_max_range(v)
        )

    def _on_simulation_updated(self):
        """Handle simulation update tick."""
        tracks_data = self.control_panel.get_track_data()
        self.ppi_display.update_tracks(tracks_data)
        self.track_table.update_tracks(tracks_data)

    def _on_tracks_changed(self):
        """Handle track addition/removal."""
        tracks_data = self.control_panel.get_track_data()
        self.ppi_display.update_tracks(tracks_data)
        self.track_table.update_tracks(tracks_data)
