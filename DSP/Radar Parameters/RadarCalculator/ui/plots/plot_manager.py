"""
Plot manager widget with plot type selection and display.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel,
    QPushButton, QStackedWidget
)
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg

from core.radar_equations import RadarCalculator, RadarParameters, TargetScenario
from ui.styles import PLOT_COLORS


# Configure pyqtgraph for dark theme
pg.setConfigOptions(antialias=True)

# Get color scheme
COLORS = PLOT_COLORS['dark']


class PlotManager(QWidget):
    """
    Manages multiple plot types with a selector to switch between them.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calculator = None
        self._current_rcs = 1.0
        self._current_range = 240 * 1852  # 240 nmi in meters
        self._range_unit = "nmi"
        self._processing_gain_db = 0.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Plot selector bar
        selector_layout = QHBoxLayout()

        selector_layout.addWidget(QLabel("Plot:"))
        self.plot_selector = QComboBox()
        self.plot_selector.addItems([
            "SNR vs Range",
            "Received Power vs Range",
            "Min Detectable RCS vs Range",
            "Max Range vs RCS",
            "Link Budget Waterfall"
        ])
        self.plot_selector.setMinimumWidth(200)
        self.plot_selector.currentIndexChanged.connect(self._on_plot_changed)
        selector_layout.addWidget(self.plot_selector)

        selector_layout.addStretch()

        self.export_btn = QPushButton("Export Plot")
        self.export_btn.setObjectName("secondaryButton")
        self.export_btn.clicked.connect(self._export_plot)
        selector_layout.addWidget(self.export_btn)

        layout.addLayout(selector_layout)

        # Stacked widget for plots
        self.plot_stack = QStackedWidget()
        layout.addWidget(self.plot_stack)

        # Create plot widgets
        self._create_plots()

    def _create_plots(self):
        """Create all plot widgets."""

        # Common plot styling
        label_style = {'color': COLORS['foreground'], 'font-size': '10pt'}
        title_style = {'color': COLORS['primary'], 'size': '11pt', 'bold': True}

        # SNR vs Range plot
        self.snr_plot = pg.PlotWidget()
        self.snr_plot.setBackground(COLORS['background'])
        self.snr_plot.setLabel('left', 'SNR', units='dB', **label_style)
        self.snr_plot.setLabel('bottom', 'Range', units='nmi', **label_style)
        self.snr_plot.setTitle('SNR vs Range', **title_style)
        self.snr_plot.showGrid(x=True, y=True, alpha=0.3)
        self.snr_plot.getAxis('bottom').setPen(pg.mkPen(color=COLORS['grid']))
        self.snr_plot.getAxis('left').setPen(pg.mkPen(color=COLORS['grid']))
        self.snr_plot.addLegend(labelTextColor=COLORS['foreground'])
        self.plot_stack.addWidget(self.snr_plot)

        # Received Power vs Range plot
        self.pr_plot = pg.PlotWidget()
        self.pr_plot.setBackground(COLORS['background'])
        self.pr_plot.setLabel('left', 'Received Power', units='dBm', **label_style)
        self.pr_plot.setLabel('bottom', 'Range', units='nmi', **label_style)
        self.pr_plot.setTitle('Received Power vs Range', **title_style)
        self.pr_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pr_plot.getAxis('bottom').setPen(pg.mkPen(color=COLORS['grid']))
        self.pr_plot.getAxis('left').setPen(pg.mkPen(color=COLORS['grid']))
        self.pr_plot.addLegend(labelTextColor=COLORS['foreground'])
        self.plot_stack.addWidget(self.pr_plot)

        # Min RCS vs Range plot
        self.rcs_plot = pg.PlotWidget()
        self.rcs_plot.setBackground(COLORS['background'])
        self.rcs_plot.setLabel('left', 'Min Detectable RCS', units='dBsm', **label_style)
        self.rcs_plot.setLabel('bottom', 'Range', units='nmi', **label_style)
        self.rcs_plot.setTitle('Minimum Detectable RCS vs Range', **title_style)
        self.rcs_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rcs_plot.getAxis('bottom').setPen(pg.mkPen(color=COLORS['grid']))
        self.rcs_plot.getAxis('left').setPen(pg.mkPen(color=COLORS['grid']))
        self.plot_stack.addWidget(self.rcs_plot)

        # Max Range vs RCS plot
        self.range_plot = pg.PlotWidget()
        self.range_plot.setBackground(COLORS['background'])
        self.range_plot.setLabel('left', 'Max Detection Range', units='nmi', **label_style)
        self.range_plot.setLabel('bottom', 'Target RCS', units='dBsm', **label_style)
        self.range_plot.setTitle('Maximum Detection Range vs RCS', **title_style)
        self.range_plot.showGrid(x=True, y=True, alpha=0.3)
        self.range_plot.getAxis('bottom').setPen(pg.mkPen(color=COLORS['grid']))
        self.range_plot.getAxis('left').setPen(pg.mkPen(color=COLORS['grid']))
        self.plot_stack.addWidget(self.range_plot)

        # Link Budget Waterfall (bar chart)
        self.budget_plot = pg.PlotWidget()
        self.budget_plot.setBackground(COLORS['background'])
        self.budget_plot.setTitle('Link Budget Waterfall', **title_style)
        self.budget_plot.showGrid(x=False, y=True, alpha=0.3)
        self.budget_plot.getAxis('bottom').setPen(pg.mkPen(color=COLORS['grid']))
        self.budget_plot.getAxis('left').setPen(pg.mkPen(color=COLORS['grid']))
        self.plot_stack.addWidget(self.budget_plot)

    def _on_plot_changed(self, index):
        """Handle plot type selection change."""
        self.plot_stack.setCurrentIndex(index)
        self._update_current_plot()

    def set_calculator(self, calculator: RadarCalculator):
        """Set the radar calculator for generating plot data."""
        self._calculator = calculator
        self._update_all_plots()

    def set_target(self, rcs_m2: float, range_m: float):
        """Set the current target parameters."""
        self._current_rcs = rcs_m2
        self._current_range = range_m
        self._update_all_plots()

    def set_range_unit(self, unit: str):
        """Set the range unit for display."""
        self._range_unit = unit
        self._update_all_plots()

    def set_processing_gain(self, gain_db: float):
        """Set the processing gain from pulse compression."""
        self._processing_gain_db = gain_db
        self._update_all_plots()

    def _update_all_plots(self):
        """Update all plot data."""
        if self._calculator is None:
            return

        self._update_snr_plot()
        self._update_pr_plot()
        self._update_rcs_plot()
        self._update_range_plot()
        self._update_budget_plot()

    def _update_current_plot(self):
        """Update only the currently visible plot."""
        if self._calculator is None:
            return

        index = self.plot_stack.currentIndex()
        if index == 0:
            self._update_snr_plot()
        elif index == 1:
            self._update_pr_plot()
        elif index == 2:
            self._update_rcs_plot()
        elif index == 3:
            self._update_range_plot()
        elif index == 4:
            self._update_budget_plot()

    def _range_to_display(self, range_m: float) -> float:
        """Convert range from meters to display unit."""
        if self._range_unit == "nmi":
            return range_m / 1852.0
        elif self._range_unit == "km":
            return range_m / 1000.0
        elif self._range_unit == "mi":
            return range_m / 1609.344
        return range_m

    def _update_snr_plot(self):
        """Update the SNR vs Range plot."""
        self.snr_plot.clear()

        # Calculate range from 1 to max range * 1.5
        max_range = self._calculator.calculate_max_range_m(self._current_rcs)
        range_max = min(max_range * 1.5, 1000 * 1852)  # Cap at 1000 nmi
        range_min = 1 * 1852  # 1 nmi

        ranges, snr_values = self._calculator.snr_vs_range(
            self._current_rcs, range_min, range_max, 500
        )

        # Convert ranges to display units
        ranges_display = np.array([self._range_to_display(r) for r in ranges])

        # Plot raw SNR curve
        pen = pg.mkPen(color=COLORS['grid'], width=2)
        self.snr_plot.plot(
            ranges_display, snr_values,
            pen=pen, name='SNR (Raw)'
        )

        # Plot effective SNR with processing gain
        if self._processing_gain_db > 0:
            effective_snr = snr_values + self._processing_gain_db
            eff_pen = pg.mkPen(color=COLORS['primary'], width=2)
            self.snr_plot.plot(
                ranges_display, effective_snr,
                pen=eff_pen, name=f'SNR (w/ +{self._processing_gain_db:.1f} dB compression)'
            )
        else:
            # If no processing gain, make raw SNR the main curve
            self.snr_plot.clear()
            pen = pg.mkPen(color=COLORS['primary'], width=2)
            self.snr_plot.plot(
                ranges_display, snr_values,
                pen=pen, name='SNR'
            )

        # Plot required SNR threshold
        req_snr = self._calculator.params.required_snr_db
        thresh_pen = pg.mkPen(color=COLORS['danger'], width=2, style=pg.QtCore.Qt.PenStyle.DashLine)
        self.snr_plot.plot(
            [ranges_display[0], ranges_display[-1]],
            [req_snr, req_snr],
            pen=thresh_pen, name=f'Required SNR ({req_snr} dB)'
        )

        # Plot MDS line (0 dB SNR)
        mds_pen = pg.mkPen(color=COLORS['foreground'], width=1, style=pg.QtCore.Qt.PenStyle.DotLine)
        self.snr_plot.plot(
            [ranges_display[0], ranges_display[-1]],
            [0, 0],
            pen=mds_pen, name='MDS (0 dB)'
        )

        # Mark current range - show effective SNR point if processing gain is applied
        current_range_display = self._range_to_display(self._current_range)
        current_snr = self._calculator.calculate_snr_db(self._current_rcs, self._current_range)
        effective_current_snr = current_snr + self._processing_gain_db

        scatter = pg.ScatterPlotItem(
            [current_range_display], [effective_current_snr],
            pen=pg.mkPen(color=COLORS['tertiary'], width=2),
            brush=pg.mkBrush(color=COLORS['tertiary']),
            size=12, symbol='o'
        )
        self.snr_plot.addItem(scatter)

        # Update axis labels
        self.snr_plot.setLabel('bottom', 'Range', units=self._range_unit)

    def _update_pr_plot(self):
        """Update the Received Power vs Range plot."""
        self.pr_plot.clear()

        max_range = self._calculator.calculate_max_range_m(self._current_rcs)
        range_max = min(max_range * 1.5, 1000 * 1852)
        range_min = 1 * 1852

        ranges, pr_values = self._calculator.received_power_vs_range(
            self._current_rcs, range_min, range_max, 500
        )

        ranges_display = np.array([self._range_to_display(r) for r in ranges])

        # Plot Pr curve
        pen = pg.mkPen(color=COLORS['primary'], width=2)
        self.pr_plot.plot(ranges_display, pr_values, pen=pen, name='Received Power')

        # Plot MDS threshold
        mds = self._calculator.calculate_mds_dbm()
        mds_pen = pg.mkPen(color=COLORS['danger'], width=2, style=pg.QtCore.Qt.PenStyle.DashLine)
        self.pr_plot.plot(
            [ranges_display[0], ranges_display[-1]],
            [mds, mds],
            pen=mds_pen, name=f'MDS ({mds:.1f} dBm)'
        )

        # Plot sensitivity threshold
        sens = self._calculator.calculate_sensitivity_dbm()
        sens_pen = pg.mkPen(color=COLORS['secondary'], width=2, style=pg.QtCore.Qt.PenStyle.DashLine)
        self.pr_plot.plot(
            [ranges_display[0], ranges_display[-1]],
            [sens, sens],
            pen=sens_pen, name=f'Sensitivity ({sens:.1f} dBm)'
        )

        # Mark current point
        current_range_display = self._range_to_display(self._current_range)
        current_pr = self._calculator.calculate_received_power_dbm(
            self._current_rcs, self._current_range
        )

        scatter = pg.ScatterPlotItem(
            [current_range_display], [current_pr],
            pen=pg.mkPen(color=COLORS['tertiary'], width=2),
            brush=pg.mkBrush(color=COLORS['tertiary']),
            size=12, symbol='o'
        )
        self.pr_plot.addItem(scatter)

        self.pr_plot.setLabel('bottom', 'Range', units=self._range_unit)

    def _update_rcs_plot(self):
        """Update the Min Detectable RCS vs Range plot."""
        self.rcs_plot.clear()

        range_min = 1 * 1852  # 1 nmi
        range_max = 500 * 1852  # 500 nmi

        ranges, rcs_values = self._calculator.min_rcs_vs_range(
            range_min, range_max, 500
        )

        ranges_display = np.array([self._range_to_display(r) for r in ranges])
        rcs_dbsm = 10 * np.log10(rcs_values)

        # Plot min RCS curve
        pen = pg.mkPen(color=COLORS['primary'], width=2)
        self.rcs_plot.plot(ranges_display, rcs_dbsm, pen=pen, name='Min Detectable RCS')

        # Mark current target RCS
        current_rcs_dbsm = 10 * np.log10(self._current_rcs)
        rcs_pen = pg.mkPen(color=COLORS['secondary'], width=2, style=pg.QtCore.Qt.PenStyle.DashLine)
        self.rcs_plot.plot(
            [ranges_display[0], ranges_display[-1]],
            [current_rcs_dbsm, current_rcs_dbsm],
            pen=rcs_pen, name=f'Target RCS ({current_rcs_dbsm:.1f} dBsm)'
        )

        # Mark current point
        current_range_display = self._range_to_display(self._current_range)
        current_min_rcs = self._calculator.calculate_min_rcs_m2(self._current_range)
        current_min_rcs_dbsm = 10 * np.log10(current_min_rcs)

        scatter = pg.ScatterPlotItem(
            [current_range_display], [current_min_rcs_dbsm],
            pen=pg.mkPen(color=COLORS['tertiary'], width=2),
            brush=pg.mkBrush(color=COLORS['tertiary']),
            size=12, symbol='o'
        )
        self.rcs_plot.addItem(scatter)

        self.rcs_plot.setLabel('bottom', 'Range', units=self._range_unit)

    def _update_range_plot(self):
        """Update the Max Range vs RCS plot."""
        self.range_plot.clear()

        rcs_min = 1e-4  # -40 dBsm
        rcs_max = 1e4   # +40 dBsm

        rcs_values, range_values = self._calculator.max_range_vs_rcs(
            rcs_min, rcs_max, 500
        )

        rcs_dbsm = 10 * np.log10(rcs_values)
        ranges_display = np.array([self._range_to_display(r) for r in range_values])

        # Plot max range curve
        pen = pg.mkPen(color=COLORS['primary'], width=2)
        self.range_plot.plot(rcs_dbsm, ranges_display, pen=pen, name='Max Detection Range')

        # Mark current target
        current_rcs_dbsm = 10 * np.log10(self._current_rcs)
        max_range = self._calculator.calculate_max_range_m(self._current_rcs)
        max_range_display = self._range_to_display(max_range)

        scatter = pg.ScatterPlotItem(
            [current_rcs_dbsm], [max_range_display],
            pen=pg.mkPen(color=COLORS['tertiary'], width=2),
            brush=pg.mkBrush(color=COLORS['tertiary']),
            size=12, symbol='o'
        )
        self.range_plot.addItem(scatter)

        self.range_plot.setLabel('left', 'Max Detection Range', units=self._range_unit)

    def _update_budget_plot(self):
        """Update the Link Budget Waterfall plot."""
        self.budget_plot.clear()

        params = self._calculator.params
        results = self._calculator.calculate_performance(
            TargetScenario(self._current_rcs, self._current_range)
        )

        # Budget items using theme colors
        items = [
            ('Tx Power', results.peak_power_dbm, COLORS['primary']),
            ('Tx Gain', params.tx_gain_dbi, COLORS['secondary']),
            ('Rx Gain', params.rx_gain_dbi, COLORS['secondary']),
            ('RCS', results.rcs_dbsm, COLORS['accent2']),
            ('Sys Loss', -params.system_loss_db, COLORS['danger']),
            ('Path Loss', -results.path_loss_db, COLORS['danger']),
        ]

        x = np.arange(len(items))
        values = [item[1] for item in items]
        colors = [item[2] for item in items]

        # Create bar chart
        bar = pg.BarGraphItem(
            x=x, height=values, width=0.6,
            brushes=[pg.mkBrush(c) for c in colors]
        )
        self.budget_plot.addItem(bar)

        # Add labels
        labels = [item[0] for item in items]
        ax = self.budget_plot.getAxis('bottom')
        ax.setTicks([[(i, labels[i]) for i in range(len(labels))]])
        ax.setTextPen(pg.mkPen(color=COLORS['foreground']))

        # Add result line
        pr_line = pg.InfiniteLine(
            pos=results.received_power_dbm,
            angle=0,
            pen=pg.mkPen(color=COLORS['tertiary'], width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            label=f'Pr = {results.received_power_dbm:.1f} dBm',
            labelOpts={'color': COLORS['tertiary'], 'position': 0.9}
        )
        self.budget_plot.addItem(pr_line)

        self.budget_plot.setLabel('left', 'Value', units='dB/dBm')

    def _export_plot(self):
        """Export the current plot to an image file."""
        from PyQt6.QtWidgets import QFileDialog
        from pyqtgraph.exporters import ImageExporter

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG Files (*.png);;All Files (*)"
        )

        if filename:
            if not filename.endswith('.png'):
                filename += '.png'

            current_plot = self.plot_stack.currentWidget()
            exporter = ImageExporter(current_plot.plotItem)
            exporter.parameters()['width'] = 1920
            exporter.export(filename)
