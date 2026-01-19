"""
Qt stylesheets for the Radar Calculator application.

Professional/Industrial Radar Console Theme
Inspired by military radar displays, aerospace interfaces, and industrial control systems.
"""

# =============================================================================
# COLOR PALETTE - Radar Console Theme
# =============================================================================
# Primary Background: Deep navy/charcoal (#0a0e14, #0d1117)
# Secondary Background: Dark blue-gray (#161b22, #1c2128)
# Panel Background: Slightly lighter (#21262d)
#
# Primary Accent: Radar Green (#00ff88, #39d353) - classic phosphor
# Secondary Accent: Amber/Gold (#ffb000, #f0a000) - warning/highlight
# Tertiary Accent: Cyan (#00d4ff, #58a6ff) - information
#
# Text Primary: High contrast (#e6edf3)
# Text Secondary: Muted (#8b949e)
# Text Dim: Very muted (#484f58)
#
# Borders: Subtle lines (#30363d)
# Danger: Red (#f85149)
# Success: Green (#3fb950)
# =============================================================================

DARK_THEME = """
/* ============================================
   RADAR CONSOLE DARK THEME
   Professional/Industrial Aerospace Style
   ============================================ */

QMainWindow {
    background-color: #0d1117;
}

QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Consolas', 'Cascadia Code', 'SF Mono', 'Monaco', monospace;
    font-size: 10pt;
}

/* ============================================
   GROUP BOXES - Panel Containers
   ============================================ */
QGroupBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #00ff88;
    border-radius: 4px;
    margin-top: 16px;
    padding: 12px;
    padding-top: 24px;
    font-weight: 500;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    left: 8px;
    background-color: #21262d;
    color: #00ff88;
    font-weight: bold;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 1px solid #30363d;
    border-radius: 3px;
}

/* ============================================
   LABELS
   ============================================ */
QLabel {
    color: #8b949e;
    background-color: transparent;
    font-weight: 500;
}

QLabel#headerLabel {
    color: #00ff88;
    font-size: 11pt;
    font-weight: bold;
}

QLabel#valueLabel {
    color: #e6edf3;
    font-family: 'Consolas', 'Cascadia Code', monospace;
    font-size: 11pt;
}

QLabel#unitLabel {
    color: #58a6ff;
    font-size: 9pt;
}

/* ============================================
   LINE EDIT
   ============================================ */
QLineEdit {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-bottom: 2px solid #30363d;
    border-radius: 3px;
    padding: 6px 10px;
    color: #e6edf3;
    font-family: 'Consolas', monospace;
    selection-background-color: #00ff88;
    selection-color: #0d1117;
}

QLineEdit:focus {
    border-bottom: 2px solid #00ff88;
}

QLineEdit:hover {
    border-color: #484f58;
}

/* ============================================
   SPIN BOXES - Numeric Input
   ============================================ */
QSpinBox, QDoubleSpinBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-bottom: 2px solid #30363d;
    border-radius: 3px;
    padding: 5px 8px;
    padding-right: 28px;
    color: #00ff88;
    font-family: 'Consolas', 'Cascadia Code', monospace;
    font-size: 10pt;
    font-weight: bold;
    selection-background-color: #00ff88;
    selection-color: #0d1117;
    min-height: 22px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-bottom: 2px solid #00ff88;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #484f58;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    border-left: 1px solid #30363d;
    background-color: #21262d;
    border-top-right-radius: 2px;
    margin: 1px;
    margin-bottom: 0;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #00ff88;
}

QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
    background-color: #00cc6a;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 6px solid #8b949e;
}

QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {
    border-bottom-color: #0d1117;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border: none;
    border-left: 1px solid #30363d;
    background-color: #21262d;
    border-bottom-right-radius: 2px;
    margin: 1px;
    margin-top: 0;
}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #00ff88;
}

QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background-color: #00cc6a;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #8b949e;
}

QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
    border-top-color: #0d1117;
}

/* ============================================
   COMBO BOXES - Dropdown Select
   ============================================ */
QComboBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-bottom: 2px solid #30363d;
    border-radius: 3px;
    padding: 6px 10px;
    padding-right: 30px;
    color: #e6edf3;
    font-family: 'Consolas', monospace;
    min-width: 100px;
    min-height: 22px;
}

QComboBox:hover {
    border-color: #484f58;
}

QComboBox:focus {
    border-bottom: 2px solid #00ff88;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 26px;
    border: none;
    border-left: 1px solid #30363d;
    background-color: #21262d;
    border-radius: 2px;
    margin: 2px;
}

QComboBox::drop-down:hover {
    background-color: #00ff88;
}

QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 7px solid #8b949e;
}

QComboBox::down-arrow:hover {
    border-top-color: #0d1117;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #00ff88;
    selection-color: #0d1117;
    outline: none;
}

QComboBox QAbstractItemView::item {
    padding: 8px 12px;
    border-radius: 3px;
    min-height: 24px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #21262d;
}

/* ============================================
   PUSH BUTTONS
   ============================================ */
QPushButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 8px 20px;
    font-weight: bold;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 1px;
    min-height: 18px;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #00ff88;
    color: #00ff88;
}

QPushButton:pressed {
    background-color: #00ff88;
    color: #0d1117;
}

QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}

QPushButton#primaryButton {
    background-color: #00ff88;
    color: #0d1117;
    border: none;
}

QPushButton#primaryButton:hover {
    background-color: #39d353;
}

QPushButton#primaryButton:pressed {
    background-color: #00cc6a;
}

QPushButton#secondaryButton {
    background-color: transparent;
    color: #58a6ff;
    border: 1px solid #58a6ff;
}

QPushButton#secondaryButton:hover {
    background-color: #58a6ff;
    color: #0d1117;
}

QPushButton#dangerButton {
    background-color: transparent;
    color: #f85149;
    border: 1px solid #f85149;
}

QPushButton#dangerButton:hover {
    background-color: #f85149;
    color: #0d1117;
}

QPushButton#warningButton {
    background-color: transparent;
    color: #f0a000;
    border: 1px solid #f0a000;
}

QPushButton#warningButton:hover {
    background-color: #f0a000;
    color: #0d1117;
}

/* ============================================
   CHECK BOXES
   ============================================ */
QCheckBox {
    color: #e6edf3;
    spacing: 8px;
    font-weight: 500;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 2px solid #30363d;
    background-color: #0d1117;
}

QCheckBox::indicator:checked {
    background-color: #00ff88;
    border-color: #00ff88;
    image: none;
}

QCheckBox::indicator:checked:after {
    content: "";
}

QCheckBox::indicator:hover {
    border-color: #00ff88;
}

QCheckBox::indicator:unchecked:hover {
    background-color: #161b22;
}

/* ============================================
   RADIO BUTTONS
   ============================================ */
QRadioButton {
    color: #e6edf3;
    spacing: 8px;
    font-weight: 500;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #30363d;
    background-color: #0d1117;
}

QRadioButton::indicator:checked {
    background-color: #00ff88;
    border-color: #00ff88;
}

QRadioButton::indicator:hover {
    border-color: #00ff88;
}

/* ============================================
   TABLES
   ============================================ */
QTableWidget {
    background-color: #0d1117;
    gridline-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
}

QTableWidget::item {
    padding: 6px 8px;
    border-bottom: 1px solid #21262d;
}

QTableWidget::item:selected {
    background-color: #00ff88;
    color: #0d1117;
}

QTableWidget::item:hover {
    background-color: #161b22;
}

QHeaderView::section {
    background-color: #161b22;
    color: #00ff88;
    padding: 8px 12px;
    border: none;
    border-bottom: 2px solid #00ff88;
    border-right: 1px solid #21262d;
    font-weight: bold;
    font-size: 9pt;
    text-transform: uppercase;
}

QHeaderView::section:last {
    border-right: none;
}

/* ============================================
   SCROLL AREAS
   ============================================ */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #0d1117;
    width: 12px;
    border-radius: 6px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #30363d;
    border-radius: 6px;
    min-height: 40px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #00ff88;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QScrollBar:horizontal {
    background-color: #0d1117;
    height: 12px;
    border-radius: 6px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #30363d;
    border-radius: 6px;
    min-width: 40px;
    margin: 2px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #00ff88;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

/* ============================================
   STATUS BAR
   ============================================ */
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #30363d;
    font-family: 'Consolas', monospace;
    font-size: 9pt;
    padding: 4px 12px;
}

QStatusBar::item {
    border: none;
}

/* ============================================
   MENU BAR
   ============================================ */
QMenuBar {
    background-color: #0d1117;
    color: #8b949e;
    border-bottom: 1px solid #30363d;
    padding: 4px 8px;
    font-size: 10pt;
}

QMenuBar::item {
    padding: 6px 12px;
    border-radius: 3px;
}

QMenuBar::item:selected {
    background-color: #21262d;
    color: #e6edf3;
}

QMenuBar::item:pressed {
    background-color: #00ff88;
    color: #0d1117;
}

QMenu {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px;
}

QMenu::item {
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    color: #e6edf3;
}

QMenu::item:selected {
    background-color: #00ff88;
    color: #0d1117;
}

QMenu::separator {
    height: 1px;
    background-color: #30363d;
    margin: 6px 8px;
}

/* ============================================
   SPLITTERS
   ============================================ */
QSplitter::handle {
    background-color: #30363d;
}

QSplitter::handle:horizontal {
    width: 3px;
    margin: 0 1px;
}

QSplitter::handle:vertical {
    height: 3px;
    margin: 1px 0;
}

QSplitter::handle:hover {
    background-color: #00ff88;
}

/* ============================================
   TAB WIDGET
   ============================================ */
QTabWidget::pane {
    border: 1px solid #30363d;
    border-top: 2px solid #00ff88;
    background-color: #0d1117;
    border-radius: 0 0 6px 6px;
}

QTabBar {
    background-color: transparent;
}

QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    padding: 12px 24px;
    margin-right: 2px;
    border: 1px solid #30363d;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    font-weight: bold;
    font-size: 10pt;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QTabBar::tab:selected {
    background-color: #0d1117;
    color: #00ff88;
    border-bottom: 2px solid #0d1117;
    margin-bottom: -2px;
}

QTabBar::tab:hover:!selected {
    background-color: #21262d;
    color: #e6edf3;
}

QTabBar::tab:first {
    margin-left: 0;
}

/* ============================================
   SLIDERS
   ============================================ */
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background-color: #21262d;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #00ff88;
    border: none;
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background-color: #39d353;
}

QSlider::sub-page:horizontal {
    background-color: #00ff88;
    border-radius: 3px;
}

/* ============================================
   TOOLTIPS
   ============================================ */
QToolTip {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #00ff88;
    border-radius: 4px;
    padding: 6px 10px;
    font-family: 'Consolas', monospace;
}

/* ============================================
   PROGRESS BAR
   ============================================ */
QProgressBar {
    background-color: #21262d;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #00ff88;
    border-radius: 4px;
}

/* ============================================
   MESSAGE BOX
   ============================================ */
QMessageBox {
    background-color: #161b22;
}

QMessageBox QLabel {
    color: #e6edf3;
}
"""


# =============================================================================
# LIGHT THEME - Industrial/Technical Light Mode
# =============================================================================
LIGHT_THEME = """
/* ============================================
   RADAR CONSOLE LIGHT THEME
   Professional/Industrial Aerospace Style
   ============================================ */

QMainWindow {
    background-color: #f0f3f6;
}

QWidget {
    background-color: #f0f3f6;
    color: #1c2128;
    font-family: 'Consolas', 'Cascadia Code', 'SF Mono', 'Monaco', monospace;
    font-size: 10pt;
}

/* ============================================
   GROUP BOXES
   ============================================ */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d1d5da;
    border-left: 3px solid #0969da;
    border-radius: 4px;
    margin-top: 16px;
    padding: 12px;
    padding-top: 24px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    left: 8px;
    background-color: #f0f3f6;
    color: #0969da;
    font-weight: bold;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 1px solid #d1d5da;
    border-radius: 3px;
}

/* ============================================
   LABELS
   ============================================ */
QLabel {
    color: #57606a;
    background-color: transparent;
    font-weight: 500;
}

/* ============================================
   INPUTS
   ============================================ */
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #d1d5da;
    border-bottom: 2px solid #d1d5da;
    border-radius: 3px;
    padding: 6px 10px;
    color: #1c2128;
    selection-background-color: #0969da;
}

QLineEdit:focus {
    border-bottom: 2px solid #0969da;
}

QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #d1d5da;
    border-bottom: 2px solid #d1d5da;
    border-radius: 3px;
    padding: 5px 8px;
    padding-right: 28px;
    color: #0969da;
    font-family: 'Consolas', monospace;
    font-weight: bold;
    min-height: 22px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-bottom: 2px solid #0969da;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #f0f3f6;
    border: none;
    border-left: 1px solid #d1d5da;
    width: 22px;
    margin: 1px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #0969da;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    border-bottom: 6px solid #57606a;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    border-top: 6px solid #57606a;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
}

QComboBox {
    background-color: #ffffff;
    border: 1px solid #d1d5da;
    border-bottom: 2px solid #d1d5da;
    border-radius: 3px;
    padding: 6px 10px;
    padding-right: 30px;
    color: #1c2128;
    min-height: 22px;
}

QComboBox:focus {
    border-bottom: 2px solid #0969da;
}

QComboBox::drop-down {
    background-color: #f0f3f6;
    border-left: 1px solid #d1d5da;
    width: 26px;
    margin: 2px;
    border-radius: 2px;
}

QComboBox::drop-down:hover {
    background-color: #0969da;
}

QComboBox::down-arrow {
    border-top: 7px solid #57606a;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #d1d5da;
    selection-background-color: #0969da;
    selection-color: #ffffff;
}

/* ============================================
   BUTTONS
   ============================================ */
QPushButton {
    background-color: #f0f3f6;
    color: #1c2128;
    border: 1px solid #d1d5da;
    border-radius: 4px;
    padding: 8px 20px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QPushButton:hover {
    background-color: #0969da;
    color: #ffffff;
    border-color: #0969da;
}

QPushButton:pressed {
    background-color: #0550ae;
}

QPushButton#primaryButton {
    background-color: #0969da;
    color: #ffffff;
    border: none;
}

QPushButton#primaryButton:hover {
    background-color: #0550ae;
}

QPushButton#dangerButton {
    color: #cf222e;
    border-color: #cf222e;
}

QPushButton#dangerButton:hover {
    background-color: #cf222e;
    color: #ffffff;
}

/* ============================================
   CHECKBOXES & RADIO
   ============================================ */
QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #d1d5da;
    background-color: #ffffff;
}

QCheckBox::indicator {
    border-radius: 3px;
}

QRadioButton::indicator {
    border-radius: 9px;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #0969da;
    border-color: #0969da;
}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #0969da;
}

/* ============================================
   TABLES
   ============================================ */
QTableWidget {
    background-color: #ffffff;
    gridline-color: #d1d5da;
    border: 1px solid #d1d5da;
}

QTableWidget::item:selected {
    background-color: #0969da;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #f0f3f6;
    color: #0969da;
    border-bottom: 2px solid #0969da;
    padding: 8px;
    font-weight: bold;
    text-transform: uppercase;
}

/* ============================================
   SCROLLBARS
   ============================================ */
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #f0f3f6;
}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #d1d5da;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background-color: #0969da;
}

QScrollBar:vertical {
    width: 12px;
}

QScrollBar:horizontal {
    height: 12px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    height: 0;
    width: 0;
}

/* ============================================
   STATUS BAR
   ============================================ */
QStatusBar {
    background-color: #f0f3f6;
    color: #57606a;
    border-top: 1px solid #d1d5da;
}

/* ============================================
   MENU BAR
   ============================================ */
QMenuBar {
    background-color: #ffffff;
    border-bottom: 1px solid #d1d5da;
    color: #57606a;
}

QMenuBar::item:selected {
    background-color: #f0f3f6;
    color: #1c2128;
}

QMenu {
    background-color: #ffffff;
    border: 1px solid #d1d5da;
    border-radius: 6px;
}

QMenu::item:selected {
    background-color: #0969da;
    color: #ffffff;
}

/* ============================================
   SPLITTERS
   ============================================ */
QSplitter::handle {
    background-color: #d1d5da;
}

QSplitter::handle:hover {
    background-color: #0969da;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}

/* ============================================
   TABS
   ============================================ */
QTabWidget::pane {
    border: 1px solid #d1d5da;
    border-top: 2px solid #0969da;
    background-color: #f0f3f6;
}

QTabBar::tab {
    background-color: #ffffff;
    color: #57606a;
    padding: 12px 24px;
    border: 1px solid #d1d5da;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #f0f3f6;
    color: #0969da;
    border-bottom: 2px solid #f0f3f6;
    margin-bottom: -2px;
}

QTabBar::tab:hover:!selected {
    background-color: #f0f3f6;
    color: #1c2128;
}

/* ============================================
   TOOLTIPS
   ============================================ */
QToolTip {
    background-color: #ffffff;
    color: #1c2128;
    border: 1px solid #0969da;
    border-radius: 4px;
    padding: 6px 10px;
}

/* ============================================
   SLIDERS
   ============================================ */
QSlider::groove:horizontal {
    background-color: #d1d5da;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #0969da;
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

QSlider::sub-page:horizontal {
    background-color: #0969da;
    border-radius: 3px;
}
"""


# =============================================================================
# RESULT CARD STYLES
# =============================================================================
DETECTED_STYLE = """
    background-color: #238636;
    color: #ffffff;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
    font-family: 'Consolas', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
"""

NOT_DETECTED_STYLE = """
    background-color: #da3633;
    color: #ffffff;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
    font-family: 'Consolas', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
"""

DETECTED_STYLE_LIGHT = """
    background-color: #1a7f37;
    color: #ffffff;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
    font-family: 'Consolas', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
"""

NOT_DETECTED_STYLE_LIGHT = """
    background-color: #cf222e;
    color: #ffffff;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
    font-family: 'Consolas', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
"""


# =============================================================================
# PLOT COLORS - For pyqtgraph
# =============================================================================
PLOT_COLORS = {
    'dark': {
        'background': '#0d1117',
        'foreground': '#e6edf3',
        'grid': '#21262d',
        'primary': '#00ff88',      # Radar green
        'secondary': '#58a6ff',    # Cyan/blue
        'tertiary': '#f0a000',     # Amber
        'danger': '#f85149',       # Red
        'accent1': '#a371f7',      # Purple
        'accent2': '#00d4ff',      # Bright cyan
    },
    'light': {
        'background': '#ffffff',
        'foreground': '#1c2128',
        'grid': '#d1d5da',
        'primary': '#0969da',
        'secondary': '#1a7f37',
        'tertiary': '#9a6700',
        'danger': '#cf222e',
        'accent1': '#8250df',
        'accent2': '#0550ae',
    }
}
