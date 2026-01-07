# Radar Parameter Calculator

A holistic radar performance analysis application built with PyQt6.

## Features

- **Multiple Radar Profiles**: Pre-configured TPS-43, TPS-70, TPS-75, and generic S/X-band templates
- **Custom Profiles**: Create, save, and load your own radar configurations
- **Real-time Calculations**: Instantly see results as you adjust parameters
- **Interactive Plots**:
  - SNR vs Range (primary)
  - Received Power vs Range
  - Minimum Detectable RCS vs Range
  - Maximum Detection Range vs RCS
  - Link Budget Waterfall
- **Unit Flexibility**: Toggle between nmi/km, MW/kW/W, GHz/MHz
- **Target Library**: 25+ predefined targets from insects to aircraft carriers
- **Export**: Save results to CSV

## Installation

### Requirements
- Python 3.10+
- PyQt6
- pyqtgraph
- numpy
- scipy
- PyYAML

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install PyQt6 pyqtgraph numpy scipy PyYAML
```

## Running the Application

### Option 1: Python
```bash
cd RadarCalculator
python main.py
```

### Option 2: Windows Batch File
Double-click `run.bat`

## Project Structure

```
RadarCalculator/
├── main.py                 # Application entry point
├── run.bat                 # Windows launch script
├── requirements.txt        # Dependencies
│
├── core/                   # Calculation engine (UI-independent)
│   ├── constants.py        # Physical constants
│   ├── units.py            # Unit conversions
│   ├── radar_equations.py  # All radar calculations
│   └── targets.py          # Target RCS library
│
├── profiles/               # Radar system profiles
│   ├── radar_profile.py    # Profile dataclass
│   ├── profile_manager.py  # Profile management
│   ├── defaults/           # Built-in profiles
│   │   ├── tps43.yaml
│   │   ├── tps70.yaml
│   │   ├── tps75.yaml
│   │   ├── generic_sband.yaml
│   │   └── generic_xband.yaml
│   └── user_profiles/      # Your saved profiles
│
└── ui/                     # PyQt6 user interface
    ├── main_window.py      # Main application window
    ├── styles.py           # Dark/light themes
    ├── widgets/            # UI components
    └── plots/              # Plot widgets
```

## Using the Core Module Standalone

The calculation engine can be used independently without the GUI:

```python
from core.radar_equations import RadarCalculator, RadarParameters, TargetScenario

# Define radar parameters
params = RadarParameters(
    frequency_hz=3.0e9,      # 3 GHz
    peak_power_w=2.4e6,      # 2.4 MW
    tx_gain_dbi=36.0,
    rx_gain_dbi=39.0,
    bandwidth_hz=1.6e6,      # 1.6 MHz
    noise_figure_db=2.0,
    system_loss_db=5.0,
    required_snr_db=13.0
)

# Create calculator
calc = RadarCalculator(params)

# Define target scenario
target = TargetScenario(
    rcs_m2=20.0,             # B737-class aircraft
    range_m=200 * 1852,      # 200 nmi
    name="B737"
)

# Calculate performance
results = calc.calculate_performance(target)

print(f"SNR: {results.snr_db:.1f} dB")
print(f"Detected: {results.detected}")
print(f"Max Range: {results.max_range_m/1852:.1f} nmi")
```

## Creating Custom Profiles

Profiles are stored as YAML files. Create a new file in `profiles/user_profiles/`:

```yaml
name: "My Custom Radar"
description: "Custom configuration"
category: "Custom"

frequency_hz: 9400000000.0   # 9.4 GHz
peak_power_w: 500000.0       # 500 kW
tx_gain_dbi: 38.0
rx_gain_dbi: 38.0
bandwidth_hz: 2000000.0      # 2 MHz
noise_figure_db: 4.0
system_loss_db: 6.0
required_snr_db: 15.0
```

## Keyboard Shortcuts

- `Ctrl+S`: Export results to CSV
- `Ctrl+Q`: Quit application

## Calculations Performed

1. **Received Power (Pr)**: Radar range equation
2. **Thermal Noise**: kTB calculation
3. **MDS**: Minimum Detectable Signal (noise floor + NF)
4. **SNR**: Signal-to-Noise Ratio
5. **Max Range**: Inverse radar equation for range
6. **Min RCS**: Inverse radar equation for RCS
7. **Detection Status**: Comparison against required SNR threshold

## Distribution

To create a standalone executable for distribution:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py
```

The executable will be in the `dist/` folder.

## License

Internal use - TSS Solutions
