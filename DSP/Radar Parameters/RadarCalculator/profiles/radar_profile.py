"""
Radar profile dataclass for storing radar system configurations.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml


@dataclass
class RadarProfile:
    """
    Complete radar system profile.

    All values stored in SI base units:
    - Frequency in Hz
    - Power in Watts
    - Bandwidth in Hz
    - Gains in dBi
    - Other values in dB
    """
    # Identification
    name: str
    description: str = ""
    category: str = "Custom"

    # Transmitter
    frequency_hz: float = 3.0e9
    peak_power_w: float = 1.0e6

    # Antenna
    tx_gain_dbi: float = 30.0
    rx_gain_dbi: float = 30.0

    # Receiver
    bandwidth_hz: float = 1.0e6
    noise_figure_db: float = 3.0

    # System
    system_loss_db: float = 5.0
    required_snr_db: float = 13.0

    # Optional metadata
    manufacturer: str = ""
    beam_count: int = 1
    notes: str = ""

    # Optional beam configuration (for multi-beam radars like TPS-43)
    beam_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert profile to dictionary."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Convert profile to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> 'RadarProfile':
        """Create profile from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_string: str) -> 'RadarProfile':
        """Create profile from YAML string."""
        data = yaml.safe_load(yaml_string)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, filepath: str) -> 'RadarProfile':
        """Load profile from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_to_yaml(self, filepath: str):
        """Save profile to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @property
    def total_gain_dbi(self) -> float:
        """Combined Tx + Rx gain."""
        return self.tx_gain_dbi + self.rx_gain_dbi

    @property
    def frequency_ghz(self) -> float:
        """Frequency in GHz."""
        return self.frequency_hz / 1e9

    @property
    def frequency_mhz(self) -> float:
        """Frequency in MHz."""
        return self.frequency_hz / 1e6

    @property
    def peak_power_kw(self) -> float:
        """Peak power in kW."""
        return self.peak_power_w / 1e3

    @property
    def peak_power_mw(self) -> float:
        """Peak power in MW."""
        return self.peak_power_w / 1e6

    @property
    def bandwidth_mhz(self) -> float:
        """Bandwidth in MHz."""
        return self.bandwidth_hz / 1e6

    def __str__(self) -> str:
        return f"{self.name} ({self.frequency_ghz:.2f} GHz, {self.peak_power_kw:.1f} kW)"

    def summary(self) -> str:
        """Return a multi-line summary of the profile."""
        return f"""
Radar Profile: {self.name}
{'=' * 40}
Category:       {self.category}
Description:    {self.description}

Transmitter:
  Frequency:    {self.frequency_ghz:.3f} GHz
  Peak Power:   {self.peak_power_kw:.1f} kW ({self.peak_power_w:.2e} W)

Antenna:
  Tx Gain:      {self.tx_gain_dbi:.1f} dBi
  Rx Gain:      {self.rx_gain_dbi:.1f} dBi
  Total Gain:   {self.total_gain_dbi:.1f} dBi

Receiver:
  Bandwidth:    {self.bandwidth_mhz:.2f} MHz
  Noise Figure: {self.noise_figure_db:.1f} dB

System:
  Losses:       {self.system_loss_db:.1f} dB
  Required SNR: {self.required_snr_db:.1f} dB
""".strip()
