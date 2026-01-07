"""
Waveform calculations for radar pulse compression.

Supports:
- Barker codes (various lengths)
- FM Chirp (Linear Frequency Modulation)
- Phase-coded waveforms
- Simple pulse (no compression)
"""

import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class WaveformType(Enum):
    """Supported waveform types."""
    SIMPLE_PULSE = "Simple Pulse"
    BARKER_2 = "Barker-2"
    BARKER_3 = "Barker-3"
    BARKER_4 = "Barker-4"
    BARKER_5 = "Barker-5"
    BARKER_7 = "Barker-7"
    BARKER_11 = "Barker-11"
    BARKER_13 = "Barker-13"
    FM_CHIRP = "FM Chirp (LFM)"
    NLFM_CHIRP = "NLFM Chirp"
    POLYPHASE = "Polyphase Code"


# Barker code sequences (optimal binary phase codes)
BARKER_CODES = {
    WaveformType.BARKER_2: [1, -1],
    WaveformType.BARKER_3: [1, 1, -1],
    WaveformType.BARKER_4: [1, 1, -1, 1],  # or [1, 1, 1, -1]
    WaveformType.BARKER_5: [1, 1, 1, -1, 1],
    WaveformType.BARKER_7: [1, 1, 1, -1, -1, 1, -1],
    WaveformType.BARKER_11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    WaveformType.BARKER_13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


@dataclass
class WaveformParameters:
    """Parameters defining a radar waveform."""
    waveform_type: WaveformType
    pulse_width_s: float  # Uncompressed pulse width (seconds)
    bandwidth_hz: float  # Signal bandwidth (Hz)

    # FM Chirp specific
    chirp_bandwidth_hz: Optional[float] = None  # Sweep bandwidth for chirp

    # Polyphase specific
    num_subpulses: Optional[int] = None

    @property
    def compression_ratio(self) -> float:
        """Calculate pulse compression ratio."""
        return calculate_compression_ratio(self)

    @property
    def processing_gain_db(self) -> float:
        """Calculate processing gain in dB."""
        return calculate_processing_gain_db(self)

    @property
    def compressed_pulse_width_s(self) -> float:
        """Calculate compressed pulse width."""
        return calculate_compressed_pulse_width(self)

    @property
    def range_resolution_m(self) -> float:
        """Calculate range resolution in meters."""
        return calculate_range_resolution(self)

    @property
    def time_bandwidth_product(self) -> float:
        """Calculate time-bandwidth product."""
        return calculate_time_bandwidth_product(self)


@dataclass
class WaveformPerformance:
    """Calculated waveform performance metrics."""
    waveform_type: WaveformType
    pulse_width_s: float
    bandwidth_hz: float
    compression_ratio: float
    processing_gain_db: float
    compressed_pulse_width_s: float
    range_resolution_m: float
    time_bandwidth_product: float
    peak_sidelobe_level_db: float
    integrated_sidelobe_ratio_db: float


def get_barker_length(waveform_type: WaveformType) -> int:
    """Get the length of a Barker code."""
    if waveform_type in BARKER_CODES:
        return len(BARKER_CODES[waveform_type])
    return 1


def calculate_compression_ratio(params: WaveformParameters) -> float:
    """
    Calculate pulse compression ratio.

    For Barker codes: N (code length)
    For FM Chirp: B * T (time-bandwidth product)
    """
    wf_type = params.waveform_type

    if wf_type == WaveformType.SIMPLE_PULSE:
        return 1.0

    elif wf_type in BARKER_CODES:
        return float(len(BARKER_CODES[wf_type]))

    elif wf_type in [WaveformType.FM_CHIRP, WaveformType.NLFM_CHIRP]:
        if params.chirp_bandwidth_hz:
            return params.chirp_bandwidth_hz * params.pulse_width_s
        return params.bandwidth_hz * params.pulse_width_s

    elif wf_type == WaveformType.POLYPHASE:
        if params.num_subpulses:
            return float(params.num_subpulses)
        return params.bandwidth_hz * params.pulse_width_s

    return 1.0


def calculate_processing_gain_db(params: WaveformParameters) -> float:
    """
    Calculate processing gain in dB from pulse compression.

    Processing gain = 10 * log10(compression_ratio)
    """
    cr = calculate_compression_ratio(params)
    if cr <= 0:
        return 0.0
    return 10 * math.log10(cr)


def calculate_compressed_pulse_width(params: WaveformParameters) -> float:
    """
    Calculate the compressed pulse width.

    tau_compressed = tau_uncompressed / compression_ratio
    Or equivalently: tau_compressed ≈ 1 / bandwidth
    """
    wf_type = params.waveform_type

    if wf_type == WaveformType.SIMPLE_PULSE:
        return params.pulse_width_s

    elif wf_type in BARKER_CODES:
        # For Barker codes, compressed width = chip width = pulse_width / N
        n = len(BARKER_CODES[wf_type])
        return params.pulse_width_s / n

    elif wf_type in [WaveformType.FM_CHIRP, WaveformType.NLFM_CHIRP]:
        # For chirp, compressed width ≈ 1 / bandwidth
        if params.chirp_bandwidth_hz:
            return 1.0 / params.chirp_bandwidth_hz
        return 1.0 / params.bandwidth_hz

    elif wf_type == WaveformType.POLYPHASE:
        if params.num_subpulses:
            return params.pulse_width_s / params.num_subpulses
        return 1.0 / params.bandwidth_hz

    return params.pulse_width_s


def calculate_range_resolution(params: WaveformParameters) -> float:
    """
    Calculate range resolution in meters.

    delta_R = c * tau_compressed / 2 = c / (2 * B)
    """
    c = 299792458.0  # Speed of light

    compressed_width = calculate_compressed_pulse_width(params)
    return c * compressed_width / 2


def calculate_time_bandwidth_product(params: WaveformParameters) -> float:
    """
    Calculate time-bandwidth product (TBP).

    TBP = pulse_width * bandwidth
    """
    wf_type = params.waveform_type

    if wf_type in [WaveformType.FM_CHIRP, WaveformType.NLFM_CHIRP]:
        if params.chirp_bandwidth_hz:
            return params.pulse_width_s * params.chirp_bandwidth_hz
    return params.pulse_width_s * params.bandwidth_hz


def get_peak_sidelobe_level_db(waveform_type: WaveformType) -> float:
    """
    Get theoretical peak sidelobe level (PSL) in dB.

    Barker codes have PSL = -20*log10(N) for length N
    FM Chirp without windowing has ~-13.3 dB sidelobes
    """
    if waveform_type == WaveformType.SIMPLE_PULSE:
        return 0.0  # No sidelobes for simple pulse

    elif waveform_type in BARKER_CODES:
        n = len(BARKER_CODES[waveform_type])
        # Barker codes have peak sidelobe of 1/N
        return -20 * math.log10(n)

    elif waveform_type == WaveformType.FM_CHIRP:
        # LFM without windowing has ~-13.3 dB sidelobes (sinc function)
        return -13.3

    elif waveform_type == WaveformType.NLFM_CHIRP:
        # NLFM can achieve lower sidelobes, typically -30 to -40 dB
        return -35.0

    elif waveform_type == WaveformType.POLYPHASE:
        # Polyphase codes (like Frank, P1-P4) typically have low sidelobes
        return -30.0

    return 0.0


def get_integrated_sidelobe_ratio_db(waveform_type: WaveformType) -> float:
    """
    Get theoretical integrated sidelobe ratio (ISLR) in dB.
    """
    if waveform_type == WaveformType.SIMPLE_PULSE:
        return 0.0

    elif waveform_type in BARKER_CODES:
        n = len(BARKER_CODES[waveform_type])
        # ISLR for Barker ≈ -10*log10(N)
        return -10 * math.log10(n)

    elif waveform_type == WaveformType.FM_CHIRP:
        return -9.5  # Typical for unweighted LFM

    elif waveform_type == WaveformType.NLFM_CHIRP:
        return -25.0

    elif waveform_type == WaveformType.POLYPHASE:
        return -20.0

    return 0.0


def calculate_waveform_performance(params: WaveformParameters) -> WaveformPerformance:
    """
    Calculate complete waveform performance metrics.
    """
    return WaveformPerformance(
        waveform_type=params.waveform_type,
        pulse_width_s=params.pulse_width_s,
        bandwidth_hz=params.bandwidth_hz,
        compression_ratio=calculate_compression_ratio(params),
        processing_gain_db=calculate_processing_gain_db(params),
        compressed_pulse_width_s=calculate_compressed_pulse_width(params),
        range_resolution_m=calculate_range_resolution(params),
        time_bandwidth_product=calculate_time_bandwidth_product(params),
        peak_sidelobe_level_db=get_peak_sidelobe_level_db(params.waveform_type),
        integrated_sidelobe_ratio_db=get_integrated_sidelobe_ratio_db(params.waveform_type)
    )


def get_effective_snr_with_compression(snr_db: float, processing_gain_db: float) -> float:
    """
    Calculate effective SNR after pulse compression processing.

    SNR_effective = SNR_raw + Processing_Gain
    """
    return snr_db + processing_gain_db


def get_waveform_types() -> list:
    """Get list of all available waveform types."""
    return list(WaveformType)


def get_waveform_description(waveform_type: WaveformType) -> str:
    """Get a description of the waveform type."""
    descriptions = {
        WaveformType.SIMPLE_PULSE: "Unmodulated pulse, no compression",
        WaveformType.BARKER_2: "2-bit Barker code, 3 dB gain",
        WaveformType.BARKER_3: "3-bit Barker code, 4.8 dB gain",
        WaveformType.BARKER_4: "4-bit Barker code, 6 dB gain",
        WaveformType.BARKER_5: "5-bit Barker code, 7 dB gain",
        WaveformType.BARKER_7: "7-bit Barker code, 8.5 dB gain",
        WaveformType.BARKER_11: "11-bit Barker code, 10.4 dB gain",
        WaveformType.BARKER_13: "13-bit Barker code, 11.1 dB gain, best sidelobes",
        WaveformType.FM_CHIRP: "Linear FM chirp, gain = 10*log10(BT)",
        WaveformType.NLFM_CHIRP: "Non-linear FM chirp, lower sidelobes",
        WaveformType.POLYPHASE: "Polyphase code (Frank, P1-P4, etc.)"
    }
    return descriptions.get(waveform_type, "Unknown waveform type")


# Convenience functions for common calculations
def barker_13_gain() -> float:
    """Get the processing gain for Barker-13."""
    return 10 * math.log10(13)  # ≈ 11.14 dB


def chirp_gain(bandwidth_hz: float, pulse_width_s: float) -> float:
    """Calculate FM chirp processing gain."""
    tbp = bandwidth_hz * pulse_width_s
    return 10 * math.log10(tbp)


def range_resolution_from_bandwidth(bandwidth_hz: float) -> float:
    """Calculate range resolution from bandwidth."""
    c = 299792458.0
    return c / (2 * bandwidth_hz)
