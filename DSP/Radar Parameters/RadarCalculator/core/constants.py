"""
Physical constants used in radar calculations.
All values in SI base units.
"""

# Speed of light in vacuum (m/s)
C = 299792458.0

# Boltzmann constant (J/K)
K_BOLTZMANN = 1.380649e-23

# Standard reference temperature (Kelvin)
T0 = 290.0

# Pi constant
PI = 3.141592653589793

# Common reference values
DBM_TO_WATTS_FACTOR = 1000.0  # 1 W = 1000 mW
NMI_TO_METERS = 1852.0
KM_TO_METERS = 1000.0
MI_TO_METERS = 1609.344

# Earth parameters
EARTH_RADIUS_M = 6371000.0  # Mean earth radius (m)
EARTH_RADIUS_43_M = 8495000.0  # 4/3 effective earth radius for RF refraction (m)

# SRTM resolution
SRTM_RESOLUTION_1_ARCSEC_M = 30.0  # ~30m at equator
SRTM_RESOLUTION_3_ARCSEC_M = 90.0  # ~90m at equator
