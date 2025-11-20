import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

def calculate_mie_rcs(diameter, frequency_hz):
    """
    Calculates the exact backscatter RCS of a PEC sphere using the Mie Series.
    """
    c = 3e8  # Speed of light (m/s)
    wavelength = c / frequency_hz
    radius = diameter / 2.0
    k = 2 * np.pi / wavelength # Wavenumber
    x = k * radius # Size parameter (ka)

    # Convergence limit for the series (Wiscombe's criterion approximation)
    # We need enough terms to ensure accuracy.
    n_stop = int(x + 4 * (x**(1/3)) + 2)
    
    summation = 0.0 + 0.0j
    
    # Iterate through the modes
    for n in range(1, n_stop + 1):
        # Spherical Bessel functions of the first kind (j)
        j_n = spherical_jn(n, x)
        # Derivative of j
        j_n_prime = spherical_jn(n, x, derivative=True)
        
        # Spherical Bessel of second kind (y) to form Hankel (h)
        y_n = spherical_yn(n, x)
        y_n_prime = spherical_yn(n, x, derivative=True)
        
        # Spherical Hankel function of the first kind: h = j + i*y
        h_n = j_n + 1j * y_n
        h_n_prime = j_n_prime + 1j * y_n_prime
        
        # Mie Coefficients for PEC Sphere
        # a_n: Related to TM modes
        a_n = j_n / h_n
        
        # b_n: Related to TE modes
        # Note: The formula involves derivatives of (x * function), requires chain rule
        # d/dx [x * f(x)] = f(x) + x * f'(x)
        psi_prime = j_n + x * j_n_prime
        zeta_prime = h_n + x * h_n_prime
        
        b_n = psi_prime / zeta_prime

        # Backscatter summation formula
        term = ((-1)**n) * (n + 0.5) * (b_n - a_n)
        summation += term

    # Final RCS calculation
    # sigma = (lambda^2 / pi) * |sum|^2
    rcs_m2 = (wavelength**2 / np.pi) * np.abs(summation)**2
    return rcs_m2

# --- Simulation Configuration ---
frequency = 3e9  # 10 GHz (X-Band Radar)
diameters = np.linspace(0.1, 6.0, 300) # 0.1m to 6m resolution
rcs_data_db = []
optical_limit_db = []

print(f"Starting Simulation: {len(diameters)} steps at {frequency/1e9} GHz...")

# --- Execution Loop ---
for d in diameters:
    # Calculate Exact Mie RCS
    sigma = calculate_mie_rcs(d, frequency)
    rcs_db = 10 * np.log10(sigma) # Convert to dBsm
    rcs_data_db.append(rcs_db)
    
    # Calculate Optical Limit (Geometric Optics approximation: pi * r^2)
    r = d / 2
    opt_limit = 10 * np.log10(np.pi * r**2)
    optical_limit_db.append(opt_limit)

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-darkgrid')

# Plot Exact Mie Series
plt.plot(diameters, rcs_data_db, label='Exact Mie Solution (PEC)', color='#1f77b4', linewidth=2)

# Plot Optical Limit (Reference)
plt.plot(diameters, optical_limit_db, label='Optical Limit ($\pi r^2$)', color='red', linestyle='--', alpha=0.6)

plt.title(f'RCS of Metal Sphere vs Diameter\nFrequency: {frequency/1e9} GHz (X-Band)', fontsize=14)
plt.xlabel('Sphere Diameter (meters)', fontsize=12)
plt.ylabel('RCS (dBsm)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, which="both", ls="-")
plt.axhline(y=0, color='k', linewidth=0.5) # 0 dBsm line

# Add text annotation for the resonance region
plt.text(0.5, -10, 'Resonance Region\n(Mie Scattering)', fontsize=9, color='#1f77b4', ha='center')
plt.text(5.0, 15, 'Optical Region\n(Geometric)', fontsize=9, color='red', ha='center')

plt.tight_layout()
plt.show()