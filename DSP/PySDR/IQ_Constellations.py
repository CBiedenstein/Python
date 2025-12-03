import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

num_symbols = 1000

x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
x_degrees = x_int*360/4 + 45 # 4 corners of the unit circle
x_radians = np.deg2rad(x_degrees)
x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians)
'''plt.plot(np.real(x_symbols), np.imag(x_symbols), '.') # produces the QPSK Modulation Contellation
plt.grid(True)
plt.show()'''

## Adding noise to constellation ##

n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/sqrt(2)  # AWGN with unity power
noise_power = 0.01
#r = x_symbols + n * sqrt(noise_power)

phase_noise = np.random.randn(len(x_symbols)) * 0.1 # adjust multiplier for strength of phase noise
r = x_symbols * np.exp(1j*phase_noise) + n*sqrt(noise_power)
plt.plot(np.real(r), np.imag(r), '.')
plt.grid(True)
plt.show()

