import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

num_symbols = 10000

# x_symbols array will contain copmplex numbers representing the QPSK symbols.Each symbol wioll be a copmplex number with a magnitude of 1 and a phase angle corresponding 
# to one of the four QPSK constellation points (45, 135, 225, or 315 degrees)

x_int = np.random.randint(0, 4, num_symbols)
x_degrees = x_int*360/4.0 + 45 # modulation points for QPSK 45, 135, 225, 315
x_radians = np.deg2rad(x_degrees) # sin and cos take radians
x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians) # this produces the complex QPSK symbols.

n = np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)/sqrt(2) # AWGN with unity power
r = x_symbols + n * sqrt(0.01) # noise power of 0.01
print(r)
plt.plot(np.real(r), np.imag(r), '.-')
plt.grid(True)
plt.show()

# Saving to an IQ file.
print(type(r[0])) # Check Data type. It will be 128 here as that is the numpy default. must convert to npcomplex64
r = r.astype(np.complex64)
print(type(r[0]))
r.tofile('qpsk_in_noise.iq')