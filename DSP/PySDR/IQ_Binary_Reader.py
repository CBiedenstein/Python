import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

samples = np.fromfile('qpsk_in_noise.iq', np.complex64) # read in file, need to say what format the file is

# Plot constellation to make sure it looks right
plt.plot(np.real(samples), np.imag(samples), '.')
plt.grid(True)
plt.show()
