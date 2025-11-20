import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

samples = np.fromfile('qpsk_in_noise.iq', np.complex64) # read in file, need to say what format the file is

# Plot constellation to make sure it looks right
plt.plot(np.real(samples), np.imag(samples), '.')
plt.grid(True)
plt.show()

## If ever dealing with int16's (aka short ints) or any other data type that numpy doesnt have an equivalent for, you must read the samples as real.
## read as real and then interleave back into IQIQIQ. Here are a couple ways to do that:

samples = np.fromfile('iq_samples_as_int16.iq', np.int16).astype(np.float32).view(np.complex64)

## OR

samples = np.fromfile('iq_samples_as_int16.iq', np.int16)
samples /= 32768 # convert to -1 to +1 (optional)
samples = samples[::2] + 1j*samples[1::2]

