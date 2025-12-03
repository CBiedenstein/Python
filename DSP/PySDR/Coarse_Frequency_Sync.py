import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

num_symbols = 1000
sps = 8
bits = np.random.randint(0, 2, num_symbols)
fs = 1e6
pulse_train = np.array([])
for bit in bits:
    pulse = np.zeros(sps)
    pulse[0] = bit*2-1
    pulse_train = np.concatenate((pulse_train, pulse)) #add the 8 samples to the signal

# raised cosine filter
num_taps = 101
beta = 0.35
Ts = sps
t = np.arange(-51, 52)
h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

# filter the sample to apply the pulse shaping
samples = np.convolve(pulse_train, h)
samples = samples**2
psd = np.fft.fftshift(np.abs(np.fft.fft(samples)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))
plt.plot(f, psd)
plt.show()

max_freq = f[np.argmax(psd)]
Ts = 1/fs
t = np.arange(0, Ts*len(samples), Ts) # create time vector
samples = samples * np.exp(-1j*2*np.pi*max_freq*t/2.0)