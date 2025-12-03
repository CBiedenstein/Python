import numpy as np
from scipy.signal import firwin2, convolve, fftconvolve, lfilter
import matplotlib.pyplot as plt

## create a test signal ##
sample_rate = 1e6
N = 1000
x = np.random.randn(N) + 1j * np.random.randn(N)

## FIR filter ##
freqs = [0, 100e3, 110e3, 190e3, 200e3, 300e3, 310e3, 500e3]
gains = [1, 1, 0, 0, 0.5, 0.5, 0, 0]
h2 = firwin2(101, freqs, gains, fs=sample_rate)

x_numpy = np.convolve(h2, x)
x_scipy = convolve(h2, x) # scipy convolve
x_fft_convolve = fftconvolve(h2, x)
x_lfilter = lfilter(h2, 1, x) # second arg is always 1 for FIR filters.

print(x_numpy[0:2])
print(x_scipy[0:2])
print(x_fft_convolve[0:2])
print(x_lfilter[0:2])

Ng = 100000 # signal length
x = np.random.randn(Ng) + 1j * np.random.randn(Ng)
PSD_input = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(x))**2)/len(x)) #saving the PSD of the input signal
x = fftconvolve(x, h2, 'same') # applying the filter

PSD_output = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(x))**2)/len(x))
f = np.linspace(-sample_rate/2/1e6, sample_rate/2/1e6, len(PSD_output))
plt.plot(f, PSD_input, alpha=0.8)
plt.plot(f, PSD_output, alpha=0.8)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.axis([-sample_rate/2/1e6, sample_rate/2/1e6, -40, 20])
plt.legend(['Input]', 'output'], loc=1)
plt.grid(True)
plt.show()


### For Stateful Filtering or Real-Time ##
'''
taps = 51
b = taps
a = 1 # 1 for FIR, but non-1 for IIR
zi = lfitler_zi(b, a) # calc initial conditions
while True:
    samples = sdr.read_samples(num_samples) # Replace with your SDR/RFSoC receive samples function
    samples_filtered, zi = lfitler(b, a, samples, zi=zi) # apply filter.
'''



