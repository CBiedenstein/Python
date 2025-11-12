import numpy as np
import matplotlib.pyplot as plt

Fs = 300
Ts = 1/Fs # sample period
N = 2048 # sample size

t = Ts*np.arange(N)
x = np.exp(1j*2*np.pi*50*t) # simulates sinusoid at 50 Hz

n = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2) # complex noise with unity power
noise_power = 2
r = x + n * np.sqrt(noise_power)

PSD = np.abs(np.fft.fft(r)**2 / (N*Fs))
PSD_log = 10.0*np.log10(PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

f = np.arange(Fs/-2, Fs/2, Fs/N)

plt.plot(f, PSD_shifted)
plt.xlabel("frequency")
plt.ylabel("magnitude")
plt.grid(True)
plt.show()
