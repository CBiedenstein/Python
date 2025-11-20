import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

num_symbols = 10
sps = 8 # symbols per sample

bits = np.random.randint(0, 2, num_symbols) # our data to be transmitted 1's and 0's

x = np.array([])
for bit in bits:
    pulse = np.zeros(sps)
    pulse[0] = bit*2-1 # set the first value to either a 1 or -1
    x = np.concatenate((x, pulse)) # add the 8 samples to the signal
plt.figure(0)
plt.plot(x, '.-')
plt.grid(True)
plt.show()

# creating raised-cosine filter
num_taps = 101
beta = 0.35
Ts = sps # assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
t = np.arange(num_taps) - (num_taps-1)//2
h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)
plt.figure(1)
plt.plot(t, h, '.')
plt.grid(True)
plt.show()

# filter our signal, in order to apply the pulse shaping
x_shpaed = np.convolve(x, h)
plt.figure(2)
plt.plot(x_shpaed, '.-')
for i in range(num_symbols):
    plt.plot([i*sps+num_taps//2, i*sps+num_taps//2], [0, x_shpaed[i*sps+num_taps//2]], linestyle='--', color='red')
plt.grid(True)
plt.show()






