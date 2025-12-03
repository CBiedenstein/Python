import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

num_symbols = 1000
sps = 8
bits = np.random.randint(0, 2, num_symbols)
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

# create and apply fractional delay filter
delay = 0.4 # fractional delay, in samples
N = 21 # number of taps keep this odd
n = np.arange(-(N-1)//2, N//2+1) # -10,-9,...,0,...,9,10
h = np.sinc(n -delay) # calculate filter taps
h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
samples = np.convolve(samples, h)
samples_interpolated = signal.resample_poly(samples, 16, 1)
# Mueller and Mueller Clock Recovery technique

mu = 0 # initial phase estimate of sample
out = np.zeros(len(samples) + 10, dtype=np.complex64)
out_rail = np.zeros(len(samples) + 10, dtype=np.complex64) # stores values, each iteration we need the previous 2 values plus current value
i_in = 0 # input samples index
i_out = 2 # output index (let first two outputs be 0)
while i_out < len(samples) and i_in+16 < len(samples):
    out[i_out] = samples_interpolated[i_in*16 + int(mu*16)] # grab what we think is the best sample
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j * int(np.imag(out[i_out]))
    x = (out_rail[i_out] -out_rail[i_out-2]) * np.conj(out[i_out-1])
    y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
    mm_val = np.real(y - x)
    mu += sps + 0.3*mm_val
    i_in += int(np.floor(mu)) # round down to the nearest int since we are using it as an index
    mu = mu - np.floor(mu) # remove the integer part of mu
    i_out += 1 #increment output index
out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
samples = out # only inlcude this line for connection to the costas loop 



# Plot the old vs the new
plt.figure('before interp')
plt.plot(samples, '.-')
plt.figure('after interp')
plt.plot(samples_interpolated, '.-')
plt.show()



