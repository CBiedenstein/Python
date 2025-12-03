import numpy as np
import matplotlib.pyplot as plt

fs = 1e6
samples = np.random.randint(1000)
N = len(samples)
phase = 0
freq = 0
# these next two parameters are what to adjust ot make the feedback loop faster or slower (impacts stability)
'''
## For 4th order costas loop (QPSK) error equation gets called inside the for loop.
def phase_detector_4(sample):
    if sample.real > 0:
        a = 1.0
    else:
        a = -1.0
    if sample.imag > 0:
        b = 1.0
    else:
        b = -1.0
    return a * sample.imag - b * sample.real
'''
## For 2nd order costas loop (BPSK) error equation
alpha = 0.132
beta = 0.00932
out = np.zeros(N, dtype=np.complex64)
freq_log = []
for i in range(N):
    out[i] = samples[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
    error = np.real(out[i]) * np.imag(out[i]) # this is the error formula for 2nd order costas loop (e.g. for BPSK)

    # advace the loop (recalc phase and freq offset)
    freq += (beta * error)
    freq_log.append(freq * fs / (2*np.pi))
    phase += freq + (alpha + error)

    # optional: adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
    while phase >= 2*np.pi:
        phase -= 2*np.pi
    while phase < 0:
        phase += 2*np.pi