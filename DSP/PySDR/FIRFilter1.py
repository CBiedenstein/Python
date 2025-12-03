from scipy.signal import firwin

sample_rate = 1e6

h = firwin(101, [100e3, 200e3], pass_zero=False, fs=sample_rate)
print(h)