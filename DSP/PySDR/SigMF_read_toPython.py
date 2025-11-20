from sigmf import SigMFFile, sigmffile
import numpy as np

# load a dataset
filename = 'qpsk_in_noise.sigmf-meta'
signal = sigmffile.fromfile(filename)
samples = signal.read_samples().view(np.complex64).flatten()
print(samples[0:10]) # looking at first 10 samples

# get some metadata and all annotations
sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
sample_count = signal.sample_count
signal_duration = sample_count/sample_rate