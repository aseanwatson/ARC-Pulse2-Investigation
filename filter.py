#!/bin/env python
import numpy as np

import matplotlib.pyplot as plt

import logging

from WaterfallPSDViewer import WaterfallPSDViewer
from iq_samples import iq_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

fc = 433.92e6
fc = 433.9214288e6 # read from waterfall
fd = 10e3
fft_size = 1 << 14
hop_size = fft_size // 2  # overlap
n_max = 100  # number of frames to track
decim = 8

numtaps = 9001  # long enough for steep transition

samples = iq_samples.load_int8(
    path = 'data/remote_dr2_f433125000_s2000000_a0_l16_g2.iq',
    fs = 20e6,
    fc = 433.125e6)

samples.save_to_cf32('raw')

# focus on 200ms to 500ms
samples = samples.time_slice(0.215, 0.340)
samples.save_to_cf32('trimmed')

# samples = samples.decimate(decim)
# samples.save_to_cf32('decimated')
# fft_size //= decim
# hop_size //= decim
# n_max //= decim

# shift from fc_capture to fc
samples = samples.recenter(fc)
samples.save_to_cf32('shifted')

#dc correct by removing the mean
samples=samples.dc_correct()
samples.save_to_cf32('dc_corrected')

# do a low-pass filter to focus on the signal
samples = samples.low_pass(numtaps=numtaps, cutoff=fd)
samples.save_to_cf32('filtered')

#samples=samples.normalize_percentile(95, min_threshold_percentile=10)
#samples.save_to_cf32('normalized')

WaterfallPSDViewer(samples)
#samples = samples.time_slice(0.05, 0.05+.105)
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(nrows=1)
ax_f_inst = fig.add_subplot(gs[0])
t = samples.time()
dphi = np.angle(samples.data[1:] * np.conjugate(samples.data[:-1]))
f_inst_raw = samples.fs / (np.pi * 2) * dphi
f_inst = f_inst_raw - np.mean(f_inst_raw)
ax_f_inst.scatter(t[:-1] * 1e3, f_inst, s=1) # plot in ms
plt.show()