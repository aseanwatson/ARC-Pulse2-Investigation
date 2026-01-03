#!/bin/env python
import numpy as np

import matplotlib.pyplot as plt

import logging

from WaterfallPSDViewer import WaterfallPSDViewer
from IQSamples import IQSamples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

fc = 433.92e6
fc = 433.9214288e6 # read from waterfall
fd = 10e3

numtaps = 9001  # long enough for steep transition

samples = IQSamples.load_int8(
    path = 'data/remote_dr2_f433125000_s2000000_a0_l16_g2.iq',
    fs = 20e6,
    fc = 433.125e6)

samples.save_to_cf32('raw')

# focus on 200ms to 500ms
samples = samples.time_slice(0.215, 0.340)
samples.save_to_cf32('trimmed')

# samples = samples.decimate(8)
# samples.save_to_cf32('decimated')

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