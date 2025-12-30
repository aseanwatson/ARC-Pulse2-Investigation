#!/bin/env python
import numpy as np
from numpy.typing import NDArray

from typing import Optional

from scipy.signal import firwin, lfilter
from scipy.signal.windows import blackmanharris

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

fc = 433.92e6
fd = 80e3
fft_size = 32768
hop_size = fft_size // 8  # overlap
n_max = 100  # number of frames to track
decim = 2

numtaps = 801  # long enough for steep transition

class iq_samples:
    def __init__(self, data, fs, fc):
        self.data = data
        self.fs = fs
        self.fc = fc

    @property
    def sample_count(self):
        return len(self.data)

    def _modified(self, data = None, fs = None, fc = None):
        if data is None:
            data = self.data
        if fs is None:
            fs = self.fs
        if fc is None:
            fc = self.fc
        return iq_samples(data = data, fs = fs, fc = fc)

    def recenter(self, fc_new:float) -> 'iq_samples':
        """
        Shifts samples so a new frequency is at the center
        
        :param self: Description
        :param fc_new: Description
        :type fc_new: float
        :return: Description
        :rtype: iq_samples
        """
        return self._modified(
            data = self.data * np.exp(-2j * np.pi * (fc_new - self.fc) * np.arange(self.sample_count) / self.fs),
            fc = fc_new)

    def dc_correct(self) -> 'iq_samples':
        """
        Subtracts the mean to remove DC bias.
        """
        return self._modified(data = self.data - np.mean(self.data))

    def normalize_percentile(self, p: float = 95.0, *, min_threshold: float = 0.0, min_threshold_percentile: float = 0.0) -> 'iq_samples':
        """
        Normalizes the samples so the absolute value is 1.0 at p95 (or some other value), excluding values less than some value (given as an absolute value or a percentile).
        
        :param p: Percentile to normalize to 1
        :type p: float
        :param min_threshold: Value below which does not count in the percentile.
        :type min_threshold: float
        :param min_threshold_percentile: Global percentile to not count
        :type min_threshold_percentile: float
        :return: An `iq_samples` with the normalization applied
        :rtype: iq_samples
        """
        trimmed_data = np.abs(self.data)

        if min_threshold_percentile > 0.0:
            if min_threshold > 0.0:
                raise ValueError("min_threshold must be zero if min_threshold_percentiles is positive")
            min_threshold = np.nanpercentile(trimmed_data, min_threshold_percentile)

        if min_threshold > 0.0:
            trimmed_data = trimmed_data[trimmed_data <= min_threshold]

        if trimmed_data.size == 0:
            raise ValueError("No data left after thresholding; cannot compute percentile.")

        scale = np.nanpercentile(trimmed_data, p)
        if scale == 0:
            raise ValueError("Percentile value is zero; cannot normalize.")

        return self._modified(data=self.data / scale)

    def low_pass(self, numtaps:int, bandwidth:float) -> 'iq_samples':
        """
        Applies an FIR low-pass filter around the center frequency.
        
        :param numtaps: Number of coefficients in the filter. (See 'numtaps' in 'firwin')
        :type numtaps: int
        :param bandwidth: Width of the frequency band to pass.
        :type bandwidth: float
        :return: An `iq_samples` with the filter applied.
        :rtype: iq_samples
        """
        lp_taps = firwin(
            numtaps = numtaps,
            cutoff = bandwidth / 2,
            fs = self.fs,
            window = "blackmanharris")

        return self._modified(
            data = lfilter(lp_taps, 1.0, self.data))

    def decimate(self, decimation_factor:int) -> 'iq_samples':
        """
        Resamples the iq_samples, keeping only one sample per `decimation_factor` samples.
        
        :param decimation_factor: Number of samples to reduce to one.
        :type decimation_factor: int
        :return: The decimated `iq_samples`
        :rtype: iq_samples
        """
        return self._modified(data = self.data[::decimation_factor], fs=self.fs/decimation_factor)

    def save_to_cf32(self, base):
        """
        Saves samples as a cf32 file. This is used for input to gqrx or inspectrum, for example.
        
        :param base: Basename to save to.
        """
        path = f'generated/{base}.cf32'
        logging.info(f'saving {self.sample_count} samples to {path}:')
        interleaved = np.empty(2 * self.sample_count, dtype=np.float32)
        interleaved[0::2] = self.data.real.astype(np.float32)
        interleaved[1::2] = self.data.imag.astype(np.float32)
        interleaved.tofile(path)

    @staticmethod
    def load_int8(path, fs:float, fc:float) -> 'iq_samples':
        """
        Docstring for load_int8
        
        :param path: Description
        :param fs: Description
        :type fs: float
        :param fc: Description
        :type fc: float
        :return: Description
        :rtype: iq_samples
        """
        raw = np.fromfile(path, dtype=np.int8)
        logging.info(f'loaded {len(raw)//2} samples from {path}:')
        iq = raw.reshape(-1, 2)
        return iq_samples(
            data = (iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)) / 128.0,
            fs = fs,
            fc = fc)

samples = iq_samples.load_int8(
    path = 'data/remote_dr2_f433125000_s2000000_a0_l16_g2.iq',
    fs = 20e6,
    fc = 433.125e6)

samples.save_to_cf32('raw')
# shift from fc_capture to fc
samples = samples.recenter(fc)
samples.save_to_cf32('shifted')
#dc correct by removing the mean
samples=samples.dc_correct()
samples.save_to_cf32('dc_corrected')

# do a low-pass filter to focus on the signal
samples = samples.low_pass(numtaps=numtaps, bandwidth=fd)
samples.save_to_cf32('filtered')

samples=samples.normalize_percentile(95, min_threshold_percentile=10)
samples.save_to_cf32('normalized')

samples = samples.decimate(decim)
samples.save_to_cf32('decimated')
fft_size //= decim
hop_size //= decim
n_max //= decim

psd_history = []
# ------------------------------------------------------------
# Precompute PSD frames
# ------------------------------------------------------------

frames = (samples.sample_count - fft_size) // hop_size

psd_frames = []
for frame in range(frames):
    if frame % 1000 == 0:
        logging.info(f'Precomputing frame {frame}/{frames}')
    start = frame * hop_size
    chunk = samples.data[start:start+fft_size]
    if len(chunk) < fft_size:
        break
    windowed = chunk * blackmanharris(len(chunk))
    psd = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(windowed))) + 1e-12)
    psd_frames.append(psd)

psd_frames = np.array(psd_frames)
frames = len(psd_frames)

xscale=1e6
xunit='MHz'

# Prepare plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=1, label="Current PSD")
max_line, = ax.plot([], [], lw=1, color='red', alpha=0.6, label=f"Max over last {n_max}")
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='gray')
ax.legend()
ax.set_ylim(50, 120)
ax.set_xlabel(f"Frequency ({xunit})")
ax.set_ylabel("Power (dB)")
ax.set_title("Live PSD")

# Frequency axis centered at fc
freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/samples.fs)) + samples.fc

# Set axis limits to fc ± fd/2
ax.set_xlim((samples.fc - fd*3/2)/xscale, (samples.fc + fd*3/2)/xscale)

# Animation update function
def update(frame):
    start = frame * hop_size
    current_time = start * 1e3 / samples.fs  # in ms
    time_text.set_text(f"Time: {current_time: 4.0f}ms; frame: {frame: 4d}")

    psd = psd_frames[frame]
    line.set_data(freqs/xscale, psd)

    # Max over last n_max frames
    start = max(0, frame - n_max + 1)
    max_psd = np.max(psd_frames[start:frame+1], axis=0)
    max_line.set_data(freqs/xscale, max_psd)

    return line, max_line, time_text

# Animate
frames = (samples.sample_count - fft_size) // hop_size
ani = FuncAnimation(fig, update, frames=frames, interval=5, blit=True)
plt.show()