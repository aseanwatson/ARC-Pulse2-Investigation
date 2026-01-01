#!/bin/env python
import numpy as np
from numpy.typing import NDArray

from typing import Optional

from scipy.signal import firwin, lfilter
from scipy.signal.windows import blackmanharris

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, Cursor

import logging

from functools import lru_cache

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

class iq_samples:
    data: NDArray[np.complex64]
    fs: float
    fc: float

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

    def _time_to_sample(self, time: float) -> int:
        """
        Converts a time (in seconds) to a sample index
        """
        return int(time * self.fs)

    def _sample_to_time(self, index: int) -> float:
        """
        Converts an sample index to a time
        """
        return index / self.fs

    def time(self) -> NDArray[np.float64]:
        return np.arange(len(self.data)) / self.fs

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
            data = self.data * np.exp(-2j * np.pi * (fc_new - self.fc) * self.time()),
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

    def low_pass(self, numtaps:int, cutoff:float) -> 'iq_samples':
        """
        Applies an FIR low-pass filter around the center frequency.
        
        :param numtaps: Number of coefficients in the filter. (See 'numtaps' in 'firwin')
        :type numtaps: int
        :param cuttoff: Width of the frequency band to pass.
        :type cuttoff: float
        :return: An `iq_samples` with the filter applied.
        :rtype: iq_samples
        """
        lp_taps = firwin(
            numtaps = numtaps,
            cutoff = cutoff,
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

    def time_slice(self, start: float, end:float) -> 'iq_samples':
        start_sample = self._time_to_sample(start)
        end_sample = self._time_to_sample(end)
        return self._modified(data=self.data[start_sample:end_sample])

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

def draw_waterfall_and_psd(samples: iq_samples, xw=30e3):
    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------
    fft_size = 1 << 14
    hop_size = fft_size // 64
    n_max = 50
    xscale = 1e6
    xunit = "MHz"

    # ------------------------------------------------------------
    # Frequency mask for zoomed band
    # ------------------------------------------------------------
    freqs_full = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/samples.fs)) + samples.fc
    freqs_scaled_full = freqs_full / xscale

    f_lo = (samples.fc - xw) / xscale
    f_hi = (samples.fc + xw) / xscale
    mask = (freqs_scaled_full >= f_lo) & (freqs_scaled_full <= f_hi)
    freqs = freqs_scaled_full[mask]

    # Total number of possible frames
    max_frames = (samples.sample_count - fft_size) // hop_size

    # ------------------------------------------------------------
    # LRU‑cached PSD computation
    # ------------------------------------------------------------
    @lru_cache(maxsize=5000)
    def compute_psd_frame(frame_index: int):
        logging.info(
            f'computing psd for frame {frame_index} '
            f'(sample_index = {frame_index * hop_size}; '
            f'time = {(frame_index * hop_size)/samples.fs})'
        )

        if frame_index < 0 or frame_index >= max_frames:
            logging.info(f'frame {frame_index} out of valid range (0 - {max_frames})')
            return None

        start = frame_index * hop_size
        chunk = samples.data[start:start+fft_size]
        if len(chunk) < fft_size:
            return None

        windowed = chunk * blackmanharris(len(chunk))
        psd_full = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(windowed))) + 1e-12)
        return psd_full[mask]

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])

    ax_psd = fig.add_subplot(gs[0])
    ax_wf  = fig.add_subplot(gs[1])

    # PSD plot
    line, = ax_psd.plot([], [], lw=1, label="Current PSD")
    max_line, = ax_psd.plot([], [], lw=1, color='red', alpha=0.6,
                            label=f"Max over last {n_max}")
    time_text = ax_psd.text(0.02, 0.95, '', transform=ax_psd.transAxes,
                            fontsize=10, color='gray')

    ax_psd.legend()
    ax_psd.set_xlabel(f"Frequency ({xunit})")
    ax_psd.set_ylabel("Power (dB)")
    ax_psd.set_xlim(freqs[0], freqs[-1])

    # Waterfall image placeholder
    im = ax_wf.imshow(
        np.zeros((2, len(freqs))),  # temporary
        aspect='auto',
        origin='lower',
        extent=[freqs[0], freqs[-1], 0, 1],
        cmap='viridis'
    )

    ax_wf.set_title("Waterfall (dynamic)")
    ax_wf.set_xlabel(f"Frequency ({xunit})")
    ax_wf.set_ylabel("Time (ms)")

    # ------------------------------------------------------------
    # Pixel‑driven dynamic waterfall rendering
    # ------------------------------------------------------------
    def render_waterfall():
        ymin, ymax = ax_wf.get_ylim()

        # Number of vertical pixels
        height_px = int(im.get_window_extent().height)
        if height_px < 2:
            return

        # Time samples for each pixel row
        times_ms = np.linspace(ymin, ymax, height_px)

        # Convert to frame indices
        frame_indices = (times_ms / 1e3 * samples.fs / hop_size).astype(int)
        frame_indices = np.clip(frame_indices, 0, max_frames - 1)

        # Compute PSD rows
        rows = []
        for f in frame_indices:
            psd = compute_psd_frame(f)
            if psd is None:
                psd = np.zeros_like(freqs)
            rows.append(psd)

        block = np.vstack(rows)

        # Adaptive contrast
        vmin = np.percentile(block, 5)
        vmax = np.percentile(block, 95)
        if vmax - vmin < 1.0e-2:
            vmax = vmin + 1.0e-2

        im.set_clim(vmin, vmax)
        ax_psd.set_ylim(vmin, vmax)

        im.set_data(block)
        im.set_extent([freqs[0], freqs[-1], ymin, ymax])
        fig.canvas.draw_idle()

    # ------------------------------------------------------------
    # PSD update when clicking a time in the waterfall
    # ------------------------------------------------------------
    def draw_frame_from_time(t_ms):
        frame = int((t_ms / 1e3) * samples.fs / hop_size)
        frame = max(0, min(max_frames - 1, frame))

        psd = compute_psd_frame(frame)
        if psd is None:
            return

        line.set_data(freqs, psd)

        start = max(0, frame - n_max + 1)
        max_psd = np.max(
            [compute_psd_frame(f) for f in range(start, frame+1)],
            axis=0
        )
        max_line.set_data(freqs, max_psd)

        time_text.set_text(f"Time: {t_ms:6.1f} ms; frame: {frame}")
        fig.canvas.draw_idle()

    # ------------------------------------------------------------
    # Mouse click handler
    # ------------------------------------------------------------
    def on_click(event):
        if event.inaxes != ax_wf:
            return
        if event.ydata is None:
            return
        draw_frame_from_time(event.ydata)

    fig.canvas.mpl_connect('button_press_event', on_click)

    # ------------------------------------------------------------
    # Redraw waterfall on zoom/pan
    # ------------------------------------------------------------
    def on_ylimits_change(event_ax):
        if event_ax is not ax_wf:
            return

        if getattr(ax_wf, "_updating_limits", False):
            return

        render_waterfall()

    # ------------------------------------------------------------
    # Prevent x-axis scroll/zoom
    # ------------------------------------------------------------
    def on_xlim_changed(event_ax):
        if event_ax is ax_wf or event_ax is ax_psd:
            if event_ax.get_xlim() != (freqs[0], freqs[-1]):
                event_ax.set_xlim(freqs[0], freqs[-1])

    ax_wf.callbacks.connect("xlim_changed", on_xlim_changed)
    ax_psd.callbacks.connect("xlim_changed", on_xlim_changed)

    # ------------------------------------------------------------
    # Re-render on window resize (DPI or size change)
    # ------------------------------------------------------------
    def on_resize(event):
        render_waterfall()

    fig.canvas.mpl_connect('resize_event', on_resize)

    initial_window = samples.sample_count / samples.fs * 1e3

    # ------------------------------------------------------------
    # Initial view
    # ------------------------------------------------------------
    ax_wf._updating_limits = True
    ax_wf.set_ylim(0, initial_window)
    del ax_wf._updating_limits

    ax_wf.set_xlim(freqs[0], freqs[-1])
    ax_psd.set_xlim(freqs[0], freqs[-1])
    render_waterfall()
    draw_frame_from_time(0)

    plt.show()


draw_waterfall_and_psd(samples)
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