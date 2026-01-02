from typing import Optional

from iq_samples import iq_samples


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal.windows import blackmanharris


import logging
from functools import lru_cache


class WaterfallPSDViewer:
    """
    Interactive SDR-style waterfall + PSD viewer.

    All navigation (zoom, pan, click-to-select) is expressed in terms of
    integer sample indices. The y-axis is derived from the current visible
    sample window and is never used as the source of truth.

    Features:
    - Pixel-driven dynamic waterfall rendering
    - LRU-cached PSD frames
    - Sample-indexed zoom (mouse wheel)
    - Sample-indexed pan (click-drag)
    - Click-to-select PSD
    - Frequency axis locked
    - Resize-aware redraw
    """

    def __init__(self, samples: "iq_samples", xw: float = 30e3) -> None:
        """
        Initialize the viewer and open the interactive window.

        Parameters
        ----------
        samples : iq_samples
            IQ buffer with fields: data, fs, fc, sample_count.
        xw : float
            Half-width of frequency span to display (Hz).
        """
        self.samples = samples
        self.fs: float = samples.fs
        self.fc: float = samples.fc
        self.data: np.ndarray = samples.data
        self.total_samples: int = samples.sample_count

        # DSP parameters
        self.fft_size: int = 1 << 14
        self.hop_size: int = self.fft_size // 64
        self.n_max: int = 50
        self.xscale: float = 1e6
        self.xunit: str = "MHz"

        # Frequency mask
        freqs_full = np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1/self.fs)) + self.fc
        freqs_scaled = freqs_full / self.xscale

        f_lo = (self.fc - xw) / self.xscale
        f_hi = (self.fc + xw) / self.xscale
        self.mask = (freqs_scaled >= f_lo) & (freqs_scaled <= f_hi)
        self.freqs: np.ndarray = freqs_scaled[self.mask]

        # Frame count
        self.max_frames: int = (self.total_samples - self.fft_size) // self.hop_size

        # Navigation state (sample-indexed)
        self.zoom_level: int = 0
        self.min_visible_sample: int = 0

        # Build UI
        self._build_figure()
        self._connect_events()

        # Initial view = full capture
        self.render_waterfall()
        self.draw_frame_from_sample(0)

        plt.show()

    # ------------------------------------------------------------
    # PSD computation (LRU cached)
    # ------------------------------------------------------------
    @lru_cache(maxsize=5000)
    def compute_psd_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Compute PSD for a single STFT frame, masked to the zoomed frequency band.

        Parameters
        ----------
        frame_index : int
            Index of the STFT frame.

        Returns
        -------
        np.ndarray or None
            PSD slice for the masked frequency region.
        """
        logging.info(
            f"computing psd for frame {frame_index} "
            f"(sample_index = {frame_index * self.hop_size}; "
            f"time = {(frame_index * self.hop_size)/self.fs})"
        )

        if frame_index < 0 or frame_index >= self.max_frames:
            return None

        start = frame_index * self.hop_size
        chunk = self.data[start:start+self.fft_size]
        if len(chunk) < self.fft_size:
            return None

        windowed = chunk * blackmanharris(len(chunk))
        psd_full = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(windowed))) + 1e-12)
        return psd_full[self.mask]

    # ------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------
    def _build_figure(self) -> None:
        """Create the Matplotlib figure and axes."""
        self.fig: Figure = plt.figure(figsize=(10, 8))
        self.fig.canvas.toolbar_visible = False

        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1.5])
        self.ax_psd: Axes = self.fig.add_subplot(gs[0])
        self.ax_wf: Axes = self.fig.add_subplot(gs[1])

        # PSD plot
        self.line, = self.ax_psd.plot([], [], lw=1, label="Current PSD")
        self.max_line, = self.ax_psd.plot([], [], lw=1, color='red', alpha=0.6,
                                          label=f"Max over last {self.n_max}")
        self.time_text = self.ax_psd.text(0.02, 0.95, '', transform=self.ax_psd.transAxes,
                                          fontsize=10, color='gray')

        self.ax_psd.legend()
        self.ax_psd.set_xlabel(f"Frequency ({self.xunit})")
        self.ax_psd.set_ylabel("Power (dB)")
        self.ax_psd.set_xlim(self.freqs[0], self.freqs[-1])

        # Waterfall placeholder
        self.im = self.ax_wf.imshow(
            np.zeros((2, len(self.freqs))),
            aspect='auto',
            origin='lower',
            extent=[self.freqs[0], self.freqs[-1], 0, 1],
            cmap='viridis'
        )

        self.ax_wf.set_title("Waterfall (dynamic)")
        self.ax_wf.set_xlabel(f"Frequency ({self.xunit})")
        self.ax_wf.set_ylabel("Time (ms)")

    # ------------------------------------------------------------
    # Event connections
    # ------------------------------------------------------------
    def _connect_events(self) -> None:
        """Connect all Matplotlib event handlers."""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)

        self.ax_wf.callbacks.connect("xlim_changed", self.on_xlim_changed)
        self.ax_psd.callbacks.connect("xlim_changed", self.on_xlim_changed)

        self._pan = {"press_sample": None, "orig_min": None}

    # ------------------------------------------------------------
    # Navigation helpers (sample-indexed)
    # ------------------------------------------------------------
    def visible_samples(self) -> int:
        """
        Compute number of samples visible at current zoom level.

        Returns
        -------
        int
            Number of samples in the visible window.
        """
        return max(1, int(self.total_samples * pow(0.9, self.zoom_level)))

    def set_min_visible_sample(self, min_sample: int) -> None:
        """
        Clamp and update the minimum visible sample index.

        Parameters
        ----------
        min_sample : int
            Proposed minimum sample index.
        """
        vis = self.visible_samples()
        min_sample = max(0, min(min_sample, self.total_samples - vis))
        self.min_visible_sample = int(min_sample)

    def set_zoom_level(self, delta: int, fixed_sample: int) -> None:
        """
        Adjust zoom level and anchor around a fixed sample index.

        Parameters
        ----------
        delta : int
            Change in zoom level (+1 zoom in, -1 zoom out).
        fixed_sample : int
            Sample index to keep visually anchored during zoom.
        """
        old_vis = self.visible_samples()

        # Update zoom level
        self.zoom_level = max(0, self.zoom_level + delta)
        new_vis = self.visible_samples()

        # Compute new min_visible_sample so fixed_sample stays anchored
        old_offset = fixed_sample - self.min_visible_sample
        zoom_ratio = new_vis / old_vis
        new_offset = int(old_offset * zoom_ratio)

        new_min = fixed_sample - new_offset
        self.set_min_visible_sample(new_min)

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
    def render_waterfall(self) -> None:
        """
        Render the waterfall image based on current sample-indexed view state.
        """
        vis = self.visible_samples()
        y0 = (self.min_visible_sample / self.fs) * 1e3
        y1 = ((self.min_visible_sample + vis) / self.fs) * 1e3

        # Update y-limits
        self.ax_wf._updating_limits = True
        self.ax_wf.set_ylim(y0, y1)
        del self.ax_wf._updating_limits

        # Pixel height
        height_px = int(self.im.get_window_extent().height)
        if height_px < 2:
            return

        # Samples per pixel
        samples_per_pixel = vis / height_px
        samples_per_pixel = max(1, samples_per_pixel)

        # Sample index for each row
        row_samples = (
            self.min_visible_sample +
            np.arange(height_px) * samples_per_pixel
        ).astype(int)

        row_samples = np.clip(row_samples, 0, self.total_samples - 1)
        frame_indices = row_samples // self.hop_size
        frame_indices = np.clip(frame_indices, 0, self.max_frames - 1)

        # Compute PSD rows
        rows = []
        for f in frame_indices:
            psd = self.compute_psd_frame(f)
            if psd is None:
                psd = np.zeros_like(self.freqs)
            rows.append(psd)

        block = np.vstack(rows)

        # Contrast
        vmin = np.percentile(block, 5)
        vmax = np.percentile(block, 95)
        if vmax - vmin < 1e-2:
            vmax = vmin + 1e-2

        self.im.set_clim(vmin, vmax)
        self.ax_psd.set_ylim(vmin, vmax)

        self.im.set_data(block)
        self.im.set_extent([self.freqs[0], self.freqs[-1], y0, y1])
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------
    # PSD update
    # ------------------------------------------------------------
    def draw_frame_from_sample(self, sample_index: int) -> None:
        """
        Update the PSD panel for a given sample index.

        Parameters
        ----------
        sample_index : int
            Sample index to display PSD for.
        """
        frame = sample_index // self.hop_size
        frame = max(0, min(frame, self.max_frames - 1))

        psd = self.compute_psd_frame(frame)
        if psd is None:
            return

        self.line.set_data(self.freqs, psd)

        start = max(0, frame - self.n_max + 1)
        max_psd = np.max(
            [self.compute_psd_frame(f) for f in range(start, frame+1)],
            axis=0
        )
        self.max_line.set_data(self.freqs, max_psd)

        t_ms = (sample_index / self.fs) * 1e3
        self.time_text.set_text(f"Time: {t_ms:6.1f} ms; frame: {frame}")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------
    def on_click(self, event) -> None:
        """Click-to-select PSD."""
        if event.inaxes != self.ax_wf or event.ydata is None:
            return
        sample = int((event.ydata / 1e3) * self.fs)
        self.draw_frame_from_sample(sample)

    def on_scroll(self, event) -> None:
        """Mouse wheel zoom (sample-indexed)."""
        if event.inaxes != self.ax_wf:
            return

        fixed_sample = int((event.ydata / 1e3) * self.fs)
        fixed_sample = max(0, min(fixed_sample, self.total_samples - 1))

        delta = +1 if event.button == 'up' else -1
        self.set_zoom_level(delta, fixed_sample)
        self.render_waterfall()

    def on_press(self, event) -> None:
        """Start panning."""
        if event.inaxes == self.ax_wf and event.button == 1:
            self._pan["press_sample"] = int((event.ydata / 1e3) * self.fs)
            self._pan["orig_min"] = self.min_visible_sample

    def on_motion(self, event) -> None:
        """Continue panning."""
        if self._pan["press_sample"] is None or event.inaxes != self.ax_wf:
            return

        current_sample = int((event.ydata / 1e3) * self.fs)
        delta = current_sample - self._pan["press_sample"]

        new_min = self._pan["orig_min"] - delta
        self.set_min_visible_sample(new_min)
        self.render_waterfall()

    def on_release(self, event) -> None:
        """End panning."""
        self._pan["press_sample"] = None

    def on_resize(self, event) -> None:
        """Re-render on window resize."""
        self.render_waterfall()

    def on_xlim_changed(self, event_ax: Axes) -> None:
        """Lock frequency axis for both PSD and waterfall."""
        if event_ax in (self.ax_wf, self.ax_psd):
            if event_ax.get_xlim() != (self.freqs[0], self.freqs[-1]):
                event_ax.set_xlim(self.freqs[0], self.freqs[-1])