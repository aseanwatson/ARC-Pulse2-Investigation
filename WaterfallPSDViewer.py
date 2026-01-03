from typing import Optional

from iq_samples import iq_samples

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scipy.signal.windows import blackmanharris

from sortedcontainers import SortedDict
from collections import OrderedDict

from concurrent.futures import ThreadPoolExecutor

import threading
import queue

import logging


class WaterfallPSDViewer:
    """
    Interactive SDR-style waterfall + PSD viewer.

    Features:
    - Pixel-driven waterfall (one FFT per vertical pixel)
    - PSDs computed at arbitrary sample indices
    - LRU PSD cache with nearest-neighbor fallback
    - Synchronous cache warm-up on startup
    - One background job per redraw (row batching)
    - Background never overlaps with foreground
    - Sample-indexed zoom/pan
    - Resize-aware redraw
    """

    def __init__(self, samples: iq_samples, xw: float = 30e3) -> None:
        self.samples = samples
        self.fs: float = samples.fs
        self.fc: float = samples.fc
        self.data: np.ndarray = samples.data
        self.total_samples: int = samples.sample_count

        # DSP parameters
        self.fft_size: int = 1 << 14
        self.n_max: int = 50
        self.xscale: float = 1e6
        self.xunit: str = "MHz"

        # Frequency mask
        freqs_full = np.fft.fftshift(
            np.fft.fftfreq(self.fft_size, d=1 / self.fs)
        ) + self.fc
        freqs_scaled = freqs_full / self.xscale

        f_lo = (self.fc - xw) / self.xscale
        f_hi = (self.fc + xw) / self.xscale
        self.mask = (freqs_scaled >= f_lo) & (freqs_scaled <= f_hi)
        self.freqs: np.ndarray = freqs_scaled[self.mask]

        # Navigation state
        self.zoom_level: int = 0
        self.min_visible_sample: int = 0

        # PSD cache (LRU)
        self.psd_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max_cache_entries: int = 20000
        self.cache_lock = threading.Lock()

        # Nearest-neighbor fast path
        self.last_lookup_key: Optional[int] = None
        self.last_lookup_psd: Optional[np.ndarray] = None

        # Background job control
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.background_lock = threading.Lock()
        self.background_busy = False
        self.background_cancel = False
        self.needs_redraw = False

        # UI
        self._build_figure()
        self._connect_events()
        self._pan = {"press_sample": None, "orig_min": None}

        # Synchronous warm-up
        self._warm_up_initial_cache()

        # Timer for background redraw
        timer = self.fig.canvas.new_timer(interval=50)
        timer.add_callback(self._check_redraw)
        timer.start()

        # Initial draw
        self.render_waterfall()
        self.draw_frame_from_sample(0)

        plt.show()

    # ------------------------------------------------------------
    # Synchronous warm-up
    # ------------------------------------------------------------
    def _warm_up_initial_cache(self) -> None:
        """
        Fill the PSD cache synchronously for the initial visible window.
        Ensures the first render is fully populated.
        """
        vis = self.visible_samples()

        # Estimate pixel height before first draw
        # Use a reasonable default (e.g., 600px)
        height_px = 600

        samples_per_pixel = max(1, vis / height_px)
        max_start = max(0, self.total_samples - self.fft_size)

        row_samples = (
            self.min_visible_sample +
            np.arange(height_px) * samples_per_pixel
        ).astype(int)
        row_samples = np.clip(row_samples, 0, max_start)

        for s in row_samples:
            if s not in self.psd_cache:
                chunk = self.data[s:s + self.fft_size]
                if len(chunk) < self.fft_size:
                    continue

                windowed = chunk * blackmanharris(len(chunk))
                psd_full = 20 * np.log10(
                    np.abs(np.fft.fftshift(np.fft.fft(windowed))) + 1e-12
                )
                psd = psd_full[self.mask]

                self.psd_cache[s] = psd
                self.psd_cache.move_to_end(s)

                if len(self.psd_cache) > self.max_cache_entries:
                    self.psd_cache.popitem(last=False)

        # Initialize nearest-neighbor state
        if len(self.psd_cache) > 0:
            last_key = next(reversed(self.psd_cache))
            self.last_lookup_key = last_key
            self.last_lookup_psd = self.psd_cache[last_key]

    # ------------------------------------------------------------
    # Foreground PSD lookup
    # ------------------------------------------------------------
    def _foreground_psd(self, start_sample: int) -> np.ndarray:
        """
        Return PSD from cache if available.
        Otherwise return nearest cached PSD or zeros.
        """
        max_start = max(0, self.total_samples - self.fft_size)
        start_sample = max(0, min(start_sample, max_start))

        with self.cache_lock:
            # Exact hit
            if start_sample in self.psd_cache:
                psd = self.psd_cache.pop(start_sample)
                self.psd_cache[start_sample] = psd
                self.last_lookup_key = start_sample
                self.last_lookup_psd = psd
                return psd

            # Fast-path nearest
            if (
                self.last_lookup_key is not None
                and abs(start_sample - self.last_lookup_key) < 2000
            ):
                return self.last_lookup_psd

            # Full nearest scan
            if len(self.psd_cache) > 0:
                nearest_key = min(
                    self.psd_cache.keys(),
                    key=lambda k: abs(k - start_sample),
                )
                nearest_psd = self.psd_cache[nearest_key]
                self.last_lookup_key = nearest_key
                self.last_lookup_psd = nearest_psd
                return nearest_psd

        return np.zeros_like(self.freqs)

    # ------------------------------------------------------------
    # Background job: compute all PSDs for current view
    # ------------------------------------------------------------
    def _background_fill(self, row_samples: np.ndarray) -> None:
        if not self.background_lock.acquire(blocking=False):
            return

        self.background_busy = True
        self.background_cancel = False

        try:
            for s in row_samples:
                if self.background_cancel:
                    return

                with self.cache_lock:
                    if s in self.psd_cache:
                        psd = self.psd_cache.pop(s)
                        self.psd_cache[s] = psd
                        continue

                # Compute PSD
                chunk = self.data[s:s + self.fft_size]
                if len(chunk) < self.fft_size:
                    continue

                windowed = chunk * blackmanharris(len(chunk))
                psd_full = 20 * np.log10(
                    np.abs(np.fft.fftshift(np.fft.fft(windowed))) + 1e-12
                )
                psd = psd_full[self.mask]

                with self.cache_lock:
                    self.psd_cache[s] = psd
                    self.psd_cache.move_to_end(s)

                    while len(self.psd_cache) > self.max_cache_entries:
                        self.psd_cache.popitem(last=False)

            self.needs_redraw = True

        finally:
            self.background_busy = False
            self.background_lock.release()

    # ------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------
    def _check_redraw(self) -> None:
        if self.needs_redraw:
            self.needs_redraw = False
            self.render_waterfall()

    # ------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------
    def _build_figure(self) -> None:
        self.fig: Figure = plt.figure(figsize=(10, 8))
        if hasattr(self.fig.canvas, "toolbar_visible"):
            self.fig.canvas.toolbar_visible = False

        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1.5])
        self.ax_psd: Axes = self.fig.add_subplot(gs[0])
        self.ax_wf: Axes = self.fig.add_subplot(gs[1])

        self.line, = self.ax_psd.plot([], [], lw=1, label="Current PSD")
        self.max_line, = self.ax_psd.plot(
            [], [], lw=1, color="red", alpha=0.6, label=f"Max over last {self.n_max}"
        )
        self.time_text = self.ax_psd.text(
            0.02, 0.95, "", transform=self.ax_psd.transAxes,
            fontsize=10, color="gray"
        )

        self.ax_psd.legend()
        self.ax_psd.set_xlabel(f"Frequency ({self.xunit})")
        self.ax_psd.set_ylabel("Power (dB)")
        self.ax_psd.set_xlim(self.freqs[0], self.freqs[-1])

        self.im = self.ax_wf.imshow(
            np.zeros((2, len(self.freqs))),
            aspect="auto",
            origin="lower",
            extent=[self.freqs[0], self.freqs[-1], 0, 1],
            cmap="viridis",
        )

        self.ax_wf.set_title("Waterfall (dynamic)")
        self.ax_wf.set_xlabel(f"Frequency ({self.xunit})")
        self.ax_wf.set_ylabel("Time (ms)")

    # ------------------------------------------------------------
    # Event connections
    # ------------------------------------------------------------
    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

        self.ax_wf.callbacks.connect("xlim_changed", self.on_xlim_changed)
        self.ax_psd.callbacks.connect("xlim_changed", self.on_xlim_changed)

    # ------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------
    def visible_samples(self) -> int:
        return max(1, int(self.total_samples * pow(0.9, self.zoom_level)))

    def set_min_visible_sample(self, min_sample: int) -> None:
        vis = self.visible_samples()
        min_sample = max(0, min(min_sample, self.total_samples - vis))
        self.min_visible_sample = int(min_sample)

    def set_zoom_level(self, delta: int, fixed_sample: int) -> None:
        old_vis = self.visible_samples()
        self.zoom_level = max(0, self.zoom_level + delta)
        new_vis = self.visible_samples()

        old_offset = fixed_sample - self.min_visible_sample
        zoom_ratio = new_vis / old_vis if old_vis > 0 else 1.0
        new_offset = int(old_offset * zoom_ratio)

        new_min = fixed_sample - new_offset
        self.set_min_visible_sample(new_min)

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
    def render_waterfall(self) -> None:
        vis = self.visible_samples()
        y0 = (self.min_visible_sample / self.fs) * 1e3
        y1 = ((self.min_visible_sample + vis) / self.fs) * 1e3

        self.ax_wf._updating_limits = True
        self.ax_wf.set_ylim(y0, y1)
        del self.ax_wf._updating_limits

        height_px = int(self.im.get_window_extent().height)
        if height_px < 2:
            return

        samples_per_pixel = max(1, vis / height_px)
        max_start = max(0, self.total_samples - self.fft_size)

        row_samples = (
            self.min_visible_sample +
            np.arange(height_px) * samples_per_pixel
        ).astype(int)
        row_samples = np.clip(row_samples, 0, max_start)

        # Foreground: approximate PSDs
        rows = [self._foreground_psd(s) for s in row_samples]
        block = np.vstack(rows)

        vmin = np.percentile(block, 5)
        vmax = np.percentile(block, 95)
        if vmax - vmin < 1e-2:
            vmax = vmin + 1e-2

        self.im.set_clim(vmin, vmax)
        self.ax_psd.set_ylim(vmin, vmax)

        self.im.set_data(block)
        self.im.set_extent([self.freqs[0], self.freqs[-1], y0, y1])
        self.fig.canvas.draw_idle()

        # Cancel any previous background job
        self.background_cancel = True

        # Schedule new background job
        self.executor.submit(self._background_fill, row_samples)

    # ------------------------------------------------------------
    # PSD panel update
    # ------------------------------------------------------------
    def draw_frame_from_sample(self, sample_index: int) -> None:
        max_start = max(0, self.total_samples - self.fft_size)
        start_sample = max(0, min(int(sample_index), max_start))

        psd = self._foreground_psd(start_sample)
        self.line.set_data(self.freqs, psd)

        step = max(1, self.fft_size // 2)
        starts = np.arange(
            max(0, start_sample - step * (self.n_max - 1)),
            start_sample + step,
            step,
            dtype=int,
        )
        starts = np.clip(starts, 0, max_start)

        max_rows = [self._foreground_psd(int(s)) for s in starts]
        max_psd = np.max(np.vstack(max_rows), axis=0)
        self.max_line.set_data(self.freqs, max_psd)

        t_ms = (sample_index / self.fs) * 1e3
        self.time_text.set_text(f"Time: {t_ms:6.1f} ms; start: {start_sample}")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------
    def on_click(self, event) -> None:
        if event.inaxes != self.ax_wf or event.ydata is None:
            return
        sample = int((event.ydata / 1e3) * self.fs)
        sample = max(0, min(sample, self.total_samples - 1))
        self.draw_frame_from_sample(sample)

    def on_scroll(self, event) -> None:
        if event.inaxes != self.ax_wf or event.ydata is None:
            return
        fixed_sample = int((event.ydata / 1e3) * self.fs)
        fixed_sample = max(0, min(fixed_sample, self.total_samples - 1))
        delta = +1 if event.button == "up" else -1
        self.set_zoom_level(delta, fixed_sample)
        self.render_waterfall()

    def on_press(self, event) -> None:
        if event.inaxes == self.ax_wf and event.button == 1 and event.ydata is not None:
            self._pan["press_sample"] = int((event.ydata / 1e3) * self.fs)
            self._pan["orig_min"] = self.min_visible_sample

    def on_motion(self, event) -> None:
        if (
            self._pan["press_sample"] is None
            or event.inaxes != self.ax_wf
            or event.ydata is None
        ):
            return
        current_sample = int((event.ydata / 1e3) * self.fs)
        delta = current_sample - self._pan["press_sample"]
        new_min = self._pan["orig_min"] - delta
        self.set_min_visible_sample(new_min)
        self.render_waterfall()

    def on_release(self, event) -> None:
        self._pan["press_sample"] = None

    def on_resize(self, event) -> None:
        self.render_waterfall()

    def on_xlim_changed(self, event_ax: Axes) -> None:
        if event_ax in (self.ax_wf, self.ax_psd):
            if event_ax.get_xlim() != (self.freqs[0], self.freqs[-1]):
                event_ax.set_xlim(self.freqs[0], self.freqs[-1])