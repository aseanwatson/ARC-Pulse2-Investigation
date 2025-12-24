#!/bin/env python
import numpy as np

from scipy.signal import firwin, lfilter
from scipy.signal.windows import blackmanharris

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fc = 433.92e6
fc_capture = 433.125e6
fd = 200e3
fs = 20e6  # sample rate
fft_size = 32768 // 16
hop_size = fft_size // 2  # overlap
n_max = 10  # number of frames to track
decim = 40

numtaps = 801  # long enough for steep transition

def load_hackrf_iq(path):
    raw = np.fromfile(path, dtype=np.int8)
    print(f'loaded {len(raw)} samples from {path}:')
    iq = raw.reshape(-1, 2)
    return (iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)) / 128.0

def save_to_gqrx_float32(x, base):
    path = f'generated/{base}.cf32'
    print(f'saving {len(x)} samples to {path}:')
    interleaved = np.empty(2 * len(x), dtype=np.float32)
    interleaved[0::2] = x.real.astype(np.float32)
    interleaved[1::2] = x.imag.astype(np.float32)
    interleaved.tofile(path)

samples = load_hackrf_iq('data/remote_dr2_f433125000_s2000000_a0_l16_g2.iq')
save_to_gqrx_float32(samples, 'raw')
# shift from fc_capture to fc
samples = samples * np.exp(-2j * np.pi * (fc - fc_capture) * np.arange(len(samples)) / fs)
save_to_gqrx_float32(samples, 'shifted')
#dc correct by removing the mean
samples=samples-np.mean(samples)
save_to_gqrx_float32(samples, 'dc_corrected')

# do a low-pass filter to focus on the signal
lp_taps = firwin(
    numtaps,
    cutoff=fd/2,
    fs=fs,
    window="blackmanharris"
)
samples = lfilter(lp_taps, 1.0, samples)
save_to_gqrx_float32(samples, 'filtered')

samples = samples[::decim]
fs = fs / decim

save_to_gqrx_float32(samples, 'decimated')

psd_history = []

xscale=1e6
xunit='MHz'

# Prepare plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=1, label="Current PSD")
max_line, = ax.plot([], [], lw=1, color='red', alpha=0.6, label=f"Max over last {n_max}")
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='gray')
ax.legend()
ax.set_ylim(-60, 0)
ax.set_xlabel(f"Frequency ({xunit})")
ax.set_ylabel("Power (dB)")
ax.set_title("Live PSD")

# Frequency axis centered at fc
freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/fs)) + fc

# Set axis limits to fc ± fd/2
ax.set_xlim((fc - fd/2)/xscale, (fc + fd/2)/xscale)

# Animation update function
def update(frame):
    start = frame * hop_size
    chunk = samples[start:start+fft_size]
    current_time = start * 1e3 / fs  # in ms
    time_text.set_text(f"Time: {current_time: 4f}ms; frame: {frame:04d}")
    if len(chunk) < fft_size:
        return line,
    windowed = chunk * blackmanharris(len(chunk))
    psd = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(windowed))) + 1e-12)
    # Update history
    psd_history.append(psd)
    if len(psd_history) > n_max:
        psd_history.pop(0)

    max_psd = np.max(psd_history, axis=0)

    line.set_data(freqs/xscale, psd)
    max_line.set_data(freqs/xscale, max_psd)
    return line, max_line, time_text

# Animate
frames = (len(samples) - fft_size) // hop_size
ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
plt.show()
