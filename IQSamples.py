import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union
from scipy.signal import firwin, lfilter

import logging

class BaseSamples:
    """Container for sample data with basic utilities.

    Attributes:
        data: Sample array (complex or real depending on subclass).
        fs: Sampling frequency in Hz.
    """

    def __init__(self, data: NDArray, fs: float) -> None:
        """Create a BaseSamples container.

        :param data: Array of samples.
        :param fs: Sampling frequency in Hz.
        """
        self.data: NDArray = data
        self.fs: float = fs

    def _modified(self, data: Optional[NDArray] = None, fs: Optional[float] = None):
        """Return a modified copy of this object (used by transformations).

        Returns an instance of the same runtime class (`self.__class__`).
        Subclasses may override when additional state (e.g. `fc`) must be preserved.
        """
        if data is None:
            data = self.data
        if fs is None:
            fs = self.fs
        return self.__class__(data=data, fs=fs)

    @property
    def sample_count(self):
        return len(self.data)

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
        """Return an array of time values (seconds) for each sample index."""
        return np.arange(len(self.data)) / self.fs

    def scale(self, factor: float) -> 'BaseSamples':
        """Scale sample values by `factor` and return a modified copy."""
        return self._modified(data=self.data * factor)

    def normalize_percentile(self, p: float = 95.0, *, min_threshold: float = 0.0, min_threshold_percentile: float = 0.0) -> 'BaseSamples':
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

        return self.scale(1.0 / scale)

    def decimate(self, decimation_factor: int) -> 'BaseSamples':
        """
        Resamples the iq_samples, keeping only one sample per `decimation_factor` samples.

        :param decimation_factor: Number of samples to reduce to one.
        :type decimation_factor: int
        :return: The decimated `iq_samples`
        :rtype: iq_samples
        """
        return self._modified(data=self.data[::decimation_factor], fs=self.fs / decimation_factor)

    def time_slice(self, start: float, end: float) -> 'BaseSamples':
        """Return a time-sliced view between `start` and `end` (seconds)."""
        start_sample = self._time_to_sample(start)
        end_sample = self._time_to_sample(end)
        return self._modified(data=self.data[start_sample:end_sample])

    def remove_mean(self) -> 'BaseSamples':
        """Subtract the mean from the data and return a modified copy."""
        return self._modified(data=self.data - np.mean(self.data))

    def abs(self) -> 'RealSamples':
        """Return a RealSamples containing the absolute value of the data."""
        arr = np.abs(self.data).astype(np.float32)
        return RealSamples(arr, fs=self.fs)

    def threshold(self, threshold: float) -> NDArray[np.bool]:
        """Return a boolean array where samples exceed the given `threshold`."""
        return np.abs(self.data) >= threshold

class ComplexSamples(BaseSamples):
    """Samples container for complex-valued signals."""

    def __init__(self, data: Union[NDArray[np.complex64], 'BaseSamples'], fs: Optional[float] = None) -> None:
        """Construct from raw complex array or from another BaseSamples instance.

        If `data` is a `BaseSamples` instance, its `data` and `fs` are used.
        """
        if isinstance(data, BaseSamples):
            super().__init__(data=data.data.astype(np.complex64), fs=data.fs)
        else:
            assert fs is not None
            super().__init__(data=data, fs=fs)

    def angle(self) -> 'AngleSamples':
        """Return the instantaneous phase as `AngleSamples`."""
        arr = np.angle(self.data).astype(np.float32)
        return AngleSamples(arr, fs=self.fs)

    def conjugate(self) -> 'ComplexSamples':
        """Return the complex conjugate as a modified copy."""
        return self._modified(data=np.conjugate(self.data))

    def real(self) -> 'RealSamples':
        """Return the real part as `RealSamples`."""
        arr = self.data.real.astype(np.float32)
        return RealSamples(arr, fs=self.fs)

    def imag(self) -> 'RealSamples':
        """Return the imaginary part as `RealSamples`."""
        arr = self.data.imag.astype(np.float32)
        return RealSamples(arr, fs=self.fs)

class RealSamples(BaseSamples):
    """Samples container for real-valued signals."""

    def __init__(self, data: Union[NDArray[np.float32], BaseSamples], fs: Optional[float] = None) -> None:
        """Construct from raw real array or from a BaseSamples instance."""
        if isinstance(data, BaseSamples):
            super().__init__(data=data.data.real.astype(np.float32), fs=data.fs)
        else:
            assert fs is not None
            super().__init__(data=data, fs=fs)
        self.data: NDArray[np.float32] = self.data.astype(np.float32)

    def percentile(self, p: float) -> float:
        """Return the `p`th percentile of the sample values."""
        return np.percentile(self.data, p)

class AngleSamples(RealSamples):
    """Angle (phase) samples derived from complex signals."""

    def __init__(self, data: Union[NDArray[np.float32], BaseSamples], fs: Optional[float] = None) -> None:
        if isinstance(data, BaseSamples):
            super().__init__(data=np.angle(data.data).astype(np.float32), fs=data.fs)
        else:
            assert fs is not None
            super().__init__(data=data, fs=fs)

    def unwrap(self) -> 'AngleSamples':
        """Return an `AngleSamples` with phase unwrapped."""
        arr = np.unwrap(self.data).astype(np.float32)
        return AngleSamples(arr, fs=self.fs)

class IQSamples(ComplexSamples):
    """IQ (complex) samples with center frequency metadata and DSP helpers."""

    def __init__(self, data: NDArray[np.complex64], fs: float, fc: float) -> None:
        """Create IQ samples.

        :param data: Complex baseband samples.
        :param fs: Sampling frequency in Hz.
        :param fc: Center frequency in Hz.
        """
        super().__init__(data, fs)
        self.fc: float = fc

    def _modified(self, data: Optional[NDArray] = None, fs: Optional[float] = None, fc: Optional[float] = None) -> 'IQSamples':
        """Return a modified IQSamples instance preserving `fc` when not provided."""
        if data is None:
            data = self.data
        if fs is None:
            fs = self.fs
        if fc is None:
            fc = self.fc
        return IQSamples(data=data, fs=fs, fc=fc)

    def recenter(self, fc_new: float) -> 'IQSamples':
        """Shift samples so a new frequency `fc_new` is at center frequency.

        Returns a new `IQSamples` instance with the frequency shift applied.
        """
        return self._modified(
            data=self.data * np.exp(-2j * np.pi * (fc_new - self.fc) * self.time()),
            fc=fc_new,
        )

    def dc_correct(self) -> 'IQSamples':
        """Subtract the mean (DC) from the complex samples."""
        return self._modified(data=self.data - np.mean(self.data))

    def low_pass(self, numtaps: int, cutoff: float) -> 'IQSamples':
        """Apply an FIR low-pass filter and return filtered samples.

        :param numtaps: Number of FIR taps.
        :param cutoff: Cutoff frequency in Hz.
        """
        lp_taps = firwin(numtaps=numtaps, cutoff=cutoff, fs=self.fs, window="blackmanharris")

        return self._modified(data=lfilter(lp_taps, 1.0, self.data))

    def freq_discriminator(self) -> ComplexSamples:
        """Return the frequency discriminator output as `ComplexSamples`."""
        return ComplexSamples(
            data = self.data[1:] * np.conjugate(self.data[:-1]),
            fs = self.fs)

    def instantaneous_frequency(self) -> RealSamples:
        """Return the instantaneous frequency as `RealSamples`."""
        return (
            self
                .freq_discriminator()
                .angle()
                .scale(self.fs / (2*np.pi))
            )

    def save_to_cf32(self, base: str) -> None:
        """Save interleaved cf32 (float32 I/Q) file to `generated/{base}.cf32`."""
        path = f'generated/{base}.cf32'
        logging.info(f'saving {self.sample_count} samples to {path}:')
        interleaved = np.empty(2 * self.sample_count, dtype=np.float32)
        interleaved[0::2] = self.data.real.astype(np.float32)
        interleaved[1::2] = self.data.imag.astype(np.float32)
        interleaved.tofile(path)

    @staticmethod
    def load_int8(path: str, fs: float, fc: float) -> 'IQSamples':
        """Load int8 I/Q interleaved file and return an `IQSamples` instance.

        Values are scaled to approximately [-1, 1].
        """
        raw = np.fromfile(path, dtype=np.int8)
        logging.info(f'loaded {len(raw)//2} samples from {path}:')
        iq = raw.reshape(-1, 2)
        return IQSamples(
            data=(iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)) / 128.0,
            fs=fs,
            fc=fc,
        )