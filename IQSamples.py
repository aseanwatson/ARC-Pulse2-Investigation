import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin, lfilter

import logging

class IQSamples:
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
        return IQSamples(data = data, fs = fs, fc = fc)

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

    def recenter(self, fc_new:float) -> 'IQSamples':
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

    def dc_correct(self) -> 'IQSamples':
        """
        Subtracts the mean to remove DC bias.
        """
        return self._modified(data = self.data - np.mean(self.data))

    def normalize_percentile(self, p: float = 95.0, *, min_threshold: float = 0.0, min_threshold_percentile: float = 0.0) -> 'IQSamples':
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

    def low_pass(self, numtaps:int, cutoff:float) -> 'IQSamples':
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

    def decimate(self, decimation_factor:int) -> 'IQSamples':
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

    def time_slice(self, start: float, end:float) -> 'IQSamples':
        start_sample = self._time_to_sample(start)
        end_sample = self._time_to_sample(end)
        return self._modified(data=self.data[start_sample:end_sample])

    @staticmethod
    def load_int8(path, fs:float, fc:float) -> 'IQSamples':
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
        return IQSamples(
            data = (iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)) / 128.0,
            fs = fs,
            fc = fc)