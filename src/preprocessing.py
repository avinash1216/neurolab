"""Preprocessing as a scikit-learn transformer so training and runtime share one code path."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin


class BandpassCrop(BaseEstimator, TransformerMixin):
    """Zero-phase Butterworth band-pass on the full epoch, then crop to the decision window.

    Filtering the whole -1..4 s epoch before cropping keeps filter edge effects
    out of the classification window. Stateless: nothing is learned from data.
    """

    def __init__(self, sfreq: float = 160.0, l_freq: float = 8.0, h_freq: float = 30.0,
                 order: int = 4, epoch_tmin: float = -1.0, window_tmin: float = 0.5,
                 window_tmax: float = 2.5):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.order = order
        self.epoch_tmin = epoch_tmin
        self.window_tmin = window_tmin
        self.window_tmax = window_tmax

    def fit(self, X, y=None):
        return self

    def window_slice(self) -> slice:
        start = int(round((self.window_tmin - self.epoch_tmin) * self.sfreq))
        stop = int(round((self.window_tmax - self.epoch_tmin) * self.sfreq)) + 1
        return slice(start, stop)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        sos = butter(self.order, [self.l_freq, self.h_freq], btype="bandpass",
                     fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)[..., self.window_slice()]


def bandpass_from_config(cfg: dict, sfreq: float) -> BandpassCrop:
    p = cfg["preprocessing"]
    return BandpassCrop(sfreq=sfreq, l_freq=p["l_freq"], h_freq=p["h_freq"],
                        order=p["filter_order"], epoch_tmin=cfg["data"]["epoch_tmin"],
                        window_tmin=p["window_tmin"], window_tmax=p["window_tmax"])
