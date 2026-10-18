"""CSP + LDA baseline pipeline."""

from __future__ import annotations

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from src.preprocessing import bandpass_from_config


def build_pipeline(cfg: dict, sfreq: float) -> Pipeline:
    """Band-pass/crop -> CSP (log-variance features) -> LDA.

    Every learned step lives inside the pipeline, so cross-validation refits
    CSP on each training fold and no test information leaks into the filters.
    """
    m = cfg["model"]
    return Pipeline([
        ("bandpass", bandpass_from_config(cfg, sfreq)),
        ("csp", CSP(n_components=m["csp_components"], reg=m["csp_reg"], log=True)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
