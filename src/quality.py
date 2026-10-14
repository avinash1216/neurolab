"""Signal-quality gate applied to an unfiltered epoch before the classifier runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


@dataclass
class QualityReport:
    ok: bool
    bad_channels: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    max_peak_to_peak_uv: float = 0.0
    min_channel_std_uv: float = 0.0
    max_line_noise_ratio: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def assess_quality(epoch_uv: np.ndarray, sfreq: float, ch_names: list[str],
                   qcfg: dict, window: slice | None = None) -> QualityReport:
    """Flag channels with artifacts, flat lines, or strong mains interference.

    epoch_uv: (n_channels, n_times) unfiltered signal in microvolts.
    window: optional slice restricting checks to the decision window.
    """
    # Amplitude checks on a 1-40 Hz version (removes slow drift); mains check on raw signal.
    sos = butter(4, [1.0, 40.0], btype="bandpass", fs=sfreq, output="sos")
    filt = sosfiltfilt(sos, epoch_uv.astype(np.float64), axis=-1)
    filt = filt[:, window] if window is not None else filt
    x = epoch_uv[:, window] if window is not None else epoch_uv
    x = x - x.mean(axis=1, keepdims=True)

    ptp = np.ptp(filt, axis=1)
    std = x.std(axis=1)
    freqs, psd = welch(x, fs=sfreq, nperseg=min(x.shape[1], int(sfreq)), axis=1)
    line = psd[:, (freqs >= 55) & (freqs <= 65)].sum(axis=1)
    band = psd[:, (freqs >= 1) & (freqs <= 40)].sum(axis=1) + 1e-12
    line_ratio = line / band

    reasons: dict[str, list[str]] = {}
    for i, ch in enumerate(ch_names):
        if ptp[i] > qcfg["max_peak_to_peak_uv"]:
            reasons.setdefault(ch, []).append(f"amplitude {ptp[i]:.0f} µV")
        if std[i] < qcfg["min_channel_std_uv"]:
            reasons.setdefault(ch, []).append("flat signal")
        if line_ratio[i] > qcfg["max_line_noise_ratio"]:
            reasons.setdefault(ch, []).append("mains interference")

    bad = list(reasons)
    summary = []
    for kind in ("amplitude", "flat signal", "mains interference"):
        chans = [ch for ch, r in reasons.items() if any(kind in s for s in r)]
        if chans:
            label = "excessive amplitude / artifact" if kind == "amplitude" else kind
            summary.append(f"{label} on {', '.join(chans[:6])}{'...' if len(chans) > 6 else ''}")

    return QualityReport(
        ok=len(bad) <= qcfg["max_bad_channels"],
        bad_channels=bad,
        reasons=summary,
        max_peak_to_peak_uv=float(ptp.max()),
        min_channel_std_uv=float(std.min()),
        max_line_noise_ratio=float(line_ratio.max()),
    )


def corrupt(epoch_uv: np.ndarray, kind: str, sfreq: float, ch_names: list[str],
            seed: int = 0) -> np.ndarray:
    """Synthetically corrupt a recorded trial for the demo. Always labelled as such in the UI."""
    rng = np.random.default_rng(seed)
    x = epoch_uv.copy()
    n_ch, n_t = x.shape
    t = np.arange(n_t) / sfreq
    idx = {c: i for i, c in enumerate(ch_names)}
    if kind == "movement_artifact":
        # Large slow deflections across frontal-central channels.
        centre = n_t // 2
        bump = 1500 * np.exp(-0.5 * ((np.arange(n_t) - centre) / (0.15 * sfreq)) ** 2)
        for c in ch_names[: max(3, n_ch // 3)]:
            x[idx[c]] += bump
    elif kind == "flat_channel":
        for c in ("C3", "C4"):
            if c in idx:
                x[idx[c]] = 0.0
    elif kind == "line_noise":
        x += 400 * np.sin(2 * np.pi * 60 * t)[None, :]
    elif kind == "broadband_noise":
        # Usually passes the quality gate but masks the mu/beta pattern; the model
        # may or may not abstain, which is shown honestly in the UI.
        x += rng.normal(0, 40.0, size=x.shape)
    else:
        raise ValueError(f"unknown corruption: {kind}")
    return x.astype(np.float32)


CORRUPTIONS = {
    "movement_artifact": "Movement / electrode-pop artifact",
    "flat_channel": "Disconnected electrode (flat C3, C4)",
    "line_noise": "Severe 60 Hz mains interference",
    "broadband_noise": "Broadband noise (passes gate; tests model score)",
}
