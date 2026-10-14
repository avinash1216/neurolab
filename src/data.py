"""Loading and epoching of the EEGMMIDB motor-imagery runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from mne.datasets import eegbci

mne.set_log_level("WARNING")


@dataclass
class EpochSet:
    """Unfiltered epochs in microvolts, shape (n_trials, n_channels, n_times)."""

    X: np.ndarray
    y: np.ndarray            # integer labels, index into class_names
    runs: np.ndarray         # recording run of each trial
    onsets: np.ndarray       # cue onset (s) within its run
    times: np.ndarray
    sfreq: float
    ch_names: list[str]
    class_names: list[str]
    subject: int


def edf_path(data_dir: str | Path, subject: int, run: int) -> Path:
    sid = f"S{subject:03d}"
    return (Path(data_dir) / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
            / sid / f"{sid}R{run:02d}.edf")


def _read_run(data_dir: str | Path, subject: int, run: int) -> mne.io.BaseRaw:
    path = edf_path(data_dir, subject, run)
    if not path.exists():
        # Falls back to MNE's downloader (slower than scripts/download_data.py).
        eegbci.load_data(subject, [run], path=str(data_dir), update_path=False)
    raw = mne.io.read_raw_edf(path, preload=True)
    eegbci.standardize(raw)
    return raw


def load_subject_epochs(subject: int, runs: list[int], cfg: dict[str, Any]) -> EpochSet:
    dcfg = cfg["data"]
    codes = list(dcfg["classes"].keys())
    class_names = [dcfg["classes"][c] for c in codes]
    event_id = {code: i + 1 for i, code in enumerate(codes)}

    X_parts, y_parts, run_parts, onset_parts = [], [], [], []
    times = sfreq = ch_names = None
    for run in runs:
        raw = _read_run(dcfg["data_dir"], subject, run)
        raw.pick(dcfg["channels"])
        events, _ = mne.events_from_annotations(raw, event_id=event_id)
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=dcfg["epoch_tmin"],
                            tmax=dcfg["epoch_tmax"], baseline=None, preload=True)
        X_parts.append(epochs.get_data(copy=True) * 1e6)
        y_parts.append(epochs.events[:, 2] - 1)
        run_parts.append(np.full(len(epochs), run))
        onset_parts.append(epochs.events[:, 0] / raw.info["sfreq"])
        times, sfreq, ch_names = epochs.times, raw.info["sfreq"], epochs.ch_names

    return EpochSet(
        X=np.concatenate(X_parts).astype(np.float32),
        y=np.concatenate(y_parts).astype(int),
        runs=np.concatenate(run_parts),
        onsets=np.concatenate(onset_parts),
        times=np.asarray(times),
        sfreq=float(sfreq),
        ch_names=list(ch_names),
        class_names=class_names,
        subject=subject,
    )
