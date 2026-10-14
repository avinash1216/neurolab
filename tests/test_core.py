import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_experiment_config  # noqa: E402
from src.evaluate import risk_coverage, select_threshold  # noqa: E402
from src.gemini_client import suggest_phrases  # noqa: E402
from src.preprocessing import bandpass_from_config  # noqa: E402
from src.quality import CORRUPTIONS, assess_quality, corrupt  # noqa: E402

CFG = load_experiment_config()
SFREQ = 160.0
CH = CFG["data"]["channels"]


def synthetic_epoch(seed=0):
    rng = np.random.default_rng(seed)
    n_t = int((CFG["data"]["epoch_tmax"] - CFG["data"]["epoch_tmin"]) * SFREQ) + 1
    t = np.arange(n_t) / SFREQ
    x = rng.normal(0, 15, (len(CH), n_t)) + 20 * np.sin(2 * np.pi * 10 * t)
    return x.astype(np.float32)


def window():
    return bandpass_from_config(CFG, SFREQ).window_slice()


def test_clean_epoch_passes_quality():
    assert assess_quality(synthetic_epoch(), SFREQ, CH, CFG["quality"], window()).ok


@pytest.mark.parametrize("kind", ["movement_artifact", "flat_channel", "line_noise"])
def test_corruptions_rejected(kind):
    x = corrupt(synthetic_epoch(), kind, SFREQ, CH)
    q = assess_quality(x, SFREQ, CH, CFG["quality"], window())
    assert not q.ok and q.reasons


def test_all_corruptions_defined():
    for k in CORRUPTIONS:
        corrupt(synthetic_epoch(), k, SFREQ, CH)


def test_bandpass_crop_shape():
    bp = bandpass_from_config(CFG, SFREQ)
    out = bp.transform(synthetic_epoch()[None])
    p = CFG["preprocessing"]
    assert out.shape[-1] == int(round((p["window_tmax"] - p["window_tmin"]) * SFREQ)) + 1


def test_risk_coverage_and_threshold():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.45, 0.55], [0.6, 0.4]])
    rows = {r["threshold"]: r for r in risk_coverage(y, proba, [0.5, 0.75])}
    assert rows[0.5]["coverage"] == 1.0 and rows[0.5]["accepted_accuracy"] == 0.5
    assert rows[0.75]["coverage"] == 0.5 and rows[0.75]["accepted_accuracy"] == 1.0
    thr, _ = select_threshold(y, proba, {"grid": [0.5, 0.75], "min_coverage": 0.25,
                                         "target_accepted_accuracy": 0.8,
                                         "fallback_threshold": 0.9})
    assert thr == 0.75


def test_gemini_fallback_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    res = suggest_phrases("Water", ["A", "B"])
    assert res.source == "fallback" and res.phrases == ["A", "B"]
