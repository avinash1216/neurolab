"""Runtime decoder: quality gate -> model score -> abstain or propose a candidate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import joblib
import numpy as np

from src.quality import QualityReport, assess_quality


class DecisionState(str, Enum):
    POOR_SIGNAL = "POOR_SIGNAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CANDIDATE = "CANDIDATE"


@dataclass
class Decision:
    state: DecisionState
    predicted_class: str | None
    score: float | None            # top-class model score (not a calibrated probability)
    scores: dict[str, float] | None
    threshold: float
    quality: QualityReport
    model_version: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        d["quality"] = self.quality.to_dict()
        return d


class Decoder:
    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.class_names = bundle["class_names"]
        self.ch_names = bundle["ch_names"]
        self.sfreq = float(bundle["sfreq"])
        self.quality_cfg = bundle["quality_cfg"]
        self.model_version = bundle["model_version"]
        self.subject = bundle["subject"]
        self.window = self.pipeline.named_steps["bandpass"].window_slice()

    @classmethod
    def load(cls, artifacts_dir: str | Path, subject: int) -> "Decoder":
        return cls(joblib.load(Path(artifacts_dir) / "models" / f"subject_{subject:03d}.joblib"))

    def decide(self, epoch_uv: np.ndarray, threshold: float | None = None) -> Decision:
        thr = self.threshold if threshold is None else float(threshold)
        quality = assess_quality(epoch_uv, self.sfreq, self.ch_names, self.quality_cfg, self.window)
        if not quality.ok:
            return Decision(DecisionState.POOR_SIGNAL, None, None, None, thr, quality,
                            self.model_version)
        proba = self.pipeline.predict_proba(epoch_uv[None].astype(np.float64))[0]
        k = int(np.argmax(proba))
        scores = {c: float(p) for c, p in zip(self.class_names, proba)}
        state = DecisionState.CANDIDATE if proba[k] >= thr else DecisionState.LOW_CONFIDENCE
        return Decision(state, self.class_names[k], float(proba[k]), scores, thr, quality,
                        self.model_version)


def load_summary(artifacts_dir: str | Path) -> dict:
    with open(Path(artifacts_dir) / "metrics.json", encoding="utf-8") as fh:
        return json.load(fh)
