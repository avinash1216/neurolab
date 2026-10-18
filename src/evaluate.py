"""Metrics, abstention threshold selection and report plots."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred, class_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                   "support": int(s[i])}
            for i, name in enumerate(class_names)
        },
    }


def risk_coverage(y_true, proba, thresholds) -> list[dict[str, float]]:
    """For each threshold on the top-class score: coverage and accuracy of accepted trials."""
    y_true = np.asarray(y_true)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    rows = []
    for t in thresholds:
        acc_mask = conf >= t
        n_acc = int(acc_mask.sum())
        rows.append({
            "threshold": float(t),
            "coverage": float(n_acc / len(y_true)) if len(y_true) else 0.0,
            "abstention_rate": float(1 - n_acc / len(y_true)) if len(y_true) else 1.0,
            "n_accepted": n_acc,
            "accepted_accuracy": float((pred[acc_mask] == y_true[acc_mask]).mean()) if n_acc else float("nan"),
            "accepted_error_rate": float((pred[acc_mask] != y_true[acc_mask]).mean()) if n_acc else float("nan"),
        })
    return rows


def select_threshold(y_val, proba_val, acfg: dict) -> tuple[float, str]:
    """Smallest grid threshold meeting the accepted-accuracy target with enough coverage."""
    for row in risk_coverage(y_val, proba_val, acfg["grid"]):
        if (row["coverage"] >= acfg["min_coverage"]
                and row["accepted_accuracy"] >= acfg["target_accepted_accuracy"]):
            return row["threshold"], "met validation target"
    return float(acfg["fallback_threshold"]), "target not reachable on validation; fallback used"


def plot_confusion(cm, class_names, title, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    short = [c.replace("_imagery", "").replace("_", " ") for c in class_names]
    ax.set_xticks(range(len(short)), short)
    ax.set_yticks(range(len(short)), short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(curves: dict[str, list[dict]], threshold: float, title: str, path):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for label, rows in curves.items():
        ax.plot([r["coverage"] for r in rows], [r["accepted_accuracy"] for r in rows],
                marker="o", label=label)
        for r in rows:
            if abs(r["threshold"] - threshold) < 1e-9:
                ax.scatter([r["coverage"]], [r["accepted_accuracy"]], s=120,
                           facecolors="none", edgecolors="red", zorder=5)
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance (2 classes)")
    ax.set_xlabel("Coverage (fraction of trials acted on)")
    ax.set_ylabel("Accuracy of accepted trials")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
