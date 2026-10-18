"""Day-1 pipeline: load -> preprocess -> CSP+LDA -> validate -> pick threshold -> test once.

Outputs
  artifacts/models/subject_XXX.joblib   decoder bundle per subject
  artifacts/demo_trials.npz             held-out (test-run) trials for UI playback
  artifacts/metrics.json                every reported number
  reports/*.png, reports/results.md     figures + human-readable results
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_experiment_config  # noqa: E402
from src.data import load_subject_epochs  # noqa: E402
from src.evaluate import (  # noqa: E402
    classification_metrics, plot_confusion, plot_risk_coverage, risk_coverage, select_threshold,
)
from src.model import build_pipeline  # noqa: E402
from src.quality import assess_quality  # noqa: E402


def validate(pipe, X, y, cfg):
    """Repeated stratified k-fold on the training runs; returns fold scores and pooled OOF proba."""
    s = cfg["split"]
    cv = RepeatedStratifiedKFold(n_splits=s["cv_folds"], n_repeats=s["cv_repeats"],
                                 random_state=cfg["seed"])
    fold_bacc, y_oof, p_oof = [], [], []
    for tr, va in cv.split(X, y):
        m = clone(pipe).fit(X[tr], y[tr])
        proba = m.predict_proba(X[va])
        fold_bacc.append(classification_metrics(y[va], proba.argmax(1), [0, 1])["balanced_accuracy"])
        y_oof.append(y[va])
        p_oof.append(proba)
    return np.array(fold_bacc), np.concatenate(y_oof), np.concatenate(p_oof)


def plot_csp_patterns(pipe, ch_names, sfreq, path, title):
    info = mne.create_info(ch_names, sfreq, "eeg")
    montage = "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005"
    info.set_montage(montage)
    csp = pipe.named_steps["csp"]
    fig = csp.plot_patterns(info, ch_type="eeg", units="a.u.", size=1.4, show=False,
                            colorbar=False)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_subject(rows, chosen, path):
    subj = [r["subject"] for r in rows]
    val = [r["validation"]["balanced_accuracy_mean"] for r in rows]
    err = [r["validation"]["balanced_accuracy_std"] for r in rows]
    test = [r["test"]["balanced_accuracy"] for r in rows]
    x = np.arange(len(subj))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - 0.2, val, 0.4, yerr=err, capsize=3, label="validation (CV on runs 4, 8)")
    ax.bar(x + 0.2, test, 0.4, label="held-out test (run 12)")
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_xticks(x, [f"S{s}{'*' if s == chosen else ''}" for s in subj])
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-subject CSP+LDA, left vs right fist imagery  (* = demo subject)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="*",
                        help="override config subjects (e.g. for a quick check)")
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_experiment_config()
    if args.subjects:
        cfg["data"]["subjects"] = args.subjects
    np.random.seed(cfg["seed"])
    art = Path(cfg["paths"]["artifacts_dir"])
    rep = Path(cfg["paths"]["reports_dir"])
    (art / "models").mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    train_runs, test_runs = cfg["split"]["train_runs"], cfg["split"]["test_runs"]
    subjects = cfg["data"]["subjects"]

    data, val_pool_y, val_pool_p = {}, [], []
    per_subject = []
    print(f"Validating {len(subjects)} subjects ...", flush=True)
    for subj in subjects:
        es = load_subject_epochs(subj, train_runs + test_runs, cfg)
        tr = np.isin(es.runs, train_runs)
        pipe = build_pipeline(cfg, es.sfreq)
        fold_bacc, y_oof, p_oof = validate(pipe, es.X[tr], es.y[tr], cfg)
        val_pool_y.append(y_oof)
        val_pool_p.append(p_oof)
        data[subj] = (es, pipe, fold_bacc, y_oof, p_oof)
        print(f"  S{subj:03d}: n_train={tr.sum()} n_test={(~tr).sum()} "
              f"val bAcc={fold_bacc.mean():.3f}±{fold_bacc.std():.3f}", flush=True)

    # One threshold for all subjects, chosen on pooled validation predictions only.
    y_val_all, p_val_all = np.concatenate(val_pool_y), np.concatenate(val_pool_p)
    threshold, thr_reason = select_threshold(y_val_all, p_val_all, cfg["abstention"])
    print(f"Threshold = {threshold:.2f} ({thr_reason})", flush=True)

    # Final fit on all training runs, single evaluation on the held-out run.
    test_pool_y, test_pool_p = [], []
    demo = {"X": [], "y": [], "subject": [], "run": [], "onset": [], "trial_index": []}
    quality_rejects = quality_total = 0
    for subj, (es, pipe, fold_bacc, y_oof, p_oof) in data.items():
        tr = np.isin(es.runs, train_runs)
        te = ~tr
        model = clone(pipe).fit(es.X[tr], es.y[tr])
        proba = model.predict_proba(es.X[te])
        pred = proba.argmax(1)
        y_te = es.y[te]
        test_pool_y.append(y_te)
        test_pool_p.append(proba)

        window = model.named_steps["bandpass"].window_slice()
        q_ok = [assess_quality(x, es.sfreq, es.ch_names, cfg["quality"], window).ok for x in es.X[te]]
        quality_rejects += int(len(q_ok) - sum(q_ok))
        quality_total += len(q_ok)

        test_m = classification_metrics(y_te, pred, es.class_names)
        test_m["binomial_p_vs_chance"] = float(
            binomtest(int((pred == y_te).sum()), len(y_te), 0.5, alternative="greater").pvalue)
        rc = risk_coverage(y_te, proba, [threshold])[0]
        per_subject.append({
            "subject": subj,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "validation": {
                "balanced_accuracy_mean": float(fold_bacc.mean()),
                "balanced_accuracy_std": float(fold_bacc.std()),
                **{k: v for k, v in risk_coverage(y_oof, p_oof, [threshold])[0].items()
                   if k != "threshold"},
            },
            "test": test_m,
            "test_at_threshold": rc,
            "quality_gate_rejects_on_clean_test": int(len(q_ok) - sum(q_ok)),
        })

        joblib.dump({
            "pipeline": model,
            "threshold": threshold,
            "class_names": es.class_names,
            "ch_names": es.ch_names,
            "sfreq": es.sfreq,
            "epoch_tmin": cfg["data"]["epoch_tmin"],
            "quality_cfg": cfg["quality"],
            "model_version": cfg["model_version"],
            "subject": subj,
            "train_runs": train_runs,
        }, art / "models" / f"subject_{subj:03d}.joblib")

        idx = np.where(te)[0]
        demo["X"].append(es.X[te])
        demo["y"].append(y_te)
        demo["subject"].append(np.full(te.sum(), subj))
        demo["run"].append(es.runs[te])
        demo["onset"].append(es.onsets[te])
        demo["trial_index"].append(np.arange(len(idx)))

    y_test_all, p_test_all = np.concatenate(test_pool_y), np.concatenate(test_pool_p)
    class_names = data[subjects[0]][0].class_names
    pooled_test = classification_metrics(y_test_all, p_test_all.argmax(1), class_names)
    pooled_test["binomial_p_vs_chance"] = float(binomtest(
        int((p_test_all.argmax(1) == y_test_all).sum()), len(y_test_all), 0.5,
        alternative="greater").pvalue)

    # Demo subject chosen on VALIDATION balanced accuracy, never on test.
    demo_subject = max(per_subject, key=lambda r: r["validation"]["balanced_accuracy_mean"])["subject"]
    demo_row = next(r for r in per_subject if r["subject"] == demo_subject)

    es0 = data[subjects[0]][0]
    np.savez_compressed(
        art / "demo_trials.npz",
        X=np.concatenate(demo["X"]).astype(np.float32),
        y=np.concatenate(demo["y"]),
        subject=np.concatenate(demo["subject"]),
        run=np.concatenate(demo["run"]),
        onset=np.concatenate(demo["onset"]),
        trial_index=np.concatenate(demo["trial_index"]),
        times=es0.times,
        sfreq=es0.sfreq,
        ch_names=np.array(es0.ch_names),
        class_names=np.array(class_names),
    )

    grid = cfg["abstention"]["grid"]
    rc_val = risk_coverage(y_val_all, p_val_all, grid)
    rc_test = risk_coverage(y_test_all, p_test_all, grid)
    demo_es = data[demo_subject][0]
    demo_te = ~np.isin(demo_es.runs, train_runs)
    demo_model = joblib.load(art / "models" / f"subject_{demo_subject:03d}.joblib")["pipeline"]

    plot_confusion(pooled_test["confusion_matrix"], class_names,
                   f"Held-out test, all {len(subjects)} subjects (n={pooled_test['n']})",
                   rep / "confusion_pooled_test.png")
    plot_confusion(demo_row["test"]["confusion_matrix"], class_names,
                   f"Held-out test, demo subject S{demo_subject} (n={demo_row['n_test']})",
                   rep / "confusion_demo_subject_test.png")
    plot_risk_coverage({"validation (pooled OOF)": rc_val, "held-out test (pooled)": rc_test},
                       threshold, f"Risk-coverage (red = chosen threshold {threshold:.2f})",
                       rep / "risk_coverage.png")
    plot_per_subject(per_subject, demo_subject, rep / "per_subject_balanced_accuracy.png")
    plot_csp_patterns(demo_model, demo_es.ch_names, demo_es.sfreq,
                      rep / "csp_patterns_demo_subject.png",
                      f"CSP spatial patterns, S{demo_subject}")

    rc_thr_test = risk_coverage(y_test_all, p_test_all, [threshold])[0]
    rc_thr_val = risk_coverage(y_val_all, p_val_all, [threshold])[0]
    metrics = {
        "model_version": cfg["model_version"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": cfg["data"]["dataset"],
        "subjects": subjects,
        "runs": {"train_validation": train_runs, "test": test_runs},
        "classes": class_names,
        "channels": demo_es.ch_names,
        "preprocessing": cfg["preprocessing"],
        "model": cfg["model"],
        "seed": cfg["seed"],
        "threshold": {"value": threshold, "selection": thr_reason, **cfg["abstention"]},
        "validation_pooled": {"n_oof_predictions": int(len(y_val_all)),
                              "at_threshold": rc_thr_val, "risk_coverage": rc_val},
        "test_pooled": {**pooled_test, "at_threshold": rc_thr_test, "risk_coverage": rc_test},
        "quality_gate": {"clean_test_trials": quality_total,
                         "rejected_clean_test_trials": quality_rejects},
        "demo_subject": {"subject": demo_subject,
                         "selection": "highest mean validation balanced accuracy"},
        "per_subject": per_subject,
        "environment": {"python": platform.python_version(), "mne": mne.__version__,
                        "scikit_learn": sklearn.__version__, "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(art / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    write_results_md(metrics, rep / "results.md")
    print(f"Done in {metrics['runtime_seconds']} s. Pooled test bAcc="
          f"{pooled_test['balanced_accuracy']:.3f}; demo subject S{demo_subject}", flush=True)


def write_results_md(m: dict, path: Path) -> None:
    t, v = m["test_pooled"], m["validation_pooled"]
    thr = m["threshold"]["value"]
    lines = [
        "# NeuroBridge — Day-1 baseline results",
        "",
        f"_Generated {m['generated_utc']} by `scripts/train.py` from `config/experiment.yaml`. "
        "All numbers below are actual outputs; nothing is illustrative._",
        "",
        "## Setup",
        f"- **Dataset:** {m['dataset']}",
        f"- **Subjects:** {', '.join('S' + str(s) for s in m['subjects'])} (one model per subject)",
        f"- **Task:** {' vs '.join(m['classes'])} (runs 4, 8, 12 = motor imagery)",
        f"- **Split:** train/validate on runs {m['runs']['train_validation']}, "
        f"test once on run {m['runs']['test']} — split by recording run, no window overlap",
        f"- **Channels:** {len(m['channels'])} sensorimotor channels",
        f"- **Preprocessing:** {m['preprocessing']['l_freq']}–{m['preprocessing']['h_freq']} Hz "
        f"Butterworth (order {m['preprocessing']['filter_order']}, zero-phase), window "
        f"{m['preprocessing']['window_tmin']}–{m['preprocessing']['window_tmax']} s after cue",
        f"- **Model:** CSP ({m['model']['csp_components']} components, {m['model']['csp_reg']}) "
        f"+ LDA · version `{m['model_version']}` · seed {m['seed']}",
        "",
        "## Held-out test (pooled over subjects)",
        f"| Metric | Value |", "|---|---|",
        f"| Trials | {t['n']} |",
        f"| Accuracy | {t['accuracy']:.3f} |",
        f"| Balanced accuracy | {t['balanced_accuracy']:.3f} |",
        f"| Macro F1 | {t['macro_f1']:.3f} |",
        f"| Binomial p vs. chance (0.5) | {t['binomial_p_vs_chance']:.2g} |",
        "",
        f"## Abstention (threshold = {thr:.2f}, {m['threshold']['selection']})",
        "| Split | Coverage | Abstention rate | Accuracy when acting | Error rate when acting |",
        "|---|---|---|---|---|",
    ]
    for name, r in (("Validation (pooled OOF)", v["at_threshold"]), ("Held-out test", t["at_threshold"])):
        lines.append(f"| {name} | {r['coverage']:.2f} | {r['abstention_rate']:.2f} | "
                     f"{r['accepted_accuracy']:.3f} | {r['accepted_error_rate']:.3f} |")
    q = m["quality_gate"]
    lines += [
        "",
        f"Quality gate rejected {q['rejected_clean_test_trials']} of {q['clean_test_trials']} "
        "clean recorded test trials.",
        "",
        "## Per subject",
        "| Subject | Val bAcc (mean ± sd) | Test bAcc | Test n | Test coverage @thr | Test acc @thr | p vs chance |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in m["per_subject"]:
        a = r["test_at_threshold"]
        acc = "—" if a["n_accepted"] == 0 else f"{a['accepted_accuracy']:.2f}"
        star = " *" if r["subject"] == m["demo_subject"]["subject"] else ""
        lines.append(
            f"| S{r['subject']}{star} | {r['validation']['balanced_accuracy_mean']:.2f} ± "
            f"{r['validation']['balanced_accuracy_std']:.2f} | {r['test']['balanced_accuracy']:.2f} | "
            f"{r['n_test']} | {a['coverage']:.2f} | {acc} | {r['test']['binomial_p_vs_chance']:.2g} |")
    lines += [
        "",
        f"\\* Demo subject S{m['demo_subject']['subject']} chosen by "
        f"{m['demo_subject']['selection']} (not by test score).",
        "",
        "## Figures",
        "![per subject](per_subject_balanced_accuracy.png)",
        "![risk coverage](risk_coverage.png)",
        "![confusion pooled](confusion_pooled_test.png)",
        "![confusion demo](confusion_demo_subject_test.png)",
        "![csp](csp_patterns_demo_subject.png)",
        "",
        "## Limitations",
        "- Very small per-subject data (~45 imagery trials); per-subject test sets have ~15 trials, "
        "so per-subject accuracies have wide uncertainty.",
        "- Within-subject, within-session models only; no cross-subject or cross-day generalisation is claimed.",
        "- The model score is LDA `predict_proba`; calibration has not been evaluated, so it is "
        "reported as a model score, not a probability.",
        "- Recorded data from a public dataset; no live EEG is used.",
        "- A motor-imagery class is a task-specific control signal, not decoded thought or language.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
