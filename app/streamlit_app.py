"""NeuroBridge Streamlit prototype: recorded EEG playback -> abstaining decoder ->
confirm-first scanning menu -> Gemini phrase suggestions -> user-chosen message."""

from __future__ import annotations

import html
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_actions_config, load_experiment_config  # noqa: E402
from src.decoder import Decoder, DecisionState  # noqa: E402
from src.gemini_client import gemini_available, suggest_phrases  # noqa: E402
from src.logging_utils import log_event  # noqa: E402
from src.quality import CORRUPTIONS, corrupt  # noqa: E402

st.set_page_config(page_title="NeuroBridge", page_icon="🧠", layout="wide")

CFG = load_experiment_config()
ACTIONS = load_actions_config()
ART = Path(CFG["paths"]["artifacts_dir"])
REP = Path(CFG["paths"]["reports_dir"])
PLOT_CHANNELS = ["C3", "Cz", "C4"]

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 18px; }
div.stButton > button { font-size: 1.1rem; padding: 0.7rem 1.2rem; border-radius: 10px;
  border: 2px solid #1f2937; font-weight: 600; min-height: 3.2rem; }
div.stButton > button:focus { outline: 4px solid #f59e0b; outline-offset: 2px; }
.nb-banner { background: #111827; color: #fff; padding: 0.6rem 1rem; border-radius: 8px;
  font-weight: 600; margin-bottom: 0.6rem; }
.nb-banner span { background: #f59e0b; color: #111827; padding: 0.1rem 0.5rem;
  border-radius: 4px; margin-left: 0.4rem; }
.nb-tile { border: 3px solid #9ca3af; border-radius: 12px; padding: 0.9rem; text-align: center;
  font-size: 1.15rem; font-weight: 600; background: #f9fafb; color: #111827; }
.nb-tile.active { border-color: #2563eb; background: #dbeafe; box-shadow: 0 0 0 4px #93c5fd; }
.nb-tile.selected { border-color: #059669; background: #d1fae5; }
.nb-state { border-radius: 12px; padding: 1rem 1.2rem; font-size: 1.15rem; font-weight: 600; }
.nb-neutral { background: #f3f4f6; border: 3px dashed #6b7280; color: #111827; }
.nb-candidate { background: #eff6ff; border: 3px solid #2563eb; color: #1e3a8a; }
.nb-message { background: #ecfdf5; border: 3px solid #059669; color: #064e3b; font-size: 1.6rem;
  padding: 1.2rem; border-radius: 12px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data/model
@st.cache_data
def load_demo_trials():
    d = np.load(ART / "demo_trials.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


@st.cache_data
def load_metrics():
    return json.loads((ART / "metrics.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_decoder(subject: int) -> Decoder:
    return Decoder.load(ART, subject)


if not (ART / "metrics.json").exists():
    st.error("No trained model found. Run `python scripts/download_data.py` and "
             "`python scripts/train.py` first.")
    st.stop()

TRIALS = load_demo_trials()
METRICS = load_metrics()
CLASS_NAMES = [str(c) for c in TRIALS["class_names"]]
CH_NAMES = [str(c) for c in TRIALS["ch_names"]]
SFREQ = float(TRIALS["sfreq"])
TIMES = TRIALS["times"]
MENU = ACTIONS["menu"]


# ---------------------------------------------------------------- state
def init_state():
    ss = st.session_state
    ss.setdefault("session_id", uuid.uuid4().hex[:12])
    ss.setdefault("stage", "idle")          # idle | candidate | neutral | phrases | message
    ss.setdefault("highlight", 0)
    ss.setdefault("selected_item", None)
    ss.setdefault("last", None)              # last trial + decision
    ss.setdefault("used", set())
    ss.setdefault("phrases", None)
    ss.setdefault("final_message", "")
    ss.setdefault("history", [])


def reset_flow(keep_highlight: bool = True):
    ss = st.session_state
    ss.stage = "idle"
    ss.selected_item = None
    ss.last = None
    ss.phrases = None
    ss.final_message = ""
    if not keep_highlight:
        ss.highlight = 0


init_state()
ss = st.session_state


def pick_trial(subject: int, want_class: str | None, rng: np.random.Generator) -> int | None:
    mask = TRIALS["subject"] == subject
    if want_class is not None:
        mask &= TRIALS["y"] == CLASS_NAMES.index(want_class)
    idx = [i for i in np.where(mask)[0] if (subject, int(i)) not in ss.used]
    if not idx:
        ss.used = {u for u in ss.used if u[0] != subject}   # recycle when exhausted
        idx = list(np.where(mask)[0])
    return int(rng.choice(idx)) if idx else None


def run_decode(subject: int, want_class: str | None, corruption: str | None):
    rng = np.random.default_rng()
    i = pick_trial(subject, want_class, rng)
    if i is None:
        st.warning("No held-out trials available for this subject.")
        return
    ss.used.add((subject, i))
    epoch = TRIALS["X"][i]
    if corruption:
        epoch = corrupt(epoch, corruption, SFREQ, CH_NAMES, seed=int(rng.integers(1e6)))
    decision = load_decoder(subject).decide(epoch)
    ss.last = {"index": i, "epoch": epoch, "decision": decision, "corruption": corruption,
               "true": CLASS_NAMES[int(TRIALS["y"][i])], "subject": subject,
               "run": int(TRIALS["run"][i]), "trial": int(TRIALS["trial_index"][i])}
    ss.stage = "candidate" if decision.state == DecisionState.CANDIDATE else "neutral"
    log_event("decode", ss.session_id, subject=subject, run=ss.last["run"],
              trial=ss.last["trial"], corruption=corruption, state=decision.state.value,
              predicted=decision.predicted_class, score=decision.score,
              threshold=decision.threshold, true_label=ss.last["true"],
              model_version=decision.model_version)


def confirm_action():
    d = ss.last["decision"]
    action = ACTIONS["classes"][d.predicted_class]["action"]
    correct = d.predicted_class == ss.last["true"]
    ss.history.append({"state": "confirmed", "action": action, "correct": correct})
    log_event("confirm", ss.session_id, action=action, predicted=d.predicted_class,
              matches_recorded_label=correct)
    if action == "next_option":
        ss.highlight = (ss.highlight + 1) % len(MENU)
        ss.stage = "idle"
        ss.last = None
    elif action == "select_option":
        ss.selected_item = MENU[ss.highlight]
        ss.stage = "phrases"


def cancel_action():
    d = ss.last["decision"] if ss.last else None
    log_event("cancel", ss.session_id, predicted=d.predicted_class if d else None)
    ss.history.append({"state": "cancelled"})
    ss.stage = "idle"
    ss.last = None


# ---------------------------------------------------------------- widgets
def signal_figure(epoch: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    offset = 0.0
    spacing = max(150.0, float(np.ptp(epoch[[CH_NAMES.index(c) for c in PLOT_CHANNELS]])) * 0.6)
    for k, ch in enumerate(PLOT_CHANNELS):
        y = epoch[CH_NAMES.index(ch)]
        y = y - y.mean() - k * spacing
        fig.add_trace(go.Scatter(x=TIMES, y=y, mode="lines", name=ch, line={"width": 1.3}))
        offset = k * spacing
    p = CFG["preprocessing"]
    fig.add_vrect(x0=p["window_tmin"], x1=p["window_tmax"], fillcolor="#fde68a", opacity=0.35,
                  line_width=0, annotation_text="decision window", annotation_position="top left")
    fig.add_vline(x=0, line_dash="dash", line_color="#111827", annotation_text="cue")
    fig.update_layout(title=title, height=300, margin={"l": 10, "r": 10, "t": 40, "b": 30},
                      xaxis_title="time from cue (s)", yaxis={"showticklabels": False},
                      legend={"orientation": "h", "y": -0.25}, template="plotly_white")
    fig.update_yaxes(range=[-offset - spacing, spacing])
    return fig


def score_bar(decision) -> go.Figure:
    names = [ACTIONS["classes"][c]["label"] for c in CLASS_NAMES]
    vals = [decision.scores[c] for c in CLASS_NAMES]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=["#6366f1", "#0ea5e9"],
                           text=[f"{v:.2f}" for v in vals], textposition="auto"))
    fig.add_vline(x=decision.threshold, line_color="#dc2626", line_width=3,
                  annotation_text=f"threshold {decision.threshold:.2f}")
    fig.update_layout(height=180, margin={"l": 10, "r": 10, "t": 30, "b": 20},
                      xaxis={"range": [0, 1], "title": "model score (uncalibrated)"},
                      template="plotly_white")
    return fig


def speak(text: str):
    safe = json.dumps(text)
    components.html(f"""<script>
      const u = new SpeechSynthesisUtterance({safe}); u.rate = 0.95;
      window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
    </script>""", height=0)


def render_menu():
    cols = st.columns(3)
    for i, item in enumerate(MENU):
        cls = "nb-tile"
        if ss.selected_item and ss.selected_item["id"] == item["id"]:
            cls += " selected"
        elif i == ss.highlight and ss.stage not in ("phrases", "message"):
            cls += " active"
        with cols[i % 3]:
            st.markdown(f'<div class="{cls}">{item["icon"]}<br>{html.escape(item["label"])}</div>',
                        unsafe_allow_html=True)
            st.write("")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Demo controls")
    st.caption("Input source")
    st.radio("Input source", ["Recorded EEG playback (PhysioNet EEGMMIDB)", "Live EEG headset"],
             index=0, disabled=True, label_visibility="collapsed",
             help="Live EEG is not implemented in this prototype.")
    st.info("Live EEG input is **not implemented**. All signals are recorded trials from a "
            "held-out run the model never saw during training.", icon="ℹ️")

    demo_subject = METRICS["demo_subject"]["subject"]
    subjects = METRICS["subjects"]
    subject = st.selectbox(
        "Recorded participant", subjects, index=subjects.index(demo_subject),
        format_func=lambda s: f"S{s:03d}" + ("  (demo, chosen on validation)" if s == demo_subject else ""))
    ss.setdefault("subject", subject)
    if subject != ss.subject:
        ss.subject = subject
        reset_flow(keep_highlight=False)

    st.divider()
    st.caption("Signal condition")
    corr_opts = ["clean"] + list(CORRUPTIONS)
    corruption = st.selectbox("Trial condition", corr_opts,
                              format_func=lambda k: "Clean recorded trial" if k == "clean"
                              else f"Synthetic: {CORRUPTIONS[k]}")
    corruption = None if corruption == "clean" else corruption

    st.divider()
    dec = load_decoder(subject)
    st.metric("Abstention threshold", f"{dec.threshold:.2f}")
    st.caption(f"Model `{dec.model_version}` · selected on validation data only. "
               f"Session `{ss.session_id}`")
    st.metric("Gemini", "connected" if gemini_available() else "not configured")
    if st.button("Reset session", width="stretch"):
        reset_flow(keep_highlight=False)
        ss.history = []
        st.rerun()


# ---------------------------------------------------------------- header
st.markdown('<div class="nb-banner">🧠 NeuroBridge — research prototype, not a medical device'
            '<span>MODE: RECORDED EEG PLAYBACK</span></div>', unsafe_allow_html=True)

tab_comm, tab_results, tab_about = st.tabs(["Communicate", "Model results", "How it works & limits"])

# ---------------------------------------------------------------- communicate
with tab_comm:
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("1 · Scanning menu")
        st.caption("Left-hand imagery → move highlight · Right-hand imagery → select. "
                   "The classes are control signals; the menu carries the meaning.")
        render_menu()

    with right:
        if ss.stage in ("idle", "candidate", "neutral"):
            st.subheader("2 · Decode a recorded trial")
            st.caption("In playback, you pick which recorded trial to replay. The model only "
                       "sees the EEG, never the label.")
            c1, c2, c3 = st.columns(3)
            cue_l = ACTIONS["classes"][CLASS_NAMES[0]]["label"]
            cue_r = ACTIONS["classes"][CLASS_NAMES[1]]["label"]
            if c1.button(f"◀ Replay a LEFT-hand trial\n({cue_l})", width="stretch"):
                run_decode(subject, CLASS_NAMES[0], corruption)
            if c2.button(f"Replay a RIGHT-hand trial ▶\n({cue_r})", width="stretch"):
                run_decode(subject, CLASS_NAMES[1], corruption)
            if c3.button("🎲 Random trial", width="stretch"):
                run_decode(subject, None, corruption)

        last = ss.last
        if last and ss.stage in ("candidate", "neutral"):
            d = last["decision"]
            src = "Recorded" if not last["corruption"] else \
                f"Recorded + SYNTHETIC {CORRUPTIONS[last['corruption']].lower()}"
            st.plotly_chart(signal_figure(last["epoch"],
                            f"{src} · S{last['subject']:03d} run {last['run']} trial {last['trial'] + 1}"),
                            width="stretch")
            true_lbl = last["true"].replace("_", " ")

            if d.state == DecisionState.POOR_SIGNAL:
                st.markdown('<div class="nb-state nb-neutral">⏸ Neutral — poor signal quality. '
                            'No action taken.</div>', unsafe_allow_html=True)
                for r in d.quality.reasons:
                    st.write(f"• {r}")
                st.caption("The classifier was not run on this trial.")
            elif d.state == DecisionState.LOW_CONFIDENCE:
                st.markdown(f'<div class="nb-state nb-neutral">⏸ Neutral — model score {d.score:.2f} '
                            f'is below the threshold {d.threshold:.2f}. No action taken.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(score_bar(d), width="stretch")
            else:
                act = ACTIONS["classes"][d.predicted_class]
                st.markdown(f'<div class="nb-state nb-candidate">Proposed action: <b>{act["label"]}</b>'
                            f'<br><small>from {d.predicted_class.replace("_", " ")} · model score '
                            f'{d.score:.2f} ≥ {d.threshold:.2f}</small></div>', unsafe_allow_html=True)
                st.plotly_chart(score_bar(d), width="stretch")

            match = "" if d.predicted_class is None else (
                " ✅ prediction matches" if d.predicted_class == last["true"] else " ❌ prediction is wrong")
            st.caption(f"Ground truth (recorded label, hidden from the model): **{true_lbl}**{match}")

            if ss.stage == "candidate":
                b1, b2 = st.columns(2)
                b1.button("✔ Confirm", type="primary", width="stretch", on_click=confirm_action)
                b2.button("✖ Cancel", width="stretch", on_click=cancel_action)
            else:
                st.button("↻ Retry", width="stretch", on_click=cancel_action)

        if ss.stage == "phrases":
            item = ss.selected_item
            st.subheader(f"3 · Phrases for: {item['icon']} {item['label']}")
            st.caption("You confirmed this item. Gemini only receives the item label and the "
                       "optional context below — no EEG data.")
            ctx = st.text_input("Optional context (e.g. who you're talking to)", key="ctx",
                                max_chars=120)
            g1, g2 = st.columns(2)
            if g1.button("✨ Suggest phrases", type="primary", width="stretch"):
                with st.spinner("Asking Gemini..."):
                    res = suggest_phrases(item["label"], item["fallback_phrases"], ctx)
                ss.phrases = res
                log_event("phrases", ss.session_id, item=item["id"], source=res.source,
                          n=len(res.phrases), note=res.note)
            if g2.button("✖ Back to menu", width="stretch"):
                reset_flow()
                st.rerun()

            if ss.phrases:
                if ss.phrases.note:
                    st.warning(ss.phrases.note)
                else:
                    st.success("Suggestions from Gemini. Pick one, then edit if needed.")
                for k, p in enumerate(ss.phrases.phrases):
                    if st.button(p, key=f"phrase_{k}", width="stretch"):
                        ss.final_message = p
                ss.final_message = st.text_area("Your message (editable)", value=ss.final_message,
                                                max_chars=200)
                if st.button("💬 Use this message", type="primary", width="stretch",
                             disabled=not ss.final_message.strip()):
                    ss.stage = "message"
                    log_event("message_selected", ss.session_id, item=item["id"],
                              source=ss.phrases.source, edited=ss.final_message not in ss.phrases.phrases)
                    st.rerun()

        if ss.stage == "message":
            st.subheader("4 · Message")
            st.markdown(f'<div class="nb-message">{html.escape(ss.final_message)}</div>',
                        unsafe_allow_html=True)
            st.write("")
            m1, m2 = st.columns(2)
            if m1.button("🔊 Speak", type="primary", width="stretch"):
                speak(ss.final_message)
                log_event("speak", ss.session_id)
            if m2.button("New message", width="stretch"):
                reset_flow(keep_highlight=False)
                st.rerun()

    if ss.history:
        n_conf = sum(h["state"] == "confirmed" for h in ss.history)
        n_wrong = sum(h["state"] == "confirmed" and not h["correct"] for h in ss.history)
        n_cancel = sum(h["state"] == "cancelled" for h in ss.history)
        st.caption(f"Session: {n_conf} confirmed ({n_wrong} where the prediction differed from the "
                   f"recorded label) · {n_cancel} cancelled / retried")

# ---------------------------------------------------------------- results
with tab_results:
    t = METRICS["test_pooled"]
    a = t["at_threshold"]
    st.subheader("Held-out test results (actual, not illustrative)")
    st.caption(f"{METRICS['dataset']} · subjects {', '.join('S' + str(s) for s in METRICS['subjects'])} · "
               f"train/validate runs {METRICS['runs']['train_validation']} · test run "
               f"{METRICS['runs']['test']} · model `{METRICS['model_version']}` · "
               f"generated {METRICS['generated_utc']}")
    c = st.columns(5)
    c[0].metric("Test trials", t["n"])
    c[1].metric("Balanced accuracy", f"{t['balanced_accuracy']:.2f}")
    c[2].metric("Macro F1", f"{t['macro_f1']:.2f}")
    c[3].metric("Abstention rate @thr", f"{a['abstention_rate']:.2f}")
    c[4].metric("Error rate when acting", "—" if a["n_accepted"] == 0 else f"{a['accepted_error_rate']:.2f}")
    st.caption(f"Chance = 0.50. Binomial p-value vs chance (pooled): {t['binomial_p_vs_chance']:.2g}. "
               f"Threshold {METRICS['threshold']['value']:.2f}: {METRICS['threshold']['selection']}.")

    rows = []
    for r in METRICS["per_subject"]:
        at = r["test_at_threshold"]
        rows.append({
            "Subject": f"S{r['subject']}" + (" (demo)" if r["subject"] == METRICS["demo_subject"]["subject"] else ""),
            "Val bAcc": f"{r['validation']['balanced_accuracy_mean']:.2f} ± {r['validation']['balanced_accuracy_std']:.2f}",
            "Test bAcc": round(r["test"]["balanced_accuracy"], 2),
            "Test n": r["n_test"],
            "Coverage @thr": round(at["coverage"], 2),
            "Acc when acting": None if at["n_accepted"] == 0 else round(at["accepted_accuracy"], 2),
            "p vs chance": float(f"{r['test']['binomial_p_vs_chance']:.2g}"),
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    figs = [("per_subject_balanced_accuracy.png", "Per-subject balanced accuracy"),
            ("risk_coverage.png", "Risk-coverage: acting less often, more accurately"),
            ("confusion_pooled_test.png", "Confusion matrix — pooled held-out test"),
            ("confusion_demo_subject_test.png", "Confusion matrix — demo subject"),
            ("csp_patterns_demo_subject.png", "CSP spatial patterns — demo subject")]
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, (f, cap) in zip(cols, figs[i:i + 2]):
            if (REP / f).exists():
                col.image(str(REP / f), caption=cap, width="stretch")

# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("What this prototype does")
    st.markdown(f"""
1. **Recorded EEG** — trials come from run {METRICS['runs']['test']} of the PhysioNet EEG Motor
   Movement/Imagery dataset, which the model never saw during training.
2. **Quality gate** — trials with gross artifacts, flat electrodes or mains interference are
   rejected before classification.
3. **CSP + LDA decoder** — {CFG['preprocessing']['l_freq']:.0f}–{CFG['preprocessing']['h_freq']:.0f} Hz
   band-pass, {len(CH_NAMES)} sensorimotor channels, trained per participant on runs
   {METRICS['runs']['train_validation']}.
4. **Abstention** — if the top model score is below the validated threshold, the system stays neutral.
5. **Configured action** — a confident prediction only proposes *next* or *select* on a fixed menu.
6. **Confirmation** — nothing happens until the user confirms.
7. **Gemini** — after the user selects a menu item, Gemini suggests phrases; the user chooses or
   edits the final message. If Gemini is unavailable, default phrases are shown.
""")
    st.subheader("Limitations")
    st.markdown("""
- **Not a medical device.** No clinical validation; no medical benefit is claimed.
- **Not mind reading.** The model distinguishes two imagined hand movements; it does not decode
  thoughts, words or intentions.
- **Small data.** Each participant has ~45 imagery trials; per-participant test sets have ~15 trials,
  so individual accuracies are highly uncertain.
- **Within-session, per-participant models.** No cross-participant or cross-day generalisation is shown.
- **Uncalibrated scores.** The displayed score is LDA's `predict_proba`, not a calibrated probability.
- **Synthetic corruptions** are added to recorded trials for demonstration and are labelled as such.
- **No live EEG.** Hardware integration is out of scope for this prototype.
""")
