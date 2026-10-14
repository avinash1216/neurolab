# NeuroBridge

**Neural intent → accessible communication**

NeuroBridge is a research prototype built for the AI Builder Cup (14–18 October 2026). It demonstrates a task-specific EEG motor-imagery intent decoder, maps confident predictions to explicitly configured interface actions, and uses Gemini to suggest communication phrases that a user confirms before anything is spoken or sent.

> ⚠️ This is a hackathon research demo — **not a medical device and not a mind-reading system.**

---

## Overview

People with severe motor impairments face significant communication barriers. NeuroBridge explores whether a lightweight, well-understood EEG classifier can be combined with a confirmation-first UI and an LLM phrase assistant to produce a safer, more accessible communication aid.

The system:

1. Loads EEG motor-imagery trials from a public dataset.
2. Classifies the trial with a documented baseline model (CSP + LDA).
3. Abstains when the signal or prediction is uncertain.
4. Maps a confident prediction to a configured interface action.
5. Asks Gemini for a small set of phrase suggestions **only after the user confirms the intent**.
6. Lets the user choose and edit the final phrase before it is displayed or spoken.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | Streamlit (accessible, high-contrast UI) |
| Signal processing | MNE-Python, NumPy, SciPy |
| Baseline model | scikit-learn (CSP + LDA) |
| LLM assistant | Gemini API (phrase suggestions only) |
| Hosting | Google Cloud Run |
| Backend API | FastAPI (optional, only if separation is useful) |

## Dataset

[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) — a small, documented subset is used for the prototype. The exact subset, preprocessing steps, train/validation/test split strategy, and random seeds are versioned alongside the model artifact.

## Status & roadmap

Working plan — four-day execution window:

- **Day 1 (14 Oct)** — Dataset, preprocessing, baseline model, initial metrics.
- **Day 2 (15 Oct)** — UI, recorded EEG playback, class-to-action mapping, confirmation + abstention states.
- **Day 3 (16 Oct)** — Gemini integration, error handling, end-to-end deploy to Google Cloud.
- **Day 4 (17–18 Oct)** — Regression test, document actual metrics and limitations, record demo, submit.

## Definition of done

- [ ] Public EEG dataset loaded; preprocessing reproducible.
- [ ] Baseline classifier trained and evaluated on held-out trials/sessions.
- [ ] UI clearly distinguishes recorded-data playback from live EEG input.
- [ ] Predictions map only to explicitly configured interface actions.
- [ ] Low-confidence input leads to a neutral/retry state, not an action.
- [ ] Gemini suggests phrases only after explicit user confirmation.
- [ ] App runs end-to-end and is deployed to Google Cloud if required.
- [ ] Real metrics, limitations, and failure cases are shown — no invented performance claims.
- [ ] Submission assets complete: proposal PDF, prototype link, demo video.

## Setup

> _Setup instructions will be added as the project takes shape._

```bash
# Planned
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

> _Usage instructions will be added once the Streamlit app exists._

## Evaluation

The project will report:

- Exact dataset subset, labels, preprocessing, split strategy, and model version.
- Accuracy plus balanced accuracy / F1 and a confusion matrix.
- Held-out trial/session results — not random-window splits that can leak similar EEG segments across train and test.
- Abstention rate and the error rate of accepted predictions.
- At least one uncertain/noisy case that correctly results in no action.

## Risks & non-goals

- Not a medical device; no clinical claims.
- The classifier is not described as reading arbitrary thoughts or natural-language intent.
- Illustrative scores are never presented as actual results.
- Gemini never converts uncertain predictions into messages without user confirmation.
- Complex multi-agent orchestration, Kubernetes, and real-time hardware integration are explicitly out of scope until the end-to-end demo works.

## References

- [AI Builder Cup](https://aibuildercup.com/)
- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [MNE-Python](https://mne.tools/stable/)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [Gemini API](https://ai.google.dev/gemini-api/docs)

## License

TBD.

---

_Author: Avinash Kumar_
