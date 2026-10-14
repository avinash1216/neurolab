"""Smoke test of the Streamlit UI flow (requires artifacts from scripts/train.py)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")

pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "metrics.json").exists(),
                                reason="run scripts/train.py first")


def _button(at, startswith):
    return next(b for b in at.button if b.label.startswith(startswith))


def test_app_loads_and_decodes():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("RECORDED EEG PLAYBACK" in m.value for m in at.markdown)

    _button(at, "◀ Replay a LEFT").click().run()
    assert not at.exception
    assert at.session_state.stage in ("candidate", "neutral")


@pytest.mark.parametrize("corruption", ["movement_artifact", "flat_channel", "line_noise"])
def test_corrupted_trial_is_neutral(corruption):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox[1].set_value(corruption).run()
    _button(at, "🎲 Random").click().run()
    assert not at.exception
    assert at.session_state.stage == "neutral"
    assert at.session_state.last["decision"].state.value == "POOR_SIGNAL"


def test_select_to_phrases_with_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    # Drive the state machine directly: a confirmed "select" lands on the phrases stage.
    at.session_state.stage = "phrases"
    at.session_state.selected_item = {"id": "water", "label": "Water / drink", "icon": "💧",
                                      "fallback_phrases": ["Water please."]}
    at.run()
    _button(at, "✨ Suggest").click().run()
    assert not at.exception
    assert at.session_state.phrases.source == "fallback"
    assert any(b.label == "Water please." for b in at.button)
