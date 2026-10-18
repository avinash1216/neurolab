"""Gemini phrase suggestions for a USER-CONFIRMED menu item, with a static fallback.

Only the confirmed menu label and optional user-typed context are sent.
No EEG data, model scores or identifiers leave the app.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Vertex AI (preferred on Google Cloud: uses the runtime service account / ADC, no API key).
# Set GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION) to enable it.
def _vertex_settings() -> tuple[bool, str, str]:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    use = os.environ.get("GEMINI_USE_VERTEX", "true" if project else "false").lower() == "true"
    return use, project, location

SYSTEM_PROMPT = (
    "You help a person who uses an assistive communication device. "
    "Given the topic they have confirmed and optional context, return between 3 and 5 short, "
    "polite, first-person phrases (at most 12 words each) they could say. "
    "Do not give medical advice, do not invent facts that were not provided, "
    "never use placeholders or brackets such as [number], "
    "and do not add anything other than the phrases."
)


@dataclass
class PhraseResult:
    phrases: list[str]
    source: str          # "gemini" or "fallback"
    note: str = ""


def gemini_backend() -> str | None:
    """'vertex', 'api_key', or None when Gemini is not configured."""
    use, project, _ = _vertex_settings()
    if use and project:
        return "vertex"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "api_key"
    return None


def gemini_available() -> bool:
    return gemini_backend() is not None


def _client():
    from google import genai

    if gemini_backend() == "vertex":
        _, project, location = _vertex_settings()
        return genai.Client(vertexai=True, project=project, location=location)
    return genai.Client()


def _call_gemini(topic: str, context: str, model: str) -> list[str]:
    from google.genai import types

    client = _client()
    prompt = f"Confirmed topic: {topic}\nContext from user (may be empty): {context or '-'}"
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            # 2.5 models count "thinking" tokens against the output budget, which can
            # truncate the JSON; thinking is unnecessary for short phrase lists.
            max_output_tokens=512,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
        ),
    )
    phrases = [p.strip() for p in json.loads(resp.text) if isinstance(p, str) and p.strip()]
    return phrases[:5]


def suggest_phrases(topic: str, fallback: list[str], context: str = "",
                    timeout_s: float = 15.0, model: str = DEFAULT_MODEL) -> PhraseResult:
    if not gemini_available():
        return PhraseResult(fallback, "fallback", "Gemini not configured - showing default phrases.")
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        phrases = pool.submit(_call_gemini, topic, context, model).result(timeout=timeout_s)
        if not phrases:
            return PhraseResult(fallback, "fallback", "Gemini returned no phrases - showing defaults.")
        return PhraseResult(phrases, "gemini")
    except FutureTimeout:
        return PhraseResult(fallback, "fallback", f"Gemini timed out after {timeout_s:.0f}s - showing defaults.")
    except Exception as exc:  # API, auth, quota, parsing
        return PhraseResult(fallback, "fallback", f"Gemini unavailable ({type(exc).__name__}) - showing defaults.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
