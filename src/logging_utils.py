"""Structured JSON event logs to stdout (picked up by Cloud Logging). Never logs raw EEG."""

from __future__ import annotations

import json
import sys
import time
from typing import Any


def log_event(event: str, session_id: str, **fields: Any) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event,
              "session_id": session_id, **fields}
    print(json.dumps(record, default=str), file=sys.stdout, flush=True)
