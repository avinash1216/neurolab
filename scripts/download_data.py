"""Parallel, resumable download of the EEGMMIDB subset defined in config/experiment.yaml.

PhysioNet can be slow from some networks, so files are fetched concurrently and
partial downloads are resumed with HTTP Range requests. Files land in the same
layout that `mne.datasets.eegbci.load_data` expects, so MNE will not re-download.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_experiment_config  # noqa: E402
from src.data import edf_path  # noqa: E402

BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def download(subject: int, run: int, data_dir: Path, retries: int = 5) -> str:
    sid = f"S{subject:03d}"
    name = f"{sid}R{run:02d}.edf"
    url = f"{BASE_URL}/{sid}/{name}"
    out = edf_path(data_dir, subject, run)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".edf.part")

    if out.exists():
        return f"{name}: cached"

    total = None
    for attempt in range(1, retries + 1):
        try:
            total = _remote_size(url)
            break
        except Exception as exc:
            print(f"  {name}: size check attempt {attempt} failed ({exc})", flush=True)
    if total is None:
        return f"{name}: FAILED"

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(part, "ab") as fh:
                while chunk := resp.read(64 * 1024):
                    fh.write(chunk)
        except Exception as exc:  # network hiccup: resume on next attempt
            print(f"  {name}: attempt {attempt} failed ({exc}); resuming", flush=True)

    if part.exists() and part.stat().st_size == total:
        part.replace(out)
        return f"{name}: ok ({total / 1e6:.1f} MB)"
    return f"{name}: FAILED"


def main() -> None:
    cfg = load_experiment_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="*", default=cfg["data"]["subjects"])
    parser.add_argument("--runs", type=int, nargs="*", default=cfg["data"]["runs"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(cfg["data"]["data_dir"])
    jobs = [(s, r) for s in args.subjects for r in args.runs]
    print(f"Downloading {len(jobs)} files with {args.workers} workers -> {data_dir}", flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, s, r, data_dir): (s, r) for s, r in jobs}
        for fut in as_completed(futures):
            msg = fut.result()
            failed += msg.endswith("FAILED")
            print(msg, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
