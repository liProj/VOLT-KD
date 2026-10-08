"""Download PTB-XL 1.0.3 from PhysioNet's official open-data S3 bucket and verify every file
against the SHA256SUMS.txt shipped with the dataset.

physionet.org itself served ~40 kB/s on the day of the download; the S3 mirror is the same
publisher's bucket (s3://physionet-open/ptb-xl/1.0.3/) and carries the identical file tree.
A file is accepted only if its sha256 matches; a size check alone is not enough after an
interrupted run.

Usage: python fetch_ptbxl.py [--workers 12] [--skip500]
"""
import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from project_paths import ROOT as PROJECT_ROOT
ROOT = os.path.join(PROJECT_ROOT, "data", "ptbxl")
BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
S = requests.Session()
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get(item):
    rel, want = item
    out = os.path.join(ROOT, rel)
    if os.path.exists(out) and (want is None or sha(out) == want):
        return rel, "cached"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(6):
        try:
            r = S.get(BASE + rel, timeout=120)
            if r.status_code == 200:
                h = hashlib.sha256(r.content).hexdigest()
                if want is None or h == want:
                    tmp = out + ".part"
                    with open(tmp, "wb") as f:
                        f.write(r.content)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, out)
                    return rel, "ok"
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return rel, "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip500", action="store_true")
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    print(get(("SHA256SUMS.txt", None)), flush=True)
    items = []
    for line in open(os.path.join(ROOT, "SHA256SUMS.txt")):
        h, rel = line.split(maxsplit=1)
        rel = rel.strip()
        if a.skip500 and rel.startswith("records500/"):
            continue
        items.append((rel, h))
    # metadata and the 100 Hz tree first, so the reproduction can start before 500 Hz is in
    items.sort(key=lambda x: (x[0].startswith("records500/"), x[0].startswith("records100/"), x[0]))
    print(len(items), "files listed", flush=True)
    n = {"ok": 0, "cached": 0, "FAIL": 0}
    fails = []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (rel, st) in enumerate(ex.map(get, items), 1):
            n[st] += 1
            if st == "FAIL":
                fails.append(rel)
            if i % 2000 == 0:
                print(i, n, f"{time.time() - t0:.0f}s", flush=True)
    print("done", n, f"{time.time() - t0:.0f}s")
    if fails:
        print("FAILED:", fails[:20])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
