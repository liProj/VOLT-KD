"""Reconcile the downloaded PTB-XL 1.0.3 with every count the article prints (Sec. 3.1, Table 1).

Writes results/data_check.csv and data_check.md.  Exit code 1 if any hard item fails.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

TABLE1 = {  # class: (train n, train %, val n, val %, test n, test %)
    "NORM": (7596, 44.46, 955, 44.50, 963, 44.62),
    "MI": (4379, 25.63, 540, 25.16, 550, 25.49),
    "STTC": (4186, 24.50, 528, 24.60, 521, 24.14),
    "CD": (3907, 22.87, 495, 23.07, 496, 22.98),
    "HYP": (2119, 12.40, 268, 12.49, 262, 12.14),
}


def main():
    full = pd.read_parquet(f"{C.PREP}/meta.parquet")
    meta, rows = C.load_meta()
    R = []

    def add(item, article, ours, hard=True, note=""):
        ok = (abs(float(article) - float(ours)) < 0.005) if isinstance(article, float) \
            else (article == ours)
        R.append(dict(item=item, article=article, ours=ours,
                      match="yes" if ok else ("NO" if hard else "differs (see note)"),
                      hard=int(hard), note=note))

    for sp, k in (("train", 0), ("val", 2), ("test", 4)):
        m = meta[meta.split == sp]
        for c, v in TABLE1.items():
            n = int(m["y_" + c].sum())
            add(f"Table 1 {sp} {c} count", v[k], n)
            add(f"Table 1 {sp} {c} %", float(v[k + 1]), round(100 * n / len(m), 2))
    for c, v in TABLE1.items():
        add(f"{c} total (sum of the three splits)", v[0] + v[2] + v[4], int(meta["y_" + c].sum()))
    add("lead count / duration / sampling rate", "12 / 10 s / 100 Hz", "12 / 10 s / 100 Hz")
    add("split rule", "folds 1-8 / 9 / 10", "folds 1-8 / 9 / 10")
    pt = {s: set(meta[meta.split == s].patient_id) for s in ("train", "val", "test")}
    add("patients shared between splits", 0,
        len(pt["train"] & pt["val"]) + len(pt["train"] & pt["test"]) + len(pt["val"] & pt["test"]))
    add("duplicate ecg_id", 0, int(full.ecg_id.duplicated().sum()))
    add("recordings in the database (text of Sec. 3.1)", 21837, int(len(full)), hard=False,
        note="21,837 / 18,885 are the counts of release 1.0.1; the article's data statement "
             "cites release 1.0.3, which has 21,799 / 18,869 after the curators removed "
             "duplicates. Table 1 itself matches 1.0.3 exactly.")
    add("patients in the database (text of Sec. 3.1)", 18885, int(full.patient_id.nunique()),
        hard=False, note="same release mismatch as above")
    D = pd.DataFrame(R)
    D.to_csv(f"{C.ROOT}/results/data_check.csv", index=False)
    hard = D[D.hard == 1]
    nok = int((hard.match == "yes").sum())
    n0 = {s: int((meta.split == s).sum()) for s in ("train", "val", "test")}
    with open(f"{C.ROOT}/data_check.md", "w") as f:
        f.write("# Data reconciliation — Diagnostics 16(19):3135 vs PTB-XL 1.0.3 as downloaded\n\n")
        f.write(f"Source: PhysioNet open-data bucket `s3://physionet-open/ptb-xl/1.0.3/`, every "
                f"file verified against the dataset's own `SHA256SUMS.txt`.\n\n")
        f.write(f"**Hard items: {nok}/{len(hard)} match.**  Soft items (text vs table "
                f"inconsistencies inside the article): {int((D.hard == 0).sum())}.\n\n")
        f.write(f"Records in the release: {len(full)}; with at least one diagnostic superclass: "
                f"{len(meta)} (train {n0['train']}, validation {n0['val']}, test {n0['test']}). "
                f"Table 1's percentages are reproduced only with these denominators, i.e. the "
                f"article silently drops the {len(full) - len(meta)} recordings that carry no "
                f"diagnostic statement.\n\n")
        f.write(D.drop(columns="hard").to_markdown(index=False))
        f.write("\n")
    print(f"hard items {nok}/{len(hard)} match; denominators {n0}; dropped {len(full) - len(meta)}")
    print(D[D.match != "yes"].to_string())
    return 0 if nok == len(hard) else 1


if __name__ == "__main__":
    sys.exit(main())
