"""Stitch teacher runs into soft-label files for the student.

  results/soft/<name>_xfit.npz      every training ECG scored by a teacher that did not see it
                                    (four cross-fitted runs g1..g4); validation/test rows hold the
                                    mean of the four runs and are used only to report the teacher.
  results/soft/<name>_insample.npz  the single teacher trained on all eight folds (ablation).

<name> is one architecture or an ensemble 'a+b' (logit average).  Also prints and stores the
teacher's own validation / test metrics.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402

T = f"{C.ROOT}/results/teachers"
G = {"g1": (1, 2), "g2": (3, 4), "g3": (5, 6), "g4": (7, 8)}


def xfit(arch, seed=0):
    meta, _ = C.load_meta()
    fold = meta.strat_fold.values
    Z = np.zeros((len(meta), 28), np.float32)
    held = np.zeros(len(meta), bool)
    rest = []
    for g, fs in G.items():
        z = np.load(f"{T}/{arch}_{g}_s{seed}/logits.npz")
        assert (z["ecg_id"] == meta.ecg_id.values).all()
        m = np.isin(fold, fs)
        assert z["oof"][m].all()
        Z[m] = z["logits"][m]
        held |= m
        rest.append(z["logits"])
    ev = fold >= 9
    Z[ev] = np.mean(rest, 0)[ev]
    assert (held | ev).all()
    return Z, held | ev


def insample(arch, seed=0):
    z = np.load(f"{T}/{arch}_none_s{seed}/logits.npz")
    return z["logits"], z["oof"]


def main():
    names = sys.argv[1:]
    meta, _ = C.load_meta()
    Y5, _ = C.labels(meta)
    va, te = (meta.split == "val").values, (meta.split == "test").values
    os.makedirs(f"{C.ROOT}/results/soft", exist_ok=True)
    rep = {}
    for name in names:
        for kind, fn in (("xfit", xfit), ("insample", insample)):
            try:
                parts = [fn(a) for a in name.split("+")]
            except FileNotFoundError as e:
                print("skip", name, kind, "-", os.path.basename(os.path.dirname(str(e.filename))))
                continue
            Z = np.mean([p[0] for p in parts], 0)
            oof = np.all([p[1] for p in parts], 0)
            np.savez_compressed(f"{C.ROOT}/results/soft/{name}_{kind}.npz", logits=Z, oof=oof,
                                ecg_id=meta.ecg_id.values)
            P = 1 / (1 + np.exp(-Z[:, :5]))
            thr = C.pick_thresholds(Y5[va], P[va])
            rep[f"{name}_{kind}"] = dict(val=C.metrics(Y5[va], P[va], thr),
                                         test=C.metrics(Y5[te], P[te], thr))
            print(f"{name:28s} {kind:9s} val macro AUC {rep[f'{name}_{kind}']['val']['macro_auc']:.4f}")
    old = {}
    if os.path.exists(f"{C.ROOT}/results/teacher_metrics.json"):
        old = json.load(open(f"{C.ROOT}/results/teacher_metrics.json"))
    old.update(rep)
    json.dump(old, open(f"{C.ROOT}/results/teacher_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
