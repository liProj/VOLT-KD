"""Collect every finished run and build the comparison tables.

results/summary.csv        one row per (tag): mean and sample SD over the five seeds, val + test
results/per_seed.csv       one row per (tag, seed)
results/vs_published.csv   ours vs every number the article prints for a configuration
results/repro_check.csv    our re-implementation of the article's configurations vs its numbers
results/paired.csv         seed-paired differences (mean, 95% t interval -- the article's own
                           device) and a paired bootstrap over test ECGs, with Holm correction
results/tally.json         how many published comparisons the proposed model wins

Usage: python evaluate.py [--main <tag of the proposed model>]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from scipy import stats                               # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score   # noqa: E402

R = f"{C.ROOT}/results"


def load_runs():
    rows = []
    for f in sorted(glob.glob(f"{C.RUNS}/*/s*/metrics.json")):
        m = json.load(open(f))
        if m["tag"].startswith("_"):
            continue
        row = dict(tag=m["tag"], seed=m["seed"], params=m["params"], best_epoch=m["best_epoch"],
                   epochs_run=m["epochs_run"], train_sec=m["train_sec"], protocol=m["protocol"],
                   prep=m["prep"])
        for sp in ("val", "test"):
            for k, v in m.get(sp, {}).items():
                row[f"{sp}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def welch(m1, s1, m2, s2, n=5):
    se = np.sqrt(s1 ** 2 / n + s2 ** 2 / n)
    if se == 0:
        return np.nan
    t = (m1 - m2) / se
    df = (s1 ** 2 / n + s2 ** 2 / n) ** 2 / ((s1 ** 2 / n) ** 2 / (n - 1) + (s2 ** 2 / n) ** 2 / (n - 1))
    return float(2 * stats.t.sf(abs(t), df))


def test_probs(tag, seeds=None):
    """(seeds, n_test, 5) probabilities and thresholds (seeds, 5) for the seeds that exist
    (peripheral ablations were run with three of the five seeds), plus the seed list."""
    P, T, used = [], [], []
    for s in (seeds or C.SEEDS):
        f = f"{C.RUNS}/{tag}/s{s}/preds.npz"
        if not os.path.exists(f) or not os.path.exists(f"{C.RUNS}/{tag}/s{s}/metrics.json"):
            continue
        z = np.load(f)
        if "test_logit" not in z:
            continue
        P.append(1 / (1 + np.exp(-z["test_logit"])))
        T.append(z["thr"])
        used.append(s)
    if not P:
        return None, None, []
    return np.stack(P), np.stack(T), used


def seedmean_metrics(P, T, y, idx):
    """Mean over seeds of the seven headline metrics on the resample `idx`."""
    out = np.zeros(7)
    yy = y[idx]
    for p, t in zip(P, T):
        pp = p[idx]
        yb = pp >= t[None]
        tp = (yb & (yy > 0)).sum(0)
        f1c = 2 * tp / np.maximum(yb.sum(0) + yy.sum(0), 1)
        out += np.array([
            roc_auc_score(yy, pp, average="macro"), roc_auc_score(yy, pp, average="micro"),
            average_precision_score(yy, pp, average="macro"), f1c.mean(),
            2 * tp.sum() / max(yb.sum() + yy.sum(), 1),
            float((f1c * yy.sum(0)).sum() / yy.sum()), float((yb == (yy > 0)).all(1).mean())])
    return out / len(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    D = load_runs()
    D.to_csv(f"{R}/per_seed.csv", index=False)
    num = [c for c in D.columns if c.startswith(("val_", "test_"))]
    g = D.groupby("tag")
    S = g[num].mean().add_suffix("_mean").join(g[num].std(ddof=1).add_suffix("_sd"))
    S["n_seeds"] = g.seed.nunique()
    S["params"] = g.params.first()
    S["best_epoch_mean"] = g.best_epoch.mean()
    S["train_sec_mean"] = g.train_sec.mean()
    S.reset_index().to_csv(f"{R}/summary.csv", index=False)
    print(S[["n_seeds", "params"] + [f"val_{m}_mean" for m in ("macro_auc", "macro_f1")]
            + ([f"test_{m}_mean" for m in ("macro_auc", "macro_ap", "macro_f1")]
               if "test_macro_auc_mean" in S else [])].round(4).to_string())

    # ---- fidelity of our re-implementation of the article's configurations
    rows = []
    for cfg, pub in C.PUBLISHED.items():
        tag = f"ref_{cfg}"
        if tag not in S.index or "test_macro_auc_mean" not in S:
            continue
        for m, (pm, psd) in pub.items():
            om, osd = S.loc[tag, f"test_{m}_mean"], S.loc[tag, f"test_{m}_sd"]
            rows.append(dict(config=cfg, metric=m, published_mean=pm, published_sd=psd,
                             ours_mean=om, ours_sd=osd, diff=om - pm, n_seeds=int(S.loc[tag, "n_seeds"]),
                             p_welch=welch(om, osd, pm, psd)))
    if rows:
        pd.DataFrame(rows).to_csv(f"{R}/repro_check.csv", index=False)

    if not a.main or a.main not in S.index or "test_macro_auc_mean" not in S:
        return 0
    # ---- proposed model vs everything the article prints
    rows = []
    for cfg, pub in C.PUBLISHED.items():
        for m, (pm, psd) in pub.items():
            om, osd = S.loc[a.main, f"test_{m}_mean"], S.loc[a.main, f"test_{m}_sd"]
            rows.append(dict(config=cfg, metric=m, published_mean=pm, published_sd=psd,
                             ours_mean=om, ours_sd=osd, diff=om - pm, win=int(om > pm),
                             p_welch=welch(om, osd, pm, psd)))
    V = pd.DataFrame(rows)
    V["p_holm"] = holm(V.p_welch.fillna(1).values)
    V["significant"] = ((V.p_holm < 0.05) & (V.win == 1)).astype(int)
    V.to_csv(f"{R}/vs_published.csv", index=False)
    # seed-42 numbers of the article (Table 5, Sec. 4.1) vs our seed-42 run
    r42 = D[(D.tag == a.main) & (D.seed == 42)].iloc[0]
    rows42 = []
    for cfg, pub in C.PUBLISHED_SEED42.items():
        for m, pm in pub.items():
            rows42.append(dict(config=cfg, metric=m, published_seed42=pm, ours_seed42=r42[f"test_{m}"],
                               diff=r42[f"test_{m}"] - pm, win=int(r42[f"test_{m}"] > pm)))
    V42 = pd.DataFrame(rows42)
    V42.to_csv(f"{R}/vs_published_seed42.csv", index=False)

    # ---- paired comparisons against our own runs of the article's configurations
    meta, _ = C.load_meta()
    Y5, _ = C.labels(meta)
    y = Y5[(meta.split == "test").values]
    rng = np.random.RandomState(0)
    boots = [rng.randint(0, len(y), len(y)) for _ in range(a.boot)]
    cache = {}
    rows = []
    for tag in S.index:
        if tag == a.main or tag.startswith("dev_"):
            continue
        Po, To, used = test_probs(tag)
        if Po is None or len(used) < 2:
            continue
        key = tuple(used)                               # the main model on the same seeds
        if key not in cache:
            Pm, Tm, um = test_probs(a.main, used)
            if um != used:
                continue
            cache[key] = np.array([seedmean_metrics(Pm, Tm, y, b) for b in boots])
        Bm = cache[key]
        Bo = np.array([seedmean_metrics(Po, To, y, b) for b in boots])
        for j, m in enumerate(C.MAIN7):
            dm = D[D.tag == a.main].set_index("seed")[f"test_{m}"]
            do = D[D.tag == tag].set_index("seed")[f"test_{m}"]
            common = [s for s in C.SEEDS if s in dm.index and s in do.index]
            d = (dm[common] - do[common]).values
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            db = Bm[:, j] - Bo[:, j]
            pb = 2 * min((db <= 0).mean(), (db >= 0).mean())
            rows.append(dict(vs=tag, metric=m, main_mean=dm[common].mean(), other_mean=do[common].mean(),
                             diff=d.mean(), t_lo=d.mean() - half, t_hi=d.mean() + half,
                             p_paired_t=float(stats.ttest_rel(dm[common], do[common]).pvalue),
                             boot_lo=np.percentile(db, 2.5), boot_hi=np.percentile(db, 97.5),
                             p_boot=max(pb, 1 / a.boot), n_seeds=len(common)))
    Pd_ = pd.DataFrame(rows)
    if len(Pd_):
        Pd_["p_boot_holm"] = holm(Pd_.p_boot.values)
        Pd_.to_csv(f"{R}/paired.csv", index=False)
    tally = dict(main=a.main,
                 published_total=int(len(V)), published_wins=int(V.win.sum()),
                 published_sig=int(V.significant.sum()),
                 main7_wins=int(V[V.config == "msca_mamba"].win.sum()),
                 seed42_total=int(len(V42)), seed42_wins=int(V42.win.sum()),
                 losses=V[V.win == 0][["config", "metric", "published_mean", "ours_mean"]]
                 .to_dict("records"),
                 losses_seed42=V42[V42.win == 0].to_dict("records"))
    json.dump(tally, open(f"{R}/tally.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in tally.items() if "loss" not in k}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
