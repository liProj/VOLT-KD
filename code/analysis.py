"""Post-hoc analyses on the held-out test fold.  Every function writes one CSV (or npz) under
results/analysis/ and touches no training state.

  python analysis.py --main <tag> --zs <z-score student tag> [--only a,b,c]

  voltage     classical voltage criteria (Sokolow-Lyon, Cornell) in mV and after z-scoring
  gain        amplitude-scaling probe: predicted probability and AUC as the ECG is rescaled
  noise       robustness to baseline wander, broadband noise and missing leads
  subgroup    discrimination by sex, age band, validation status, signal quality, device
  subclass    superclass sensitivity within each of the 23 diagnostic subclasses + subclass AUC
  calib       expected calibration error, Brier score and reliability curves
  dca         decision-curve net benefit
  curves      ROC / PR curve points and confusion matrices
  ensemble    five-seed ensembles
  efficiency  parameters, size, latency, throughput, memory, operation count
  saliency    gradient x input attribution for example ECGs
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import torch                                          # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve,   # noqa: E402
                             roc_auc_score, roc_curve)

import models as M                                    # noqa: E402

A = f"{C.ROOT}/results/analysis"
REF = "ref_msca_mamba"
torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ loading helpers
def run_info(tag, seed):
    return json.load(open(f"{C.RUNS}/{tag}/s{seed}/metrics.json"))


def load_model(tag, seed):
    info = run_info(tag, seed)
    a = info["args"]
    if info["model"] == "student":
        net = M.Student(C.child_index(), c1=a["c1"], c2=a["c2"], hid=a["hid"], n1=a["n1"],
                        n2=a["n2"], head=a["head"], rnn=not a["no_rnn"])
    else:
        net = M.MSCA(**M.REF_CONFIGS[info["model"]][0])
    net.load_state_dict(torch.load(f"{C.RUNS}/{tag}/s{seed}/model.pt", map_location="cpu"))
    return net.cuda().eval(), C.PREPS[info["prep"]], np.array(info["thresholds"]), info


@torch.no_grad()
def probs(net, prep, X, transform=None, bs=256):
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs]
        if transform is not None:
            x = transform(x, i)
        o = net(prep(x))
        o = o[0] if isinstance(o, tuple) else o
        out.append(torch.sigmoid(o.float()).cpu())
    return torch.cat(out).numpy()


def saved_probs(tag, split="test"):
    P, T = [], []
    for s in C.SEEDS:
        z = np.load(f"{C.RUNS}/{tag}/s{s}/preds.npz")
        P.append(1 / (1 + np.exp(-z[f"{split}_logit"])))
        T.append(z["thr"])
    return np.stack(P), np.stack(T)


class Data:
    def __init__(self):
        self.meta, self.rows = C.load_meta()
        self.Y5, self.Ysub = C.labels(self.meta)
        self.te = np.where(self.meta.split == "test")[0]
        self.mt = self.meta.iloc[self.te].reset_index(drop=True)
        self.y = self.Y5[self.te]
        self.ysub = self.Ysub[self.te]
        self._X = None

    @property
    def X(self):                                      # int16 test ECGs on the GPU (n, 1000, 12)
        if self._X is None:
            self._X = C.load_x(self.rows[self.te])
        return self._X


def macro_auc(y, p):
    return float(roc_auc_score(y, p, average="macro"))


# ------------------------------------------------------------------ analyses
def voltage(D, a):
    """Sokolow-Lyon (S_V1 + max(R_V5, R_V6)) and Cornell (R_aVL + S_V3) voltages, measured as
    robust extremes of the baseline-corrected lead, in mV and after per-lead z-scoring."""
    X = np.load(f"{C.PREP}/X100.npy", mmap_mode="r")
    x = np.asarray(X[D.rows[D.te]], np.float32) / 1000
    x = x - np.median(x, 1, keepdims=True)
    z = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    li = {l: i for i, l in enumerate(C.maps()["leads"])}

    def idx(v):
        hi = lambda l: np.percentile(v[:, :, li[l]], 99, axis=1)       # noqa: E731
        lo = lambda l: -np.percentile(v[:, :, li[l]], 1, axis=1)       # noqa: E731
        return lo("V1") + np.maximum(hi("V5"), hi("V6")), hi("aVL") + lo("V3")
    sok, cor = idx(x)
    sokz, corz = idx(z)
    subs = C.maps()["subs"]
    rows = []
    for name, y in (("HYP", D.y[:, 4]), ("LVH", D.ysub[:, subs.index("LVH")])):
        for crit, v, vz in (("Sokolow-Lyon", sok, sokz), ("Cornell", cor, corz)):
            rows.append(dict(target=name, criterion=crit, n_pos=int(y.sum()),
                             auc_mv=roc_auc_score(y, v), auc_zscore=roc_auc_score(y, vz)))
    # the same on the development folds, where the hypothesis was examined first
    dv = np.where(D.meta.split != "test")[0]
    xd = np.asarray(X[np.sort(D.rows[dv])], np.float32) / 1000
    xd = xd - np.median(xd, 1, keepdims=True)
    zd = (xd - xd.mean(1, keepdims=True)) / (xd.std(1, keepdims=True) + 1e-6)
    sd_, cd_ = idx(xd)
    sdz, cdz = idx(zd)
    for name, y in (("HYP", D.Y5[dv, 4]), ("LVH", D.Ysub[dv, subs.index("LVH")])):
        for crit, v, vz in (("Sokolow-Lyon", sd_, sdz), ("Cornell", cd_, cdz)):
            rows.append(dict(target=name, criterion=crit, n_pos=int(y.sum()), fold="development",
                             auc_mv=roc_auc_score(y, v), auc_zscore=roc_auc_score(y, vz)))
    lv = D.ysub[:, subs.index("LVH")] > 0
    extra = dict(sok_median_lvh=float(np.median(sok[lv])), sok_median_nohyp=float(np.median(sok[D.y[:, 4] == 0])),
                 share35_lvh=float((sok[lv] >= 3.5).mean()), share35_nohyp=float((sok[D.y[:, 4] == 0] >= 3.5).mean()))
    json.dump(extra, open(f"{A}/voltage_extra.json", "w"), indent=1)
    T_ = pd.DataFrame(rows)
    T_["fold"] = T_["fold"].fillna("test")
    T_.to_csv(f"{A}/voltage.csv", index=False)
    rows = T_[T_.fold == "test"].to_dict("records")
    pd.DataFrame(dict(ecg_id=D.mt.ecg_id, sokolow_mv=sok, cornell_mv=cor, sokolow_z=sokz,
                      hyp=D.y[:, 4], lvh=D.ysub[:, subs.index("LVH")],
                      sex=D.mt.sex)).to_csv(f"{A}/voltage_index.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def gain(D, a):
    """Rescale every test ECG by a common factor g.  A network that reads absolute voltage must
    respond; one fed z-scored leads cannot."""
    gains = [0.4, 0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5]
    rows = []
    for tag in [a.main, a.zs, REF]:
        for s in C.SEEDS:
            net, prep, thr, _ = load_model(tag, s)
            for g in gains:
                p = probs(net, prep, D.X, lambda x, i: (x.float() * g))
                r = dict(tag=tag, seed=s, gain=g, macro_auc=macro_auc(D.y, p))
                for j, c in enumerate(C.SUPER):
                    r[f"auc_{c}"] = roc_auc_score(D.y[:, j], p[:, j])
                    r[f"mean_p_{c}"] = float(p[:, j].mean())
                    r[f"flag_rate_{c}"] = float((p[:, j] >= thr[j]).mean())
                rows.append(r)
    pd.DataFrame(rows).to_csv(f"{A}/gain.csv", index=False)


def noise(D, a):
    gen = torch.Generator(device="cuda").manual_seed(0)
    n, T = D.X.shape[0], D.X.shape[1]
    t = torch.arange(T, device="cuda").float() / 100.0
    f = 0.15 + 0.35 * torch.rand(n, 1, 12, device="cuda", generator=gen)
    ph = 6.2832 * torch.rand(n, 1, 12, device="cuda", generator=gen)
    wander = torch.sin(6.2832 * f * t.view(1, T, 1) + ph)             # unit-amplitude drift
    white = torch.randn(n, T, 12, device="cuda", generator=gen)
    order = torch.stack([torch.randperm(12, device="cuda", generator=gen) for _ in range(n)])

    def tf(kind, level):
        def fn(x, i):
            x = x.float()
            b = x.shape[0]
            if kind == "wander":
                return x + 1000 * level * wander[i:i + b]
            if kind == "white":
                return x + 1000 * level * white[i:i + b]
            keep = (order[i:i + b] >= level).float().unsqueeze(1)      # drop `level` leads
            return x * keep
        return fn
    levels = dict(wander=[0, 0.1, 0.2, 0.5, 1.0, 2.0], white=[0, 0.01, 0.02, 0.05, 0.1, 0.2],
                  drop=[0, 1, 2, 4, 6, 8])
    rows = []
    for tag in [a.main, REF]:
        for s in C.SEEDS:
            net, prep, thr, _ = load_model(tag, s)
            for kind, ls in levels.items():
                for lv in ls:
                    p = probs(net, prep, D.X, tf(kind, lv))
                    rows.append(dict(tag=tag, seed=s, kind=kind, level=lv,
                                     macro_auc=macro_auc(D.y, p),
                                     macro_f1=C.metrics(D.y, p, thr)["macro_f1"]))
    pd.DataFrame(rows).to_csv(f"{A}/noise.csv", index=False)


def subgroup(D, a):
    m = D.mt
    q = m[[c for c in m.columns if c.startswith("q_") and c not in ("q_extra_beats", "q_pacemaker")]]
    groups = {
        "All": np.ones(len(m), bool),
        "Male": (m.sex == 0).values, "Female": (m.sex == 1).values,
        "Age < 40": (m.age < 40).values, "Age 40-59": ((m.age >= 40) & (m.age < 60)).values,
        "Age 60-74": ((m.age >= 60) & (m.age < 75)).values, "Age >= 75": (m.age >= 75).values,
        "Human-validated": (m.validated_by_human == True).values,       # noqa: E712
        "Not human-validated": (m.validated_by_human != True).values,   # noqa: E712
        "Clean signal": (q.sum(1) == 0).values, "Any noise flag": (q.sum(1) > 0).values,
        "Single superclass": (m.n_super == 1).values, "Multiple superclasses": (m.n_super > 1).values,
    }
    for dev, cnt in m.device.value_counts().head(4).items():
        groups[f"Device: {dev.strip()}"] = (m.device == dev).values
    rng = np.random.RandomState(0)
    rows = []
    for tag in [a.main, REF]:
        P, _ = saved_probs(tag)
        for gname, mask in groups.items():
            idx = np.where(mask)[0]
            if len(idx) < 50 or (D.y[idx].sum(0) < 5).any():
                continue
            pt = np.mean([macro_auc(D.y[idx], p[idx]) for p in P])
            bs = []
            for _ in range(300):
                b = idx[rng.randint(0, len(idx), len(idx))]
                if (D.y[b].sum(0) < 2).any():
                    continue
                bs.append(np.mean([macro_auc(D.y[b], p[b]) for p in P]))
            rows.append(dict(tag=tag, group=gname, n=len(idx), macro_auc=pt,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    pd.DataFrame(rows).to_csv(f"{A}/subgroup.csv", index=False)


def subclass(D, a):
    mp = C.maps()
    rows = []
    for tag in [a.main, REF]:
        P, T = saved_probs(tag)
        flag = np.stack([p >= t[None] for p, t in zip(P, T)])           # (seeds, n, 5)
        for j, s in enumerate(mp["subs"]):
            sup = C.SUPER.index(mp["sub_sup"][s])
            pos = D.ysub[:, j] > 0
            if pos.sum() == 0:
                continue
            rows.append(dict(tag=tag, subclass=s, superclass=mp["sub_sup"][s], n=int(pos.sum()),
                             sens_mean=float(flag[:, pos, sup].mean()),
                             sens_sd=float(flag[:, pos, sup].mean(1).std(ddof=1))))
    pd.DataFrame(rows).to_csv(f"{A}/subclass_sens.csv", index=False)
    rows = []                                           # the student's own subclass outputs
    for s in C.SEEDS:
        z = np.load(f"{C.RUNS}/{a.main}/s{s}/preds.npz")
        if "test_sub" not in z:
            return
        for j, sc in enumerate(mp["subs"]):
            if D.ysub[:, j].sum() >= 5:
                rows.append(dict(seed=s, subclass=sc, superclass=mp["sub_sup"][sc],
                                 n=int(D.ysub[:, j].sum()),
                                 auc=roc_auc_score(D.ysub[:, j], z["test_sub"][:, j]),
                                 ap=average_precision_score(D.ysub[:, j], z["test_sub"][:, j])))
    pd.DataFrame(rows).to_csv(f"{A}/subclass_auc.csv", index=False)


def calib(D, a):
    rows, curves = [], []
    edges = np.linspace(0, 1, 11)
    for tag in [a.main, REF]:
        P, _ = saved_probs(tag)
        for s, p in zip(C.SEEDS, P):
            for j, c in enumerate(C.SUPER):
                y, q = D.y[:, j], p[:, j]
                b = np.clip(np.digitize(q, edges) - 1, 0, 9)
                ece = sum((b == k).mean() * abs(y[b == k].mean() - q[b == k].mean())
                          for k in range(10) if (b == k).any())
                rows.append(dict(tag=tag, seed=s, cls=c, ece=ece, brier=float(((q - y) ** 2).mean())))
                for k in range(10):
                    if (b == k).any():
                        curves.append(dict(tag=tag, seed=s, cls=c, bin=k, n=int((b == k).sum()),
                                           conf=float(q[b == k].mean()), obs=float(y[b == k].mean())))
    pd.DataFrame(rows).to_csv(f"{A}/calib.csv", index=False)
    pd.DataFrame(curves).to_csv(f"{A}/calib_curve.csv", index=False)


def dca(D, a):
    ths = np.round(np.arange(0.02, 0.81, 0.02), 2)
    rows = []
    for tag in [a.main, REF]:
        P, _ = saved_probs(tag)
        for j, c in enumerate(C.SUPER):
            y = D.y[:, j]
            for t in ths:
                nb = np.mean([((p[:, j] >= t) & (y > 0)).mean()
                              - ((p[:, j] >= t) & (y == 0)).mean() * t / (1 - t) for p in P])
                rows.append(dict(tag=tag, cls=c, threshold=t, net_benefit=nb,
                                 treat_all=y.mean() - (1 - y.mean()) * t / (1 - t)))
    pd.DataFrame(rows).to_csv(f"{A}/dca.csv", index=False)


def curves(D, a):
    out, cm = {}, []
    for tag in [a.main, REF]:
        P, T = saved_probs(tag)
        for j, c in enumerate(C.SUPER):
            grid = np.linspace(0, 1, 201)
            tprs, precs = [], []
            for p in P:
                fpr, tpr, _ = roc_curve(D.y[:, j], p[:, j])
                tprs.append(np.interp(grid, fpr, tpr))
                pr, rc, _ = precision_recall_curve(D.y[:, j], p[:, j])
                precs.append(np.interp(grid, rc[::-1], pr[::-1]))
            out[f"{tag}|{c}|roc"] = np.stack(tprs)
            out[f"{tag}|{c}|pr"] = np.stack(precs)
        for s, p, t in zip(C.SEEDS, P, T):
            yb = p >= t[None]
            for j, c in enumerate(C.SUPER):
                y = D.y[:, j] > 0
                cm.append(dict(tag=tag, seed=s, cls=c, tp=int((yb[:, j] & y).sum()),
                               fp=int((yb[:, j] & ~y).sum()), fn=int((~yb[:, j] & y).sum()),
                               tn=int((~yb[:, j] & ~y).sum())))
    np.savez_compressed(f"{A}/curves.npz", **out)
    pd.DataFrame(cm).to_csv(f"{A}/confusion.csv", index=False)


def ensemble(D, a):
    va = (D.meta.split == "val").values
    rows = []
    tags = sorted(t for t in os.listdir(C.RUNS) if not t.startswith(("_", "dev")))
    for tag in tags:
        try:
            P, _ = saved_probs(tag)
            Pv, _ = saved_probs(tag, "val")
        except (FileNotFoundError, KeyError):
            continue
        thr = C.pick_thresholds(D.Y5[va], Pv.mean(0))
        r = C.metrics(D.y, P.mean(0), thr)
        r["tag"] = tag
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{A}/ensemble.csv", index=False)


def efficiency(D, a):
    from torch.utils.flop_counter import FlopCounterMode
    rows = []
    x1 = D.X[:1]
    tags = [a.main] + [f"ref_{k}" for k in ("msca_mamba", "bimamba", "m_conv", "m_gru", "m_lstm",
                                             "m_transformer", "ms_cnn")]
    for tag in tags:
        if os.path.exists(f"{C.RUNS}/{tag}/s42/metrics.json"):
            net, prep, _, info = load_model(tag, 42)
            trained = 1
        else:                                           # timing does not depend on the weights:
            net = M.MSCA(**M.REF_CONFIGS[tag[4:]][0]).cuda().eval()   # an untrained copy is timed
            prep, info, trained = C.prep_zscore, dict(params=M.n_params(net)), 0
        with torch.no_grad():
            for _ in range(50):
                net(prep(x1))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ts = []
            for _ in range(500):
                t0 = time.perf_counter()
                net(prep(x1))
                torch.cuda.synchronize()
                ts.append(1000 * (time.perf_counter() - t0))
            peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
            xb = D.X[:256]
            t0 = time.perf_counter()
            for _ in range(10):
                net(prep(xb))
            torch.cuda.synchronize()
            thr_b = 2560 / (time.perf_counter() - t0)
            try:
                with FlopCounterMode(display=False) as fc:
                    net(prep(x1))
                flops = fc.get_total_flops()
            except Exception:
                flops = float("nan")
        cpu = net.cpu()
        xc = prep(x1).cpu()
        torch.set_num_threads(1)
        tc = []
        with torch.no_grad():
            try:
                for _ in range(5):
                    cpu(xc)
                for _ in range(30):
                    t0 = time.perf_counter()
                    cpu(xc)
                    tc.append(1000 * (time.perf_counter() - t0))
            except Exception:                          # CUDA-only kernels (mamba_ssm)
                tc = [float("nan")]
        dep = info["params"]
        if hasattr(net, "head") and hasattr(net.head, "sub"):       # the auxiliary subclass layer is not deployed
            dep -= sum(p.numel() for p in net.head.sub.parameters())
        rows.append(dict(tag=tag, params=dep, trained_weights=trained, size_mb=dep * 4 / 2 ** 20,
                         lat_ms=float(np.mean(ts)), lat_sd=float(np.std(ts)),
                         p95_ms=float(np.percentile(ts, 95)), thr_b1=1000 / float(np.mean(ts)),
                         thr_b256=thr_b, peak_mb=peak, flops_m=flops / 1e6,
                         cpu_ms=float(np.median(tc))))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(f"{A}/efficiency.csv", index=False)


def saliency(D, a):
    """Gradient x input for the seed-42 student on one confidently-detected ECG per class."""
    net, prep, thr, _ = load_model(a.main, 42)
    P, _ = saved_probs(a.main)
    out = {}
    for j, c in enumerate(C.SUPER):
        only = (D.y.sum(1) == 1) & (D.y[:, j] == 1)
        i = int(np.where(only)[0][np.argmax(P[0][only, j])])
        x = prep(D.X[i:i + 1]).clone().requires_grad_(True)
        net.train()                                     # cuDNN RNN backward needs train mode
        for mod in net.modules():
            if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.Dropout)):
                mod.eval()
        z = net(x)[0][0, j]
        g, = torch.autograd.grad(z, x)
        net.eval()
        out[f"{c}_x"] = x.detach().cpu().numpy()[0]
        out[f"{c}_attr"] = (g * x).detach().cpu().numpy()[0]
        out[f"{c}_id"] = int(D.mt.ecg_id[i])
        out[f"{c}_p"] = float(P[0][i, j])
    np.savez_compressed(f"{A}/saliency.npz", **out)


ALL = dict(voltage=voltage, gain=gain, noise=noise, subgroup=subgroup, subclass=subclass,
           calib=calib, dca=dca, curves=curves, ensemble=ensemble, efficiency=efficiency,
           saliency=saliency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--zs", default="abl_zscore")
    ap.add_argument("--only", default="")
    ap.add_argument("--ref", default="")                # dry runs only: stand-in reference tag
    ap.add_argument("--out", default="")                # dry runs only: scratch output directory
    a = ap.parse_args()
    global A, REF
    if a.ref:
        REF = a.ref
    if a.out:
        A = a.out
    os.makedirs(A, exist_ok=True)
    D = Data()
    for name in (a.only.split(",") if a.only else ALL):
        t0 = time.time()
        ALL[name](D, a)
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
