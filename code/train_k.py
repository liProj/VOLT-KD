"""Seed-batched student training: the five seeds of one configuration in a single process.

`models.StudentK` holds five weight-disjoint students; each has its own initialisation, its own
shuffling (generator seeded with the article's seed), its own augmentation draws, its own
checkpoint (best validation Macro ROC-AUC) and its own validation-selected thresholds.  One
AdamW instance updates all of them, which is identical to five separate AdamW instances because
the update is element-wise and no gradient crosses networks.

Outputs are written per seed in the same layout as train.py:
    results/runs/<tag>/s<seed>/{metrics.json, preds.npz, curve.csv, model.pt}
model.pt is the extracted stand-alone `Student` state dict.

With --dev the test fold is never scored.
"""
import argparse
import copy
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402

import models as M                                    # noqa: E402

warnings.filterwarnings("ignore", message="RNN module weights")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--prep", default="mv", choices=list(C.PREPS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--head", default="flat", choices=["noisyor", "flat"])
    ap.add_argument("--w_sub", type=float, default=1.0)
    ap.add_argument("--kd", default="")
    ap.add_argument("--w_kd", type=float, default=1.0)
    ap.add_argument("--w_hard", type=float, default=1.0)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--c1", type=int, default=48)
    ap.add_argument("--c2", type=int, default=96)
    ap.add_argument("--hid", type=int, default=56)
    ap.add_argument("--n1", type=int, default=1)
    ap.add_argument("--n2", type=int, default=2)
    ap.add_argument("--no_rnn", action="store_true")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--leads", default="")
    return ap.parse_args()


def augment(x):
    """x (B, K, 12, T): independent circular shift per network, +-10% gain and noise per ECG."""
    B, K, _, T = x.shape
    sh = torch.randint(0, T, (K,))
    x = torch.stack([torch.roll(x[:, k], int(sh[k]), -1) for k in range(K)], 1)
    g = 1 + 0.1 * (2 * torch.rand(B, K, 1, 1, device=x.device) - 1)
    return x * g + 0.01 * torch.randn_like(x)


@torch.no_grad()
def predict(model, X, prep, K, mask=None, bs=256):
    model.eval()
    sup, sub = [], []
    for i in range(0, len(X), bs):
        x = prep(X[i:i + bs])
        if mask is not None:
            x = x * mask
        a, b = model(x.unsqueeze(1).expand(-1, K, -1, -1))
        sup.append(a.float().cpu())
        sub.append(b.float().cpu())
    return torch.cat(sup).numpy(), torch.cat(sub).numpy()      # (n, K, 5), (n, K, 23)


def main():
    a = get_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    K = len(seeds)
    outs = [f"{C.RUNS}/{a.tag}/s{s}" for s in seeds]
    if all(os.path.exists(f"{o}/metrics.json") for o in outs) and not a.force:
        print("done already:", a.tag)
        return 0
    for o in outs:
        os.makedirs(o, exist_ok=True)
    prep = C.PREPS[a.prep]
    torch.manual_seed(sum(seeds))
    np.random.seed(sum(seeds) % (2 ** 31))
    meta, rows = C.load_meta()
    Y5, Ysub = C.labels(meta)
    tr0 = np.where(meta.split == "train")[0]
    va = np.where(meta.split == "val")[0]
    te = np.where(meta.split == "test")[0]
    trs = []
    for s in seeds:                                    # per-seed labelled subset (if --frac < 1)
        if a.frac < 1:
            rng = np.random.RandomState(1000 + s)
            trs.append(np.sort(rng.choice(tr0, int(round(a.frac * len(tr0))), replace=False)))
        else:
            trs.append(tr0)
    X = C.load_x(rows)
    Y5t = torch.tensor(Y5, device="cuda")
    Ysubt = torch.tensor(Ysub, device="cuda")
    mask = None
    if a.leads:
        mask = torch.zeros(1, 12, 1, device="cuda")
        mask[0, [int(k) for k in a.leads.split(",")], 0] = 1
    soft = None
    if a.kd:
        z = np.load(a.kd)
        assert (z["ecg_id"] == meta.ecg_id.values).all()
        assert bool(z["oof"][tr0].all()) or "insample" in a.kd, "teacher saw its own targets"
        soft = torch.sigmoid(torch.tensor(z["logits"], device="cuda"))

    model = M.StudentK(K, C.child_index(), c1=a.c1, c2=a.c2, hid=a.hid, n1=a.n1, n2=a.n2,
                       head=a.head, rnn=not a.no_rnn).cuda()
    npar = M.n_params(model) // K
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    nb = len(trs[0]) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb,
                                                pct_start=0.15)
    ema = None
    if a.ema > 0:
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    best = [-1.0] * K
    best_ep = [0] * K
    best_sd = [None] * K
    curves = [[] for _ in range(K)]
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perms = torch.stack([torch.tensor(trs[k])[torch.randperm(len(trs[k]), generator=gens[k])]
                             for k in range(K)], 1).cuda()          # (n, K)
        tot = torch.zeros(K, device="cuda")
        for b in range(nb):
            idx = perms[b * a.bs:(b + 1) * a.bs]                     # (bs, K)
            x = prep(X[idx.reshape(-1)]).view(a.bs, K, 12, -1)
            if mask is not None:
                x = x * mask
            if a.aug:
                x = augment(x)
            zs, zsub = model(x)                                      # (bs, K, 5/23)
            lk = a.w_hard * F.binary_cross_entropy_with_logits(
                zs, Y5t[idx], reduction="none").mean((0, 2))
            if a.w_sub > 0:
                lk = lk + a.w_sub * F.binary_cross_entropy_with_logits(
                    zsub, Ysubt[idx], reduction="none").mean((0, 2))
            if soft is not None:
                t = soft[idx]
                lk = lk + a.w_kd * (
                    F.binary_cross_entropy_with_logits(zs, t[..., :5], reduction="none").mean((0, 2))
                    + F.binary_cross_entropy_with_logits(zsub, t[..., 5:], reduction="none").mean((0, 2)))
            opt.zero_grad(set_to_none=True)
            lk.sum().backward()
            opt.step()
            sched.step()
            if ema is not None:
                d = min(a.ema, (1 + ep * nb + b) / (10 + ep * nb + b))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            tot += lk.detach()
        ev = ema if ema is not None else model
        zv, _ = predict(ev, X[va], prep, K, mask)
        pv = 1 / (1 + np.exp(-zv))
        line = []
        for k in range(K):
            auc = roc_auc_score(Y5[va], pv[:, k], average="macro")
            thr = C.pick_thresholds(Y5[va], pv[:, k])
            yb = pv[:, k] >= thr[None]
            tp = (yb & (Y5[va] > 0)).sum(0)
            mf1 = float(np.mean(2 * tp / np.maximum(yb.sum(0) + Y5[va].sum(0), 1)))
            curves[k].append(dict(epoch=ep + 1, train_loss=float(tot[k]) / nb, val_macro_auc=auc,
                                  val_macro_f1=mf1, sec=time.time() - t0))
            line.append(auc)
            if auc > best[k]:
                best[k], best_ep[k] = auc, ep + 1
                best_sd[k] = {n: v.clone() for n, v in ev.extract(k).state_dict().items()}
        print(f"ep {ep + 1:2d} loss {float(tot.mean()) / nb:.4f} val AUC "
              + " ".join(f"{v:.4f}" for v in line) + f" | mean {np.mean(line):.4f} "
              f"{time.time() - t0:.0f}s", flush=True)

    kw = dict(c1=a.c1, c2=a.c2, hid=a.hid, n1=a.n1, n2=a.n2, head=a.head, rnn=not a.no_rnn)
    Xva, Xte = X[va], X[te]
    summ = []
    for k, s in enumerate(seeds):
        net = M.Student(C.child_index(), **kw).cuda()
        net.load_state_dict(best_sd[k])
        net.eval()

        @torch.no_grad()
        def run(Xs):
            sup, sub = [], []
            for i in range(0, len(Xs), 256):
                x = prep(Xs[i:i + 256])
                if mask is not None:
                    x = x * mask
                o = net(x)
                sup.append(o[0].float().cpu())
                sub.append(o[1].float().cpu())
            return torch.cat(sup).numpy(), torch.cat(sub).numpy()
        zv, sv = run(Xva)
        pv = 1 / (1 + np.exp(-zv))
        thr = C.pick_thresholds(Y5[va], pv)
        res = dict(tag=a.tag, model="student", seed=s, protocol="ours", prep=a.prep, params=npar,
                   best_epoch=best_ep[k], epochs_run=a.epochs, train_n=int(len(trs[k])),
                   train_sec=(time.time() - t0) / K, thresholds=thr.tolist(), args=vars(a),
                   val=C.metrics(Y5[va], pv, thr))
        save = dict(val_logit=zv, val_sub=sv, thr=thr, val_id=meta.ecg_id.values[va])
        if not a.dev:
            zt, st = run(Xte)
            res["test"] = C.metrics(Y5[te], 1 / (1 + np.exp(-zt)), thr)
            save.update(test_logit=zt, test_sub=st, test_id=meta.ecg_id.values[te])
        np.savez_compressed(f"{outs[k]}/preds.npz", **save)
        torch.save(best_sd[k], f"{outs[k]}/model.pt")
        pd.DataFrame(curves[k]).to_csv(f"{outs[k]}/curve.csv", index=False)
        json.dump(res, open(f"{outs[k]}/metrics.json", "w"), indent=1)
        summ.append(res["val"])
    print("params/model", npar, "| val mean",
          {m: round(float(np.mean([r[m] for r in summ])), 4) for m in C.MAIN7})
    return 0


if __name__ == "__main__":
    sys.exit(main())
