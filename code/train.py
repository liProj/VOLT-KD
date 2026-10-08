"""Train one model on PTB-XL (official folds: 1-8 train, 9 validation, 10 test).

Two protocols share this loop.
  --protocol ref   the article's recipe: per-record per-lead z-score, AdamW 1e-4 / wd 1e-4,
                   batch 16, <= 30 epochs, early stopping on validation Macro ROC-AUC
                   (patience 7, min_delta 1e-4), unweighted BCE, best-validation checkpoint.
  --protocol ours  the student recipe (see the flags).
Model selection and threshold selection use the validation fold only.  With --dev the test
fold is never scored, which is how every design decision in this round was taken.

Output: results/runs/<tag>/s<seed>/{metrics.json, preds.npz, curve.csv, model.pt}
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402  (sets thread env first)
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402

import models as M                                    # noqa: E402

import warnings                                       # noqa: E402
warnings.filterwarnings("ignore", message="RNN module weights")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)         # a REF_CONFIGS key or "student"
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocol", default="ref", choices=["ref", "ours"])
    ap.add_argument("--prep", default=None)           # zscore | mv ; default by protocol
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--force", action="store_true")
    # student-only switches (each is one row of the ablation table)
    ap.add_argument("--head", default="flat", choices=["noisyor", "flat"])
    ap.add_argument("--w_sub", type=float, default=1.0)   # subclass supervision weight
    ap.add_argument("--kd", default="")                   # soft-label file (results/soft/*.npz)
    ap.add_argument("--w_kd", type=float, default=1.0)
    ap.add_argument("--w_hard", type=float, default=1.0)  # weight of the hard superclass labels
    ap.add_argument("--kd_sub", type=int, default=1)      # distil the 23 subclass outputs too
    ap.add_argument("--ema_every", type=int, default=4)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--c1", type=int, default=48)
    ap.add_argument("--c2", type=int, default=96)
    ap.add_argument("--hid", type=int, default=56)
    ap.add_argument("--n1", type=int, default=1)
    ap.add_argument("--n2", type=int, default=2)
    ap.add_argument("--no_rnn", action="store_true")
    ap.add_argument("--frac", type=float, default=1.0)    # fraction of labelled training ECGs
    ap.add_argument("--leads", default="")                # comma list of lead indices to keep
    return ap.parse_args()


def augment(x):
    """Light, voltage-respecting augmentation: circular time shift, +-10% gain, small noise."""
    B, _, T = x.shape
    s = int(torch.randint(0, T, (1,)))
    x = torch.roll(x, s, -1)
    g = 1 + 0.1 * (2 * torch.rand(B, 1, 1, device=x.device) - 1)
    return x * g + 0.01 * torch.randn_like(x)


@torch.no_grad()
def predict(model, X, prep, bs=256, student=False, leadmask=None):
    model.eval()
    sup, sub = [], []
    for i in range(0, len(X), bs):
        x = prep(X[i:i + bs])
        if leadmask is not None:
            x = x * leadmask
        o = model(x)
        if student:
            sup.append(o[0].float().cpu())
            sub.append(o[1].float().cpu())
        else:
            sup.append(o.float().cpu())
    return torch.cat(sup).numpy(), (torch.cat(sub).numpy() if sub else None)


def main():
    a = get_args()
    out = f"{C.RUNS}/{a.tag}/s{a.seed}"
    if os.path.exists(f"{out}/metrics.json") and not a.force:
        print("done already:", out)
        return 0
    os.makedirs(out, exist_ok=True)
    ref = a.protocol == "ref"
    student = a.model == "student"
    prep_name = a.prep or ("zscore" if ref else "mv")
    prep = C.PREPS[prep_name]
    epochs = a.epochs or (30 if ref else 24)
    bs = a.bs or (16 if ref else 64)
    lr = a.lr or (1e-4 if ref else 4e-3)
    wd = a.wd if a.wd is not None else (1e-4 if ref else 1e-2)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    meta, rows = C.load_meta()
    Y5, Ysub = C.labels(meta)
    tr = np.where(meta.split == "train")[0]
    va = np.where(meta.split == "val")[0]
    te = np.where(meta.split == "test")[0]
    if a.frac < 1:                                    # label-efficiency experiments
        rng = np.random.RandomState(1000 + a.seed)
        tr = np.sort(rng.choice(tr, int(round(a.frac * len(tr))), replace=False))
    X = C.load_x(rows)                                # int16 on the GPU, (N, 1000, 12)
    Y5t = torch.tensor(Y5, device="cuda")
    Ysubt = torch.tensor(Ysub, device="cuda")
    leadmask = None
    if a.leads:
        keep = [int(k) for k in a.leads.split(",")]
        leadmask = torch.zeros(1, 12, 1, device="cuda")
        leadmask[0, keep, 0] = 1

    soft = None
    if a.kd:
        z = np.load(a.kd)
        assert (z["ecg_id"] == meta.ecg_id.values).all()
        soft = torch.tensor(z["logits"], device="cuda")           # (N, 5 + 23) teacher logits
        assert bool(z["oof"][tr].all()) or "insample" in a.kd, "teacher saw its own targets"

    if student:
        model = M.Student(C.child_index(), c1=a.c1, c2=a.c2, hid=a.hid, n1=a.n1, n2=a.n2,
                          head=a.head, rnn=not a.no_rnn).cuda()
    else:
        model = M.MSCA(**M.REF_CONFIGS[a.model][0]).cuda()
    npar = M.n_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * (len(tr) // bs + (0 if not ref else 1))
    sched = None if ref else torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * (len(tr) // bs), pct_start=0.15)
    ema = None
    if not ref and a.ema > 0:
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        ema_p, model_p = list(ema.parameters()), list(model.parameters())

    best, best_ep, bad, curve = -1.0, -1, 0, []
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.tensor(tr)[torch.randperm(len(tr), generator=g)]
        nb = len(perm) // bs if not ref else int(np.ceil(len(perm) / bs))
        tot = 0.0
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs].cuda()
            x = prep(X[idx])
            if leadmask is not None:
                x = x * leadmask
            if not ref and a.aug:
                x = augment(x)
            if student:
                zs, zsub = model(x)
                loss = a.w_hard * F.binary_cross_entropy_with_logits(zs, Y5t[idx])
                if a.w_sub > 0:
                    loss = loss + a.w_sub * F.binary_cross_entropy_with_logits(zsub, Ysubt[idx])
                if soft is not None:
                    t = torch.sigmoid(soft[idx])
                    loss = loss + a.w_kd * F.binary_cross_entropy_with_logits(zs, t[:, :5])
                    if a.kd_sub:
                        loss = loss + a.w_kd * F.binary_cross_entropy_with_logits(zsub, t[:, 5:])
            else:
                loss = F.binary_cross_entropy_with_logits(model(x), Y5t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not ref:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None and (b + 1) % a.ema_every == 0:
                it = (ep * nb + b + 1) // a.ema_every
                d = min(a.ema ** a.ema_every, (1 + it) / (10 + it))
                with torch.no_grad():
                    torch._foreach_lerp_(ema_p, model_p, 1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            tot += float(loss.detach())
        ev = ema if ema is not None else model
        zv, _ = predict(ev, X[va], prep, student=student, leadmask=leadmask)
        pv = 1 / (1 + np.exp(-zv))
        auc = roc_auc_score(Y5[va], pv, average="macro")
        thr = C.pick_thresholds(Y5[va], pv)
        mf1 = C.metrics(Y5[va], pv, thr)["macro_f1"]
        curve.append(dict(epoch=ep + 1, train_loss=tot / nb, val_macro_auc=auc, val_macro_f1=mf1,
                          sec=time.time() - t0))
        print(f"ep {ep + 1:2d} loss {tot / nb:.4f} val AUC {auc:.4f} F1 {mf1:.4f} "
              f"{time.time() - t0:.0f}s", flush=True)
        if auc > best + (1e-4 if ref else 0):
            best, best_ep, bad = auc, ep + 1, 0
            torch.save(ev.state_dict(), f"{out}/model.pt")
        else:
            bad += 1
            if ref and bad >= 7:
                break
    pd.DataFrame(curve).to_csv(f"{out}/curve.csv", index=False)

    ev = ema if ema is not None else model
    ev.load_state_dict(torch.load(f"{out}/model.pt"))
    zv, sv = predict(ev, X[va], prep, student=student, leadmask=leadmask)
    pv = 1 / (1 + np.exp(-zv))
    thr = C.pick_thresholds(Y5[va], pv)
    res = dict(tag=a.tag, model=a.model, seed=a.seed, protocol=a.protocol, prep=prep_name,
               params=npar, best_epoch=best_ep, epochs_run=len(curve), train_n=int(len(tr)),
               train_sec=time.time() - t0, thresholds=thr.tolist(), args=vars(a),
               mamba_impl=M.MAMBA_IMPL, val=C.metrics(Y5[va], pv, thr))
    save = dict(val_logit=zv, thr=thr, val_id=meta.ecg_id.values[va])
    if sv is not None:
        save["val_sub"] = sv
    if not a.dev:
        zt, st = predict(ev, X[te], prep, student=student, leadmask=leadmask)
        res["test"] = C.metrics(Y5[te], 1 / (1 + np.exp(-zt)), thr)
        save.update(test_logit=zt, test_id=meta.ecg_id.values[te])
        if st is not None:
            save["test_sub"] = st
    np.savez_compressed(f"{out}/preds.npz", **save)
    json.dump(res, open(f"{out}/metrics.json", "w"), indent=1)
    print("params", npar, "best epoch", best_ep, "val", {k: round(v, 4) for k, v in res["val"].items()
                                                          if k in C.MAIN7})
    return 0


if __name__ == "__main__":
    sys.exit(main())
