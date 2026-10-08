"""Train one large teacher and write its logits for every ECG.

Cross-fitting: the eight training folds are cut into four groups ({1,2},{3,4},{5,6},{7,8}).
A teacher run with --holdout g never sees group g, so the logits it writes for group g are
out-of-fold.  `make_soft.py` stitches the four runs into one soft-label file in which every
training ECG is scored by a teacher that was not trained on it.  --holdout none trains on all
eight folds (the in-sample teacher used only as an ablation of cross-fitting).

The validation fold (9) selects the checkpoint; the test fold (10) is scored but its labels are
never used here.

  --arch ecgfounder   ECGFounder Net1D, 500 Hz, pretrained weights (fine-tuned end to end)
  --arch ecgfounder_scratch   same network, random initialisation (pretraining ablation)
  --arch resnet100    xresnet-style network trained from scratch on the 100 Hz signal in mV
Targets: 5 superclasses + 23 subclasses (28 logits).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402

import models as M                                    # noqa: E402

GROUPS = {"g1": (1, 2), "g2": (3, 4), "g3": (5, 6), "g4": (7, 8), "none": ()}
CKPT = f"{C.ROOT}/data/weights/ecgfounder/12_lead_ECGFounder.pth"


def augment(x, hr):
    B, _, T = x.shape
    x = torch.roll(x, int(torch.randint(0, T, (1,))), -1)
    g = 1 + 0.1 * (2 * torch.rand(B, 1, 1, device=x.device) - 1)
    return x * g


@torch.no_grad()
def predict(model, X, prep, bs=128):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(prep(X[i:i + bs])).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--holdout", default="none", choices=list(GROUPS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = f"{C.ROOT}/results/teachers/{a.arch}_{a.holdout}_s{a.seed}"
    if os.path.exists(f"{out}/logits.npz") and not a.force:
        print("done already:", out)
        return 0
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    hr = a.arch.startswith("ecgfounder")
    meta, rows = C.load_meta()
    Y5, Ysub = C.labels(meta)
    Y = np.concatenate([Y5, Ysub], 1)
    fold = meta.strat_fold.values
    tr = np.where((fold <= 8) & ~np.isin(fold, GROUPS[a.holdout]))[0]
    va = np.where(fold == 9)[0]
    X = C.load_x(rows, hr=hr)
    Yt = torch.tensor(Y, device="cuda")
    if hr:
        model, info = M.ecgfounder(28, CKPT if a.arch == "ecgfounder" else None)
        prep, lr = C.prep_global, a.lr or (3e-4 if a.arch == "ecgfounder" else 1e-3)
    else:
        model, info = M.BigResNet(28), "random init"
        prep, lr = C.prep_mv, a.lr or 2e-3
    model = model.cuda().to(memory_format=torch.contiguous_format)
    print(a.arch, a.holdout, "params", M.n_params(model), info, "train n", len(tr), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    nb = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.epochs * nb,
                                                pct_start=0.2)
    g = torch.Generator().manual_seed(a.seed)
    best, log, t0 = -1, [], time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.tensor(tr)[torch.randperm(len(tr), generator=g)]
        tot = 0.0
        for b in range(nb):
            idx = perm[b * a.bs:(b + 1) * a.bs].cuda()
            x = augment(prep(X[idx]), hr)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x)
            loss = F.binary_cross_entropy_with_logits(z.float(), Yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        zv = predict(model, X[va], prep)
        auc = roc_auc_score(Y5[va], zv[:, :5], average="macro")
        log.append(dict(epoch=ep + 1, loss=tot / nb, val_macro_auc=auc, sec=time.time() - t0))
        print(f"ep {ep + 1:2d} loss {tot / nb:.4f} val AUC {auc:.4f} {time.time() - t0:.0f}s",
              flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), f"{out}/model.pt")
    model.load_state_dict(torch.load(f"{out}/model.pt"))
    Z = predict(model, X, prep)
    oof = ~np.isin(fold, [f for f in range(1, 9) if f not in GROUPS[a.holdout]])
    np.savez_compressed(f"{out}/logits.npz", logits=Z.astype(np.float32), oof=oof,
                        ecg_id=meta.ecg_id.values)
    json.dump(dict(arch=a.arch, holdout=a.holdout, seed=a.seed, params=M.n_params(model),
                   init=info, best_val_macro_auc=float(best), log=log, train_n=int(len(tr)),
                   train_sec=time.time() - t0), open(f"{out}/log.json", "w"), indent=1)
    os.remove(f"{out}/model.pt") if a.holdout != "none" else None
    print("best val", round(best, 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
