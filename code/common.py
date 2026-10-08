"""Shared data access, preprocessing, metrics and threshold selection."""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from project_paths import ROOT
PREP = f"{ROOT}/data/prep"
RUNS = f"{ROOT}/results/runs"
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
SEEDS = [42, 123, 456, 789, 2025]                    # the article's five seeds
GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)    # the article's threshold grid


def maps():
    return json.load(open(f"{PREP}/label_maps.json"))


def child_index():
    m = maps()
    return [[j for j, s in enumerate(m["subs"]) if m["sub_sup"][s] == S] for S in SUPER]


def load_meta():
    """Records carrying at least one diagnostic superclass (the article's 17,084/2146/2158)."""
    meta = pd.read_parquet(f"{PREP}/meta.parquet")
    keep = (meta.n_super > 0).values
    return meta[keep].reset_index(drop=True), np.where(keep)[0]


def labels(meta):
    m = maps()
    Y5 = meta[["y_" + s for s in SUPER]].values.astype(np.float32)
    Ysub = meta[["sub_" + s for s in m["subs"]]].values.astype(np.float32)
    return Y5, Ysub


def load_x(rows, hr=False, device="cuda"):
    X = np.load(f"{PREP}/{'X500' if hr else 'X100'}.npy", mmap_mode="r")
    out = torch.empty((len(rows), X.shape[1], 12), dtype=torch.int16)
    step = 2000
    for i in range(0, len(rows), step):
        out[i:i + step] = torch.from_numpy(np.ascontiguousarray(X[rows[i:i + step]]))
    return out.to(device)


def prep_zscore(x):
    """Article protocol: each lead standardised with its own mean/SD within the recording."""
    x = x.float()
    x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
    return x.transpose(1, 2)


def prep_mv(x):
    """Voltage-preserving: ADC units -> millivolts, per-lead baseline (median) removed."""
    x = x.float() / 1000.0
    x = x - x.median(1, keepdim=True).values
    return x.transpose(1, 2)


def prep_global(x):
    """ECGFounder pipeline: one mean/SD over all leads of the recording."""
    x = x.float()
    m = x.mean((1, 2), keepdim=True)
    s = x.std((1, 2), keepdim=True)
    return ((x - m) / (s + 1e-6)).transpose(1, 2)


PREPS = {"zscore": prep_zscore, "mv": prep_mv, "global": prep_global}


def pick_thresholds(y, p):
    """Per class, the grid threshold maximising F1 on the validation fold (article Sec. 3.7)."""
    yb = y.astype(bool)[None]                                   # (1, n, C)
    pred = p[None] >= GRID[:, None, None]                       # (G, n, C)
    tp = (pred & yb).sum(1)
    f1 = 2 * tp / np.maximum(pred.sum(1) + yb.sum(1), 1)        # (G, C); first max wins
    return GRID[f1.argmax(0)]


def metrics(y, p, thr, names=SUPER):
    yb = (p >= thr[None]).astype(int)
    out = {
        "macro_auc": roc_auc_score(y, p, average="macro"),
        "micro_auc": roc_auc_score(y, p, average="micro"),
        "macro_ap": average_precision_score(y, p, average="macro"),
        "macro_f1": f1_score(y, yb, average="macro", zero_division=0),
        "micro_f1": f1_score(y, yb, average="micro", zero_division=0),
        "weighted_f1": f1_score(y, yb, average="weighted", zero_division=0),
        "subset_acc": float((yb == y).all(1).mean()),
    }
    for j, n in enumerate(names):
        tp = int(((yb[:, j] == 1) & (y[:, j] == 1)).sum())
        fp = int(((yb[:, j] == 1) & (y[:, j] == 0)).sum())
        fn = int(((yb[:, j] == 0) & (y[:, j] == 1)).sum())
        tn = int(((yb[:, j] == 0) & (y[:, j] == 0)).sum())
        out[f"auc_{n}"] = roc_auc_score(y[:, j], p[:, j])
        out[f"ap_{n}"] = average_precision_score(y[:, j], p[:, j])
        out[f"f1_{n}"] = f1_score(y[:, j], yb[:, j], zero_division=0)
        out[f"prec_{n}"] = tp / max(tp + fp, 1)
        out[f"rec_{n}"] = tp / max(tp + fn, 1)
        out[f"spec_{n}"] = tn / max(tn + fp, 1)
        out[f"tp_{n}"], out[f"fp_{n}"], out[f"fn_{n}"], out[f"tn_{n}"] = tp, fp, fn, tn
    return {k: float(v) for k, v in out.items()}


MAIN7 = ["macro_auc", "micro_auc", "macro_ap", "macro_f1", "micro_f1", "weighted_f1",
         "subset_acc"]
MAIN7_LABEL = ["Macro ROC-AUC", "Micro ROC-AUC", "Macro PR-AUC", "Macro F1", "Micro F1",
               "Weighted F1", "Subset accuracy"]

# Every number the article prints for a model configuration (mean, SD over its five seeds).
PUBLISHED = {
    "msca_mamba": dict(macro_auc=(.8998, .0022), micro_auc=(.9119, .0052), macro_ap=(.7651, .0036),
                       macro_f1=(.7038, .0045), micro_f1=(.7474, .0074),
                       weighted_f1=(.7494, .0039), subset_acc=(.5691, .0143)),
    "msca_noSE": dict(macro_auc=(.8997, .0016), micro_auc=(.9128, .0031), macro_ap=(.7644, .0037),
                      macro_f1=(.7012, .0024), micro_f1=(.7462, .0044)),
    "ms_cnn": dict(macro_auc=(.9012, .0013), micro_auc=(.9175, .0025), macro_ap=(.7644, .0019),
                   macro_f1=(.7010, .0034), micro_f1=(.7492, .0065)),
    "single_scale": dict(macro_auc=(.8964, .0011), micro_auc=(.9140, .0020),
                         macro_ap=(.7579, .0017), macro_f1=(.6977, .0029),
                         micro_f1=(.7438, .0024)),
    "m_conv": dict(macro_auc=(.9043, .0014), micro_auc=(.9225, .0018), macro_ap=(.7719, .0027),
                   macro_f1=(.7094, .0055), micro_f1=(.7540, .0075)),
    "m_gru": dict(macro_auc=(.9062, .0014), micro_auc=(.9251, .0010), macro_ap=(.7764, .0026),
                  macro_f1=(.7115, .0036), micro_f1=(.7559, .0024)),
    "m_lstm": dict(macro_auc=(.9044, .0010), micro_auc=(.9227, .0026), macro_ap=(.7706, .0030),
                   macro_f1=(.7070, .0018), micro_f1=(.7524, .0041)),
    "m_transformer": dict(macro_auc=(.8971, .0008), micro_auc=(.9161, .0006),
                          macro_ap=(.7569, .0018), macro_f1=(.7018, .0043),
                          micro_f1=(.7484, .0043)),
    "bimamba": dict(macro_auc=(.9047, .0018), macro_ap=(.7712, .0038), macro_f1=(.7098, .0032)),
}
PUBLISHED_SEED42 = {                                 # article Table 5 and Sec. 4.1 (seed 42)
    "msca_noSE": dict(macro_auc=.8986, micro_auc=.9115, macro_ap=.7631, macro_f1=.7016,
                      micro_f1=.7465, weighted_f1=.7464),
    "msca_mamba": dict(macro_auc=.9020, micro_auc=.9188, macro_ap=.7693, macro_f1=.7100,
                       micro_f1=.7571, weighted_f1=.7553,
                       auc_NORM=.9394, auc_MI=.9142, auc_STTC=.9305, auc_CD=.9144,
                       auc_HYP=.8115, f1_NORM=.8538, ap_HYP=.4662, f1_HYP=.4682,
                       rec_NORM=.9335, rec_MI=.7855, rec_STTC=.7774, rec_CD=.6714,
                       rec_HYP=.4924),
    "se_postfusion": dict(macro_auc=.9028, micro_auc=.9185, macro_ap=.7661, macro_f1=.7070,
                          micro_f1=.7519, weighted_f1=.7501),
    "se_postseq": dict(macro_auc=.9006, micro_auc=.9190, macro_ap=.7639, macro_f1=.7140,
                       micro_f1=.7599, weighted_f1=.7586),
}
PUBLISHED_EFF = {                                    # article Table 8 (NVIDIA L4, batch 1)
    "msca_mamba": dict(params=121149, mb=0.462, lat=2.286, p95=2.356, thr=437.48, mem=36.40),
    "bimamba": dict(params=214077, mb=0.817, lat=2.904, p95=2.998, thr=344.37, mem=37.96),
    "m_conv": dict(params=120653, mb=0.460, lat=1.843, p95=2.164, thr=542.46, mem=35.29),
    "m_gru": dict(params=120906, mb=0.461, lat=2.061, p95=2.118, thr=485.09, mem=48.49),
    "m_lstm": dict(params=120953, mb=0.461, lat=2.010, p95=2.066, thr=497.56, mem=50.27),
    "m_transformer": dict(params=121405, mb=0.463, lat=2.037, p95=2.117, thr=490.97, mem=36.63),
}
