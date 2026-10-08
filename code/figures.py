"""All figures of the paper.  Each function reads result files only and is skipped (with a
message) when its inputs are not there yet, so the module can be run at any stage.

  python figures.py --main <tag> [--only f03,f11]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

import figstyle as S                                  # noqa: E402
from figstyle import plt                              # noqa: E402

R = f"{C.ROOT}/results"
A = f"{R}/analysis"
F = f"{C.ROOT}/figures"
REF = "ref_msca_mamba"
ARGS = None


def per_seed():
    return pd.read_csv(f"{R}/per_seed.csv")


def summ():
    return pd.read_csv(f"{R}/summary.csv").set_index("tag")


def have(*paths):
    return all(os.path.exists(p) for p in paths)


def lab(ax, s):
    ax.set_title(s, loc="left", fontweight="bold")


# ------------------------------------------------------------------ data
def f01_dataset():
    meta, _ = C.load_meta()
    Y5, _ = C.labels(meta)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[1.3, 1, 0.9]))
    w = 0.26
    for k, (sp, col) in enumerate((("train", S.CAT[0]), ("val", S.CAT[2]), ("test", S.CAT[1]))):
        m = (meta.split == sp).values
        v = 100 * Y5[m].mean(0)
        b = ax[0].bar(np.arange(5) + (k - 1) * w, v, w, color=col,
                      label=f"{ {'train': 'Training', 'val': 'Validation', 'test': 'Test'}[sp]} (n = {m.sum():,})")
        for r, n in zip(b, Y5[m].sum(0)):
            ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.6, f"{int(n)}", ha="center",
                       va="bottom", fontsize=6, rotation=90, color=S.INK2)
    ax[0].set_xticks(range(5), C.SUPER)
    ax[0].set_ylabel("ECGs carrying the label (%)")
    ax[0].set_ylim(0, 56)
    ax[0].legend(loc="upper right")
    S.clean(ax[0])
    lab(ax[0], "(a) Label prevalence by split")
    co = (Y5.T @ Y5) / Y5.sum(0)[:, None]
    im = ax[1].imshow(100 * co, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", S.SEQ),
                      vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax[1].text(j, i, f"{100 * co[i, j]:.0f}", ha="center", va="center", fontsize=8,
                       color="white" if co[i, j] > 0.55 else S.INK)
    ax[1].set_xticks(range(5), C.SUPER)
    ax[1].set_yticks(range(5), C.SUPER)
    ax[1].set_xlabel("... also carry this label (%)")
    ax[1].set_ylabel("Of ECGs with this label ...")
    lab(ax[1], "(b) Label co-occurrence")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    card = pd.Series(Y5.sum(1)).value_counts().sort_index()
    ax[2].bar(card.index.astype(int), 100 * card.values / len(Y5), color=S.CAT[0], width=0.6)
    for x, v in zip(card.index, card.values):
        ax[2].text(x, 100 * v / len(Y5) + 1, f"{v:,}", ha="center", fontsize=7, color=S.INK2)
    ax[2].set_xlabel("Superclasses per ECG")
    ax[2].set_ylabel("ECGs (%)")
    ax[2].set_xticks(card.index.astype(int))
    S.clean(ax[2])
    lab(ax[2], "(c) Label cardinality")
    fig.tight_layout()
    S.save(fig, f"{F}/f01_dataset")


def f02_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec=None, fs=8, bold=False):
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec or S.INK2, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=S.INK,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, col=S.INK2, ls="-"):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, ls=ls))
    blue, org, vio, grn = "#dbe9fa", "#fbe3d8", "#e3dff3", "#d8f0e6"
    ax.text(1, 44, "Training time only (privileged information)", fontsize=9, fontweight="bold", color=S.CAT[6])
    box(1, 33, 17, 8, "12-lead ECG\n500 Hz, 10 s", vio)
    box(22, 33, 24, 8, "ECGFounder teacher (30.7 M)\npretrained on 10 M ECGs,\nfine-tuned, 4-fold cross-fitted", vio)
    box(50, 33, 22, 8, "Out-of-fold soft labels\n5 superclasses +\n23 subclasses", vio)
    box(76, 33, 30, 8, "Loss = BCE(superclass, label)\n+ BCE(subclass, label)\n+ BCE(both, teacher soft label)", vio, bold=True)
    arrow(18.5, 37, 21.5, 37, S.CAT[6])
    arrow(46.5, 37, 49.5, 37, S.CAT[6])
    arrow(72.5, 37, 75.5, 37, S.CAT[6])
    ax.plot([0, 110], [29.5, 29.5], color=S.GRID, lw=1, ls="--")
    ax.text(1, 26.5, "Deployed student (%s)" % S.NAME, fontsize=9, fontweight="bold", color=S.CAT[0])
    box(1, 12, 13, 10, "12-lead ECG\n100 Hz, 10 s\nin millivolts\n(no z-score)", blue, bold=True)
    box(17, 12, 11, 10, "Stem conv\nk = 7, stride 2\n12 -> 48", blue)
    box(31, 12, 15, 10, "Multi-scale\nDS block\ndilation 1 | 2 | 4\n(depthwise)", blue)
    box(49, 12, 11, 10, "Conv\nk = 5, stride 2\n48 -> 96", blue)
    box(63, 12, 13, 10, "2 x multi-scale\nDS block", blue)
    box(79, 12, 11, 10, "Bi-GRU\n2 x 56\n(T = 250)", blue)
    box(93, 12, 14, 10, "LayerNorm\nGAP || GMP\nlinear heads", blue)
    for x0, x1 in ((14.5, 16.5), (28.5, 30.5), (46.5, 48.5), (60.5, 62.5), (76.5, 78.5), (90.5, 92.5)):
        arrow(x0, 17, x1, 17, S.CAT[0])
    box(79, 1, 13, 7, "5 superclass\nlogits (output)", grn, bold=True)
    box(94, 1, 13, 7, "23 subclass\nlogits (auxiliary)", org)
    arrow(98, 11.5, 86, 8.5, S.CAT[0])
    arrow(100, 11.5, 100, 8.5, S.CAT[1])
    arrow(91, 33, 100, 22.5, S.CAT[6], ls="--")
    ax.text(2, 4.5, "The teacher, the 500 Hz signal, the subclass targets and the soft labels are used only while "
                    "training.\nAt inference the student reads the 100 Hz ECG in millivolts and nothing else.",
            fontsize=8, color=S.INK2, va="center")
    S.save(fig, f"{F}/f02_architecture")


def f03_voltage():
    if not have(f"{A}/voltage_index.csv"):
        return print("  skip f03 (no voltage_index.csv)")
    v = pd.read_csv(f"{A}/voltage_index.csv")
    t = pd.read_csv(f"{A}/voltage.csv")
    t = t[t.fold == "test"].reset_index(drop=True)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for k, (col, xl, rng) in enumerate((("sokolow_mv", "Sokolow-Lyon voltage (mV)", (0, 7)),
                                         ("sokolow_z", "Same index after per-lead z-score (SD units)", (2, 16)))):
        bins = np.linspace(*rng, 45)
        ax[k].hist(v[col][v.hyp == 0], bins, density=True, color=S.MUTED, alpha=0.75, label="No hypertrophy label")
        ax[k].hist(v[col][v.lvh == 1], bins, density=True, color=S.CAT[4], alpha=0.75, label="LVH")
        ax[k].set_xlabel(xl)
        ax[k].set_ylabel("Density")
        S.clean(ax[k])
        ax[k].legend(loc="upper right")
    ax[0].axvline(3.5, color=S.INK, lw=1, ls="--")
    ax[0].text(3.55, ax[0].get_ylim()[1] * 0.6, "3.5 mV\ncriterion", fontsize=7)
    lab(ax[0], "(a) Millivolts")
    lab(ax[1], "(b) After the reference preprocessing")
    lbl = [f"{r.criterion}\n{r.target}" for r in t.itertuples()]
    x = np.arange(len(t))
    b1 = ax[2].bar(x - 0.2, t.auc_mv, 0.4, color=S.CAT[0], label="Millivolts")
    b2 = ax[2].bar(x + 0.2, t.auc_zscore, 0.4, color=S.CAT[1], label="Per-lead z-score")
    for b in list(b1) + list(b2):
        ax[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.2f}", ha="center", fontsize=7)
    ax[2].axhline(0.5, color=S.INK2, lw=0.8, ls=":")
    ax[2].set_xticks(x, lbl, fontsize=7)
    ax[2].set_ylim(0.4, 1.0)
    ax[2].set_ylabel("ROC-AUC of the voltage index")
    ax[2].legend(loc="upper right")
    S.clean(ax[2])
    lab(ax[2], "(c) Discrimination of one voltage index")
    fig.tight_layout()
    S.save(fig, f"{F}/f03_voltage")


# ------------------------------------------------------------------ main results
def f04_curves():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ok = False
    for tag, key in ((ARGS.main, "ours"), (REF, "ref")):
        for i, s in enumerate(C.SEEDS):
            p = f"{C.RUNS}/{tag}/s{s}/curve.csv"
            if not os.path.exists(p):
                continue
            c = pd.read_csv(p)
            ok = True
            for k, col in enumerate(("val_macro_auc", "val_macro_f1")):
                ax[k].plot(c.epoch, c[col], color=S.COLOR[key], lw=1.1, alpha=0.75,
                           label=S.LABEL[key] if i == 0 else None)
    if not ok:
        plt.close(fig)
        return print("  skip f04")
    for k, yl in enumerate(("Validation Macro ROC-AUC", "Validation Macro F1")):
        ax[k].set_xlabel("Epoch")
        ax[k].set_ylabel(yl)
        S.clean(ax[k])
    ax[0].legend(loc="lower right")
    lab(ax[0], "(a) Five seeds per model")
    lab(ax[1], "(b)")
    fig.tight_layout()
    S.save(fig, f"{F}/f04_training_curves")


def f05_main():
    D = per_seed()
    if "test_macro_auc" not in D or ARGS.main not in set(D.tag):
        return print("  skip f05")
    fig, ax = plt.subplots(1, 7, figsize=(12, 3.0))
    pub = C.PUBLISHED["msca_mamba"]
    for k, (m, ml) in enumerate(zip(C.MAIN7, C.MAIN7_LABEL)):
        a = ax[k]
        a.errorbar([0], [pub[m][0]], yerr=[pub[m][1]], fmt=S.MARKER["pub"], color=S.COLOR["pub"],
                   capsize=3, ms=6, mfc="white")
        for x, (tag, key) in enumerate(((REF, "ref"), (ARGS.main, "ours")), start=1):
            v = D[D.tag == tag][f"test_{m}"].values
            if len(v) == 0:
                continue
            a.scatter(np.full(len(v), x) + np.linspace(-0.12, 0.12, len(v)), v, s=12, color=S.COLOR[key], alpha=0.55, lw=0)
            a.errorbar([x], [v.mean()], yerr=[v.std(ddof=1) if len(v) > 1 else 0], fmt=S.MARKER[key],
                       color=S.COLOR[key], capsize=3, ms=6)
            a.text(x, v.mean(), f"  {v.mean():.3f}", fontsize=7, va="center", ha="left", color=S.COLOR[key])
        a.text(0, pub[m][0], f"  {pub[m][0]:.3f}", fontsize=7, va="center", ha="left", color=S.COLOR["pub"])
        a.set_xticks([0, 1, 2], ["Published", "Re-run", "Ours"], fontsize=7, rotation=30)
        a.set_xlim(-0.5, 2.9)
        a.set_title(ml, fontsize=8)
        S.clean(a)
    fig.suptitle("Test fold, mean ± SD over the article's five seeds (small dots: individual seeds)", fontsize=9, y=1.02)
    fig.tight_layout()
    S.save(fig, f"{F}/f05_main_comparison")


def f06_vs_published():
    if not have(f"{R}/vs_published.csv"):
        return print("  skip f06")
    V = pd.read_csv(f"{R}/vs_published.csv")
    names = {"msca_mamba": "MSCA-Mamba (proposed in the article)", "msca_noSE": "MSCA-Mamba w/o SE",
             "ms_cnn": "Multi-scale CNN w/o SE/Mamba", "single_scale": "Single-scale + SE + Mamba",
             "m_conv": "Matched Conv1D", "m_gru": "Matched GRU", "m_lstm": "Matched LSTM",
             "m_transformer": "Matched Transformer", "bimamba": "BiMamba (214 k)"}
    ml = dict(zip(C.MAIN7, C.MAIN7_LABEL))
    V = V.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.19 * len(V) + 1.2))
    se = np.sqrt(V.published_sd ** 2 / 5 + V.ours_sd ** 2 / 5)
    col = [S.CAT[0] if w else S.CAT[7] for w in V.win]
    ax.errorbar(V["diff"], np.arange(len(V)), xerr=2.31 * se, fmt="none", ecolor=S.MUTED, lw=1, capsize=2)
    ax.scatter(V["diff"], np.arange(len(V)), c=col, s=22, zorder=3)
    ax.axvline(0, color=S.INK, lw=0.8)
    ax.set_yticks(np.arange(len(V)), [f"{names[c]} · {ml[m]}" for c, m in zip(V.config, V.metric)], fontsize=6.5)
    ax.set_xlabel(f"{S.NAME} minus published value (test fold; bars: approximate 95% interval from the two SDs)")
    ax.set_ylim(-0.7, len(V) - 0.3)
    S.clean(ax, grid="x")
    fig.tight_layout()
    S.save(fig, f"{F}/f06_vs_published")


def f07_per_class():
    D = per_seed()
    if "test_auc_HYP" not in D or ARGS.main not in set(D.tag):
        return print("  skip f07")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    p42 = C.PUBLISHED_SEED42["msca_mamba"]
    for k, (pre, yl) in enumerate((("auc", "ROC-AUC"), ("ap", "PR-AUC"), ("f1", "F1-score"))):
        for x, (tag, key) in enumerate(((REF, "ref"), (ARGS.main, "ours"))):
            d = D[D.tag == tag]
            if not len(d):
                continue
            mu = [d[f"test_{pre}_{c}"].mean() for c in C.SUPER]
            sd = [d[f"test_{pre}_{c}"].std(ddof=1) for c in C.SUPER]
            ax[k].bar(np.arange(5) + (x - 0.5) * 0.36, mu, 0.36, yerr=sd, color=S.COLOR[key], capsize=2,
                      label=S.LABEL[key], error_kw=dict(lw=0.8))
            for i, v in enumerate(mu):
                ax[k].text(i + (x - 0.5) * 0.36, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.5)
        pts = [(i, p42[f"{pre}_{c}"]) for i, c in enumerate(C.SUPER) if f"{pre}_{c}" in p42]
        if pts:
            ax[k].scatter([p[0] - 0.18 for p in pts], [p[1] for p in pts], marker="_", s=180, color=S.INK, zorder=4,
                          label="Published (seed 42)")
        ax[k].set_xticks(range(5), C.SUPER)
        ax[k].set_ylabel(yl)
        ax[k].set_ylim(0.3 if pre != "auc" else 0.7, 1.0)
        S.clean(ax[k])
        lab(ax[k], f"({'abc'[k]}) {yl}")
    ax[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    S.save(fig, f"{F}/f07_per_class")


def _curve(kind, name):
    if not have(f"{A}/curves.npz"):
        return print(f"  skip {name}")
    z = np.load(f"{A}/curves.npz")
    D = per_seed()
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(1, 5, figsize=(13, 2.9), sharey=True)
    for j, c in enumerate(C.SUPER):
        for tag, key in ((REF, "ref"), (ARGS.main, "ours")):
            k = f"{tag}|{c}|{kind}"
            if k not in z:
                continue
            y = z[k]
            col = "auc" if kind == "roc" else "ap"
            v = D[D.tag == tag][f"test_{col}_{c}"].mean()
            ax[j].plot(grid, y.mean(0), color=S.COLOR[key], lw=1.6, label=f"{S.LABEL[key].split(' (')[0]}: {v:.3f}")
            ax[j].fill_between(grid, y.min(0), y.max(0), color=S.COLOR[key], alpha=0.18, lw=0)
        if kind == "roc":
            ax[j].plot([0, 1], [0, 1], color=S.MUTED, lw=0.7, ls=":")
        ax[j].set_title(c)
        ax[j].set_xlabel("False-positive rate" if kind == "roc" else "Recall")
        ax[j].legend(loc="lower right" if kind == "roc" else "lower left", fontsize=6.5)
        S.clean(ax[j], grid="both")
    ax[0].set_ylabel("True-positive rate" if kind == "roc" else "Precision")
    fig.tight_layout()
    S.save(fig, f"{F}/{name}")


def f08_roc():
    _curve("roc", "f08_roc")


def f09_pr():
    _curve("pr", "f09_pr")


def f10_confusion():
    if not have(f"{A}/confusion.csv"):
        return print("  skip f10")
    cm = pd.read_csv(f"{A}/confusion.csv")
    cm = cm[cm.seed == 42]
    fig, ax = plt.subplots(2, 5, figsize=(11, 4.4))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", S.SEQ)
    for r, (tag, key) in enumerate(((REF, "ref"), (ARGS.main, "ours"))):
        for j, c in enumerate(C.SUPER):
            d = cm[(cm.tag == tag) & (cm.cls == c)]
            a = ax[r, j]
            if not len(d):
                a.axis("off")
                continue
            d = d.iloc[0]
            M_ = np.array([[d.tn, d.fp], [d.fn, d.tp]])
            a.imshow(M_ / M_.sum(1, keepdims=True), cmap=cmap, vmin=0, vmax=1)
            for i in range(2):
                for k in range(2):
                    a.text(k, i, f"{M_[i, k]}", ha="center", va="center", fontsize=9,
                           color="white" if M_[i, k] / M_[i].sum() > 0.55 else S.INK)
            a.set_xticks([0, 1], ["Pred −", "Pred +"], fontsize=7)
            a.set_yticks([0, 1], ["True −", "True +"], fontsize=7)
            a.set_title(f"{c}  (recall {d.tp / (d.tp + d.fn):.3f})", fontsize=8)
        ax[r, 0].set_ylabel(S.LABEL[key], fontsize=8, color=S.COLOR[key], fontweight="bold")
    fig.tight_layout()
    S.save(fig, f"{F}/f10_confusion")


# ------------------------------------------------------------------ ablations
LADDER = [("abl_arch_z", "Student, z-scored input,\nplain BCE"), ("abl_plain", "+ millivolt input"),
          ("abl_sub", "+ subclass supervision"), ("MAIN", "+ cross-fitted distillation\n(= %s)" % S.NAME)]
LOO = [("abl_zscore", "z-scored input"), ("abl_nosub", "no subclass targets"),
       ("abl_insample", "in-sample teacher (no cross-fitting)"), ("abl_noisyor", "noisy-OR hierarchy head"),
       ("abl_kdonly", "soft labels only (no hard labels)"), ("abl_noaug", "no augmentation"),
       ("abl_noema", "no weight averaging"), ("abl_nornn", "no recurrent layer"),
       ("abl_half", "half-width student")]


def f11_ladder():
    D = per_seed()
    if "test_macro_auc" not in D:
        return print("  skip f11")
    tags = [(ARGS.main if t == "MAIN" else t, l) for t, l in LADDER]
    tags = [(t, l) for t, l in tags if t in set(D.tag)]
    if len(tags) < 2:
        return print("  skip f11 (ladder incomplete)")
    cols = [("test_macro_auc", "Macro ROC-AUC"), ("test_macro_ap", "Macro PR-AUC"), ("test_macro_f1", "Macro F1"),
            ("test_auc_HYP", "HYP ROC-AUC")]
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
    for k, (col, yl) in enumerate(cols):
        for i, (t, l) in enumerate(tags):
            v = D[D.tag == t][col].values
            ax[k].scatter(v, np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), s=12, color=S.CAT[0], alpha=0.5, lw=0)
            ax[k].errorbar([v.mean()], [i], xerr=[v.std(ddof=1) if len(v) > 1 else 0], fmt="D", color=S.CAT[0], capsize=3, ms=5)
            ax[k].text(v.mean(), i + 0.3, f"{v.mean():.4f}", ha="center", fontsize=7)
        pub = C.PUBLISHED["msca_mamba"].get(col[5:])
        if pub:
            ax[k].axvline(pub[0], color=S.COLOR["pub"], lw=1, ls="--")
            ax[k].text(pub[0], -0.42, " published\n MSCA-Mamba", fontsize=6.5, ha="left", va="center", color=S.COLOR["pub"])
        ax[k].set_xlabel(yl)
        S.clean(ax[k], grid="x")
    ax[0].set_yticks(range(len(tags)), [l for _, l in tags], fontsize=7.5)
    ax[0].set_ylim(len(tags) - 0.2, -0.5)
    fig.tight_layout()
    S.save(fig, f"{F}/f11_ablation_ladder")


def f12_loo():
    D = per_seed()
    if "test_macro_auc" not in D or ARGS.main not in set(D.tag):
        return print("  skip f12")
    tags = [(t, l) for t, l in LOO if t in set(D.tag)]
    if not tags:
        return print("  skip f12 (no leave-one-out runs)")
    base = D[D.tag == ARGS.main].set_index("seed")
    cols = [("test_macro_auc", "Δ Macro ROC-AUC"), ("test_macro_ap", "Δ Macro PR-AUC"), ("test_macro_f1", "Δ Macro F1"),
            ("test_auc_HYP", "Δ HYP ROC-AUC")]
    fig, ax = plt.subplots(1, 4, figsize=(12, 0.36 * len(tags) + 1.3), sharey=True)
    from scipy import stats
    for k, (col, xl) in enumerate(cols):
        for i, (t, l) in enumerate(tags):
            d = D[D.tag == t].set_index("seed")
            cs = [s for s in base.index if s in d.index]
            diff = (d.loc[cs, col] - base.loc[cs, col]).values
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0
            c = S.CAT[7] if diff.mean() + half < 0 else (S.CAT[5] if diff.mean() - half > 0 else S.MUTED)
            ax[k].errorbar([diff.mean()], [i], xerr=[half], fmt="o", color=c, capsize=3, ms=5)
            ax[k].text(diff.mean(), i - 0.32, f"{diff.mean():+.4f}", ha="center", fontsize=6.5, color=c)
        ax[k].axvline(0, color=S.INK, lw=0.8)
        ax[k].set_xlabel(xl, fontsize=8)
        S.clean(ax[k], grid="x")
    ax[0].set_yticks(range(len(tags)), [l for _, l in tags], fontsize=7.5)
    ax[0].set_ylim(len(tags) - 0.4, -0.7)
    fig.suptitle(f"Variant minus {S.NAME}, paired by seed, 95% t interval (red: variant worse, green: better, "
                 "grey: interval includes zero)", fontsize=8.5, y=1.01)
    fig.tight_layout()
    S.save(fig, f"{F}/f12_ablation_leave_one_out")


def f13_gain():
    if not have(f"{A}/gain.csv"):
        return print("  skip f13")
    G = pd.read_csv(f"{A}/gain.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for tag, key in ((ARGS.main, "ours"), (ARGS.zs, "zs"), (REF, "ref")):
        g = G[G.tag == tag].groupby("gain")
        if not len(g):
            continue
        for k, col in enumerate(("flag_rate_HYP", "auc_HYP", "macro_auc")):
            mu, sd = g[col].mean(), g[col].std(ddof=1)
            ax[k].plot(mu.index, mu.values, marker=S.MARKER[key], color=S.COLOR[key], label=S.LABEL[key], ms=4)
            ax[k].fill_between(mu.index, mu - sd, mu + sd, color=S.COLOR[key], alpha=0.18, lw=0)
    for k, yl in enumerate(("Share of test ECGs flagged HYP", "HYP ROC-AUC", "Macro ROC-AUC")):
        ax[k].set_xscale("log")
        ax[k].set_xticks([0.4, 0.63, 1, 1.6, 2.5], ["0.4", "0.63", "1", "1.6", "2.5"])
        ax[k].minorticks_off()
        ax[k].axvline(1, color=S.MUTED, lw=0.8, ls=":")
        ax[k].set_xlabel("Amplitude gain applied to the test ECG")
        ax[k].set_ylabel(yl)
        S.clean(ax[k])
        lab(ax[k], f"({'abc'[k]})")
    ax[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    S.save(fig, f"{F}/f13_gain_probe")


def f14_pareto():
    Sm = summ()
    if "test_macro_auc_mean" not in Sm:
        return print("  skip f14")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    names = {"msca_mamba": "MSCA-Mamba", "msca_noSE": "w/o SE", "ms_cnn": "MS-CNN", "single_scale": "Single-scale",
             "m_conv": "Conv1D", "m_gru": "GRU", "m_lstm": "LSTM", "m_transformer": "Transformer", "bimamba": "BiMamba"}
    for k, v in C.PUBLISHED.items():
        n = M_PARAMS[k]
        ax.errorbar([n], [v["macro_auc"][0]], yerr=[v["macro_auc"][1]], fmt="o", color=S.COLOR["pub"], mfc="white", ms=5, capsize=2)
        left = k in ("m_lstm", "msca_noSE", "single_scale")          # twins share x and almost share y
        ax.annotate(names[k], (n, v["macro_auc"][0]), xytext=(-6 if left else 6, -3),
                    textcoords="offset points", fontsize=6.5, color=S.INK2, ha="right" if left else "left")
    for t, key, name in ((ARGS.main, "ours", S.NAME), ("abl_half", "ours", S.NAME + " (half width)")):
        if t in Sm.index:
            ax.errorbar([Sm.loc[t, "params"]], [Sm.loc[t, "test_macro_auc_mean"]], yerr=[Sm.loc[t, "test_macro_auc_sd"]],
                        fmt=S.MARKER[key], color=S.COLOR[key], ms=7, capsize=3)
            ax.annotate(name, (Sm.loc[t, "params"], Sm.loc[t, "test_macro_auc_mean"]), xytext=(6, 2),
                        textcoords="offset points", fontsize=8, color=S.COLOR[key], fontweight="bold")
    if have(f"{R}/teacher_metrics.json"):
        tm = json.load(open(f"{R}/teacher_metrics.json"))
        if "ecgfounder_xfit" in tm:
            ax.scatter([30693964], [tm["ecgfounder_xfit"]["test"]["macro_auc"]], marker=S.MARKER["teacher"], color=S.COLOR["teacher"], s=45)
            ax.annotate("ECGFounder teacher\n(4-model cross-fit mean)", (30693964, tm["ecgfounder_xfit"]["test"]["macro_auc"]),
                        xytext=(-8, -22), textcoords="offset points", fontsize=7, color=S.COLOR["teacher"], ha="right")
    ax.scatter([5.31e6], [0.9167], marker="x", color=S.MUTED, s=30)
    ax.annotate("ECGMamba (literature, 5.31 M)", (5.31e6, 0.9167), xytext=(-6, 5), textcoords="offset points", fontsize=6.5, color=S.MUTED, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test Macro ROC-AUC")
    S.clean(ax, grid="both")
    fig.tight_layout()
    S.save(fig, f"{F}/f14_pareto")


M_PARAMS = {"msca_mamba": 121149, "msca_noSE": 119493, "ms_cnn": 51141, "single_scale": 114813, "m_conv": 120653,
            "m_gru": 120906, "m_lstm": 120953, "m_transformer": 121405, "bimamba": 214077}


def f15_efficiency():
    if not have(f"{A}/efficiency.csv"):
        return print("  skip f15")
    E = pd.read_csv(f"{A}/efficiency.csv")
    nm = {ARGS.main: S.NAME, "ref_msca_mamba": "MSCA-Mamba", "ref_bimamba": "BiMamba", "ref_m_conv": "Conv1D",
          "ref_m_gru": "GRU", "ref_m_lstm": "LSTM", "ref_m_transformer": "Transformer", "ref_ms_cnn": "MS-CNN"}
    E["name"] = E.tag.map(nm)
    col = [S.COLOR["ours"] if t == ARGS.main else (S.COLOR["ref"] if t == REF else S.MUTED) for t in E.tag]
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0))
    for k, (c, yl, fmt) in enumerate((("lat_ms", "GPU latency, batch 1 (ms)", "{:.2f}"), ("thr_b256", "GPU throughput, batch 256 (ECG/s)", "{:.0f}"),
                                      ("flops_m", "Operations per ECG (M FLOPs)", "{:.0f}"), ("cpu_ms", "CPU latency, 1 thread (ms)", "{:.1f}"))):
        v = E[c].values
        b = ax[k].bar(range(len(E)), np.nan_to_num(v), color=col)
        for r, x in zip(b, v):
            ax[k].text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(x) if np.isfinite(x) else "n/a", ha="center", va="bottom", fontsize=6.5)
        ax[k].set_xticks(range(len(E)), E.name, rotation=40, ha="right", fontsize=7)
        ax[k].set_ylabel(yl, fontsize=8)
        S.clean(ax[k])
        lab(ax[k], f"({'abcd'[k]})")
    fig.tight_layout()
    S.save(fig, f"{F}/f15_efficiency")


# ------------------------------------------------------------------ clinical analyses
def f16_subclass():
    if not have(f"{A}/subclass_sens.csv"):
        return print("  skip f16")
    T = pd.read_csv(f"{A}/subclass_sens.csv")
    P = T.pivot(index="subclass", columns="tag", values="sens_mean")
    n = T.drop_duplicates("subclass").set_index("subclass")
    order = n.sort_values(["superclass", "n"], ascending=[True, False]).index
    order = [s for s in order if n.loc[s, "n"] >= 5]
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(order) + 1))
    for i, s in enumerate(order):
        a, b = P.loc[s].get(REF, np.nan), P.loc[s].get(ARGS.main, np.nan)
        ax.plot([a, b], [i, i], color=S.GRID, lw=2, zorder=1)
        ax.scatter([a], [i], color=S.COLOR["ref"], marker=S.MARKER["ref"], s=28, zorder=2, label=S.LABEL["ref"] if i == 0 else None)
        ax.scatter([b], [i], color=S.COLOR["ours"], marker=S.MARKER["ours"], s=28, zorder=3, label=S.LABEL["ours"] if i == 0 else None)
    ax.set_yticks(range(len(order)), [f"{n.loc[s, 'superclass']} · {s.strip('_')}  (n = {n.loc[s, 'n']})" for s in order], fontsize=7.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("Share of test ECGs with the subclass whose parent superclass is flagged (sensitivity)")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="lower left", fontsize=7.5)
    S.clean(ax, grid="x")
    fig.tight_layout()
    S.save(fig, f"{F}/f16_subclass_sensitivity")


def f17_subgroup():
    if not have(f"{A}/subgroup.csv"):
        return print("  skip f17")
    G = pd.read_csv(f"{A}/subgroup.csv")
    groups = list(G[G.tag == ARGS.main].group)
    fig, ax = plt.subplots(figsize=(6.8, 0.3 * len(groups) + 1))
    for off, (tag, key) in zip((0.16, -0.16), ((REF, "ref"), (ARGS.main, "ours"))):
        g = G[G.tag == tag].set_index("group")
        ys = [i + off for i, n in enumerate(groups) if n in g.index]
        gg = g.loc[[n for n in groups if n in g.index]]
        ax.errorbar(gg.macro_auc, ys, xerr=[gg.macro_auc - gg.lo, gg.hi - gg.macro_auc], fmt=S.MARKER[key], color=S.COLOR[key],
                    capsize=2, ms=4.5, lw=1, label=S.LABEL[key])
    g = G[G.tag == ARGS.main].set_index("group")
    ax.set_yticks(range(len(groups)), [f"{n}  (n = {g.loc[n, 'n']})" for n in groups], fontsize=7.5)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlabel("Test Macro ROC-AUC (seed mean; bars: 95% bootstrap interval)")
    ax.legend(loc="lower left", fontsize=7.5)
    S.clean(ax, grid="x")
    fig.tight_layout()
    S.save(fig, f"{F}/f17_subgroups")


def f18_noise():
    if not have(f"{A}/noise.csv"):
        return print("  skip f18")
    N = pd.read_csv(f"{A}/noise.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    xl = dict(wander="Baseline-wander amplitude (mV)", white="Broadband noise SD (mV)", drop="Leads set to zero (of 12)")
    for k, kind in enumerate(("wander", "white", "drop")):
        for tag, key in ((REF, "ref"), (ARGS.main, "ours")):
            g = N[(N.tag == tag) & (N.kind == kind)].groupby("level").macro_auc
            mu, sd = g.mean(), g.std(ddof=1)
            x = np.arange(len(mu))
            ax[k].plot(x, mu.values, marker=S.MARKER[key], color=S.COLOR[key], label=S.LABEL[key], ms=4)
            ax[k].fill_between(x, mu - sd, mu + sd, color=S.COLOR[key], alpha=0.18, lw=0)
            ax[k].set_xticks(x, [f"{v:g}" for v in mu.index])
        ax[k].set_xlabel(xl[kind])
        S.clean(ax[k])
        lab(ax[k], f"({'abc'[k]})")
    ax[0].set_ylabel("Test Macro ROC-AUC")
    ax[0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    S.save(fig, f"{F}/f18_noise_robustness")


def f19_calibration():
    if not have(f"{A}/calib_curve.csv"):
        return print("  skip f19")
    Cc = pd.read_csv(f"{A}/calib_curve.csv")
    Ce = pd.read_csv(f"{A}/calib.csv")
    fig, ax = plt.subplots(1, 6, figsize=(14, 2.8))
    for j, c in enumerate(C.SUPER):
        ax[j].plot([0, 1], [0, 1], color=S.MUTED, lw=0.8, ls=":")
        for tag, key in ((REF, "ref"), (ARGS.main, "ours")):
            g = Cc[(Cc.tag == tag) & (Cc.cls == c)].groupby("bin")[["conf", "obs"]].mean()
            e = Ce[(Ce.tag == tag) & (Ce.cls == c)].ece.mean()
            ax[j].plot(g.conf, g.obs, marker=S.MARKER[key], color=S.COLOR[key], ms=3.5, lw=1.3, label=f"ECE {e:.3f}")
        ax[j].set_title(c)
        ax[j].set_xlabel("Predicted probability")
        ax[j].legend(loc="upper left", fontsize=6.5)
        S.clean(ax[j], grid="both")
    ax[0].set_ylabel("Observed frequency")
    for x, (tag, key) in enumerate(((REF, "ref"), (ARGS.main, "ours"))):
        e = Ce[Ce.tag == tag].groupby("seed")[["ece", "brier"]].mean()
        ax[5].bar([x - 0.0], [e.ece.mean()], 0.6, yerr=[e.ece.std(ddof=1)], color=S.COLOR[key], capsize=3)
        ax[5].text(x, e.ece.mean() + e.ece.std(ddof=1) + 0.001, f"{e.ece.mean():.3f}", ha="center", fontsize=7)
    ax[5].set_xticks([0, 1], ["MSCA-Mamba\n(re-run)", S.NAME], fontsize=7)
    ax[5].set_title("Mean ECE over classes")
    S.clean(ax[5])
    fig.tight_layout()
    S.save(fig, f"{F}/f19_calibration")


def f20_dca():
    if not have(f"{A}/dca.csv"):
        return print("  skip f20")
    Dc = pd.read_csv(f"{A}/dca.csv")
    fig, ax = plt.subplots(1, 5, figsize=(13, 2.8))
    for j, c in enumerate(C.SUPER):
        d = Dc[Dc.cls == c]
        ta = d[d.tag == ARGS.main]
        ax[j].plot(ta.threshold, ta.treat_all, color=S.MUTED, lw=1, ls="--", label="Flag everyone")
        ax[j].axhline(0, color=S.MUTED, lw=1, ls=":")
        for tag, key in ((REF, "ref"), (ARGS.main, "ours")):
            g = d[d.tag == tag]
            ax[j].plot(g.threshold, g.net_benefit, color=S.COLOR[key], lw=1.6, label=S.LABEL[key].split(" (")[0])
        ax[j].set_ylim(-0.02, max(0.05, float(d.net_benefit.max()) * 1.08))
        ax[j].set_title(c)
        ax[j].set_xlabel("Threshold probability")
        S.clean(ax[j], grid="both")
    ax[0].set_ylabel("Net benefit")
    ax[0].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    S.save(fig, f"{F}/f20_decision_curves")


def f21_leads():
    D = per_seed()
    sets = [(ARGS.main, "12 leads"), ("lead_limb6", "6 limb leads"), ("lead_3", "I, II, V2"), ("lead_ii", "Lead II only")]
    sets = [(t, l) for t, l in sets if t in set(D.tag)]
    if len(sets) < 2 or "test_macro_auc" not in D:
        return print("  skip f21")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.1), gridspec_kw=dict(width_ratios=[1, 1.6]))
    mu = [D[D.tag == t].test_macro_auc.mean() for t, _ in sets]
    sd = [D[D.tag == t].test_macro_auc.std(ddof=1) for t, _ in sets]
    ax[0].bar(range(len(sets)), mu, yerr=sd, color=S.CAT[0], capsize=3, width=0.6)
    for i, v in enumerate(mu):
        ax[0].text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5)
    ax[0].axhline(C.PUBLISHED["msca_mamba"]["macro_auc"][0], color=S.COLOR["pub"], lw=1, ls="--")
    ax[0].text(len(sets) - 0.5, C.PUBLISHED["msca_mamba"]["macro_auc"][0] + 0.002, "published MSCA-Mamba, 12 leads", fontsize=6.5, ha="right", color=S.COLOR["pub"])
    ax[0].set_xticks(range(len(sets)), [l for _, l in sets], fontsize=7.5)
    ax[0].set_ylim(0.8, 0.95)
    ax[0].set_ylabel("Test Macro ROC-AUC")
    S.clean(ax[0])
    lab(ax[0], "(a)")
    w = 0.8 / len(sets)
    for i, (t, l) in enumerate(sets):
        v = [D[D.tag == t][f"test_auc_{c}"].mean() for c in C.SUPER]
        ax[1].bar(np.arange(5) + (i - (len(sets) - 1) / 2) * w, v, w, color=S.SEQ[::-1][1 + i], label=l)
    ax[1].set_xticks(range(5), C.SUPER)
    ax[1].set_ylim(0.7, 1.0)
    ax[1].set_ylabel("Test ROC-AUC")
    ax[1].legend(ncol=len(sets), fontsize=7, loc="upper center")
    S.clean(ax[1])
    lab(ax[1], "(b)")
    fig.tight_layout()
    S.save(fig, f"{F}/f21_reduced_leads")


def f22_labels():
    D = per_seed()
    if "test_macro_auc" not in D:
        return print("  skip f22")
    fr = [(0.1, "10"), (0.25, "25"), (0.5, "50"), (1.0, "100")]
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.1))
    ok = False
    for kd, key, labl in ((1, "ours", S.NAME + " (with distillation)"), (0, "ctrl", "Same student, no distillation")):
        for k, col in enumerate(("test_macro_auc", "test_macro_f1")):
            xs, mu, sd = [], [], []
            for f, s in fr:
                tag = (ARGS.main if kd else "abl_sub") if f == 1.0 else f"frac{s}_{'kd' if kd else 'nokd'}"
                v = D[D.tag == tag][col].values
                if len(v):
                    xs.append(100 * f)
                    mu.append(v.mean())
                    sd.append(v.std(ddof=1) if len(v) > 1 else 0)
            if len(xs) > 1:
                ok = True
                ax[k].errorbar(xs, mu, yerr=sd, marker=S.MARKER[key], color=S.COLOR[key], capsize=3, label=labl)
    if not ok:
        plt.close(fig)
        return print("  skip f22 (no label-fraction runs)")
    for k, yl in enumerate(("Test Macro ROC-AUC", "Test Macro F1")):
        pub = C.PUBLISHED["msca_mamba"][("macro_auc", "macro_f1")[k]][0]
        ax[k].axhline(pub, color=S.COLOR["pub"], lw=1, ls="--")
        ax[k].text(100, pub, "published MSCA-Mamba\n(100% of labels)", fontsize=6.5, ha="right", va="top", color=S.COLOR["pub"])
        ax[k].set_xscale("log")
        ax[k].set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
        ax[k].minorticks_off()
        ax[k].set_xlabel("Labelled training ECGs used (%)")
        ax[k].set_ylabel(yl)
        S.clean(ax[k])
        lab(ax[k], f"({'ab'[k]})")
    ax[0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    S.save(fig, f"{F}/f22_label_efficiency")


def f23_saliency():
    if not have(f"{A}/saliency.npz"):
        return print("  skip f23")
    z = np.load(f"{A}/saliency.npz")
    leads = C.maps()["leads"]
    show = {"NORM": ["II", "V2", "V5"], "MI": ["II", "III", "V2"], "STTC": ["II", "V4", "V5"], "CD": ["I", "V1", "V6"], "HYP": ["V1", "V5", "aVL"]}
    fig, ax = plt.subplots(3, 5, figsize=(14, 4.6), sharex=True)
    t = np.arange(1000) / 100
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("d", [S.CAT[0], "#f0efec", S.CAT[1]])
    for j, c in enumerate(C.SUPER):
        x, at = z[f"{c}_x"], z[f"{c}_attr"]
        sc = np.abs(at).max() + 1e-9
        for r, ld in enumerate(show[c]):
            i = leads.index(ld)
            a = ax[r, j]
            sm = np.convolve(at[i], np.ones(5) / 5, mode="same") / sc
            a.pcolormesh(t, [x[i].min() - 0.1, x[i].max() + 0.1], sm[None, :-1], cmap=cmap, vmin=-0.6, vmax=0.6, shading="flat", rasterized=True)
            a.plot(t, x[i], color=S.INK, lw=0.7)
            a.set_xlim(0, 5)
            a.text(0.01, 0.96, ld, transform=a.transAxes, fontsize=8, fontweight="bold", va="top")
            a.tick_params(labelsize=6.5)
            if r == 0:
                a.set_title(f"{c}  (ECG {int(z[f'{c}_id'])}, p = {float(z[f'{c}_p']):.2f})", fontsize=8.5)
            if j == 0:
                a.set_ylabel("mV", fontsize=7.5)
        ax[2, j].set_xlabel("Time (s)")
    fig.tight_layout()
    S.save(fig, f"{F}/f23_attribution")


def f24_teacher():
    if not have(f"{R}/teacher_metrics.json"):
        return print("  skip f24")
    tm = json.load(open(f"{R}/teacher_metrics.json"))
    D = per_seed()
    if "test_auc_HYP" not in D or ARGS.main not in set(D.tag):
        return print("  skip f24")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.1), gridspec_kw=dict(width_ratios=[1.5, 1]))
    series = [("Published MSCA-Mamba (seed 42)", S.COLOR["pub"], [C.PUBLISHED_SEED42["msca_mamba"][f"auc_{c}"] for c in C.SUPER])]
    if "abl_sub" in set(D.tag):
        series.append(("Student without teacher", S.COLOR["ctrl"], [D[D.tag == "abl_sub"][f"test_auc_{c}"].mean() for c in C.SUPER]))
    series.append((S.LABEL["ours"], S.COLOR["ours"], [D[D.tag == ARGS.main][f"test_auc_{c}"].mean() for c in C.SUPER]))
    if "ecgfounder_xfit" in tm:
        series.append(("ECGFounder teacher (30.7 M, 500 Hz)", S.COLOR["teacher"], [tm["ecgfounder_xfit"]["test"][f"auc_{c}"] for c in C.SUPER]))
    w = 0.8 / len(series)
    for i, (l, col, v) in enumerate(series):
        ax[0].bar(np.arange(5) + (i - (len(series) - 1) / 2) * w, v, w, color=col, label=l)
    ax[0].set_xticks(range(5), C.SUPER)
    ax[0].set_ylim(0.78, 0.97)
    ax[0].set_ylabel("Test ROC-AUC")
    ax[0].legend(fontsize=6.5, loc="lower left")
    S.clean(ax[0])
    lab(ax[0], "(a) Per-class discrimination")
    gap = []
    for l, t in (("no teacher", "abl_sub"), ("in-sample\nteacher", "abl_insample"), ("cross-fitted\nteacher", ARGS.main)):
        if t in set(D.tag):
            v = D[D.tag == t].test_macro_auc
            gap.append((l, v.mean(), v.std(ddof=1)))
    ax[1].bar(range(len(gap)), [g[1] for g in gap], yerr=[g[2] for g in gap], color=[S.COLOR["ctrl"], S.CAT[3], S.COLOR["ours"]][:len(gap)], capsize=3, width=0.6)
    for i, g in enumerate(gap):
        ax[1].text(i, g[1] + g[2] + 0.0008, f"{g[1]:.4f}", ha="center", fontsize=7.5)
    if "ecgfounder_xfit" in tm:
        ax[1].axhline(tm["ecgfounder_xfit"]["test"]["macro_auc"], color=S.COLOR["teacher"], lw=1, ls="--")
        ax[1].text(len(gap) - 0.5, tm["ecgfounder_xfit"]["test"]["macro_auc"], "teacher", fontsize=7, color=S.COLOR["teacher"], ha="right", va="bottom")
    ax[1].set_xticks(range(len(gap)), [g[0] for g in gap], fontsize=7.5)
    lo = min(g[1] for g in gap) - 0.012
    ax[1].set_ylim(lo, None)
    ax[1].set_ylabel("Student test Macro ROC-AUC")
    S.clean(ax[1])
    lab(ax[1], "(b) What the teacher adds")
    fig.tight_layout()
    S.save(fig, f"{F}/f24_teacher_student")


def f25_repro():
    if not have(f"{R}/repro_check.csv"):
        return print("  skip f25")
    Rp = pd.read_csv(f"{R}/repro_check.csv")
    ml = dict(zip(C.MAIN7, C.MAIN7_LABEL))
    ms = [m for m in C.MAIN7 if m in set(Rp.metric)][:5]
    fig, ax = plt.subplots(1, len(ms), figsize=(2.6 * len(ms), 2.9))
    for k, m in enumerate(ms):
        d = Rp[Rp.metric == m]
        lo, hi = min(d.published_mean.min(), d.ours_mean.min()) - 0.004, max(d.published_mean.max(), d.ours_mean.max()) + 0.004
        ax[k].plot([lo, hi], [lo, hi], color=S.MUTED, lw=0.8, ls=":")
        ax[k].errorbar(d.published_mean, d.ours_mean, xerr=d.published_sd, yerr=d.ours_sd, fmt="o", color=S.COLOR["ref"], ms=4, lw=0.8, capsize=2)
        for r in d.itertuples():
            ax[k].annotate(r.config.replace("m_", "").replace("msca_", ""), (r.published_mean, r.ours_mean), xytext=(3, 3), textcoords="offset points", fontsize=6)
        ax[k].set_title(ml[m], fontsize=8.5)
        ax[k].set_xlabel("Published")
        S.clean(ax[k], grid="both")
    ax[0].set_ylabel("Our re-implementation")
    fig.tight_layout()
    S.save(fig, f"{F}/f25_reproduction")


ALL = [f01_dataset, f02_architecture, f03_voltage, f04_curves, f05_main, f06_vs_published, f07_per_class, f08_roc,
       f09_pr, f10_confusion, f11_ladder, f12_loo, f13_gain, f14_pareto, f15_efficiency, f16_subclass, f17_subgroup,
       f18_noise, f19_calibration, f20_dca, f21_leads, f22_labels, f23_saliency, f24_teacher, f25_repro]


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="voltkd")
    ap.add_argument("--zs", default="abl_zscore")
    ap.add_argument("--only", default="")
    ARGS = ap.parse_args()
    os.makedirs(F, exist_ok=True)
    only = set(ARGS.only.split(",")) if ARGS.only else None
    for fn in ALL:
        if only and fn.__name__.split("_")[0] not in only:
            continue
        try:
            fn()
        except Exception as e:                          # one broken figure must not hide the rest
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
            if only:
                raise


if __name__ == "__main__":
    main()
