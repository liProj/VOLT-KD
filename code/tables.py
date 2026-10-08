"""Result tables in a neutral form: name -> dict(caption, header, rows, note).  Cells are plain
strings ("0.9321 ± 0.0012"); write_paper.py renders them to LaTeX and write_report.py to Word,
so the two documents can never disagree.  A table whose inputs are missing is simply absent.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

R = f"{C.ROOT}/results"
A = f"{R}/analysis"
REF = "ref_msca_mamba"
NAME = "VOLT-KD"
ML = dict(zip(C.MAIN7, C.MAIN7_LABEL))
CFG = {"msca_mamba": "MSCA-Mamba", "msca_noSE": "MSCA-Mamba w/o SE", "ms_cnn": "Multi-scale CNN w/o SE/Mamba",
       "single_scale": "Single-scale + SE + Mamba", "se_postfusion": "Post-fusion SE",
       "se_postseq": "Post-Mamba SE", "m_conv": "Matched Conv1D", "m_gru": "Matched GRU",
       "m_lstm": "Matched LSTM", "m_transformer": "Matched Transformer", "bimamba": "BiMamba"}
LADDER = [("abl_arch_z", "Student, z-scored input, plain BCE"), ("abl_plain", "+ millivolt input"),
          ("abl_sub", "+ subclass supervision"), ("MAIN", "+ cross-fitted distillation = " + NAME)]
LOO = [("abl_zscore", "z-scored input"), ("abl_nosub", "no subclass targets"),
       ("abl_insample", "in-sample teacher (no cross-fitting)"), ("abl_noisyor", "noisy-OR hierarchy head"),
       ("abl_kdonly", "soft labels only (no hard labels)"), ("abl_noaug", "no augmentation"),
       ("abl_noema", "no weight averaging"), ("abl_nornn", "no recurrent layer"),
       ("abl_half", "half-width student")]


def f4(x):
    return "–" if x is None or not np.isfinite(x) else f"{x:.4f}"


def f3(x):
    return "–" if x is None or not np.isfinite(x) else f"{x:.3f}"


def pm(mu, sd, d=4):
    if mu is None or not np.isfinite(mu):
        return "–"
    return f"{mu:.{d}f} ± {sd:.{d}f}" if sd is not None and np.isfinite(sd) else f"{mu:.{d}f}"


def sg(x, d=4):
    return "–" if not np.isfinite(x) else f"{x:+.{d}f}"


def pv(p):
    return "–" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def rd(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def build(main="voltkd"):
    T = {}
    D, S = rd(f"{R}/per_seed.csv"), rd(f"{R}/summary.csv")
    if D is None or "test_macro_auc" not in D or main not in set(D.tag):
        return T
    S = S.set_index("tag")
    tags = set(D.tag)

    def ms(tag, m, d=4):
        return pm(S.loc[tag, f"test_{m}_mean"], S.loc[tag, f"test_{m}_sd"], d) if tag in S.index else "–"

    # ---------------------------------------------------------------- main
    V = rd(f"{R}/vs_published.csv")
    if V is not None:
        v = V[V.config == "msca_mamba"].set_index("metric")
        rows = []
        for m in C.MAIN7:
            rows.append([ML[m], pm(v.loc[m, "published_mean"], v.loc[m, "published_sd"]),
                         ms(REF, m), ms(main, m), sg(v.loc[m, "diff"]), pv(v.loc[m, "p_holm"])])
        T["main"] = dict(
            caption=f"Test-fold performance on the PTB-XL diagnostic superclasses (mean ± SD over five seeds). "
                    f"Difference and Holm-adjusted Welch p-value are {NAME} against the published MSCA-Mamba.",
            header=["Metric", "MSCA-Mamba (published)", "MSCA-Mamba (our re-run)", NAME, "Difference", "p (Holm)"],
            rows=rows)
        rows = []
        for cfg in C.PUBLISHED:
            for m in C.PUBLISHED[cfg]:
                r = V[(V.config == cfg) & (V.metric == m)].iloc[0]
                rows.append([CFG[cfg], ML[m], pm(r.published_mean, r.published_sd), pm(r.ours_mean, r.ours_sd),
                             sg(r["diff"]), pv(r.p_holm), "yes" if r.win else "no"])
        T["vs_all"] = dict(
            caption=f"{NAME} against every model configuration for which the reference article prints a mean ± SD "
                    f"(its Tables 2, 4, 6 and 7). p-values: Welch's test on the summary statistics, Holm-adjusted "
                    f"over all {len(V)} comparisons.",
            header=["Published configuration", "Metric", "Published", NAME, "Difference", "p (Holm)", "Higher"],
            rows=rows)
    V42 = rd(f"{R}/vs_published_seed42.csv")
    if V42 is not None:
        lab = lambda m: ML.get(m, m.replace("auc_", "ROC-AUC ").replace("ap_", "PR-AUC ").replace("f1_", "F1 ")   # noqa: E731
                               .replace("rec_", "Recall "))
        T["seed42"] = dict(
            caption=f"Seed-42 results printed in the reference article (its Table 5 and Section 4.1) against the "
                    f"seed-42 run of {NAME}.",
            header=["Published configuration", "Metric", "Published (seed 42)", NAME + " (seed 42)", "Difference"],
            rows=[[CFG[r.config], lab(r.metric), f4(r.published_seed42), f4(r.ours_seed42), sg(r.diff)]
                  for r in V42.itertuples()])

    # ---------------------------------------------------------------- per class
    # Compared with the numbers printed in the reference article (its Sec. 4.1, seed 42); the
    # article prints class-wise ROC-AUC and recall for all classes, and PR-AUC / F1 for some.
    p42 = C.PUBLISHED_SEED42["msca_mamba"]
    r42 = D[(D.tag == main) & (D.seed == 42)].iloc[0]
    keys = ("auc", "ap", "f1", "prec", "rec", "spec")
    rows = []
    for c in C.SUPER:
        rows.append([c, "MSCA-Mamba (published, seed 42)"] + [f4(p42[f"{k}_{c}"]) if f"{k}_{c}" in p42 else "–" for k in keys])
        rows.append([c, NAME + " (seed 42)"] + [f4(float(r42[f"test_{k}_{c}"])) for k in keys])
        rows.append([c, "Difference (seed 42)"] + [sg(float(r42[f"test_{k}_{c}"]) - p42[f"{k}_{c}"]) if f"{k}_{c}" in p42 else "–" for k in keys])
        rows.append([c, NAME + " (five seeds)"] + [ms(main, f"{k}_{c}", 3) for k in keys])
    T["per_class"] = dict(caption="Class-wise test-fold results against the values printed in the reference article "
                                  "(its Section 4.1, representative seed-42 run; a dash marks a quantity the article "
                                  f"does not print). {NAME} is shown for the same seed and as mean ± SD over five seeds, "
                                  "at the validation-selected thresholds.",
                          header=["Class", "Model", "ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Specificity"],
                          rows=rows)

    # ---------------------------------------------------------------- reproduction
    Rp = rd(f"{R}/repro_check.csv")
    if Rp is not None and len(Rp):
        rows = []
        for cfg in C.PUBLISHED:
            d = Rp[Rp.config == cfg].set_index("metric")
            if not len(d):
                continue
            rows.append([CFG[cfg]] + [f"{pm(d.loc[m, 'published_mean'], d.loc[m, 'published_sd'])} / "
                                      f"{pm(d.loc[m, 'ours_mean'], d.loc[m, 'ours_sd'])}" if m in d.index else "–"
                                      for m in ("macro_auc", "macro_ap", "macro_f1")]
                        + [str(int(d.n_seeds.iloc[0]))])
        T["repro"] = dict(caption="Reproduction of the reference configurations under the published protocol. "
                                  "Each cell: published / our re-implementation (mean ± SD over seeds).",
                          header=["Configuration", "Macro ROC-AUC", "Macro PR-AUC", "Macro F1", "Seeds"], rows=rows)

    # ---------------------------------------------------------------- ablation
    cols = ["macro_auc", "micro_auc", "macro_ap", "macro_f1", "subset_acc", "auc_HYP"]
    hdr = ["Variant", "Parameters", "Macro ROC-AUC", "Micro ROC-AUC", "Macro PR-AUC", "Macro F1", "Subset acc.",
           "HYP ROC-AUC"]
    rows = []
    for t, l in LADDER:
        t = main if t == "MAIN" else t
        if t in tags:
            rows.append([l, f"{int(S.loc[t, 'params']):,}"] + [ms(t, m) for m in cols])
    if len(rows) > 1:
        T["ladder"] = dict(caption="Component ladder: each row adds one ingredient to the row above "
                                   "(test fold, mean ± SD over five seeds).", header=hdr, rows=rows)
    rows = [[NAME + " (full)", "5", f"{int(S.loc[main, 'params']):,}"] + [ms(main, m) for m in cols]]
    for t, l in LOO:
        if t in tags:
            rows.append(["  " + l, str(int(S.loc[t, "n_seeds"])), f"{int(S.loc[t, 'params']):,}"] + [ms(t, m) for m in cols])
    if len(rows) > 1:
        T["loo"] = dict(caption=f"Leave-one-out ablation: each row changes one thing in the full {NAME} "
                                "(test fold, mean ± SD over the seeds run; the column Seeds gives their number).",
                        header=[hdr[0], "Seeds"] + hdr[1:], rows=rows)
    P = rd(f"{R}/paired.csv")
    if P is not None and len(P):
        rows = []
        names = dict(LOO + [("abl_arch_z", "Plain student, z-scored input"), ("abl_plain", "Plain student, millivolt input"),
                            ("abl_sub", "Student + subclass supervision, no teacher")] + [(REF, "MSCA-Mamba (our re-run)"),
                     ("ref_msca_mamba_mv", "MSCA-Mamba, millivolt input")]
                     + [(f"ref_{k}", v + " (our re-run)") for k, v in CFG.items() if k != "msca_mamba"])
        for vs in [REF] + [t for t, _ in LADDER[:-1]] + [t for t, _ in LOO]:
            d = P[P.vs == vs].set_index("metric")
            if not len(d):
                continue
            for m in ("macro_auc", "macro_ap", "macro_f1"):
                r = d.loc[m]
                rows.append([names.get(vs, vs), str(int(r.n_seeds)), ML[m], f4(r.main_mean), f4(r.other_mean), sg(r["diff"]),
                             f"[{r.t_lo:+.4f}, {r.t_hi:+.4f}]", f"[{r.boot_lo:+.4f}, {r.boot_hi:+.4f}]",
                             pv(r.p_boot_holm)])
        T["paired"] = dict(
            caption=f"Paired comparisons of {NAME} with models run in this study on the same test ECGs. "
                    "t interval: seed-paired differences over the seeds both models were run with, as in the "
                    "reference article; bootstrap: 1000 "
                    "resamples of test ECGs, seed-averaged metric; p: bootstrap (bounded below by 1/1000), Holm-adjusted over "
                    "all rows computed.",
            header=["Compared with", "Seeds", "Metric", NAME, "Other", "Difference", "95% t interval", "95% bootstrap interval",
                    "p (Holm)"], rows=rows)

    # ---------------------------------------------------------------- voltage / transfer
    vt = rd(f"{A}/voltage.csv")
    if vt is not None:
        T["voltage"] = dict(
            caption="ROC-AUC of two classical voltage indices computed directly from the signal, in millivolts and "
                    "after per-lead per-recording z-scoring (no model involved).",
            header=["Folds", "Target", "Criterion", "Positives", "Millivolts", "After z-score"],
            rows=[[r.fold, r.target, r.criterion, str(int(r.n_pos)), f3(r.auc_mv), f3(r.auc_zscore)]
                  for r in vt.sort_values(["fold", "target"], ascending=[False, True]).itertuples()])
    rows = []
    for tag, nm in ((REF, "MSCA-Mamba, z-scored input (published recipe)"),
                    ("ref_msca_mamba_mv", "MSCA-Mamba, millivolt input"),
                    ("abl_zscore", NAME + ", z-scored input"), (main, NAME + ", millivolt input")):
        if tag in tags:
            rows.append([nm] + [ms(tag, m) for m in ("macro_auc", "macro_ap", "macro_f1", "auc_HYP", "rec_HYP",
                                                     "auc_NORM", "auc_MI", "auc_STTC", "auc_CD")])
    if len(rows) >= 3:
        T["transfer"] = dict(
            caption="Effect of the input representation on both architectures (test fold, mean ± SD over five "
                    "seeds). Everything except the input scaling is identical within each pair.",
            header=["Model and input", "Macro ROC-AUC", "Macro PR-AUC", "Macro F1", "HYP ROC-AUC", "HYP recall",
                    "NORM AUC", "MI AUC", "STTC AUC", "CD AUC"], rows=rows)

    # ---------------------------------------------------------------- teacher
    if os.path.exists(f"{R}/teacher_metrics.json"):
        tm = json.load(open(f"{R}/teacher_metrics.json"))
        rows = []
        for k, nm in (("ecgfounder_xfit", "ECGFounder teacher, mean of the 4 cross-fitted models"),
                      ("ecgfounder_insample", "ECGFounder teacher, trained on all 8 folds")):
            if k in tm:
                t = tm[k]["test"]
                rows.append([nm, "30,693,964", "500 Hz"] + [f4(t[m]) for m in ("macro_auc", "macro_ap", "macro_f1",
                                                                              "auc_HYP")])
        for tag, nm in (("abl_sub", "Student, no teacher"), ("abl_insample", "Student, in-sample teacher"),
                        (main, NAME + " (cross-fitted teacher)")):
            if tag in tags:
                rows.append([nm, f"{int(S.loc[tag, 'params']):,}", "100 Hz"] + [ms(tag, m) for m in
                                                                               ("macro_auc", "macro_ap", "macro_f1", "auc_HYP")])
        T["teacher"] = dict(caption="Teacher and students on the test fold. Teacher rows are single evaluations; "
                                    "student rows are mean ± SD over five seeds.",
                            header=["Model", "Parameters", "Input", "Macro ROC-AUC", "Macro PR-AUC", "Macro F1",
                                    "HYP ROC-AUC"], rows=rows)

    # ---------------------------------------------------------------- efficiency
    E = rd(f"{A}/efficiency.csv")
    if E is not None:
        nm = {main: NAME, REF: "MSCA-Mamba", "ref_bimamba": "MSCA-BiMamba", "ref_m_conv": "Matched Conv1D",
              "ref_m_gru": "Matched GRU", "ref_m_lstm": "Matched LSTM", "ref_m_transformer": "Matched Transformer",
              "ref_ms_cnn": "Multi-scale CNN"}
        pub = {"ref_" + k: v for k, v in C.PUBLISHED_EFF.items()}
        rows = []
        for r in E.itertuples():
            p = pub.get(r.tag)
            rows.append([nm.get(r.tag, r.tag), f"{int(r.params):,}", f"{r.size_mb:.3f}",
                         f"{r.lat_ms:.2f} ± {r.lat_sd:.2f}", f"{r.p95_ms:.2f}", f"{r.thr_b1:.0f}", f"{r.thr_b256:.0f}",
                         f"{r.peak_mb:.1f}", f"{r.flops_m:.1f}" if np.isfinite(r.flops_m) else "–",
                         f"{r.cpu_ms:.1f}" if np.isfinite(r.cpu_ms) else "n/a",
                         f"{p['lat']:.3f}" if p else "–"])
        T["efficiency"] = dict(
            caption="Computational characteristics measured in this study on one NVIDIA GB10 (batch size 1 unless "
                    "stated); the parameter-matched controls and BiMamba, which were not retrained here, were timed with "
                    "untrained weights, which does not affect timing. For the student the deployed network (without "
                    "the auxiliary subclass layer) is counted. The last column repeats the latency the reference "
                    "article measured on an NVIDIA L4; it "
                    "is not comparable with our timings and is shown only for orientation.",
            header=["Model", "Parameters", "FP32 size (MB)", "Latency (ms)", "P95 (ms)", "Throughput, b=1 (ECG/s)",
                    "Throughput, b=256 (ECG/s)", "Peak GPU memory (MB)", "FLOPs (M)", "CPU latency, 1 thread (ms)",
                    "Published latency, L4 (ms)"], rows=rows)

    # ---------------------------------------------------------------- clinical analyses
    G = rd(f"{A}/subgroup.csv")
    if G is not None:
        g = G.pivot(index="group", columns="tag")
        order = list(G[G.tag == main].group)
        rows = []
        for n in order:
            def cell(tag):
                try:
                    return f"{g.loc[n, ('macro_auc', tag)]:.3f} [{g.loc[n, ('lo', tag)]:.3f}, {g.loc[n, ('hi', tag)]:.3f}]"
                except KeyError:
                    return "–"
            d = g.loc[n, ("macro_auc", main)] - g.loc[n, ("macro_auc", REF)] if ("macro_auc", REF) in g.columns else np.nan
            rows.append([n, str(int(g.loc[n, ("n", main)])), cell(REF), cell(main), sg(d, 3)])
        T["subgroup"] = dict(caption="Test Macro ROC-AUC by subgroup (seed mean, 95% bootstrap interval).",
                             header=["Subgroup", "ECGs", "MSCA-Mamba (re-run)", NAME, "Difference"], rows=rows)
    Sc = rd(f"{A}/subclass_sens.csv")
    if Sc is not None:
        p = Sc.pivot(index="subclass", columns="tag", values="sens_mean")
        info = Sc.drop_duplicates("subclass").set_index("subclass")
        rows = []
        for s in info.sort_values(["superclass", "n"], ascending=[True, False]).index:
            if info.loc[s, "n"] < 5:
                continue
            a, b = p.loc[s].get(REF, np.nan), p.loc[s].get(main, np.nan)
            rows.append([info.loc[s, "superclass"], s.strip("_"), str(int(info.loc[s, "n"])), f3(a), f3(b), sg(b - a, 3)])
        T["subclass"] = dict(caption="Sensitivity within diagnostic subclasses: share of test ECGs carrying the "
                                     "subclass for which the parent superclass is flagged (seed mean; subclasses with at "
                                     "least five test ECGs).",
                             header=["Superclass", "Subclass", "Test ECGs", "MSCA-Mamba (re-run)", NAME, "Difference"],
                             rows=rows)
    Cb = rd(f"{A}/calib.csv")
    if Cb is not None:
        rows = []
        for c in C.SUPER + ["Mean"]:
            cells = [c]
            for tag in (REF, main):
                d = Cb[Cb.tag == tag]
                d = d if c == "Mean" else d[d.cls == c]
                e = d.groupby("seed")[["ece", "brier"]].mean()
                cells += [pm(e.ece.mean(), e.ece.std(ddof=1), 3), pm(e.brier.mean(), e.brier.std(ddof=1), 3)]
            rows.append(cells)
        T["calib"] = dict(caption="Calibration on the test fold (mean ± SD over five seeds; lower is better).",
                          header=["Class", "ECE, MSCA-Mamba (re-run)", "Brier, MSCA-Mamba (re-run)", "ECE, " + NAME,
                                  "Brier, " + NAME], rows=rows)
    N = rd(f"{A}/noise.csv")
    if N is not None:
        lab = dict(wander="Baseline wander (mV)", white="Broadband noise SD (mV)", drop="Leads zeroed")
        rows = []
        for kind in ("wander", "white", "drop"):
            for lv in sorted(N[N.kind == kind].level.unique()):
                cells = [lab[kind], f"{lv:g}"]
                for tag in (REF, main):
                    v = N[(N.tag == tag) & (N.kind == kind) & (N.level == lv)].macro_auc
                    cells.append(pm(v.mean(), v.std(ddof=1)))
                rows.append(cells)
        T["noise"] = dict(caption="Test Macro ROC-AUC under synthetic perturbations of the test ECGs (mean ± SD over "
                                  "five seeds; no model was trained with these perturbations).",
                          header=["Perturbation", "Level", "MSCA-Mamba (re-run)", NAME], rows=rows)
    rows = []
    for t, l in ((main, "12 leads"), ("lead_limb6", "6 limb leads"), ("lead_3", "I, II, V2"), ("lead_ii", "Lead II")):
        if t in tags:
            rows.append([l, str(int(S.loc[t, "n_seeds"]))] + [ms(t, m) for m in ("macro_auc", "macro_ap", "macro_f1")]
                        + [ms(t, f"auc_{c}", 3) for c in C.SUPER])
    if len(rows) > 1:
        T["leads"] = dict(caption=f"Reduced-lead {NAME} students, all taught by the 12-lead teacher (test fold, "
                                  "mean ± SD over the seeds run).",
                          header=["Input leads", "Seeds", "Macro ROC-AUC", "Macro PR-AUC", "Macro F1"] + [f"{c} AUC" for c in C.SUPER],
                          rows=rows)
    En = rd(f"{A}/ensemble.csv")
    if En is not None:
        en = En.set_index("tag")
        rows = []
        for tag, nmm in ((REF, "MSCA-Mamba (re-run)"), (main, NAME)):
            if tag in en.index:
                rows.append([nmm + ", mean of single models"] + [ms(tag, m) for m in C.MAIN7])
                rows.append([nmm + ", 5-seed ensemble"] + [f4(en.loc[tag, m]) for m in C.MAIN7])
        T["ensemble"] = dict(caption="Single models and five-seed ensembles (probability average) on the test fold.",
                             header=["Model"] + C.MAIN7_LABEL, rows=rows)
    return T


if __name__ == "__main__":
    for k, v in build(sys.argv[1] if len(sys.argv) > 1 else "voltkd").items():
        print(f"\n== {k}: {v['caption'][:90]}")
        print(pd.DataFrame(v["rows"], columns=v["header"]).to_string(index=False)[:2500])
