# VOLT-KD

Amplitude-Preserving Learning for Lightweight Multi-Label ECG Classification.

Research code, aggregate experimental results, and the revised Chinese and polished English LaTeX manuscripts. VOLT-KD combines millivolt ECG inputs, a compact multiscale backbone, diagnostic subclass supervision, and cross-fitted ECGFounder soft labels.

## Manuscripts

- [Polished English LaTeX](paper/en/manuscript_en.tex) and [PDF](paper/en/manuscript_en.pdf)
- [Chinese LaTeX](paper/zh/manuscript_zh_v4.tex) and [PDF](paper/zh/manuscript_zh_v4.pdf)
- The English main manuscript includes 13 figures and 12 tables; its [standalone supplement](paper/supplement_en/supplementary_en.tex) ([PDF](paper/supplement_en/supplementary_en.pdf)) contains figures S1–S9 and tables S1–S9. Each compiles independently with its own bibliography. The Chinese edition retains all 22 figures in one document.
- The English main manuscript and supplement together translate the Chinese edition. Their 21 tables preserve the original numerical results.

Literature revision (8 October 2026): Related Work incorporates ten additional papers from 2025–2026. Both manuscripts contain 30 references, including 10 from 2026 and 8 from 2025. Table 3 identifies its published and preprint sources. See the [source audit and update record](docs/literature_update_2026-10-08.md).

Figure 1 update (9 October 2026): both manuscripts use the user-supplied vector main figure and matching four-panel captions. Current length: English main 24 pages, independent English supplement 10 pages; Chinese combined 35 pages.

## Main results

PTB-XL 1.0.3, official patient-disjoint folds 1--8 / 9 / 10 for training / validation / testing; five seeds unless specified otherwise.

| Quantity | Complete VOLT-KD |
|---|---:|
| Macro ROC-AUC | 0.9298 ± 0.0012 |
| Macro average precision | 0.8274 ± 0.0020 |
| Macro F1 | 0.7542 ± 0.0009 |
| Training parameters | 112,364 |
| Deployed parameters | 107,189 |
| Estimated computation | 30.1 MFLOPs |

These are existing study results, not new runs performed while preparing this repository. Published MSCA-Mamba values and local measurements are distinguished in the manuscript. The main comparison uses published values; local predictions support input, subgroup, perturbation, and efficiency analyses.

## Repository structure

- code/: preprocessing, models, teachers, student training, evaluation, analysis, and plotting.
- results/: aggregate per-seed results and analysis tables.
- configs/run_arguments.json: recorded arguments from completed experiments, with machine-specific paths made relative.
- paper/en/, paper/supplement_en/, and paper/zh/: self-contained English main manuscript, English supplement, and Chinese manuscript with their vector figures.
- docs/: configuration mapping and editorial/validation notes.

Raw recordings, patient-level metadata/predictions, checkpoints, teacher logits, temporary files, and downloaded third-party papers are not distributed here. Obtain data and teacher resources from their original publishers.

## Environment

Use Python 3.10 or later. Install a PyTorch build appropriate for your CUDA environment before the remaining dependencies:

    python -m pip install -r requirements.txt

Training and most model-based analyses expect a CUDA device. Historical MSCA-Mamba measurements used mamba_ssm with CUDA selective scan; install a compatible mamba-ssm build to reproduce those measurements. The mambapy fallback supports importing model definitions but may differ in speed and numerical behavior. This release does not claim a reconstructed lockfile for the historical environment.

Paths default to the repository root. Set VOLTKD_ROOT to another project directory when data and results are stored elsewhere. The release changes project-path resolution only; original training and model logic is retained.

## Data preparation

The source dataset is [PTB-XL 1.0.3 on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/). The downloader verifies the publisher's SHA-256 checksums.

    python code/fetch_ptbxl.py --workers 12
    python code/prep.py --hr

This produces X100.npy, X500.npy, meta.parquet, and label_maps.json under data/prep/. Preprocessing checks the 1000 ADC units/mV gain and lead order. Training uses 17,084 labeled records, validation 2,146, and testing 2,158.

## Teacher preparation and cross-fitting

Obtain net1d.py and the twelve-lead pretrained checkpoint from the [official ECGFounder repository](https://github.com/PKUDigitalHealth/ECGFounder) and [model release](https://huggingface.co/PKUDigitalHealth/ECGFounder). Place them in:

    data/weights/ecgfounder/net1d.py
    data/weights/ecgfounder/12_lead_ECGFounder.pth

These upstream files and weights are not bundled; their original terms apply.

    python code/teacher.py --arch ecgfounder --holdout g1 --epochs 8 --seed 0
    python code/teacher.py --arch ecgfounder --holdout g2 --epochs 8 --seed 0
    python code/teacher.py --arch ecgfounder --holdout g3 --epochs 8 --seed 0
    python code/teacher.py --arch ecgfounder --holdout g4 --epochs 8 --seed 0
    python code/make_soft.py ecgfounder

Groups g1--g4 exclude folds (1,2), (3,4), (5,6), and (7,8), respectively. Validation fold 9 selects checkpoints. Each training record receives logits from the teacher that excluded its group. For the in-sample control, also train with --holdout none before stitching soft labels.

## Student training

Run from the repository root. A complete single-seed experiment is:

    python code/train.py --model student --protocol ours --tag voltkd --seed 42 --epochs 24 --kd results/soft/ecgfounder_xfit.npz

Repeat for seeds 123, 456, 789, and 2025. Default architecture: 48/96 channels, GRU hidden size 56 per direction, millivolt inputs, and independent superclass/subclass heads.

Input and supervision controls without a teacher:

    python code/train.py --model student --protocol ours --tag abl_arch_z --seed 42 --prep zscore --w_sub 0
    python code/train.py --model student --protocol ours --tag abl_plain --seed 42 --prep mv --w_sub 0
    python code/train.py --model student --protocol ours --tag abl_sub --seed 42 --prep mv

Local MSCA-Mamba controls:

    python code/train.py --model msca_mamba --protocol ref --tag ref_msca_mamba --seed 42
    python code/train.py --model msca_mamba --protocol ref --tag ref_msca_mamba_mv --seed 42 --prep mv

See [configuration mapping](docs/configurations.md) and recorded arguments for structural, teacher, and reduced-lead variants. Completed runs are skipped unless --force is supplied. Optional train_k.py batches seeds; pass --epochs 24 explicitly because its historical default differs.

## Evaluation and figures

After training the required runs:

    python code/evaluate.py --main voltkd --boot 1000
    python code/analysis.py --main voltkd
    python code/figures.py --main voltkd

These scripts read prepared data, checkpoints, and predictions. Included aggregate tables permit inspection without downloading patient-level data, but do not replace inputs required to rerun all analyses. Plot scripts regenerate experimental plots; manuscript PDFs include subsequent label and layout refinements.

## Compile the manuscripts

English, from paper/en/:

    xelatex -interaction=nonstopmode -halt-on-error manuscript_en.tex
    xelatex -interaction=nonstopmode -halt-on-error manuscript_en.tex

Supplement, independently from paper/supplement_en/:

    xelatex -interaction=nonstopmode -halt-on-error supplementary_en.tex
    xelatex -interaction=nonstopmode -halt-on-error supplementary_en.tex

Chinese, from paper/zh/, uses XeLaTeX and the Windows Chinese fonts specified in the source. On another operating system, select an installed CTeX font set before compiling.

## Release validation

The English manuscript was compiled and visually reviewed. All 21 tables retain their numerical cells and displayed equations are unchanged. Both language editions now contain 30 matching references; all 22 English figures have English labels. Python files passed syntax compilation and path-resolution checks. Full training was not repeated during repository preparation. See [editorial notes](docs/english_polishing.md).
