"""Shared figure style: one palette, one rc block, used by every figure in the paper.

Colours are the validated categorical order of the project's data-visualisation reference
(blue, orange, aqua, yellow, magenta, green, violet, red), assigned to models in a fixed order
and never cycled, so a model keeps its colour in every figure.  Where more than three series
share a panel the model is also encoded by marker shape and a direct label.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#eb6834")
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8980"
SURFACE = "#ffffff"
GRID = "#e3e2dd"

COLOR = {"ours": CAT[0], "ref": CAT[1], "teacher": CAT[6], "zs": CAT[3], "pub": INK2,
         "ctrl": CAT[2], "other": MUTED}
MARKER = {"ours": "D", "ref": "^", "teacher": "s", "zs": "v", "pub": "o", "ctrl": "P",
          "other": "o"}
NAME = "VOLT-KD"
LABEL = {"ours": NAME + " (ours)", "ref": "MSCA-Mamba (our re-run)",
         "pub": "MSCA-Mamba (published)", "teacher": "ECGFounder teacher",
         "zs": NAME + ", z-scored input"}
CLS = {"NORM": CAT[0], "MI": CAT[1], "STTC": CAT[2], "CD": CAT[3], "HYP": CAT[4]}

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 2.0, "lines.markersize": 5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
}
plt.rcParams.update(RC)


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, axis=grid if grid != "both" else "both", color=GRID, linewidth=0.6)
        if grid != "both":
            ax.grid(False, axis="x" if grid == "y" else "y")
    return ax


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.basename(path)}.pdf/.png", flush=True)
