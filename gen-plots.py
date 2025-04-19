#!/usr/bin/env python3


import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches

# color-blind friendly palette
from seaborn.palettes import color_palette

if len(sys.argv) != 2:
    print("Usage: python gen-plots.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

plots_dir = "plots"

# Load data
df = pd.read_csv(csv_file)

# Prepare tile combination labels
combos = sorted(
    df[["c_tile", "a_tile", "b_tile"]].drop_duplicates().apply(tuple, axis=1)
)

labels = [
    f"{c.split('_')[2]}/{a.split('_')[2]}/{b.split('_')[2]}" for c, a, b in combos
]

datasets = sorted(df["dataset"].unique())
frameworks = sorted(df["framework"].unique())
WIDTH = 0.2
x = np.arange(len(labels))

# Color-blind friendly palette
PALETTE = color_palette("colorblind", len(frameworks))
BORDER_COLOR = "black"  # same for valid and invalid

for ds in datasets:
    subset = df[df["dataset"] == ds]

    pivot_time = subset.pivot_table(
        index=["c_tile", "a_tile", "b_tile"],
        columns="framework",
        values="time",
        aggfunc="mean",
    ).reindex(index=combos)

    pivot_valid = subset.pivot_table(
        index=["c_tile", "a_tile", "b_tile"],
        columns="framework",
        values="valid",
        aggfunc="max",
    ).reindex(index=combos)

    fig, ax = plt.subplots(figsize=(4, 3))
    # Plot bars with uniform border color
    for i, fw in enumerate(frameworks):
        times = pivot_time[fw].values
        valids = pivot_valid[fw].values

        bars = ax.bar(
            x + i * WIDTH,
            times,
            WIDTH,
            facecolor=PALETTE[i],
            edgecolor=BORDER_COLOR,
            linewidth=1,
            label=fw,
        )

        for bar, valid in zip(bars, valids):
            if valid == 0:
                bar.set_alpha(0.4)
                bar.set_hatch("///")

    # Framework legend
    framework_patches = [
        mpatches.Patch(facecolor=PALETTE[i], edgecolor=BORDER_COLOR, label=fw)
        for i, fw in enumerate(frameworks)
    ]

    legend1 = ax.legend(handles=framework_patches, title="Framework", loc="upper left", fontsize="small", framealpha=0.5, title_fontsize="small")
    ax.add_artist(legend1)

    # Validation legend
    valid_patch = mpatches.Patch(
        facecolor="none", edgecolor=BORDER_COLOR, label="Valid"
    )
    invalid_patch = mpatches.Patch(
        facecolor="none", edgecolor=BORDER_COLOR, hatch="///", label="Invalid", alpha=0.4
    )
    legend2 = ax.legend(
        handles=[valid_patch, invalid_patch], title="Validation", loc="upper right", fontsize="small", framealpha=0.5, title_fontsize="small"
    )
    ax.add_artist(legend2)

    # Formatting
    ax.set_xticks(x + (len(frameworks) - 1) * WIDTH / 2)
    ax.set_xticklabels(labels, ha="center", fontsize="small")
    ax.set_xlabel("Major dimensions of C/A/B tiles", loc="center", fontsize="medium")
    ax.set_ylabel("Runtime [s]", fontsize="medium")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    os.makedirs(plots_dir, exist_ok=True)

    plt.tight_layout(pad=0.5)
    plt.savefig(f"{plots_dir}/runtime_by_framework_{ds}.pdf")

    plt.close(fig)
