#!/usr/bin/env python3
"""
Usage:
    gen_plots.py <csv_file>

Plot runtime comparisons for different frameworks and tile configurations
based on a CSV input. The CSV should contain columns:
  - framework
  - dataset
  - c_tile, a_tile, b_tile
  - time
  - valid

Example:
    python gen_plots.py compare.csv
"""

import os

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch

# ─── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Plot runtime comparisons from a CSV of framework benchmarks"
)

parser.add_argument(
    '--show-sdev',
    action='store_true',
    help='Show standard deviation as error bars'
)

parser.add_argument(
    'csv_file',
    help='Path to the input CSV file containing the benchmark data'
)

parser.add_argument(
    '--use-lines',
    action='store_true',
    help='Use lines instead of bars for plotting'
)

parser.add_argument(
    '--use-points',
    action='store_true',
    help='Do not show the legend'
)

parser.add_argument(
    '--no-validation',
    action='store_true',
    help='Do not show the validation status'
)

parser.add_argument(
    '--no-boostP2P',
    action='store_true',
    help='Do not show the Boost.MPI implementation that uses point-to-point communication'
)

args = parser.parse_args()

# ─── Load & prepare data ────────────────────────────────────────────────────────
df = pd.read_csv(args.csv_file)

if df.empty:
    raise ValueError("Input CSV file is empty or not found.")

# Ultra-concise labels: strip prefixes/suffixes to just letter codes
df['c_code'] = (
    df['c_tile']
      .str.replace('_MAJOR', '', regex=False)
      .str.replace('C_TILE_', '', regex=False)
)
df['a_code'] = (
    df['a_tile']
      .str.replace('_MAJOR', '', regex=False)
      .str.replace('A_TILE_', '', regex=False)
)
df['b_code'] = (
    df['b_tile']
      .str.replace('_MAJOR', '', regex=False)
      .str.replace('B_TILE_', '', regex=False)
)

if args.no_boostP2P:
    df = df[~df['framework'].str.contains('boostP2P', na=False)]

df['framework'] = (df['framework']
                   .str.replace('C_SCATTER_', '', regex=False)
                   .str.replace('C_GATHER_', '', regex=False)
                   .str.replace('A_SCATTER_', '', regex=False)
                   .str.replace('A_GATHER_', '', regex=False)
                   .str.replace('B_SCATTER_', '', regex=False)
                   .str.replace('B_GATHER_', '', regex=False)
                   .str.replace('_MAJOR', '', regex=False)
)

df['tile_label'] = df['c_code'] + '/' + df['a_code'] + '/' + df['b_code']

# Automatically discover unique frameworks & datasets
frameworks = df['framework'].unique()
if 'noarr' in frameworks:
    frameworks = ['noarr'] + [fw for fw in frameworks if fw != 'noarr']
if 'boost' in frameworks:
    frameworks = [fw for fw in frameworks if fw != 'boost'] + ['boost']

datasets   = df['dataset'].unique()

if 'MINI' in datasets and 'MEDIUM' in datasets and 'EXTRALARGE' in datasets:
    datasets = ['MINI', 'MEDIUM', 'EXTRALARGE'] + [ds for ds in datasets if ds not in ['MINI', 'MEDIUM', 'EXTRALARGE']]

# ─── Build color map ────────────────────────────────────────────────────────────
palette = sns.color_palette("colorblind", len(frameworks))
colors  = {fw: palette[i] for i, fw in enumerate(frameworks)}

raw_shapes = ['o', '^', 'D', 'v', 'x', 'P', '*']
shapes = {fw: raw_shapes[i % len(raw_shapes)] for i, fw in enumerate(frameworks)}

# ─── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, len(datasets),
    figsize=(3.5 * len(datasets), 3),
    sharey=False,
    layout='constrained',
)
bar_width = 0.18

for ax, ds in zip(axes, datasets) if len(datasets) > 1 else [(axes, datasets[0])]:
    sub = df[df['dataset'] == ds]
    table = sub.pivot_table(
        index='tile_label',
        columns=['framework', 'valid'],
        values=['mean_time', 'sd_time']
    )

    labels = table.index.tolist()
    x      = np.arange(len(labels))

    for i, fw in enumerate(frameworks):
        offset = (i - (len(frameworks)-1)/2) * bar_width
        for valid in (1, 0):
            col = (fw, valid)

            if col in table['mean_time'].columns:
                vals = table['mean_time'][col].reindex(labels).values
            else:
                vals = np.full(len(labels), np.nan)

            if args.show_sdev and col in table['sd_time'].columns:
                sdev = table['sd_time'][col].reindex(labels).values
            elif args.show_sdev:
                sdev = np.full(len(labels), np.nan)
            else:
                sdev = None

            if args.use_lines:
                ax.plot(
                    x, vals,
                    color=colors[fw],
                    label=fw if valid == 1 else None
                )

            if args.use_points:
                ax.scatter(
                    x, vals,
                    color=colors[fw],
                    label=fw if valid == 1 else None,
                    marker=shapes[fw],
                    zorder=5
                )

            if args.use_lines or args.use_points:
                if sdev is not None:
                    ax.errorbar(
                        x, vals,
                        yerr=sdev,
                        color=colors[fw],
                        alpha=0.5,
                        marker=None,
                        zorder=4
                    )

                continue

            if valid == 0:
                ax.bar(
                    x + offset, vals, bar_width,
                    yerr=sdev,
                    color='white',
                    hatch=None,
                    alpha=1.0,
                    edgecolor='white',
                    label=None,
                    zorder=4
                )

            ax.bar(
                x + offset, vals, bar_width,
                yerr=sdev,
                color=colors[fw],
                hatch=None if valid == 1 else '//',
                alpha=1.0 if valid == 1 else 0.5,
                edgecolor='black',
                label=fw if valid == 1 else None,
                zorder=5
            )

    ax.set_ylim(bottom=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(ds)
    ax.set_xlabel("Major dimension of C/A/B")
    ax.grid(axis='y', linestyle=':', linewidth=0.5)

    # Scientific notation formatter with "×10ⁿ"
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(fmt)

# Common y‑label
aq = axes[0] if len(datasets) > 1 else axes
aq.set_ylabel("Runtime [s]")

# Framework legend (first subplot)
fw_handles = [plt.Rectangle((0,0),1,1, color=colors[fw]) for fw in frameworks]
fig.legend(
    fw_handles, frameworks,
    title="Framework",
    loc='upper right',
    bbox_to_anchor=(1., 0.94),
)

if not args.no_validation:
    # Valid/Invalid legend (on the rightmost subplot)
    valid_patch   = Patch(facecolor='white', edgecolor='black', label='Valid')
    invalid_patch = Patch(facecolor='white', edgecolor='black', hatch='//', alpha=0.5, label='Invalid')

    if len(datasets) > 1:
        aend = axes[-1]
    else:
        aend = axes.twinx()
        aend.set_yticklabels([])
        aend.set_yticks([])

    aend.legend(
        [valid_patch, invalid_patch],
        ['Valid', 'Invalid'],
        title="Validation",
        loc='upper right'
    )

os.makedirs('plots', exist_ok=True)

file_name = os.path.splitext(os.path.basename(args.csv_file))[0]
file_path = f'plots/{file_name}.pdf'

# plt.tight_layout(pad=0.2)
plt.savefig(file_path, bbox_inches='tight')
