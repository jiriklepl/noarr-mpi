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
    '--datasets',
    nargs='+',
    default=None, # Default to None to allow automatic detection
    help='List of datasets to plot (e.g., MINI, MEDIUM, EXTRALARGE). If not specified, all datasets will be used.'
)

parser.add_argument(
    '--output',
    default=None,
    help='Output file name for the plot (default: plots/<first_csv_file_name>.pdf)'
)

parser.add_argument(
    'csv_files',
    nargs='+',
    help='Path to the input CSV file(s) containing the benchmark data'
)

parser.add_argument(
    '--mark-noarr',
    action='store_true',
    help='Mark the Noarr-MPI framework in the plot'
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
    '--no-validation-legend',
    action='store_true',
    help='Do not show the validation legend'
)

parser.add_argument(
    '--no-renaming',
    action='store_true',
    help='Do not rename the frameworks to official names'
)

parser.add_argument(
    '--no-boostP2P',
    action='store_true',
    help='Do not show the Boost.MPI implementation that uses point-to-point communication'
)

args = parser.parse_args()

# ─── Load & prepare data ────────────────────────────────────────────────────────
if len(args.csv_files) > 1:
    # If multiple CSV files are provided, concatenate them into a single DataFrame
    df = pd.concat([pd.read_csv(f) for f in args.csv_files], ignore_index=True)
elif len(args.csv_files) == 1:
    # If a single CSV file is provided, read it directly
    df = pd.read_csv(args.csv_files[0])
else:
    # If no CSV file is provided, raise an error
    parser.print_help()
    raise ValueError("No CSV file provided. Please specify a CSV file.")

if df.empty:
    raise ValueError("Input CSV file is empty or not found.")

if args.no_boostP2P:
    df = df[~df['framework'].str.contains('boostP2P', na=False)]

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

df['framework'] = (df['framework']
                   .str.replace('C_SCATTER_', '', regex=False)
                   .str.replace('C_GATHER_', '', regex=False)
                   .str.replace('A_SCATTER_', '', regex=False)
                   .str.replace('A_GATHER_', '', regex=False)
                   .str.replace('B_SCATTER_', '', regex=False)
                   .str.replace('B_GATHER_', '', regex=False)
                   .str.replace('_MAJOR', '', regex=False)
)

df['mean_time'] = df['mean_time'].astype(float)
df['sd_time'] = df['sd_time'].astype(float)
df['valid'] = df['valid'].astype(int)

if not args.no_renaming:
    df['framework'] = df['framework'].str.replace('noarr', 'Noarr-MPI', regex=False)
    df['framework'] = df['framework'].str.replace('boostP2P', 'Boost.MPI (P2P)', regex=False)
    df['framework'] = df['framework'].str.replace('boost', 'Boost.MPI', regex=False)
    df['framework'] = df['framework'].str.replace('mpi', 'MPI', regex=False)

df['tile_label'] = df['c_code'] + '/' + df['a_code'] + '/' + df['b_code']

# Automatically discover unique frameworks & datasets
frameworks = df['framework'].unique()
frameworks.sort()

if 'noarr' in frameworks:
    frameworks = ['noarr'] + [fw for fw in frameworks if fw != 'noarr']
elif 'Noarr-MPI' in frameworks:
    frameworks = ['Noarr-MPI'] + [fw for fw in frameworks if fw != 'Noarr-MPI']
if 'boost' in frameworks:
    frameworks = [fw for fw in frameworks if fw != 'boost'] + ['boost']
elif 'Boost.MPI' in frameworks:
    frameworks = [fw for fw in frameworks if fw != 'Boost.MPI'] + ['Boost.MPI']
if 'boostP2P' in frameworks:
    frameworks = [fw for fw in frameworks if fw != 'boostP2P'] + ['boostP2P']
elif 'Boost.MPI (P2P)' in frameworks:
    frameworks = [fw for fw in frameworks if fw != 'Boost.MPI (P2P)'] + ['Boost.MPI (P2P)']

if len(frameworks) == 0:
    raise ValueError("No valid frameworks found in the input data.")

datasets   = df['dataset'].unique()

if args.datasets is not None:
    datasets = [ds for ds in args.datasets if ds in datasets]
else:
    if 'MINI' in datasets and 'MEDIUM' in datasets and 'EXTRALARGE' in datasets:
        datasets = ['MINI', 'MEDIUM', 'EXTRALARGE'] + [ds for ds in datasets if ds not in ['MINI', 'MEDIUM', 'EXTRALARGE']]

if len(datasets) == 0:
    exit()

# ─── Build color map ────────────────────────────────────────────────────────────
palette = sns.color_palette("colorblind", len(frameworks))
colors  = {fw: palette[i] for i, fw in enumerate(frameworks)}

raw_shapes = ['o', '^', 'D', 'v', 'x', 'P', '*']
shapes = {fw: raw_shapes[i % len(raw_shapes)] for i, fw in enumerate(frameworks)}

# ─── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, len(datasets),
    figsize=(len(frameworks) * len(datasets), 2.5),
    sharey=False,
    layout='constrained',
)
BAR_WIDTH = 1 / (len(frameworks) + .75)

# If multiple records are present for the same algorithm+dataset+tile configuration,
# take the mean and standard deviation
def aggregate(group):
    mean = group['mean_time'].mean()
    variance = (group['sd_time']**2 + group['mean_time']**2).mean() - mean**2
    return pd.Series({'mean_time': mean, 'sd_time': np.sqrt(variance)})

df = df.groupby(['framework', 'dataset', 'tile_label', 'valid']).apply(aggregate, include_groups=False).reset_index()

for ax, ds in zip(axes, datasets) if len(datasets) > 1 else [(axes, datasets[0])]:
    sub = df[df['dataset'] == ds]
    table = sub.pivot_table(
        index='tile_label',
        columns=['framework', 'valid'],
        values=['mean_time', 'sd_time']
    )

    labels = table.index.tolist()
    x      = np.arange(len(labels))

    for i, fw in reversed([(i, fw) for i, fw in enumerate(frameworks)]):
        offset = (i - (len(frameworks)-1)/2) * BAR_WIDTH
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
                    x + offset, vals, BAR_WIDTH,
                    yerr=sdev,
                    color='white',
                    hatch=None,
                    alpha=1.0,
                    edgecolor=None,
                    label=None,
                    zorder=4
                )

            ax.bar(
                x + offset, vals, BAR_WIDTH,
                yerr=sdev,
                color=colors[fw],
                hatch=None if valid == 1 else '//',
                alpha=1.0 if valid == 1 else 0.5,
                edgecolor='black' if args.mark_noarr and fw == 'Noarr-MPI' else None,
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
fw_handles = [plt.Rectangle(
    (0,0), 1, 1,
    color=None,
    facecolor=colors[fw],
    edgecolor='black' if args.mark_noarr and fw == 'Noarr-MPI' else None,
) for fw in frameworks]
fig.legend(
    fw_handles, frameworks,
    title="Framework",
    loc='upper left',
    bbox_to_anchor=(0.05, 0.935),
)

if not args.no_validation_legend:
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

if args.output is not None:
    # If an output file name is specified, use it
    FILE_PATH = args.output
else:
    if args.csv_files[0].endswith('.csv'):
        FILE_NAME = os.path.splitext(os.path.basename(args.csv_files[0]))[0]
    else:
        # If the CSV file name is not provided, use a default name
        FILE_NAME = 'plot'

    os.makedirs('plots', exist_ok=True)
    FILE_PATH = f'plots/{FILE_NAME}.pdf'

plt.savefig(FILE_PATH, bbox_inches='tight')
