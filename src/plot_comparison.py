#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison plots for redesigned benchmarks (Fuzzy IB-ME vs Silent-Share).

Reads:
  - fuzzy_ibme_bn254_fair_results/benchmark_redesign.csv
  - silent_share_results/benchmark_redesign.csv
  - Optional: fuzzy_ibme_bn254_fair_results/threshold_extra.csv
  - Optional: silent_share_results/threshold_extra.csv

Generates PDF figures in figures/:
  - enc_time_vs_policy.pdf       : encryption time vs |P|
  - ct_size_vs_policy.pdf        : ciphertext size vs |P| (in MB)
  - dec_time_vs_attr.pdf         : decryption time vs |A|
  - dk_size_vs_attr.pdf          : decryption key size vs |A|
  - dec_time_vs_threshold.pdf    : decryption time vs d
  - search_space_comparison.pdf  : theoretical search space comparison
Also outputs combined CSV for Origin.
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
FUZZY_CSV = os.path.join('fuzzy_ibme_bn254_fair_results', 'benchmark_redesign.csv')
SILENT_CSV = os.path.join('silent_share_results', 'benchmark_redesign.csv')
FUZZY_EXTRA_CSV = os.path.join('fuzzy_ibme_bn254_fair_results', 'threshold_extra.csv')
SILENT_EXTRA_CSV = os.path.join('silent_share_results', 'threshold_extra.csv')
OUTPUT_DIR = 'figures'
DATA_DIR = 'data_csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load raw CSV files
# ------------------------------------------------------------
fuzzy_raw = pd.read_csv(FUZZY_CSV)
silent_raw = pd.read_csv(SILENT_CSV)

# ------------------------------------------------------------
# Reshape data into a unified long-format DataFrame
# ------------------------------------------------------------
def reshape_df(raw, scheme):
    """
    Convert raw CSV (which has misaligned columns due to DictWriter) into
    a clean DataFrame with columns:
      experiment, policy_size, attr_size, d, overlap, enc_ms_mean, enc_ms_std,
      ct_bytes_mean, ct_bytes_std, dec_ms_mean, dec_ms_std, dk_bytes_mean,
      dk_bytes_std, expected_attempts
    """
    rows = []
    for _, row in raw.iterrows():
        exp = row['experiment']
        base = {
            'experiment': exp,
            'policy_size': row['policy_size'],
            'attr_size': row['attr_size'],
            'd': row['d'],
        }
        if exp == 'encryption_scaling':
            base['enc_ms_mean'] = row['enc_ms_mean']
            base['enc_ms_std'] = row['enc_ms_std']
            base['ct_bytes_mean'] = row['ct_bytes_mean']
            base['ct_bytes_std'] = row['ct_bytes_std']
        elif exp == 'decryption_scaling':
            # In the raw file, decryption times and key sizes are mistakenly
            # placed in enc_ms_* and ct_bytes_* columns.
            base['dec_ms_mean'] = row['enc_ms_mean']
            base['dec_ms_std'] = row['enc_ms_std']
            base['dk_bytes_mean'] = row['ct_bytes_mean']
            base['dk_bytes_std'] = row['ct_bytes_std']
        elif exp == 'threshold_scaling':
            base['dec_ms_mean'] = row['enc_ms_mean']
            base['dec_ms_std'] = row['enc_ms_std']
        elif exp == 'search_complexity':
            # The raw CSV has extra columns not in header; need to find them.
            vals = [row[k] for k in row.index if k not in ['experiment', 'policy_size', 'attr_size', 'd']]
            vals = [v for v in vals if pd.notna(v)]
            if len(vals) >= 2:
                base['overlap'] = float(vals[-2])
                base['expected_attempts'] = float(vals[-1])
            else:
                print(f"Warning: search_complexity row has insufficient data: {row}")
        rows.append(base)

    df = pd.DataFrame(rows)
    df['scheme'] = scheme
    return df

fuzzy_df = reshape_df(fuzzy_raw, 'Fuzzy IB-ME')
silent_df = reshape_df(silent_raw, 'Silent-Share')

# Combine original data
combined = pd.concat([fuzzy_df, silent_df], ignore_index=True, sort=False)

# ------------------------------------------------------------
# Load optional threshold extra CSV files if they exist
# ------------------------------------------------------------
if os.path.exists(FUZZY_EXTRA_CSV):
    fuzzy_extra = pd.read_csv(FUZZY_EXTRA_CSV)
    fuzzy_extra['scheme'] = 'Fuzzy IB-ME'
    combined = pd.concat([combined, fuzzy_extra], ignore_index=True, sort=False)
    print(f"Loaded extra threshold data from {FUZZY_EXTRA_CSV}")
else:
    print(f"Extra threshold CSV not found: {FUZZY_EXTRA_CSV} (skipped)")

if os.path.exists(SILENT_EXTRA_CSV):
    silent_extra = pd.read_csv(SILENT_EXTRA_CSV)
    silent_extra['scheme'] = 'Silent-Share'
    combined = pd.concat([combined, silent_extra], ignore_index=True, sort=False)
    print(f"Loaded extra threshold data from {SILENT_EXTRA_CSV}")
else:
    print(f"Extra threshold CSV not found: {SILENT_EXTRA_CSV} (skipped)")

# Save combined CSV
combined.to_csv(os.path.join(DATA_DIR, 'combined_redesign.csv'), index=False)
print("Combined CSV saved to data_csv/combined_redesign.csv")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def save_fig(fig, filename):
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def plot_metric(df, x_col, y_col, xlabel, ylabel, title, filename,
                yerr_col=None, log_y=False):
    """Plot a metric vs variable for both schemes, with data sorted by x."""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Skipping {filename}: missing columns {x_col} or {y_col}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for scheme, grp in df.groupby('scheme'):
        grp = grp.sort_values(by=x_col)
        yerr = grp[yerr_col] if yerr_col and yerr_col in grp.columns else None
        ax.errorbar(grp[x_col], grp[y_col], yerr=yerr, label=scheme,
                    marker='o', capsize=4, linestyle='-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both')
    save_fig(fig, filename)

def binom(n, k):
    """Exact binomial coefficient."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    res = 1
    for i in range(1, k + 1):
        res = res * (n - k + i) // i
    return res

# ------------------------------------------------------------
# Experiment-specific plots
# ------------------------------------------------------------

# 1. Encryption time vs policy size
enc_df = combined[combined['experiment'] == 'encryption_scaling'].copy()
plot_metric(enc_df, 'policy_size', 'enc_ms_mean',
            r'Policy Size $|\mathcal{P}|$',
            r'Encryption Time (ms)',
            r'Encryption Time vs Policy Size',
            'enc_time_vs_policy.pdf',
            yerr_col='enc_ms_std')

# 2. Ciphertext size vs policy size (in MB)
enc_df['ct_mb_mean'] = enc_df['ct_bytes_mean'] / (1024 * 1024)
enc_df['ct_mb_std'] = enc_df['ct_bytes_std'] / (1024 * 1024)
plot_metric(enc_df, 'policy_size', 'ct_mb_mean',
            r'Policy Size $|\mathcal{P}|$',
            r'Ciphertext Size (MB)',
            r'Ciphertext Size vs Policy Size',
            'ct_size_vs_policy.pdf',
            yerr_col='ct_mb_std')

# 3. Decryption time vs receiver attribute size
dec_df = combined[combined['experiment'] == 'decryption_scaling'].copy()
plot_metric(dec_df, 'attr_size', 'dec_ms_mean',
            r'Receiver Attribute Size $|\mathcal{A}|$',
            r'Decryption Time (ms)',
            r'Decryption Time vs Receiver Attribute Size',
            'dec_time_vs_attr.pdf',
            yerr_col='dec_ms_std')

# 4. Decryption key size vs receiver attribute size
plot_metric(dec_df, 'attr_size', 'dk_bytes_mean',
            r'Receiver Attribute Size $|\mathcal{A}|$',
            r'Decryption Key Size (bytes)',
            r'Decryption Key Size vs Receiver Attribute Size',
            'dk_size_vs_attr.pdf',
            yerr_col='dk_bytes_std')

# 5. Decryption time vs threshold
thr_df = combined[combined['experiment'] == 'threshold_scaling'].copy()
plot_metric(thr_df, 'd', 'dec_ms_mean',
            r'Threshold $d$',
            r'Decryption Time (ms)',
            r'Decryption Time vs Threshold',
            'dec_time_vs_threshold.pdf',
            yerr_col='dec_ms_std')

# ------------------------------------------------------------
# 6. Theoretical search space comparison
# ------------------------------------------------------------
user_attr_size = 30
d = 5
policy_sizes = list(range(10, 1001, 10))

fuzzy_search = [binom(n, d) for n in policy_sizes]
silent_search = [binom(user_attr_size, d)] * len(policy_sizes)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(policy_sizes, fuzzy_search, label=r'Fuzzy IB-ME: $\binom{|\mathcal{P}|}{d}$',
        marker='o', markersize=4)
ax.plot(policy_sizes, silent_search, label=r'Silent-Share: $\binom{|\mathcal{A}|}{d}$',
        linestyle='--', marker='s', markersize=4)
ax.set_xlabel(r'Policy Size $|\mathcal{P}|$')
ax.set_ylabel(r'Search Space Size (log scale)')
ax.set_yscale('log')
ax.set_title(rf'Search Space Comparison ($d={d}$, $|\mathcal{{A}}|={user_attr_size}$)')
ax.legend()
ax.grid(True, which='both')
save_fig(fig, 'search_space_comparison.pdf')

print("All available figures saved to figures/")