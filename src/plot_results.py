#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Automatically reads CSV results and generates multiple standalone PDF figures,
while also outputting processed CSV data.

Generated figures:
  1. fig_gokvs_dense_sparse_encode_compare.pdf
  2. fig_gokvs_dense_sparse_decode_compare.pdf
  3. fig_sparse_param_success_rate.pdf
  4. fig_sparse_param_encode_time.pdf
  5. fig_one_sided_decode_vs_n.pdf
  6. fig_one_sided_pairing_vs_A.pdf
  7. fig_reconstruction_worst_case.pdf
  8. fig_bilateral_heatmap.pdf
  9. fig_one_sided_total_decap_vs_A.pdf
  10. fig_ct_size_vs_policy.pdf
  11. fig_throughput_encap.pdf
  12. fig_throughput_decap.pdf
  13. fig_bilateral_comm_heatmap.pdf
  14. fig_compare_enc_policy.pdf          (new)
  15. fig_compare_dec_attr.pdf           (new)
  16. fig_compare_ct_size_policy.pdf     (new)
  17. fig_compare_throughput_enc.pdf     (new)
  18. fig_compare_throughput_dec.pdf     (new)

All exported figures include standard deviation information (on applicable columns).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
OUTPUT_DIR = 'figures'
DATA_DIR = 'data_csv'

# Unified LaTeX symbols
LABEL_P = r'Policy size $|\mathcal{P}|$'
LABEL_A = r'Attribute count $|\mathcal{A}|$'


def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {path}")


def save_plot_data(df, filename):
    """Save DataFrame as CSV (for Origin)."""
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved plot data: {path}")


def agg_mean_std(df, group_cols, value_col):
    """
    Group by group_cols and return a DataFrame with mean, std, count.
    Column names use value_col as prefix, e.g., encap_wall_ms_mean, encap_wall_ms_std.
    """
    grouped = df.groupby(group_cols)[value_col].agg(['mean', 'std', 'count']).reset_index()
    grouped.columns = [col if col in group_cols else f"{value_col}_{col}" for col in grouped.columns]
    return grouped


def plot_gokvs_dense_sparse_compare():
    # Read data
    sparse_csv = os.path.join(RESULTS_DIR, 'sparse_gokvs.csv')
    dense_csv = os.path.join(RESULTS_DIR, 'gokvs_group.csv')
    scalar_csv = os.path.join(RESULTS_DIR, 'gokvs_scalar.csv')

    if not os.path.exists(sparse_csv):
        print("sparse_gokvs.csv not found, skip dense/sparse comparison")
        return

    sparse_df = pd.read_csv(sparse_csv)
    sparse_enc = sparse_df.groupby('n')['encode_ms'].agg(['mean', 'std']).reset_index()
    sparse_dec = sparse_df.groupby('n')['decode_ms'].agg(['mean', 'std']).reset_index()

    # Read dense data (if exists)
    dense_enc = dense_dec = None
    if os.path.exists(dense_csv):
        dense_df = pd.read_csv(dense_csv)
        dense_enc = dense_df.groupby('n')['encode_wall_ms'].agg(['mean', 'std']).reset_index()
        dense_dec = dense_df.groupby('n')['decode_wall_ms'].agg(['mean', 'std']).reset_index()

    # Read scalar data (if exists)
    scalar_enc = scalar_dec = None
    if os.path.exists(scalar_csv):
        scalar_df = pd.read_csv(scalar_csv)
        scalar_enc = scalar_df.groupby('n')['encode_wall_ms'].agg(['mean', 'std']).reset_index()
        scalar_dec = scalar_df.groupby('n')['decode_wall_ms'].agg(['mean', 'std']).reset_index()

    # ---------- Encoding time comparison (log scale) ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    if scalar_enc is not None:
        ax.errorbar(scalar_enc['n'], scalar_enc['mean'], yerr=scalar_enc['std'],
                    marker='^', label='Scalar OKVS', capsize=3)
    if dense_enc is not None:
        ax.errorbar(dense_enc['n'], dense_enc['mean'], yerr=dense_enc['std'],
                    marker='o', label='Dense GOKVS', capsize=3)
    ax.errorbar(sparse_enc['n'], sparse_enc['mean'], yerr=sparse_enc['std'],
                marker='s', label='Sparse GOKVS', capsize=3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Policy size $n$ (log scale)')
    ax.set_ylabel('Encoding time (ms, log scale)')
    ax.set_title('Encoding time comparison')
    ax.legend()
    ax.grid(True, which='both')
    save_fig(fig, 'fig_gokvs_dense_sparse_encode_compare.pdf')

    # Save data (with std)
    data_enc = sparse_enc.rename(columns={'mean': 'sparse_encode_ms', 'std': 'sparse_encode_std'})
    if dense_enc is not None:
        dense_enc_ren = dense_enc.rename(columns={'mean': 'dense_encode_ms', 'std': 'dense_encode_std'})
        data_enc = data_enc.merge(dense_enc_ren, on='n', how='outer')
    if scalar_enc is not None:
        scalar_enc_ren = scalar_enc.rename(columns={'mean': 'scalar_encode_ms', 'std': 'scalar_encode_std'})
        data_enc = data_enc.merge(scalar_enc_ren, on='n', how='outer')
    save_plot_data(data_enc, 'gokvs_encode_comparison.csv')

    # ---------- Decoding time comparison ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    if scalar_dec is not None:
        ax.errorbar(scalar_dec['n'], scalar_dec['mean'], yerr=scalar_dec['std'],
                    marker='^', label='Scalar OKVS', capsize=3)
    if dense_dec is not None:
        ax.errorbar(dense_dec['n'], dense_dec['mean'], yerr=dense_dec['std'],
                    marker='o', label='Dense GOKVS', capsize=3)
    ax.errorbar(sparse_dec['n'], sparse_dec['mean'], yerr=sparse_dec['std'],
                marker='s', label='Sparse GOKVS', capsize=3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Policy size $n$ (log scale)')
    ax.set_ylabel('Decode time per attribute (ms, log scale)')
    ax.set_title('Decoding time comparison')
    ax.legend()
    ax.grid(True, which='both')
    save_fig(fig, 'fig_gokvs_dense_sparse_decode_compare.pdf')

    # Export data
    data_dec = sparse_dec.rename(columns={'mean': 'sparse_decode_ms', 'std': 'sparse_decode_std'})
    if dense_dec is not None:
        dense_dec_ren = dense_dec.rename(columns={'mean': 'dense_decode_ms', 'std': 'dense_decode_std'})
        data_dec = data_dec.merge(dense_dec_ren, on='n', how='outer')
    if scalar_dec is not None:
        scalar_dec_ren = scalar_dec.rename(columns={'mean': 'scalar_decode_ms', 'std': 'scalar_decode_std'})
        data_dec = data_dec.merge(scalar_dec_ren, on='n', how='outer')
    save_plot_data(data_dec, 'gokvs_decode_comparison.csv')


def plot_sparse_param_sensitivity():
    param_csv = os.path.join(RESULTS_DIR, 'sparse_gokvs_param.csv')
    if not os.path.exists(param_csv):
        print("sparse_gokvs_param.csv not found, skip parameter sensitivity")
        return

    df = pd.read_csv(param_csv)

    n_values = sorted(df['n'].unique())
    rho_values = sorted(df['rho'].unique())
    m_ratios = sorted(df['m_ratio'].unique())

    # ---------- Encoding time vs m/n, using only n=300 ----------
    n_chosen = 300
    if n_chosen not in n_values:
        n_chosen = n_values[len(n_values)//2]

    sub = df[df['n'] == n_chosen]
    fig, ax = plt.subplots(figsize=(8, 5))

    # Store valid data for all rho for export
    all_valid_frames = []

    for rho in rho_values:
        subset = sub[sub['rho'] == rho]
        valid = subset[~subset['encode_ms'].isna()]
        if valid.empty:
            continue
        all_valid_frames.append(valid)

        mean_enc = valid.groupby('m_ratio')['encode_ms'].agg(['mean', 'std']).reset_index()
        ax.errorbar(mean_enc['m_ratio'], mean_enc['mean'], yerr=mean_enc['std'],
                    marker='o', label=f'$\\rho={rho}$', capsize=3)

    ax.set_xlabel('$m/n$ ratio')
    ax.set_ylabel('Encoding time (ms)')
    ax.set_title(f'Encoding time vs $m/n$ ($n={n_chosen}$)')
    ax.grid(True)
    ax.legend()
    save_fig(fig, 'fig_sparse_param_encode_time.pdf')

    # Export data: merge valid data for all rho
    if all_valid_frames:
        data_enc = pd.concat(all_valid_frames)
        data_enc = data_enc.groupby(['rho', 'm_ratio'])['encode_ms'].agg(['mean', 'std']).reset_index()
        data_enc.columns = ['rho', 'm_ratio', 'encode_ms_mean', 'encode_ms_std']
        save_plot_data(data_enc, f'sparse_param_encode_time_n{n_chosen}.csv')
    else:
        print("No valid encoding data for the chosen n.")

    # ---------- Success rate heatmap ----------
    fig, axes = plt.subplots(len(n_values), 1, figsize=(8, 4 * len(n_values)), sharex=True)
    if len(n_values) == 1:
        axes = [axes]

    cmap = plt.cm.viridis
    for ax, n_val in zip(axes, n_values):
        sub_n = df[df['n'] == n_val]
        pivot = sub_n.pivot_table(index='rho', columns='m_ratio', values='success', aggfunc='mean')
        pivot = pivot.reindex(index=rho_values, columns=m_ratios)
        im = ax.imshow(pivot.values, cmap=cmap, origin='lower', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(m_ratios)))
        ax.set_xticklabels([f'{x:.2f}' for x in m_ratios])
        ax.set_yticks(range(len(rho_values)))
        ax.set_yticklabels(rho_values)
        ax.set_ylabel('$\\rho$')
        ax.set_title(f'$n={n_val}$')
        for i in range(len(rho_values)):
            for j in range(len(m_ratios)):
                val = pivot.values[i, j]
                text = f'{val:.1f}' if not np.isnan(val) else 'NA'
                ax.text(j, i, text, ha='center', va='center',
                        color='white' if val < 0.7 else 'black', fontsize=8)

    ax.set_xlabel('$m/n$ ratio')
    fig.suptitle('Peeling success rate', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.colorbar(im, ax=axes, label='Success rate')
    save_fig(fig, 'fig_sparse_param_success_rate.pdf')

    data_success = df.groupby(['n', 'rho', 'm_ratio'])['success'].agg(['mean', 'std']).reset_index()
    data_success.columns = ['n', 'rho', 'm_ratio', 'success_mean', 'success_std']
    save_plot_data(data_success, 'sparse_param_success_rate.csv')


def plot_one_sided():
    csv_file = os.path.join(RESULTS_DIR, 'one_sided.csv')
    if not os.path.exists(csv_file):
        print("one_sided.csv not found, skip one-sided")
        return

    df = pd.read_csv(csv_file)

    # ---------- Figure 1: Decode wall time vs policy size ----------
    target_A = sorted(df['A'].unique())
    # Only plot a subset of A to avoid too many curves
    A_plot = [A for A in target_A if A in [2, 4, 10, 20, 40]] or target_A[:5]

    sub_df = df[df['A'].isin(A_plot)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for A in A_plot:
        sub = sub_df[sub_df['A'] == A]
        mean = sub.groupby('n')['decode_wall_ms'].agg(['mean', 'std']).reset_index()
        ax.errorbar(mean['n'], mean['mean'], yerr=mean['std'], marker='o',
                    label=f'$|\\mathcal{{A}}|={A}$', capsize=3)
    ax.set_xlabel(LABEL_P)
    ax.set_ylabel('Decode wall time (ms)')
    ax.set_title('Decode wall time vs policy size')
    ax.legend()
    ax.grid(True)
    save_fig(fig, 'fig_one_sided_decode_vs_n.pdf')

    data_decode = sub_df.groupby(['A', 'n'])['decode_wall_ms'].agg(['mean', 'std']).reset_index()
    data_decode.columns = ['A', 'n', 'decode_wall_ms_mean', 'decode_wall_ms_std']
    save_plot_data(data_decode, 'one_sided_decode_vs_n.csv')

    # ---------- Figure 2: Pairing time vs attribute count ----------
    avg_pairing = df.groupby('A')['pairing_ms'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(avg_pairing['A'], avg_pairing['mean'], yerr=avg_pairing['std'],
                marker='o', color='tab:blue', capsize=3)
    ax.set_xlabel(LABEL_A)
    ax.set_ylabel('Pairing time (ms)')
    ax.set_title('Pairing time vs attribute count')
    ax.grid(True)
    save_fig(fig, 'fig_one_sided_pairing_vs_A.pdf')
    avg_pairing.columns = ['A', 'pairing_ms_mean', 'pairing_ms_std']
    save_plot_data(avg_pairing, 'one_sided_pairing_vs_A.csv')

    # ---------- Figure 3: Total decapsulation time vs A ----------
    target_n = [n for n in sorted(df['n'].unique()) if n in [100, 200, 400, 800, 1600]]
    if not target_n:
        target_n = sorted(df['n'].unique())

    sub_df_n = df[df['n'].isin(target_n)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in target_n:
        sub = sub_df_n[sub_df_n['n'] == n]
        mean_total = sub.groupby('A')['total_decap_wall_ms'].agg(['mean', 'std']).reset_index()
        ax.errorbar(mean_total['A'], mean_total['mean'], yerr=mean_total['std'],
                    marker='o', label=f'$|\\mathcal{{P}}|={n}$', capsize=3)
    ax.set_xlabel(LABEL_A)
    ax.set_ylabel('Total decapsulation time (ms)')
    ax.set_title('Total decapsulation time vs attribute count')
    ax.legend()
    ax.grid(True)
    save_fig(fig, 'fig_one_sided_total_decap_vs_A.pdf')
    data_total = sub_df_n.groupby(['n', 'A'])['total_decap_wall_ms'].agg(['mean', 'std']).reset_index()
    data_total.columns = ['n', 'A', 'total_decap_wall_ms_mean', 'total_decap_wall_ms_std']
    save_plot_data(data_total, 'one_sided_total_decap_vs_A.csv')

    # ---------- Figure: Ciphertext size vs policy size (MB) ----------
    BYTES_PER_MB = 1024 * 1024
    avg_ct_size = df.groupby('n')['ct_size_bytes'].agg(['mean', 'std']).reset_index()
    avg_ct_size['ct_size_mb_mean'] = avg_ct_size['mean'] / BYTES_PER_MB
    avg_ct_size['ct_size_mb_std'] = avg_ct_size['std'] / BYTES_PER_MB

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(avg_ct_size['n'], avg_ct_size['ct_size_mb_mean'],
                yerr=avg_ct_size['ct_size_mb_std'],
                marker='s', color='tab:red', capsize=3)
    ax.set_xlabel(LABEL_P)
    ax.set_ylabel('Ciphertext size (MB)')
    ax.set_title('Ciphertext size vs policy size')
    ax.grid(True)
    save_fig(fig, 'fig_ct_size_vs_policy.pdf')

    # Export data (preserve both bytes and MB)
    save_plot_data(avg_ct_size, 'ct_size_vs_policy.csv')

    # ---------- Figure: Encapsulation throughput ----------
    avg_enc_ops = df.groupby('n')['encap_throughput_ops_per_sec'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(avg_enc_ops['n'], avg_enc_ops['mean'], yerr=avg_enc_ops['std'],
                marker='o', color='tab:green', capsize=3)
    ax.set_xlabel(LABEL_P)
    ax.set_ylabel('Encapsulation throughput (ops/s)')
    ax.set_title('Encapsulation throughput vs policy size')
    ax.grid(True)
    save_fig(fig, 'fig_throughput_encap.pdf')
    avg_enc_ops.columns = ['n', 'encap_throughput_ops_per_sec_mean', 'encap_throughput_ops_per_sec_std']
    save_plot_data(avg_enc_ops, 'throughput_encap.csv')

    # ---------- Figure: Decapsulation throughput ----------
    avg_dec_ops = df.groupby('A')['decap_throughput_ops_per_sec'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(avg_dec_ops['A'], avg_dec_ops['mean'], yerr=avg_dec_ops['std'],
                marker='o', color='tab:orange', capsize=3)
    ax.set_xlabel(LABEL_A)
    ax.set_ylabel('Decapsulation throughput (ops/s)')
    ax.set_title('Decapsulation throughput vs attribute count')
    ax.grid(True)
    save_fig(fig, 'fig_throughput_decap.pdf')
    avg_dec_ops.columns = ['A', 'decap_throughput_ops_per_sec_mean', 'decap_throughput_ops_per_sec_std']
    save_plot_data(avg_dec_ops, 'throughput_decap.csv')


def plot_reconstruction():
    csv_file = os.path.join(RESULTS_DIR, 'reconstruction.csv')
    if not os.path.exists(csv_file):
        print("reconstruction.csv not found")
        return

    df = pd.read_csv(csv_file)

    worst_df = df[df['scenario'] == 'worst']
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in sorted(worst_df['d'].unique()):
        sub = worst_df[worst_df['d'] == d]
        sub = sub[sub['mode'] == 'precompute']
        mean = sub.groupby('A')['wall_ms'].agg(['mean', 'std']).reset_index()
        ax.errorbar(mean['A'], mean['mean'], yerr=mean['std'],
                    marker='o', label=f'd={d}', capsize=3)
    ax.set_xlabel(LABEL_A)
    ax.set_ylabel('Worst-case reconstruction time (ms)')
    ax.set_title('Threshold reconstruction worst case (precompute)')
    ax.legend()
    ax.grid(True)
    save_fig(fig, 'fig_reconstruction_worst_case.pdf')

    data_worst = worst_df[worst_df['mode'] == 'precompute'].groupby(['d', 'A'])['wall_ms'].agg(['mean', 'std']).reset_index()
    data_worst.columns = ['d', 'A', 'wall_ms_mean', 'wall_ms_std']
    save_plot_data(data_worst, 'reconstruction_worst_case.csv')


def plot_bilateral():
    csv_file = os.path.join(RESULTS_DIR, 'bilateral.csv')
    if not os.path.exists(csv_file):
        print("bilateral.csv not found")
        return

    df = pd.read_csv(csv_file)

    # Total time heatmap
    pivot = df.pivot_table(index='nO', columns='nR', values='total_wall_ms', aggfunc='mean')
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap='turbo', origin='lower', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Requester policy size $|\\mathcal{P}_R|$')
    ax.set_ylabel('Owner policy size $|\\mathcal{P}_O|$')
    ax.set_title('Total bilateral time (ms)')
    plt.colorbar(im, ax=ax, label='Total time (ms)')
    save_fig(fig, 'fig_bilateral_heatmap.pdf')

    pivot_reset = pivot.reset_index().melt(id_vars='nO', var_name='nR', value_name='total_wall_ms')
    save_plot_data(pivot_reset, 'bilateral_heatmap.csv')

    # Communication heatmap (MB)
    BYTES_PER_MB = 1024 * 1024
    pivot_comm = df.pivot_table(index='nO', columns='nR', values='total_comm_bytes', aggfunc='mean')
    pivot_comm = pivot_comm / BYTES_PER_MB   # Convert to MB
    pivot_comm = pivot_comm.reindex(sorted(pivot_comm.index), axis=0)
    pivot_comm = pivot_comm.reindex(sorted(pivot_comm.columns), axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot_comm.values, cmap='turbo', origin='lower', aspect='auto')
    ax.set_xticks(range(len(pivot_comm.columns)))
    ax.set_xticklabels(pivot_comm.columns, rotation=45)
    ax.set_yticks(range(len(pivot_comm.index)))
    ax.set_yticklabels(pivot_comm.index)
    ax.set_xlabel('Requester policy size $|\\mathcal{P}_R|$')
    ax.set_ylabel('Owner policy size $|\\mathcal{P}_O|$')
    ax.set_title('Total communication (MB)')
    plt.colorbar(im, ax=ax, label='Communication (MB)')
    save_fig(fig, 'fig_bilateral_comm_heatmap.pdf')

    # Export data: keep original bytes and add MB column
    pivot_comm_reset = pivot_comm.reset_index().melt(id_vars='nO', var_name='nR', value_name='total_comm_mb')
    # If bytes are needed, they can be taken from df; here we simply export MB
    save_plot_data(pivot_comm_reset, 'bilateral_comm_heatmap.csv')



def main():
    ensure_output_dirs()
    plot_gokvs_dense_sparse_compare()
    plot_sparse_param_sensitivity()
    plot_one_sided()
    plot_reconstruction()
    plot_bilateral()
    print("All figures and data CSVs generated.")


if __name__ == '__main__':
    main()