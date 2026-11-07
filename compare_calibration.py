"""
compare_calibration.py
======================
Loads chain_fit_full_b*.csv and chain_fit_nopos_b*.csv (or matching pkl files),
then produces plots/calibration_comparison.png demonstrating whether joint
position+dEdx+Ab fitting improves Ab recovery vs dEdx+Ab-only fitting.

Usage:
    python3 compare_calibration.py [--batches 6]
"""
import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

AB_TRUE   = 0.8348813503927325
DEDX_NOM  = 1.887   # MIP mean dEdx MeV/cm

def load_csv(fit_mode, n_batches):
    pattern = f'chain_fit_summary_{fit_mode}_b{n_batches}.csv'
    if not os.path.exists(pattern):
        raise FileNotFoundError(f'Missing: {pattern}')
    df = pd.read_csv(pattern)
    # ensure numeric columns
    for col in ['Ab', 'dEdx_S2', 'Ri', 'Rs3', 'L']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def track_label(row):
    return f"b{int(row['b'])}e{int(row['ev'])}t{int(row['tr'])}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, default=6)
    ap.add_argument('--pos-mode', default='joint',
                    help='fit_mode for the position-fitted run: joint or full')
    args = ap.parse_args()
    N = args.batches

    full  = load_csv(args.pos_mode, N)
    nopos = load_csv('no_pos', N)

    # Merge on (b, ev, tr) — inner join keeps only tracks present in both runs
    merged = pd.merge(
        full.rename(columns={'Ab':'Ab_full','dEdx_S2':'dEdx_full',
                              'Rs3':'Rs3_full','Ri':'Ri_full','L_S3':'LS3_full'}),
        nopos.rename(columns={'Ab':'Ab_nopos','dEdx_S2':'dEdx_nopos',
                               'Rs3':'Rs3_nopos','Ri':'Ri_nopos','L_S3':'LS3_nopos'}),
        on=['b','ev','tr'], suffixes=('','_np')
    )
    merged = merged.dropna(subset=['Ab_full','Ab_nopos'])
    if merged.empty:
        print('No matching tracks found between the two CSV files.')
        return

    # Derived quantities
    merged['delta_resid']    = merged['Ri_full'] - merged['Rs3_full']     # improvement in cm (pos. = better)
    merged['Ab_bias_full']   = merged['Ab_full']  - AB_TRUE
    merged['Ab_bias_nopos']  = merged['Ab_nopos'] - AB_TRUE
    merged['Ab_bias_delta']  = np.abs(merged['Ab_bias_nopos']) - np.abs(merged['Ab_bias_full'])  # pos = full is better

    # Sort by delta_resid descending (tracks where position fitting helped most first)
    merged = merged.sort_values('delta_resid', ascending=False).reset_index(drop=True)
    labels = [track_label(r) for _, r in merged.iterrows()]
    x = np.arange(len(merged))

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Ab fitted per track ──────────────────────────────────────────
    ax1.axhline(AB_TRUE, color='black', lw=1.5, ls='--', label=f'Ab_true = {AB_TRUE:.4f}')
    ax1.scatter(x, merged['Ab_full'],  s=60, color='steelblue',  zorder=3, label='full (S1+S2+S3)')
    ax1.scatter(x, merged['Ab_nopos'], s=60, color='darkorange', zorder=3, label='no_pos (S2+S3)')
    for i in x:
        ax1.plot([i, i], [merged['Ab_full'].iloc[i], merged['Ab_nopos'].iloc[i]],
                 color='gray', lw=0.8, alpha=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Ab fitted'); ax1.set_title('Ab fitted per track\n(sorted by position improvement)')
    ax1.legend(fontsize=8)

    # ── Panel 2: Ab bias reduction vs position residual improvement ───────────
    track_len = merged['L'] if 'L' in merged.columns else None
    sc_kwargs = dict(c=track_len, cmap='viridis') if track_len is not None else dict(color='steelblue')
    sc = ax2.scatter(merged['delta_resid'], merged['Ab_bias_delta'],
                     s=80, edgecolors='k', linewidths=0.5, zorder=3, **sc_kwargs)
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.axvline(0, color='black', lw=1, ls='--')
    for i, row in merged.iterrows():
        ax2.annotate(labels[i], (row['delta_resid'], row['Ab_bias_delta']),
                     fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax2.set_xlabel('Δresid = Ri − Rs3 [cm]  (positive = position improved)')
    ax2.set_ylabel('|Ab_bias_nopos| − |Ab_bias_full|\n(positive = full mode is better)')
    ax2.set_title('Ab bias reduction vs position improvement')
    if track_len is not None:
        plt.colorbar(sc, ax=ax2, label='Track length [cm]')

    # ── Panel 3: dEdx per track ───────────────────────────────────────────────
    ax3.axhline(DEDX_NOM, color='black', lw=1.5, ls='--', label=f'Nominal {DEDX_NOM} MeV/cm')
    ax3.scatter(x, merged['dEdx_full'],  s=60, color='steelblue',  zorder=3, label='full')
    ax3.scatter(x, merged['dEdx_nopos'], s=60, color='darkorange', zorder=3, label='no_pos')
    for i in x:
        ax3.plot([i, i], [merged['dEdx_full'].iloc[i], merged['dEdx_nopos'].iloc[i]],
                 color='gray', lw=0.8, alpha=0.5)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Mean dEdx S2 [MeV/cm]'); ax3.set_title('dEdx fitted per track')
    ax3.legend(fontsize=8)

    # ── Panel 4: Summary statistics ───────────────────────────────────────────
    def stats(vals):
        v = vals.dropna()
        return np.mean(np.abs(v)), np.std(np.abs(v))

    categories = [('All tracks', merged),
                  ('Δresid > 0.05 cm', merged[merged['delta_resid'] > 0.05])]
    bar_x = np.arange(len(categories))
    w = 0.3

    for j, (label_cat, sub) in enumerate(categories):
        if sub.empty:
            continue
        m_full,  s_full  = stats(sub['Ab_bias_full'])
        m_nopos, s_nopos = stats(sub['Ab_bias_nopos'])
        ax4.bar(j - w/2, m_full,  w, yerr=s_full,  color='steelblue',  label='full' if j==0 else '', capsize=4)
        ax4.bar(j + w/2, m_nopos, w, yerr=s_nopos, color='darkorange', label='no_pos' if j==0 else '', capsize=4)

    ax4.set_xticks(bar_x)
    ax4.set_xticklabels([c[0] for c in categories])
    ax4.set_ylabel('Mean |Ab − Ab_true| ± std')
    ax4.set_title('Ab bias summary')
    ax4.legend(fontsize=8)

    fig.suptitle(f'Calibration comparison: {args.pos_mode} vs no-pos fitting  ({len(merged)} tracks)',
                 fontsize=13, fontweight='bold')

    os.makedirs('plots', exist_ok=True)
    out = f'plots/calibration_comparison_b{N}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    # ── Text summary ──────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'{"Track":<14} {"Ab_full":>9} {"Ab_nopos":>9} {"Δresid":>8} {"bias_Δ":>8}')
    print('-'*60)
    for i, row in merged.iterrows():
        print(f'{labels[i]:<14} {row["Ab_full"]:9.5f} {row["Ab_nopos"]:9.5f} '
              f'{row["delta_resid"]:8.3f} {row["Ab_bias_delta"]:+8.4f}')
    print('-'*60)
    m_f, s_f = stats(merged['Ab_bias_full'])
    m_n, s_n = stats(merged['Ab_bias_nopos'])
    print(f'Mean |Ab bias| — full: {m_f:.5f} ± {s_f:.5f}   no_pos: {m_n:.5f} ± {s_n:.5f}')
    print(f'Ab_true = {AB_TRUE:.6f}')

if __name__ == '__main__':
    main()
