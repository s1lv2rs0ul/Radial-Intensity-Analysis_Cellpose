#!/usr/bin/env python3
"""Batch PDF report for the radial intensity pipeline.

Structure:
  SECTION 1 — REPORT:          per colony: raw max projections + segmentation + profiles
  SECTION 2 — TROUBLESHOOTING: per colony: all QC/diagnostic panels
  COMBINED:                    all colonies fused into one weighted graph
  COMPARISON (optional):       overlay combined SMAD2 curves of several batches
                               (e.g. drug vs control)

Usage:
    python scripts/deep_dive_report.py BASE_DIR \
        [--pixel-sizes '{"pattern": um_per_px}'] \
        [--label "Control"] \
        [--compare "Drug=/path/to/other/batch" ...]
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpecFromSubplotSpec
from PIL import Image
from scipy.stats import mannwhitneyu, wilcoxon
from skimage.filters import threshold_otsu
from statsmodels.nonparametric.smoothers_lowess import lowess

A_EXP, LO, HI, CAP = 113.1, 45.2, 282.7, 400.0
MAX_R_UM = 250   # micropattern radius: beyond live only off-pattern objects
PRED_DENS = 0.6 / A_EXP * 1000
N_SEC = 12
CHANNEL = {'C0': ('ZO-1', '#00A087'), 'C1': ('DAPI', '#3C5488'),
           'C2': ('SMAD2', '#E64B35')}
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']


def load_png(p, w=1400):
    im = Image.open(p)
    im.thumbnail((w, w))
    return np.asarray(im)


def pixel_size_for(name, overrides, default=0.5):
    for pat, um in overrides.items():
        if pat in name:
            return float(um)
    return default


def divider_page(pdf, title, subtitle=''):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.55, title, ha='center', fontsize=34, fontweight='bold',
             color='#3C5488')
    if subtitle:
        fig.text(0.5, 0.44, subtitle, ha='center', fontsize=14, color='#555555')
    pdf.savefig(fig)
    plt.close(fig)


def colony_data(base_dir, img, overrides):
    """Load and derive everything one colony's pages need."""
    stem = img.replace('.tif', '')
    OUT = os.path.join(base_dir, 'radial_outputs')
    nuc = pd.read_csv(os.path.join(OUT, f'{stem}_nuclei.csv'))
    pix = pd.read_csv(os.path.join(OUT, f'{stem}_radial.csv'))
    if 'imputed' not in nuc.columns:
        nuc['imputed'] = False
    sc = pixel_size_for(img, overrides)
    kept = nuc[nuc['size_ok'] & (nuc['distance_um'] <= MAX_R_UM)].copy()
    synth = kept[kept['imputed']]
    cx, cy = kept['x_px'].mean(), kept['y_px'].mean()
    kept['x_um'] = (kept['x_px'] - cx) * sc
    kept['y_um'] = (kept['y_px'] - cy) * sc
    kept['bin'] = (kept['distance_um'] // 20) * 20 + 10
    ang = np.arctan2(kept['y_px'] - cy, kept['x_px'] - cx)
    kept['sector'] = np.clip(((ang + np.pi) / (2 * np.pi) * N_SEC).astype(int),
                             0, N_SEC - 1)
    return dict(stem=stem, nuc=nuc, pix=pix, sc=sc, kept=kept, synth=synth,
                rej=nuc[~nuc['size_ok']], cx=cx, cy=cy, ang=ang,
                r_col=kept['distance_um'].quantile(0.99))


def report_page(pdf, base_dir, d):
    """SECTION 1: raw images + segmentation + per-pixel profiles."""
    PLOTS = os.path.join(base_dir, 'plots')
    fig = plt.figure(figsize=(22, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.0, 1.15], wspace=0.15)
    fig.suptitle(f"{d['stem']}   ({len(d['kept'])} kept nuclei, "
                 f"{d['sc']:.3f} um/px)", fontsize=15, fontweight='bold', y=1.02)

    ax = fig.add_subplot(gs[0, 0])
    p = os.path.join(PLOTS, d['stem'], 'max_projection.png')
    if os.path.exists(p):
        ax.imshow(load_png(p, 1800))
    ax.axis('off')
    ax.set_title('RAW max projections (ZO-1 | DAPI | SMAD2)', fontsize=11)

    ax = fig.add_subplot(gs[0, 1])
    p = os.path.join(PLOTS, d['stem'], 'segmentation.png')
    if os.path.exists(p):
        ax.imshow(load_png(p))
    ax.axis('off')
    ax.set_title('DAPI + segmentation + center', fontsize=11)

    ax = fig.add_subplot(gs[0, 2])
    for ccol, (cn, cc) in CHANNEL.items():
        ax.plot(d['pix']['Radius'],
                pd.Series(d['pix'][ccol]).interpolate(limit_direction='both'),
                lw=1.8, color=cc, label=cn)
    ax.set_xlim(0, MAX_R_UM + 10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlabel('Distance from Center (um)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title('Radial profiles (per-pixel)', fontsize=11)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def trouble_page(pdf, base_dir, d, feats, cov):
    """SECTION 2: every QC/diagnostic panel."""
    kept, synth, pix = d['kept'], d['synth'], d['pix']
    rej, sc, cx, cy = d['rej'], d['sc'], d['cx'], d['cy']
    r_col, ang = d['r_col'], d['ang']
    nuc = d['nuc']
    fr = feats[feats['Image'] == d['stem'] + '.tif']
    f0 = fr.iloc[0]
    fs_ = fr[fr['Channel'] == 'SMAD2'].iloc[0]
    wrow = cov[cov['Image'] == d['stem'] + '.tif'].sort_values('bin')

    fig = plt.figure(figsize=(26, 21))
    gs = fig.add_gridspec(4, 4, hspace=0.42, wspace=0.3,
                          height_ratios=[1, 1, 1, 0.55])
    fig.suptitle(f"{d['stem']} — troubleshooting "
                 f"({len(kept)} kept / {len(rej)} rejected)",
                 fontsize=17, fontweight='bold', y=0.995)

    ax = fig.add_subplot(gs[0, 0])
    s1 = ax.scatter(kept['x_um'], -kept['y_um'], c=kept['C2'], s=7, cmap='inferno')
    ax.set_aspect('equal')
    plt.colorbar(s1, ax=ax, shrink=0.7)
    ax.set_title('A  SMAD2 per nucleus (spatial)')

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(kept['x_um'], -kept['y_um'], s=6, c='#00A087',
               label=f'kept {len(kept)}')
    if len(rej):
        rx = (rej['x_px'] - cx) * sc
        ry = (rej['y_px'] - cy) * sc
        big = rej['area_um2'] > CAP
        ax.scatter(rx[~big], -ry[~big], s=8, c='#E64B35', marker='x',
                   label=f'small/odd {int((~big).sum())}')
        ax.scatter(rx[big], -ry[big], s=22, c='purple', marker='s',
                   label=f'>cap {int(big.sum())}')
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    ax.set_title('B  kept vs rejected')

    ax = fig.add_subplot(gs[0, 2])
    kr = kept[~kept['imputed']]
    ax.scatter(kr['distance_um'], kr['C2'], s=4, alpha=0.25, color='gray')
    if len(synth):
        ax.scatter(synth['distance_um'], synth['C2'], s=14, facecolor='none',
                   edgecolor='#E64B35', lw=0.7, label=f'{len(synth)} imputed')
        ax.legend(fontsize=7)
    if len(kept) > 30:
        sm = lowess(kept['C2'], kept['distance_um'], frac=0.25, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color='#E64B35', lw=2.5)
    ax.set_xlim(0, MAX_R_UM + 10)
    ax.grid(alpha=0.3)
    ax.set_title('D  SMAD2 per-nucleus + LOWESS')

    ax = fig.add_subplot(gs[0, 3])
    ax.hist(nuc['area_um2'].clip(0, 600), bins=60, color='#4DBBD5')
    for v in (LO, A_EXP, HI, CAP):
        ax.axvline(v, ls='--', lw=1, color='k')
    ax.set_title('E  nucleus areas (um2) + filter bounds')

    ax = fig.add_subplot(gs[1, 0])
    r_edges = np.arange(0, 320, 20)
    cnts, _ = np.histogram(kept['distance_um'], bins=r_edges)
    annulus = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    ax.bar(r_edges[:-1] + 10, cnts / annulus * 1000, width=18, color='#8491B4')
    ax.axhline(PRED_DENS, color='k', ls='--', label='marble-in-jar max')
    ax.legend(fontsize=7)
    ax.set_title('F  density vs prediction (1D)')

    re2 = np.arange(0, r_col + 20, 20)
    te = np.linspace(-np.pi, np.pi, N_SEC + 1)
    T, R = np.meshgrid(te, re2)
    ax = fig.add_subplot(gs[1, 1], projection='polar')
    obs, _, _ = np.histogram2d(kept['distance_um'], ang, bins=[re2, te])
    pred = 0.6 * np.pi * (re2[1:] ** 2 - re2[:-1] ** 2) / N_SEC / A_EXP
    occ = np.clip(obs / pred[:, None], 0, 1.5)
    pc = ax.pcolormesh(T, R, occ, cmap='RdYlGn', vmin=0, vmax=1.5)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title('G  occupancy vs prediction (2D of F)', pad=10)
    plt.colorbar(pc, ax=ax, shrink=0.6)

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(wrow['bin'], wrow['coverage'].clip(0, 1), width=9, color='#00A087')
    ax.set_ylim(0, 1.05)
    ax.set_title('H  technical weight per bin')

    ax = fig.add_subplot(gs[1, 3])
    bm = kept.groupby('bin')[['eccentricity', 'cos2_radial']].mean()
    ax.plot(bm.index, bm['eccentricity'], color='#4DBBD5', lw=2,
            label='eccentricity')
    ax.plot(bm.index, bm['cos2_radial'], color='#7E6148', lw=2,
            label='cos2 radial')
    ax.axhline(0.5, color='gray', ls=':')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, MAX_R_UM + 10)
    ax.legend(fontsize=7)
    ax.set_title('I  nuclear shape vs radius')

    ax = fig.add_subplot(gs[2, 0])
    ax.hexbin(kept['C1'], kept['C2'], gridsize=40, cmap='inferno', bins='log')
    ta = threshold_otsu(kept['C1'].values)
    tb = threshold_otsu(kept['C2'].values)
    ax.axvline(ta, color='w', ls='--', lw=0.8)
    ax.axhline(tb, color='w', ls='--', lw=0.8)
    ax.set_title('J  DAPI vs SMAD2')

    ax = fig.add_subplot(gs[2, 1])
    if 'C2_over_DAPI' in kept.columns:
        rb = kept.groupby('bin')['C2_over_DAPI'].agg(['mean', 'sem'])
        ax.plot(rb.index, rb['mean'], color='#E64B35', lw=2)
        ax.fill_between(rb.index, rb['mean'] - rb['sem'], rb['mean'] + rb['sem'],
                        color='#E64B35', alpha=0.25)
    ax.set_xlim(0, MAX_R_UM + 10)
    ax.grid(alpha=0.3)
    ax.set_title('K  SMAD2/DAPI vs radius')

    ax = fig.add_subplot(gs[2, 2], projection='polar')
    smap = kept.groupby([pd.cut(kept['distance_um'], re2, labels=False),
                         'sector'])['C2'].mean()
    M = np.full((len(re2) - 1, N_SEC), np.nan)
    for (bi, si), v in smap.items():
        if not np.isnan(bi):
            M[int(bi), int(si)] = v
    pc = ax.pcolormesh(T, R, M, cmap='inferno')
    if len(synth):
        s_ang = np.arctan2(synth['y_px'] - cy, synth['x_px'] - cx)
        s_sec = np.clip(((s_ang + np.pi) / (2 * np.pi) * N_SEC).astype(int),
                        0, N_SEC - 1)
        s_bin = np.digitize(synth['distance_um'], re2) - 1
        for bi_s, si_s in set(zip(s_bin, s_sec)):
            if 0 <= bi_s < len(re2) - 1:
                ax.plot(-np.pi + (si_s + 0.5) * 2 * np.pi / N_SEC,
                        (re2[bi_s] + re2[bi_s + 1]) / 2, 'o',
                        mfc='none', mec='#00E5FF', mew=1.4, ms=9)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title('M  SMAD2 by direction\n(cyan rings = imputed)', pad=10)
    plt.colorbar(pc, ax=ax, shrink=0.6)

    gsn = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 3],
                                  width_ratios=[1.35, 1.0], wspace=0.02)
    ax = fig.add_subplot(gsn[0, 0])
    edge_means = {}
    for si in range(N_SEC):
        g = kept[kept['sector'] == si]
        if len(g) < 15:
            continue
        prof = g.groupby('bin')['C2'].mean()
        ax.plot(prof.index, prof.values, lw=1.2,
                color=plt.cm.hsv(si / N_SEC), alpha=0.85)
        edge_means[si] = g[g['distance_um'] >= 0.75 * r_col]['C2'].mean()
    allp = kept.groupby('bin')['C2'].mean()
    ax.plot(allp.index, allp.values, color='k', lw=2.5, label='all pies')
    ax.set_xlim(0, MAX_R_UM + 10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title('N  SMAD2 per pie (hue = direction)')

    axp = fig.add_subplot(gsn[0, 1], projection='polar')
    if edge_means:
        evs = pd.Series(edge_means)
        emin_, emax_ = evs.min(), evs.max()
        rng_e = (emax_ - emin_) or 1.0
        imp_secs = set()
        if len(synth):
            s_ang = np.arctan2(synth['y_px'] - cy, synth['x_px'] - cx)
            imp_secs = set(np.clip(((s_ang + np.pi) / (2 * np.pi) * N_SEC)
                                   .astype(int), 0, N_SEC - 1))
        for si, ev in edge_means.items():
            th_c = -np.pi + (si + 0.5) * 2 * np.pi / N_SEC
            is_imp = si in imp_secs
            axp.bar(th_c, 0.3 + 0.7 * (ev - emin_) / rng_e,
                    width=2 * np.pi / N_SEC * 0.95,
                    color=plt.cm.hsv(si / N_SEC), alpha=0.95,
                    edgecolor='#00B8CC' if is_imp else 'white',
                    linewidth=2.0 if is_imp else 0.6,
                    hatch='///' if is_imp else None)
            axp.text(th_c, 1.28, str(si + 1), ha='center', va='center',
                     fontsize=8, fontweight='bold', color='#333333')
        axp.plot(np.linspace(-np.pi, np.pi, 100), np.ones(100),
                 color='gray', lw=0.7, ls=':')
    axp.set_ylim(0, 1.45)
    axp.set_yticklabels([])
    axp.set_xticklabels([])
    axp.grid(False)
    axp.set_title('PIE KEY: hue = its line in N\nlength = edge SMAD2 (rank)',
                  fontsize=8, pad=6)

    ax = fig.add_subplot(gs[3, 0:4])
    ax.axis('off')
    em = pd.Series(edge_means)
    low = em[em < em.mean() - em.std()].index.tolist() if len(em) else []
    flags = []
    if f0['pct_rejected_size'] > 15:
        flags.append('high size-rejection')
    if f0['occupancy_vs_capacity'] < 0.3:
        flags.append('sparse vs capacity')
    if f0['dapi_bin_cv'] > 0.25:
        flags.append('DAPI non-uniform')
    if f0['angular_cv'] > 0.25:
        flags.append('angular asymmetry')
    if f0['n_directional_gap_bins'] >= 3:
        flags.append('directional gaps')
    txt = (f"kept/rej {int(f0['n_nuclei'])}/{int(f0['n_rejected_size'])} "
           f"(+{len(synth)} imputed)  ({f0['pct_rejected_size']:.0f}%)  "
           f"occupancy {f0['occupancy_vs_capacity']:.2f}  "
           f"areaCV {f0['area_cv']:.2f}  dapiCV {f0['dapi_bin_cv']:.2f}  "
           f"dirGaps {int(f0['n_directional_gap_bins'])}\n"
           f"SMAD2: peak {fs_['peak_radius_um']:.0f} um  half-max "
           f"{fs_['halfmax_radius_um']:.0f} um  center:edge "
           f"{fs_['center_edge_ratio']:.2f}\n"
           f"PIE RANKING (edge SMAD2): " +
           '  '.join(f'pie{s + 1}:{em[s]:.2f}' for s in sorted(em.index)) +
           f"\nLOW PIES: {[f'pie{p + 1}' for p in low] if low else 'none'}"
           f"\nFLAGS: {', '.join(flags) if flags else 'none - healthy'}")
    ax.text(0.01, 0.9, txt, va='top', family='monospace', fontsize=11.5)
    ax.set_title('L  QC card + pie ranking', loc='left')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def batch_curve(base_dir, channel='C2'):
    """Weighted + LOWESS-smoothed combined curve for one batch."""
    S = os.path.join(base_dir, 'summary')
    nuc = pd.read_csv(os.path.join(S, 'combined_per_nucleus.csv'))
    W = pd.read_csv(os.path.join(S, 'bin_weights.csv'), index_col=0)
    W.columns = W.columns.astype(float)
    if 'imputed' not in nuc.columns:
        nuc['imputed'] = False
    nuc = nuc[nuc['size_ok'] & (nuc['distance_um'] <= MAX_R_UM)].copy()
    nuc['bin'] = (nuc['distance_um'] // 10) * 10 + 5

    piv = nuc.groupby(['Image', 'bin'])[channel].mean().unstack('bin')
    w = W.reindex(index=piv.index, columns=piv.columns).fillna(0).values
    P = piv.values
    w = np.where(np.isnan(P), 0, w)

    def wm(rows):
        num = np.nansum(np.nan_to_num(P[rows]) * w[rows], axis=0)
        den = w[rows].sum(axis=0)
        return np.where(den > 0, num / den, np.nan)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(piv), size=(1000, len(piv)))
    boots = np.array([wm(r) for r in idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    bins = piv.columns.values
    mean = wm(np.arange(len(piv)))
    ok = np.isfinite(mean)
    sm = lowess(mean[ok], bins[ok], frac=0.18, return_sorted=True)
    lo_s = pd.Series(lo).interpolate(limit_direction='both').values
    hi_s = pd.Series(hi).interpolate(limit_direction='both').values

    # per-colony center/edge pairs on REAL nuclei only (stats never see imputed)
    pairs = []
    for img, g in nuc[~nuc['imputed']].groupby('Image'):
        q = g['distance_um'].quantile(0.99)
        pairs.append((g[g['distance_um'] <= 0.4 * q][channel].mean(),
                      g[g['distance_um'] >= 0.75 * q][channel].mean()))
    return bins, sm, lo_s, hi_s, len(piv), pairs


def apply_rise_style(ax, y0, ymax, head=0.30):
    """The approved SMAD2 presentation rules: curve start anchored at the origin
    corner, no y numbers, fine invisible y segments, x labeled every 50 um,
    range cropped to the data."""
    rng_y = (ymax - y0) or 1.0
    ax.set_ylim(y0 - 0.01 * rng_y, ymax + head * rng_y)
    ax.set_yticks([])
    for gy in np.linspace(y0, ymax, 11)[1:]:
        ax.axhline(gy, color='gray', lw=0.4, alpha=0.18, zorder=0)
    ax.set_xlim(0, MAX_R_UM + 10)
    ax.set_xticks(np.arange(0, MAX_R_UM + 10, 50))


def combined_page(pdf, base_dir, label, png_out=None):
    """All colonies fused into one weighted graph (all three channels, peak = 1)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    n_col = 0
    curves = {}
    stats_txt = ''
    for ccol, (cn, cc) in CHANNEL.items():
        bins, sm, lo_s, hi_s, n_col, pairs = batch_curve(base_dir, ccol)
        peak = np.nanmax(sm[:, 1]) or 1
        curves[ccol] = (bins, sm[:, 0], sm[:, 1] / peak, lo_s / peak, hi_s / peak)
        if ccol == 'C2' and len(pairs) >= 5:
            ce = pd.DataFrame(pairs, columns=['center', 'edge'])
            _, p = wilcoxon(ce['edge'], ce['center'], alternative='greater')
            fold = (ce['edge'] / ce['center']).median()
            stars = ('***' if p < 0.001 else '**' if p < 0.01
                     else '*' if p < 0.05 else 'n.s.')
            stats_txt = (f"{stars}   SMAD2 edge vs center: {fold:.2f}-fold, "
                         f"p = {p:.3f} (paired Wilcoxon, n = {len(ce)}, real only)")
    y0 = min(float(np.nanmin(v[2])) for v in curves.values())
    ymax = max(float(np.nanmax(v[4])) for v in curves.values())
    for ccol, (cn, cc) in CHANNEL.items():
        bins, sx, sy, lo_n, hi_n = curves[ccol]
        ax.fill_between(bins, lo_n, hi_n, color=cc, alpha=0.14, linewidth=0)
        ax.plot(sx, sy, color=cc, lw=2.8, label=cn)
    apply_rise_style(ax, y0, ymax)
    ax.axvspan(0, 95, color='gray', alpha=0.05)
    ax.axvspan(180, 240, color='#E64B35', alpha=0.05)
    rng_y = ymax - y0
    ax.text(47, ymax + 0.20 * rng_y, 'center', ha='center', fontsize=10,
            color='#555555')
    ax.text(210, ymax + 0.20 * rng_y, 'edge', ha='center', fontsize=10,
            color='#E64B35')
    if stats_txt:
        ybr = ymax + 0.10 * rng_y
        ax.plot([47, 47, 210, 210],
                [ybr - 0.02 * rng_y, ybr, ybr, ybr - 0.02 * rng_y],
                color='k', lw=1)
        ax.text(128, ybr + 0.02 * rng_y, stats_txt, ha='center', fontsize=10)
    ax.set_xlabel('Distance from Center (um)', fontsize=12)
    ax.set_ylabel('Normalized Intensity (a.u.)', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_title(f'COMBINED — {label}: all {n_col} colonies as one\n'
                 f'(QC-weighted, LOWESS-smoothed, band = colony bootstrap 95% CI)',
                 fontsize=13, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    if png_out:
        fig.savefig(png_out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def comparison_page(pdf, entries, png_out=None):
    """Overlay combined SMAD2 curves of several batches (drug vs control etc.)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    stats = []
    drawn = []
    for k, (label, path) in enumerate(entries):
        bins, sm, lo_s, hi_s, n_col, pairs = batch_curve(path, 'C2')
        peak = np.nanmax(sm[:, 1]) or 1
        drawn.append((bins, sm[:, 0], sm[:, 1] / peak, lo_s / peak,
                      hi_s / peak, NPG[k % len(NPG)], f'{label} (n = {n_col})'))
        ce = pd.DataFrame(pairs, columns=['center', 'edge'])
        stats.append((label, (ce['edge'] / ce['center']).values))
    y0 = min(float(np.nanmin(d[2])) for d in drawn)
    ymax = max(float(np.nanmax(d[4])) for d in drawn)
    for bins, sx, sy, lo_n, hi_n, col, lab in drawn:
        ax.fill_between(bins, lo_n, hi_n, color=col, alpha=0.14, linewidth=0)
        ax.plot(sx, sy, color=col, lw=2.8, label=lab)
    apply_rise_style(ax, y0, ymax, head=0.22)
    rng_y = ymax - y0
    if len(stats) == 2 and min(len(s_[1]) for s_ in stats) >= 3:
        _, p = mannwhitneyu(stats[0][1], stats[1][1], alternative='two-sided')
        ax.text(0.03, 0.97,
                f'edge/center fold: {stats[0][0]} median '
                f'{np.median(stats[0][1]):.2f} vs {stats[1][0]} median '
                f'{np.median(stats[1][1]):.2f}   Mann-Whitney p = {p:.3f}',
                transform=ax.transAxes, fontsize=10, va='top')
    ax.set_xlabel('Distance from Center (um)', fontsize=12)
    ax.set_ylabel('Normalized SMAD2 (peak = 1)', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_title('COMPARISON — combined SMAD2 per batch\n'
                 '(each curve: QC-weighted, smoothed, peak-normalized; '
                 'band = colony bootstrap 95% CI)',
                 fontsize=13, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    if png_out:
        fig.savefig(png_out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def build(base_dir, pixel_overrides, label=None, compare=()):
    S = os.path.join(base_dir, 'summary')
    feats = pd.read_csv(os.path.join(S, 'colony_features.csv'))
    cov = pd.read_csv(os.path.join(S, 'bin_coverage.csv'))
    images = sorted(feats['Image'].unique())
    label = label or os.path.basename(os.path.normpath(base_dir))
    pdf_path = os.path.join(S, 'per_colony_deep_dive.pdf')

    data = [colony_data(base_dir, img, pixel_overrides) for img in images]

    with PdfPages(pdf_path) as pdf:
        divider_page(pdf, 'SECTION 1 — REPORT',
                     f'{label}: images, segmentation, radial profiles '
                     f'({len(images)} colonies)')
        for d in data:
            report_page(pdf, base_dir, d)
            print(f"  report: {d['stem']}")

        divider_page(pdf, 'SECTION 2 — TROUBLESHOOTING',
                     'QC and diagnostic panels per colony')
        for d in data:
            trouble_page(pdf, base_dir, d, feats, cov)
            print(f"  troubleshooting: {d['stem']}")

        combined_page(pdf, base_dir, label,
                      png_out=os.path.join(S, 'combined_single_graph.png'))
        print('  combined graph')

        if compare:
            entries = [(label, base_dir)] + list(compare)
            comparison_page(pdf, entries,
                            png_out=os.path.join(S, 'comparison_graph.png'))
            print(f'  comparison: {" vs ".join(e[0] for e in entries)}')

    print(f'PDF: {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.1f} MB)')
    return pdf_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('base_dir')
    ap.add_argument('--pixel-sizes', default='{}',
                    help='JSON: {"filename-substring": um_per_px}')
    ap.add_argument('--label', default=None, help='name of this batch')
    ap.add_argument('--compare', action='append', default=[],
                    metavar='LABEL=PATH',
                    help='other finished batch to overlay in the comparison '
                         'page (repeatable), e.g. --compare "Drug=/path/b"')
    args = ap.parse_args()
    cmp_entries = []
    for c in args.compare:
        lbl, _, pth = c.partition('=')
        cmp_entries.append((lbl, pth))
    build(args.base_dir, json.loads(args.pixel_sizes),
          label=args.label, compare=cmp_entries)
