#!/usr/bin/env python3
"""Per-colony deep-dive PDF report for the radial intensity pipeline.

Usage:
    python scripts/deep_dive_report.py BASE_DIR [--pixel-sizes '{"pattern": um_per_px}']

Reads a finished batch run (radial_outputs/, plots/, summary/) and writes
BASE_DIR/summary/per_colony_deep_dive.pdf — one page per colony with:
real images + segmentation, spatial maps, profiles, size QC, density vs the
marble-in-jar prediction, technical weights, morphometrics, co-expression,
polar occupancy, SMAD2/DAPI ratio, directional pie intensity map, per-pie
profiles, and a QC card with pie ranking.
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
from PIL import Image
from skimage.filters import threshold_otsu
from statsmodels.nonparametric.smoothers_lowess import lowess

A_EXP, LO, HI, CAP = 113.1, 45.2, 282.7, 400.0
PRED_DENS = 0.6 / A_EXP * 1000
N_SEC = 12
CHANNEL = {'C0': ('ZO-1', '#00A087'), 'C1': ('DAPI', '#3C5488'),
           'C2': ('SMAD2', '#E64B35')}


def load_png(p, w=1400):
    im = Image.open(p)
    im.thumbnail((w, w))
    return np.asarray(im)


def pixel_size_for(name, overrides, default=0.5):
    for pat, um in overrides.items():
        if pat in name:
            return float(um)
    return default


def build(base_dir, pixel_overrides):
    S = os.path.join(base_dir, 'summary')
    OUT = os.path.join(base_dir, 'radial_outputs')
    PLOTS = os.path.join(base_dir, 'plots')
    feats = pd.read_csv(os.path.join(S, 'colony_features.csv'))
    cov = pd.read_csv(os.path.join(S, 'bin_coverage.csv'))
    images = sorted(feats['Image'].unique())
    pdf_path = os.path.join(S, 'per_colony_deep_dive.pdf')

    with PdfPages(pdf_path) as pdf:
        for img in images:
            stem = img.replace('.tif', '')
            nuc = pd.read_csv(os.path.join(OUT, f'{stem}_nuclei.csv'))
            pix = pd.read_csv(os.path.join(OUT, f'{stem}_radial.csv'))
            sc = pixel_size_for(img, pixel_overrides)
            kept = nuc[nuc['size_ok']].copy()
            rej = nuc[~nuc['size_ok']]
            cx, cy = kept['x_px'].mean(), kept['y_px'].mean()
            kept['x_um'] = (kept['x_px'] - cx) * sc
            kept['y_um'] = (kept['y_px'] - cy) * sc
            kept['bin'] = (kept['distance_um'] // 20) * 20 + 10
            ang = np.arctan2(kept['y_px'] - cy, kept['x_px'] - cx)
            kept['sector'] = np.clip(((ang + np.pi) / (2 * np.pi) * N_SEC).astype(int),
                                     0, N_SEC - 1)
            fr = feats[feats['Image'] == img]
            f0 = fr.iloc[0]
            fs_ = fr[fr['Channel'] == 'SMAD2'].iloc[0]
            wrow = cov[cov['Image'] == img].sort_values('bin')
            short = stem.split('_')[0]
            r_col = kept['distance_um'].quantile(0.99)

            fig = plt.figure(figsize=(26, 24))
            gs = fig.add_gridspec(5, 4, hspace=0.45, wspace=0.3,
                                  height_ratios=[1.15, 1, 1, 1, 0.7])
            fig.suptitle(f'{stem}  ({len(kept)} kept / {len(rej)} rejected, '
                         f'{sc:.3f} um/px)', fontsize=18, fontweight='bold', y=0.995)

            ax = fig.add_subplot(gs[0, 0:2])
            p = os.path.join(PLOTS, stem, 'max_projection.png')
            if os.path.exists(p):
                ax.imshow(load_png(p, 1800))
            ax.axis('off')
            ax.set_title('RAW max projections', fontsize=12)
            ax = fig.add_subplot(gs[0, 2])
            p = os.path.join(PLOTS, stem, 'segmentation.png')
            if os.path.exists(p):
                ax.imshow(load_png(p))
            ax.axis('off')
            ax.set_title('DAPI + segmentation + center', fontsize=12)
            ax = fig.add_subplot(gs[0, 3])
            s1 = ax.scatter(kept['x_um'], -kept['y_um'], c=kept['C2'], s=7, cmap='inferno')
            ax.set_aspect('equal')
            plt.colorbar(s1, ax=ax, shrink=0.7)
            ax.set_title('A  SMAD2 per nucleus (spatial)')

            ax = fig.add_subplot(gs[1, 0])
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
            ax = fig.add_subplot(gs[1, 1])
            for ccol, (cn, cc) in CHANNEL.items():
                ax.plot(pix['Radius'],
                        pd.Series(pix[ccol]).interpolate(limit_direction='both'),
                        lw=1.6, color=cc, label=cn)
            ax.set_xlim(0, 300)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.set_title('C  per-pixel profiles')
            ax = fig.add_subplot(gs[1, 2])
            ax.scatter(kept['distance_um'], kept['C2'], s=4, alpha=0.25, color='gray')
            if len(kept) > 30:
                sm = lowess(kept['C2'], kept['distance_um'], frac=0.25,
                            return_sorted=True)
                ax.plot(sm[:, 0], sm[:, 1], color='#E64B35', lw=2.5)
            ax.set_xlim(0, 300)
            ax.grid(alpha=0.3)
            ax.set_title('D  SMAD2 per-nucleus + LOWESS')
            ax = fig.add_subplot(gs[1, 3])
            ax.hist(nuc['area_um2'].clip(0, 600), bins=60, color='#4DBBD5')
            for v in (LO, A_EXP, HI, CAP):
                ax.axvline(v, ls='--', lw=1, color='k')
            ax.set_title('E  nucleus areas (um2) + filter bounds')

            ax = fig.add_subplot(gs[2, 0])
            r_edges = np.arange(0, 320, 20)
            cnts, _ = np.histogram(kept['distance_um'], bins=r_edges)
            annulus = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
            ax.bar(r_edges[:-1] + 10, cnts / annulus * 1000, width=18, color='#8491B4')
            ax.axhline(PRED_DENS, color='k', ls='--', label='marble-in-jar max')
            ax.legend(fontsize=7)
            ax.set_title('F  density vs prediction')
            ax = fig.add_subplot(gs[2, 1])
            ax.bar(wrow['bin'], wrow['coverage'].clip(0, 1), width=9, color='#00A087')
            ax.set_ylim(0, 1.05)
            ax.set_title('G  technical weight per bin')
            ax = fig.add_subplot(gs[2, 2])
            bm = kept.groupby('bin')[['eccentricity', 'cos2_radial']].mean()
            ax.plot(bm.index, bm['eccentricity'], color='#4DBBD5', lw=2,
                    label='eccentricity')
            ax.plot(bm.index, bm['cos2_radial'], color='#7E6148', lw=2,
                    label='cos2 radial')
            ax.axhline(0.5, color='gray', ls=':')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 300)
            ax.legend(fontsize=7)
            ax.set_title('H  nuclear shape vs radius')
            ax = fig.add_subplot(gs[2, 3])
            ax.hexbin(kept['C1'], kept['C2'], gridsize=40, cmap='inferno', bins='log')
            ta = threshold_otsu(kept['C1'].values)
            tb = threshold_otsu(kept['C2'].values)
            ax.axvline(ta, color='w', ls='--', lw=0.8)
            ax.axhline(tb, color='w', ls='--', lw=0.8)
            ax.set_title('I  DAPI vs SMAD2')

            ax = fig.add_subplot(gs[3, 0], projection='polar')
            re2 = np.arange(0, r_col + 20, 20)
            te = np.linspace(-np.pi, np.pi, N_SEC + 1)
            obs, _, _ = np.histogram2d(kept['distance_um'], ang, bins=[re2, te])
            pred = 0.6 * np.pi * (re2[1:] ** 2 - re2[:-1] ** 2) / N_SEC / A_EXP
            occ = np.clip(obs / pred[:, None], 0, 1.5)
            T, R = np.meshgrid(te, re2)
            pc = ax.pcolormesh(T, R, occ, cmap='RdYlGn', vmin=0, vmax=1.5)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_title('J  occupancy vs prediction', pad=10)
            plt.colorbar(pc, ax=ax, shrink=0.6)

            ax = fig.add_subplot(gs[3, 1])
            if 'C2_over_DAPI' in kept.columns:
                rb = kept.groupby('bin')['C2_over_DAPI'].agg(['mean', 'sem'])
                ax.plot(rb.index, rb['mean'], color='#E64B35', lw=2)
                ax.fill_between(rb.index, rb['mean'] - rb['sem'],
                                rb['mean'] + rb['sem'], color='#E64B35', alpha=0.25)
            ax.set_xlim(0, 300)
            ax.grid(alpha=0.3)
            ax.set_title('K  SMAD2/DAPI vs radius')

            ax = fig.add_subplot(gs[3, 2], projection='polar')
            smap = kept.groupby([pd.cut(kept['distance_um'], re2, labels=False),
                                 'sector'])['C2'].mean()
            M = np.full((len(re2) - 1, N_SEC), np.nan)
            for (bi, si), v in smap.items():
                if not np.isnan(bi):
                    M[int(bi), int(si)] = v
            pc = ax.pcolormesh(T, R, M, cmap='inferno')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_title('M  SMAD2 intensity by direction', pad=10)
            plt.colorbar(pc, ax=ax, shrink=0.6)

            ax = fig.add_subplot(gs[3, 3])
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
            ax.set_xlim(0, 300)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.set_title('N  SMAD2 per pie (hue = direction)')

            ax = fig.add_subplot(gs[4, 0:4])
            ax.axis('off')
            em = pd.Series(edge_means)
            low = em[em < em.mean() - em.std()].index.tolist()
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
                   f"({f0['pct_rejected_size']:.0f}%)  occupancy "
                   f"{f0['occupancy_vs_capacity']:.2f}  areaCV {f0['area_cv']:.2f}  "
                   f"dapiCV {f0['dapi_bin_cv']:.2f}  "
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
            print(f'  page: {short} ({len(kept)} kept)')
    print(f'PDF: {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.1f} MB)')
    return pdf_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('base_dir')
    ap.add_argument('--pixel-sizes', default='{}',
                    help='JSON: {"filename-substring": um_per_px}')
    args = ap.parse_args()
    build(args.base_dir, json.loads(args.pixel_sizes))
