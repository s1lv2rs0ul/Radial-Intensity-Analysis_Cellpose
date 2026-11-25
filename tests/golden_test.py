#!/usr/bin/env python3
"""Golden (characterization) test — curriculum step 1 (docs/ENGINEERING.md).

Freezes the pipeline's observable scientific outputs on a pinned reference
image set, so that any future change that shifts the science fails loudly
BEFORE it touches a real analysis. This is Feathers' recipe for legacy code:
characterize current behavior first, refactor second.

The pipeline still lives in a notebook, so this harness execs its code cells
by position — deliberately ugly, and itself the argument for step 2 (module
extraction). After extraction this file switches to `import radial_pipeline`,
and the reference values must NOT move: that is the mechanical proof that the
refactor preserved behavior.

Usage
-----
    RADIAL_GOLDEN_DIR="/path/to/folder/with/reference/tiffs" \\
        python tests/golden_test.py              # compare against reference
    ... python tests/golden_test.py --record     # (re)record the reference

Also discoverable by pytest (skips cleanly when RADIAL_GOLDEN_DIR is unset).
Re-record deliberately (and say why in the commit) after an *intended* science
change or an environment rebuild — never to silence a failure you don't
understand.

Comparison policy
-----------------
Counts match exactly. Floats match to rel 1e-3 / abs 1e-4: CPU thread
scheduling may wiggle the last bits of float reductions, so bitwise equality
is the wrong contract for floating point — the tolerance IS the spec.

What is fingerprinted: kept/rejected/imputed counts per image, per-colony
features (peak radius, center:edge ratio), per-pixel profile values at fixed
radii, and per-image mean bin weight. Figures are NOT fingerprinted — golden
tests pin the science, not the pixels of plots.
"""
import glob
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NOTEBOOK = os.path.join(REPO, 'radial_profile_pipeline.ipynb')
REFERENCE = os.path.join(HERE, 'golden_reference.json')

# ---- the frozen test definition: images, config, sampling ------------------
# The 4-image reference set: two 1024^2 @ 0.518 um/px (one soft-focus, with its
# per-image override) and two 512^2 @ 1.036 um/px (upsample-rescue cases) — it
# exercises rescue, overrides, and watershed splitting in one run.
GOLDEN_IMAGES = [
    '72+ActA_488 ZO1_555 SMAD2_20x.tif',
    '72+ActA2_488 ZO1_555 SMAD2_20x.tif',
    '72CAG+ActA_488 ZO1_555 SMAD2_20x.tif',
    '72CAG+ActA2_488 ZO1_555 SMAD2_20x.tif',
]
GOLDEN_CONFIG = {
    'TUNING_PROFILE': 'standard',
    'PIXEL_SIZE_OVERRIDES': {'72+ActA': 1.0358, '72CAG+ActA': 0.5179},
    'SEG_OVERRIDES': {'72CAG+ActA_': {'unsharp': True, 'prob_thresh': 0.35}},
}
SAMPLE_RADII_UM = [50, 100, 150, 200, 240]
REL_TOL, ABS_TOL = 1e-3, 1e-4


def run_pipeline(image_dir, work_dir):
    """Exec the notebook's code cells and run its batch cell on work_dir."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    with open(NOTEBOOK) as f:
        nb = json.load(f)
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    # code-cell order (kept in sync with the notebook; step 2 removes this):
    # 0 imports, 1 params, 2 tuning profiles, 3 seg backend, 4 helpers,
    # 5 load/seg, 6 per-pixel, 7 per-nucleus, 8 impute, 9 dapi-norm,
    # 10 full_analysis, 11 features, 12 ppt, 13 troubleshooting, 14 batch
    g = {'__name__': 'golden'}
    exec(code[0], g)
    exec(code[1], g)
    g.update(GOLDEN_CONFIG)
    for i in range(2, 14):
        exec(code[i], g)

    for name in GOLDEN_IMAGES:
        src = os.path.join(image_dir, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f'reference image not in RADIAL_GOLDEN_DIR: {name}')
        os.symlink(src, os.path.join(work_dir, name))

    batch = code[14].replace("BASE_DIR = 'PATH_TO_YOUR_TIFF_FOLDER'",
                             f'BASE_DIR = {work_dir!r}')
    assert work_dir in batch, 'batch-cell BASE_DIR anchor moved — update golden_test.py'
    exec(batch, g)


def fingerprint(work_dir):
    """Compact dict of the scientific outputs of a finished run."""
    import numpy as np
    import pandas as pd

    summary = os.path.join(work_dir, 'summary')
    fp = {'counts': {}, 'features': {}, 'profiles': {}, 'weights': {}}

    nuc = pd.read_csv(os.path.join(summary, 'combined_per_nucleus.csv'))
    imputed = nuc['imputed'] if 'imputed' in nuc.columns else False
    for img, grp in nuc.groupby('Image'):
        gi = grp['imputed'] if 'imputed' in grp.columns else pd.Series(False, index=grp.index)
        fp['counts'][img] = {
            'kept': int((grp['size_ok'] & ~gi).sum()),
            'rejected': int((~grp['size_ok']).sum()),
            'imputed': int(gi.sum()),
        }

    feats = pd.read_csv(os.path.join(summary, 'colony_features.csv'))
    for _, r in feats.iterrows():
        key = f"{r['Image']}|{r['Channel']}"
        fp['features'][key] = {
            'peak_radius_um': round(float(r['peak_radius_um']), 3),
            'center_edge_ratio': round(float(r['center_edge_ratio']), 5),
        }

    for img in fp['counts']:
        stem = img[:-4] if img.lower().endswith('.tif') else img
        prof = pd.read_csv(os.path.join(work_dir, 'radial_outputs', f'{stem}_radial.csv'))
        for c in (0, 1, 2):
            for r_um in SAMPLE_RADII_UM:
                i = (prof['Radius'] - r_um).abs().idxmin()
                v = prof.loc[i, f'C{c}']
                fp['profiles'][f'{img}|C{c}|r{r_um}'] = (
                    None if pd.isna(v) else round(float(v), 6))

    w = pd.read_csv(os.path.join(summary, 'bin_weights.csv'), index_col=0)
    for img, row in w.iterrows():
        fp['weights'][img] = round(float(np.nanmean(row.values)), 5)
    return fp


def compare(ref, new):
    """Return a list of human-readable mismatches (empty = pass)."""
    bad = []
    for section in ('counts', 'features', 'profiles', 'weights'):
        r, n = ref.get(section, {}), new.get(section, {})
        for key in sorted(set(r) | set(n)):
            if key not in r or key not in n:
                bad.append(f'{section}/{key}: only in '
                           + ('reference' if key in r else 'new run'))
                continue
            rv, nv = r[key], n[key]
            if isinstance(rv, dict):
                for f in sorted(set(rv) | set(nv)):
                    a, b = rv.get(f), nv.get(f)
                    if isinstance(a, int) and isinstance(b, int):
                        if a != b:
                            bad.append(f'{section}/{key}/{f}: {a} != {b}')
                    elif not _close(a, b):
                        bad.append(f'{section}/{key}/{f}: {a} !~ {b}')
            elif not _close(rv, nv):
                bad.append(f'{section}/{key}: {rv} !~ {nv}')
    return bad


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=ABS_TOL)


def run_once():
    image_dir = os.environ.get('RADIAL_GOLDEN_DIR')
    if not image_dir:
        raise RuntimeError('set RADIAL_GOLDEN_DIR to the folder holding the '
                           'reference TIFFs (see GOLDEN_IMAGES in this file)')
    work = tempfile.mkdtemp(prefix='radial_golden_')
    try:
        run_pipeline(image_dir, work)
        return fingerprint(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_golden():
    """pytest entry point — compares a fresh run against the committed reference."""
    if not os.environ.get('RADIAL_GOLDEN_DIR'):
        import pytest
        pytest.skip('RADIAL_GOLDEN_DIR not set (golden test needs local reference images)')
    assert os.path.exists(REFERENCE), 'no golden_reference.json — run with --record first'
    with open(REFERENCE) as f:
        ref = json.load(f)
    bad = compare(ref['fingerprint'], run_once())
    assert not bad, 'golden mismatches:\n' + '\n'.join(bad)


def main():
    record = '--record' in sys.argv
    fp = run_once()
    if record or not os.path.exists(REFERENCE):
        import datetime
        import platform
        payload = {
            'recorded': datetime.datetime.now().isoformat(timespec='seconds'),
            'machine': platform.platform(),
            'python': sys.version.split()[0],
            'images': GOLDEN_IMAGES,
            'config': GOLDEN_CONFIG,
            'sample_radii_um': SAMPLE_RADII_UM,
            'tolerance': {'rel': REL_TOL, 'abs': ABS_TOL},
            'fingerprint': fp,
        }
        with open(REFERENCE, 'w') as f:
            json.dump(payload, f, indent=1, sort_keys=True)
        print(f'RECORDED {sum(len(v) for v in fp.values())} values -> {REFERENCE}')
        return 0
    with open(REFERENCE) as f:
        ref = json.load(f)
    bad = compare(ref['fingerprint'], fp)
    if bad:
        print(f'GOLDEN FAIL — {len(bad)} mismatch(es):')
        for b in bad:
            print('  ' + b)
        return 1
    print(f"GOLDEN PASS — {sum(len(v) for v in fp.values())} values within "
          f"tolerance (rel {REL_TOL}, abs {ABS_TOL})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
