# Runtime & parameter-tuning trade-offs — high-content batches

What each knob costs in wall-clock and buys in fidelity, measured on real data
(Nov 2026). Every batch run re-measures itself: the batch cell writes
`summary/runtime_profile.csv` and `runtime_profile.png` (per-image, per-stage,
plus a scale projection), and `run_parameters.json` records `TUNING_PROFILE` —
so these numbers are reproducible on any batch, on any machine.

## Tuning profiles

`TUNING_PROFILE` in the parameters cell switches only the expensive knobs
(`None` = knobs exactly as set):

| profile | upsample rescue | splitting | bootstrap | intended use |
|---|---|---|---|---|
| `'screening'` | off | off | 200 | first pass over a big batch |
| `'standard'` | on | on | 1000 | day-to-day (= defaults) |
| `'publication'` | on | on | 2000 | final figures |

Deliberately **not** in any profile: `SEGMENTER` (the model choice changes the
science, not just the runtime) and every QC threshold.
**Rule: screening triages, never decides.** Its QC flags name the images that
need the expensive knobs; nothing is silently dropped by running fast.

## Reference measurements

10-core Apple-silicon Mac, CPU-only TensorFlow/StarDist. Bench = 4 images:
two 1024² @ 0.518 µm/px (one soft-focus, with its `SEG_OVERRIDES` entry
`unsharp + prob_thresh 0.35`) and two 512² @ 1.036 µm/px (coarse — the rescue
cases from `VALIDATION.md`). Run-to-run jitter ≈ ±10 %.

Per-image seconds by stage, `'publication'` profile (model load 5–7 s, once per
session):

| stage | 512² coarse (rescued) | 1024² fine | share of per-image time |
|---|---|---|---|
| segmentation | 2.8 – 3.2 | 3.7 – 6.6 | **48 %** |
| overview figures | 2.7 – 2.9 | 3.1 – 3.3 | **35 %** |
| profiles + outputs | 1.0 | 1.1 – 1.3 | 13 % |
| load + preprocess | 0.1 | 0.2 – 0.3 | 2 % |
| nucleus table + QC | 0.1 | 0.2 | 2 % |

## Measured profile trade-off (same 4 images)

| | `'screening'` | `'publication'` | difference |
|---|---|---|---|
| mean wall-clock / image | 7.3 s | 8.5 s | **+16 %** |
| kept nuclei, coarse pair | 76 / 165 | 822 / 894 | **~10×** (upsample rescue) |
| kept nuclei, fine pair | 864 / 937 | 869 / 939 | ±1 % |
| QC verdict | flags exactly the coarse pair: *far below capacity*, packing 0.09 / 0.26 → down-weighted | flags clear, packing 0.71–0.77 | triage works |

The headline: the publication profile is only ~16 % slower on a mixed batch,
because the expensive-looking knob (rescue) upsamples **only** images that are
coarser than 1.3× target — and an upsampled 512² costs about the same as a
native 1024². Where it does fire, ~2.6 s/image bought ~10× the nuclei.

## What each knob costs

| knob | runtime effect (measured here) | what it buys |
|---|---|---|
| `SEGMENTER` | StarDist: seconds/image on CPU. Cellpose-SAM: **minutes/image on CPU** — the single biggest lever; practical only with a GPU | robustness on dense/irregular nuclei |
| `SEG_UPSAMPLE_TARGET_UM` | segmenter sees factor² more pixels, but only when px > 1.3× target; +2.6 s on 512² here, zero on at-target images | 76→822 / 165→894 kept nuclei |
| `STARDIST_PROB_THRESH` / `SEG_OVERRIDES` | lower threshold → more candidates → slower NMS: the 0.35 soft-focus override ran 6.6–6.8 s vs 3.7–3.9 s for its healthy sibling | recovers soft-focus images (738→1075 detections) |
| `SPLIT_OVERSIZED` | per over-cap blob (EDT + watershed on its bbox); ≤0.2 s/image here, grows only on heavily fused images | merged blobs become nuclei instead of rejects |
| `BOOTSTRAP_N` 200↔2000 | batch-level, vectorized, once per run; <2 s either way at n=4 colonies; linear in N × colonies — not a bottleneck below thousands | CI tightness on the group curve |
| `IMPUTE_GAPS` | sub-second per colony | see PIPELINE.md; off by default |
| `DAPI_FIELD_NORM` | one masked Gaussian per channel, ~0.1–0.3 s/image | optical-artifact correction |
| figure rendering | **the hidden second cost**: ~4 s/image, constant (overview ~3 s + profile PNGs ~1 s) = 35 % of per-image time; on 300 images ≈ 20 min of matplotlib | the per-image record — kept, because the reports are the product; if a headless pass is ever needed, gate this first |
| `scripts/deep_dive_report.py` | 8.3 s for 4 colonies ≈ ~2 s/colony + overhead; PDF ≈ 0.5 MB/colony | Section 1/2 reports — version-stamp, don't regenerate per tweak |

## Projections (serial, this machine, mixed-resolution batch)

| batch | `'screening'` | `'publication'` |
|---|---|---|
| 30 images (full 6-group experiment, e.g. BRD9+ACT) | ~3.8 min | ~4.4 min |
| 96 images (one plate) | ~12 min | ~14 min |
| 300 images (plate scan) | ~37 min | ~43 min |

Plus model load once and ~2 s/colony for the deep-dive PDF. Per-image work is
independent, but CPU TensorFlow already uses the cores during segmentation
(~50 % of the time), so a process pool would mainly parallelize the matplotlib
share — expect ~2–3×, not cores×. Measure with `runtime_profile.csv` before
engineering anything.

## The intended high-content loop

1. `TUNING_PROFILE = 'screening'` over the whole batch → read the QC flags and
   `summary/runtime_profile.png`.
2. Fix what the flags name: `PIXEL_SIZE_OVERRIDES` from CZI metadata **first**
   (the founding incident, VALIDATION.md §1), then per-image `SEG_OVERRIDES`
   for soft focus.
3. `TUNING_PROFILE = 'publication'` for the final run; archive `summary/`
   with its `run_parameters.json`.

## Reproduce

Point the notebook's batch cell at a folder, set `TUNING_PROFILE`, run, and read
`summary/runtime_profile.csv` / `.png`. The two-profile comparison above is the
same run repeated with `'screening'` and `'publication'`.
