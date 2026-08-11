# Radial Intensity Profiling of Micropatterned Colonies

Quantitative radial analysis of immunofluorescence in micropatterned stem-cell colonies
(e.g. RUES2 + Activin A → SMAD2 edge signaling). Dual segmentation backends, dual analysis
modes, per-colony statistics, and publication-ready figures out of one notebook.

## The flagship notebook: `radial_profile_pipeline.ipynb`

| Capability | Detail |
|---|---|
| **Segmentation** | StarDist (default, ~2–3 s/image on CPU) or Cellpose-SAM (better for dense/irregular nuclei; practical with GPU) — one parameter switch |
| **Per-pixel mode** | Classic masked radial binning, identical truncation behaviour to the original V3 pipeline |
| **Per-nucleus mode** | One data point per nucleus + LOWESS trend — eliminates the small-bin noise spike near r = 0 and matches how the micropattern field quantifies nuclear factors (Etoc 2016, Chhabra 2019) |
| **DAPI ratio** | Optional Cn/DAPI per nucleus, controls for density and imaging depth |
| **Statistics** | Colony-level bootstrap 95% CI on the mean profile (colonies, not nuclei, are the resampling unit) |
| **Feature extraction** | Per colony: peak radius, peak value, half-max boundary radius, center:edge ratio → `colony_features.csv`, ready for group comparisons |
| **QC** | Angular asymmetry metric (CV across 12 sectors) flags off-center/asymmetric colonies; failed images logged, never silently skipped |
| **Figures** | Per-image diagnostics + a multi-panel publication figure (300 dpi PNG + vector PDF, npg palette) |
| **Input flexibility** | Accepts 4D `(Z, C, Y, X)` stacks or already max-projected 3D `(C, Y, X)` TIFFs |

**v1.1 — methods adopted from the Tapenade paper** (Gros et al., *eLife* 2026,
[doi:10.7554/eLife.107154](https://doi.org/10.7554/eLife.107154) — the group whose
`tapenade` package this pipeline already builds on):

| Addition | What it gives you |
|---|---|
| Nuclear morphometrics | Per-nucleus eccentricity + radial-alignment (cos² of major axis vs radial direction) profiles — nuclei as proxies for cell deformation |
| Positive-fraction profiles | Otsu threshold per colony → fraction of marker-positive nuclei vs radius (their sparse-marker/FoxA2 approach) |
| Co-expression histograms | Per-nucleus pairwise 2D histograms with Otsu quadrants (their Fig. 5f) |
| DAPI-field re-normalization | Optional masked-Gaussian nuclear-stain field correction for uneven illumination (2D version of their Fig. 5; `DAPI_FIELD_NORM`) |

The original StarDist notebooks (`Normalization_code_V3.ipynb`,
`Non-Normalization_Code_V3.ipynb`) are retained for provenance and comparison.

## Input assumptions

3-channel TIFFs, ordered:

| Channel | Marker |
|---|---|
| C0 | ZO-1 |
| C1 | DAPI |
| C2 | SMAD2 |

Different markers/order/pixel size? Edit `CHANNEL_NAMES`, `DAPI_IDX`, `SCALE_UM_PER_PX`
in the parameters cell.

## Outputs

```
BASE_DIR/
├── radial_outputs/   <image>_radial.csv (per-pixel) + <image>_nuclei.csv (per-nucleus)
├── plots/<image>/    max projection, segmentation overlay, heatmaps,
│                     per-pixel profiles, per-nucleus scatter + LOWESS
├── summary/          combined CSVs, per-channel cross-colony figures,
│                     colony_features.csv, publication_figure.png/.pdf,
│                     failed_images.log
└── ppt_reports/      batch_report.pptx
```

## Install

Python 3.10, tested on macOS.

```bash
conda env create -f environment.yml
conda activate radial-cellpose
```

or

```bash
pip install -r requirements.txt
```

## Run

1. Open `radial_profile_pipeline.ipynb`
2. Set `BASE_DIR` in the batch cell to your TIFF folder
3. Run all cells

## Tuning segmentation

| Symptom | Fix |
|---|---|
| Missing nuclei (Cellpose) | lower `CELLPOSE_CELLPROB_THRESHOLD` (try −1) |
| False detections | raise `CELLPOSE_CELLPROB_THRESHOLD` (+1) or `CELLPOSE_MIN_SIZE` |
| Merged nuclei | lower `CELLPOSE_FLOW_THRESHOLD` (0.3) |
| Split nuclei | raise `CELLPOSE_FLOW_THRESHOLD` (0.5) |
| StarDist under/over-detecting | adjust `STARDIST_NORM_PERCENTILES` |

## Method notes

- Contrast enhancement uses `tapenade.global_contrast_enhancement` (percentiles 0.5/99.5)
  applied per channel after Gaussian smoothing (σ = 1 px).
- Profiles are reported in enhanced-intensity arbitrary units by default
  (`NORMALIZE_PROFILES = False`), preserving inter-colony amplitude differences.
  Set `True` for per-colony max-normalization when only the shape matters.
- The per-pixel truncation past `MAX_RADIUS_UM` reproduces the original V3 behaviour
  exactly, for backward comparability.

## Related tools for 3D / gastruloid work

For whole-mount 3D gastruloids, the Tapenade ecosystem provides a pretrained StarDist3D
nuclei model ([Zenodo 14748083](https://zenodo.org/records/14748083)), dual-view
registration/fusion, spectral unmixing, and napari plugins:
[GuignardLab/tapenade](https://github.com/GuignardLab/tapenade).

## Citation

If this pipeline contributes to a publication, please cite this repository
(see `CITATION.cff` — GitHub's "Cite this repository" button uses it).

## License

MIT — see [LICENSE](LICENSE).
