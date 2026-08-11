# Radial Intensity Analysis — Cellpose Fork

Fork of [Radial-Intensity-Analysis_V3](https://github.com/s1lv2rs0ul/Radial-Intensity-Analysis_V3) with **Cellpose-SAM** as the nucleus segmenter instead of StarDist. Cellpose handles densely packed and irregularly shaped stem cell nuclei more reliably.

## Notebooks

| Notebook | Segmenter | Signal region | Notes |
|----------|-----------|----------------|-------|
| **`radial_profile_cellpose.ipynb`** | Cellpose-SAM (v4) | Inside nuclear mask | New — cleaner code, tunable parameters at top, honest error logging |
| `Normalization_code_V3.ipynb` | StarDist 2D | Inside nuclear mask | Original — normalized profiles |
| `Non-Normalization_Code_V3.ipynb` | StarDist 2D | All pixels (no mask) | Original — absolute intensity |

**Start with `radial_profile_cellpose.ipynb`.** The two `_V3` notebooks are kept for reference and comparison.

## What each pipeline does

For each input TIFF:

1. Max-projects the Z-stack to 2D
2. Contrast-enhances the DAPI channel (CLAHE in the Cellpose version, tapenade global stretch in the V3 versions)
3. Segments nuclei on DAPI
4. Computes the colony center from the nuclear mask (Cellpose + V3 normalized) or as the geometric image center (V3 non-normalized)
5. Bins pixel intensities by distance from the center → per-channel radial profile
6. Saves per-image plots, CSV, and a PowerPoint summary

## Input

A folder of 4D TIFFs in (Z, channel, Y, X) order. Assumes 3 channels:

| Channel | Marker |
|---------|--------|
| C0 | ZO-1 |
| C1 | DAPI |
| C2 | SMAD2 |

Pixel scale defaults to `0.5 µm/px` — change `SCALE_UM_PER_PX` at the top of the Cellpose notebook for your microscope.

## Output

For an input folder `images/` the pipeline creates:

```
images/
├── radial_outputs/      # <image>_radial.csv per image (all channels)
├── plots/<image>/       # per-image PNGs
├── summary/             # combined CSV + per-channel cross-image plots
│                        # + failed_images.log if anything errored
└── ppt_reports/         # batch_report.pptx
```

## Install

Tested on macOS with Python 3.10. Using conda:

```bash
conda env create -f environment.yml
conda activate radial-cellpose
```

Or with pip:

```bash
pip install -r requirements.txt
```

Cellpose downloads the CPSAM model on first run (~50 MB, cached under `~/.cellpose/`).

## Tuning Cellpose

Parameters live at the top of the notebook. Common adjustments:

| Symptom | Fix |
|---------|-----|
| Missing nuclei (undersegmentation) | Lower `CELLPOSE_CELLPROB_THRESHOLD` (try -1) |
| Extra false nuclei | Raise `CELLPOSE_CELLPROB_THRESHOLD` (try +1) or `CELLPOSE_MIN_SIZE` |
| Nuclei merged | Lower `CELLPOSE_FLOW_THRESHOLD` (try 0.3) |
| Nuclei split | Raise `CELLPOSE_FLOW_THRESHOLD` (try 0.5) |
| Wrong size range | Set `CELLPOSE_DIAMETER_PX` explicitly (in pixels) |

## Run

1. Open `radial_profile_cellpose.ipynb` in Jupyter.
2. Set `BASE_DIR` at the bottom to your folder of TIFFs.
3. Run all cells.

## Citation

If you use this in published work, please cite this repository.

## License

MIT — see [LICENSE](LICENSE).
