"""Add 3D/gastruloid tapenade-StarDist3D section to the notebook. Not committed."""
import json

NB = "/tmp/rc2/radial_profile_pipeline.ipynb"
with open(NB) as f:
    nb = json.load(f)

md = {"cell_type": "markdown", "metadata": {}, "source": """\
---
## Future work: 3D gastruloid segmentation with the pretrained tapenade StarDist3D

The Tapenade group provides their custom **StarDist3D** model (`tapenade_stardist`), trained
on 4,414 annotated gastruloid nuclei (F1 = 85±3%, constant across >200 µm depth). The cells
below download it from Zenodo (12.6 MB, cached locally), load it, and provide a 3D
per-nucleus table that mirrors this pipeline's 2D analysis — with **distance to the sample
border** (the paper's radial coordinate for variable-size 3D samples) instead of distance
to a 2D colony center.

**Model constraints (from their readme):** isotropic voxels, nuclei ≈ 15 px diameter
(so ≈ 1 µm/voxel for typical 10–15 µm nuclei), intensity normalized to [0, 1].
Set `Z_STEP_UM` to your acquisition's z-spacing.

These cells are self-contained and do not run in the 2D batch above.
""".splitlines(keepends=True)}

code1 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
"source": '''\
TAPENADE_ZENODO_API = 'https://zenodo.org/api/records/14748083'
TAPENADE_MODEL_ZIP  = 'stardist_tapenade_model.zip'   # 12.6 MB (ignore the 9.4 GB data zip)


def load_tapenade_stardist3d(model_root='models'):
    """Download (once, cached) and load the pretrained tapenade StarDist3D model
    (Gros et al., eLife 2026; Zenodo 14748083)."""
    import urllib.request, zipfile
    os.makedirs(model_root, exist_ok=True)
    basedir = os.path.join(model_root, 'stardist_tapenade_model')
    if not os.path.isdir(os.path.join(basedir, 'tapenade_stardist')):
        url = f'{TAPENADE_ZENODO_API}/files/{TAPENADE_MODEL_ZIP}/content'
        zip_path = os.path.join(model_root, TAPENADE_MODEL_ZIP)
        print(f'Downloading {TAPENADE_MODEL_ZIP} from Zenodo…')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(model_root)
        os.remove(zip_path)
    from stardist.models import StarDist3D
    return StarDist3D(None, name='tapenade_stardist', basedir=basedir)


def segment_nuclei_3d(dapi_zyx, z_step_um, xy_um_per_px=SCALE_UM_PER_PX,
                      target_um_per_vox=1.0, model=None):
    """Segment nuclei in a 3D (Z, Y, X) stack with the tapenade StarDist3D model.

    Rescales to isotropic voxels of `target_um_per_vox` (their model expects nuclei of
    ~15 px diameter, i.e. ~1 um/voxel for 10-15 um nuclei), normalizes, predicts, and
    returns (labels_isotropic, voxel_size_um)."""
    from scipy.ndimage import zoom as _zoom
    if model is None:
        model = load_tapenade_stardist3d()
    factors = (z_step_um / target_um_per_vox,
               xy_um_per_px / target_um_per_vox,
               xy_um_per_px / target_um_per_vox)
    iso = _zoom(dapi_zyx.astype(np.float32), factors, order=1)
    norm = global_contrast_enhancement(iso, perc_low=1, perc_high=99)
    labels, _ = model.predict_instances(norm)
    return labels, target_um_per_vox
'''.splitlines(keepends=True)}

code2 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
"source": '''\
def per_nucleus_table_3d(channels_iso, labels, voxel_um):
    """3D analogue of per_nucleus_table for gastruloids.

    channels_iso : (C, Z, Y, X) intensity channels at the SAME isotropic voxel size
                   as `labels` (rescale them with the same zoom factors).
    labels       : 3D label image from segment_nuclei_3d.
    Returns one row per nucleus: centroid, volume, distance to the sample border
    (the Tapenade paper's radial coordinate), and per-channel mean intensities."""
    from scipy.ndimage import distance_transform_edt, binary_fill_holes, binary_closing

    ids = np.arange(1, int(labels.max()) + 1)
    ones = np.ones(labels.shape, dtype=np.float32)
    coms = ndimage.center_of_mass(ones, labels, ids)
    vol_vox = ndimage.sum_labels(ones, labels, ids)

    # Sample mask: closed + filled union of nuclei; EDT gives distance to border
    sample = binary_fill_holes(binary_closing(labels > 0, np.ones((5, 5, 5))))
    edt = distance_transform_edt(sample) * voxel_um

    zs = np.array([c[0] for c in coms]); ys = np.array([c[1] for c in coms])
    xs = np.array([c[2] for c in coms])
    zi = np.clip(zs.round().astype(int), 0, labels.shape[0]-1)
    yi = np.clip(ys.round().astype(int), 0, labels.shape[1]-1)
    xi = np.clip(xs.round().astype(int), 0, labels.shape[2]-1)

    out = {
        'nucleus_id': ids, 'z_vox': zs, 'y_vox': ys, 'x_vox': xs,
        'volume_um3': vol_vox * voxel_um**3,
        'dist_to_border_um': edt[zi, yi, xi],
    }
    for c in range(channels_iso.shape[0]):
        out[f'C{c}'] = ndimage.mean(channels_iso[c], labels, ids)
    return pd.DataFrame(out)


# --- Example usage on a (Z, C, Y, X) two-photon stack ------------------------------
# from scipy.ndimage import zoom
# Z_STEP_UM = 1.0                                    # <- your z spacing!
# stack = tifffile.imread('my_gastruloid.tif')       # (Z, C, Y, X)
# labels3d, vox = segment_nuclei_3d(stack[:, DAPI_IDX], Z_STEP_UM)
# factors = (Z_STEP_UM / vox, SCALE_UM_PER_PX / vox, SCALE_UM_PER_PX / vox)
# chans = np.stack([zoom(stack[:, c].astype(np.float32), factors, order=1)
#                   for c in range(stack.shape[1])])
# nuc3d = per_nucleus_table_3d(chans, labels3d, vox)
# nuc3d.to_csv('gastruloid_nuclei_3d.csv', index=False)
# # then profile any channel vs nuc3d['dist_to_border_um'] exactly like the 2D LOWESS plots
'''.splitlines(keepends=True)}

nb['cells'].extend([md, code1, code2])
with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\\n')
print(f"Now {len(nb['cells'])} cells")
