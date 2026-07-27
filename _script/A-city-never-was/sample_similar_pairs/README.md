# DINOv3 Similar-Pair Sampling

These scripts create reproducible cross-city samples from the DINOv3 outputs.
They use FAISS `IndexFlatIP`; because the DINOv3 vectors are L2-normalized,
the returned inner-product score is exact cosine similarity within the selected
input rows.

- `sample_image_pairs_faiss.py` searches spatially sampled source images against
  a sampled target city.
- `sample_h3_pairs_faiss.py` searches all selected-resolution H3 vectors for
  each target city.
- `build_image_pair_gallery.py` copies the selected source images to a portable
  gallery folder and writes a side-by-side HTML preview with location maps.

Both scripts will write a Parquet result and a JSON audit sidecar. Their inputs
and output paths are configurable for the remote Slurm environment.

## Remote runs

Install a compatible `faiss-cpu` package in the remote Python environment, then
submit the two independent jobs after DINOv3 embedding/H3 aggregation finishes:

```bash
cd /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was
sbatch slurm/dinov3_sample_image_pairs.cmd
sbatch slurm/dinov3_sample_h3_pairs.cmd
```

The jobs resolve their default interpreter to the absolute repository path
`uvi-time-machine/.venv/bin/python`, so they use the `uv` environment that
contains `faiss-cpu`. To force an interpreter at submission time, pass it through
Slurm explicitly:

```bash
sbatch --export=ALL,VENV_PYTHON=/lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/.venv/bin/python \
  slurm/dinov3_sample_image_pairs.cmd
```

The job log begins with `Using Python: …`; verify that it names the expected
absolute path before relying on the result.

After the image-pair sample job completes, build a portable preview package:

```bash
sbatch slurm/dinov3_build_sample_gallery.cmd
```

It writes `sample_similar_pairs/output/image_gallery/` by default. Open
`index.html` in a browser; image files are copied into its `images/` folder,
and each side of every pair has its own Leaflet/OpenStreetMap location map.

The defaults search these directed pairs: Paris→London, London→Hong Kong,
Hong Kong→Singapore, London→Sydney, and New York→London. `IndexFlatIP` gives
exact scores only within the deterministic spatial sample (up to `100` images
per H3 cell); it exports the top `10` image pairs per requested city pair. The
default `-1.0` threshold retains every cosine candidate before ranking, and
diversity caps are disabled by default. The H3 script searches all
resolution-8 cells.

Override pairs with a semicolon-delimited variable, preserving spaces in city
names. For example:

```bash
CITY_PAIRS='Paris|London;New York|London' \
  DINO_THRESHOLD=0.88 PAIRS_PER_CITY_PAIR=50 \
  sbatch slurm/dinov3_sample_image_pairs.cmd
```

Set `OUTPUT` to redirect either Parquet output. Each output has a sibling JSON
file that records input sample sizes, retrieved candidates, threshold hits, and
accepted pair counts for every city pair.
