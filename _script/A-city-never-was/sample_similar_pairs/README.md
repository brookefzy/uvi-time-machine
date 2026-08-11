# Cross-City Similar-Pair Sampling

These scripts create reproducible cross-city image samples from either DINOv3
embeddings or city-classifier probability profiles. They use exact FAISS
`IndexFlatIP` search over deterministic spatial samples. DINOv3 inputs are
already L2-normalized; classifier probability rows are L2-normalized by the
loader. In both cases the returned inner-product score is exact cosine
similarity within the selected input rows.

- `sample_image_pairs_faiss.py` searches spatially sampled source images against
  a sampled target city using DINOv3 embeddings.
- `sample_classifier_image_pairs_faiss.py` performs the same image sampling,
  eligibility, exact search, and diversity selection using classifier
  probability profiles.
- `sample_h3_pairs_faiss.py` searches all selected-resolution H3 vectors for
  each target city using DINOv3 summaries. There is no classifier-probability
  H3-pair sampler.
- `build_image_pair_gallery.py` copies the selected source images to a portable
  gallery folder and writes a side-by-side HTML preview with location maps; its
  title and similarity label are configurable by modality.

The sampling scripts write a Parquet result and a JSON audit sidecar. Their
inputs and output paths are configurable for the remote Slurm environment.

## Classifier-probability image pipeline

### Input contract and similarity meaning

`stage1_classifier/B3_inference_city_prob.py` writes one city directory of
Parquet shards under the established production path (including its historical
`classifiier` spelling):

```text
/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob/
  paris/*.parquet
  london/*.parquet
  hongkong/*.parquet
  singapore/*.parquet
  sydney/*.parquet
  newyork/*.parquet
```

Each shard must contain numeric probability columns `0` through `126` and a
non-null `name` column. The sampler reads the corresponding city Parquet under
`--image-index-root` and treats its full `<panoid>_<angle>.jpg` basenames as the
authoritative image identifiers. It retains probability rows whose `name`
matches that set, requires complete and unique image-index coverage, and derives
`panoid` from the first 22 characters of the retained full name.

The production probability folders may also contain an older complete run in
which every viewing angle was stored under its 22-character panoid only. Those
recognized legacy rows are ignored rather than mistaken for duplicate images;
any other unrecognized name still fails validation. This lets the sampler use
the later full-filename run without rewriting or deleting the legacy shards.
The JSON audit records total input rows, image-index rows, retained full-name
rows, and ignored legacy-panoid rows for every city.

After name selection, the sampler validates that all probabilities are finite
and nonnegative, rejects zero-mass rows, and L2-normalizes each 127-dimensional
row before search. The audit also records min/mean/max raw probability sums.

Classifier-probability cosine has a different meaning from DINOv3 cosine:

- DINOv3 compares learned visual representations.
- Classifier-probability cosine compares distributions over the classifier's
  127 output classes.
- Classifier MMR promotes diversity between probability profiles, not general
  visual-scene diversity.

All probability shards in one run must come from the same checkpoint with the
same class ordering. Legacy shards do not store that provenance. The
`--vector-schema-id`/`CLASSIFIER_SCHEMA_ID` value is therefore an explicit audit
assertion by the operator, not proof inferred from the files. Do not combine
shards from different checkpoints merely because both have 127 columns.

The classifier sampler reuses the existing urban-core H3 pools. Their current
directory name contains `dinov3`, but the pools themselves contain only
geographic `hex_id` eligibility lists and are modality-independent.

### Full remote run

From the repository directory on the Slurm host, submit the pair sampler:

```bash
cd /lustre1/g/geog_pyloo/05_timemachine/uvi-time-machine/_script/A-city-never-was
sbatch slurm/classifier_sample_image_pairs.cmd
```

The default job searches these directed pairs: Paris→London, London→Hong Kong,
Hong Kong→Singapore, London→Sydney, and New York→London. It writes:

```text
sample_similar_pairs/output/classifier_image_pairs.parquet
sample_similar_pairs/output/classifier_image_pairs.json
```

After that job succeeds, build the portable side-by-side gallery:

```bash
sbatch slurm/classifier_build_sample_gallery.cmd
```

The gallery job reads the classifier pair Parquet and writes:

```text
sample_similar_pairs/output/classifier_image_gallery/index.html
sample_similar_pairs/output/classifier_image_gallery/manifest.parquet
sample_similar_pairs/output/classifier_image_gallery/manifest.json
sample_similar_pairs/output/classifier_image_gallery/images/
```

Open `index.html` in a browser. Images are copied into the package, while the
Leaflet/OpenStreetMap location maps require an internet connection.

### Recommended bounded pilot

Before the full five-pair run, validate schema, provenance, score behavior, and
gallery quality with one bounded city pair:

```bash
CITY_PAIRS='Paris|London' \
MAX_IMAGES_PER_CITY=5000 \
PAIRS_PER_CITY_PAIR=20 \
OUTPUT=sample_similar_pairs/output/classifier_paris_london_pilot.parquet \
sbatch slurm/classifier_sample_image_pairs.cmd
```

Build a separate pilot gallery after the sampling job succeeds:

```bash
PAIRS=sample_similar_pairs/output/classifier_paris_london_pilot.parquet \
OUTPUT_DIR=sample_similar_pairs/output/classifier_paris_london_pilot_gallery \
sbatch slurm/classifier_build_sample_gallery.cmd
```

Begin with `CLASSIFIER_THRESHOLD=-1.0`, the job default, so candidates are
selected by rank, hard diversity caps, and MMR. Inspect score quantiles and the
pilot gallery before introducing a cutoff. Do not copy a DINOv3 threshold into
this pipeline because the two vector spaces have different score
distributions.

### Classifier job overrides

The sampling job accepts these environment variables:

```text
CITY_PAIRS                 semicolon-delimited directed CITY1|CITY2 values
CLASSIFIER_PROB_ROOT       probability-shard root
IMAGE_INDEX_ROOT           city image-index Parquets with full image paths
CLASSIFIER_EXPECTED_DIM    expected probability width (default 127)
CLASSIFIER_SCHEMA_ID       operator-asserted checkpoint/class-order identity
CLASSIFIER_THRESHOLD       minimum classifier-profile cosine (default -1.0)
ROOTFOLDER                 Street View project root
TRAIN_TEST_FOLDER          classifier train/test image root
RES_EXCLUDE                optional high-resolution exclusion level
MIN_YEAR / MAX_YEAR        inclusive panorama year range
H3_RESOLUTION              spatial sampling/core-pool resolution
CORE_H3_POOL_ROOT          core-H3 eligibility pool, or none to disable
CORE_H3_PROFILE            core-H3 profile identifier
MAX_IMAGES_PER_H3          deterministic per-cell image cap
MAX_IMAGES_PER_CITY        optional total city cap; 0 disables it
TOP_K                      FAISS neighbors per source image
QUERY_BATCH_SIZE           FAISS query batch size
MAX_PAIRS_PER_SOURCE_IMAGE hard source-image diversity cap
MAX_PAIRS_PER_HEX_PAIR     hard unordered H3-pair diversity cap
MMR_CANDIDATE_POOL         high-score pool retained before MMR
MMR_RELEVANCE_WEIGHT       cosine relevance weight versus profile novelty
PAIRS_PER_CITY_PAIR        final accepted pairs per directed city pair
OUTPUT                     result Parquet path
VENV_PYTHON                explicit Python interpreter override
```

For example, override the schema assertion and probability root, then submit
the gallery job with an `afterok` dependency so it starts only after pair
sampling succeeds:

```bash
CLASSIFIER_SAMPLE_JOB_ID=$(
  CLASSIFIER_PROB_ROOT=/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob \
  CLASSIFIER_SCHEMA_ID=city-classifier-train4-probabilities-v1 \
  OUTPUT=sample_similar_pairs/output/classifier_image_pairs.parquet \
  sbatch --parsable slurm/classifier_sample_image_pairs.cmd
)

PAIRS=sample_similar_pairs/output/classifier_image_pairs.parquet \
OUTPUT_DIR=sample_similar_pairs/output/classifier_image_gallery \
sbatch --dependency="afterok:${CLASSIFIER_SAMPLE_JOB_ID}" \
  slurm/classifier_build_sample_gallery.cmd
```

The gallery entry point is `slurm/classifier_build_sample_gallery.cmd`. It
copies the selected images and writes `index.html`, `manifest.parquet`, and
`manifest.json` under `OUTPUT_DIR`.

## Urban-core filter

All sample jobs default to the resolution-8 POI urban-core pool generated from
the `pct5_sub30_z1_m05` profile. The expected Lustre layout is:

```text
/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_dinov3_core_hex_ids/
  res=8/profile=pct5_sub30_z1_m05/
    paris.parquet
    london.parquet
    hongkong.parquet
    singapore.parquet
    sydney.parquet
    newyork.parquet
    core_h3_pool_audit.json
```

Build those compact `hex_id` pools from the local Stage 3 tier profile with:

```bash
python sample_similar_pairs/export_core_h3_pools.py \
  --source-root '/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_data/_transformed/landuse_poi_res=8/profile=pct5_sub30_z1_m05' \
  --output-root /tmp/dinov3-core-h3-pools \
  --resolution 8 --profile-id pct5_sub30_z1_m05 \
  --cities Paris London 'Hong Kong' Singapore Sydney 'New York'
```

The output JSON audits core-H3 counts and each sample job's JSON audit records
how many rows survive the per-city pool. Use
`CORE_H3_POOL_ROOT=none` only for unfiltered exploratory runs.

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

The jobs intentionally ignore a generic inherited `REPO_DIR` (which may point
to another project). If the sample-script directory must be overridden, use
`UVI_SAMPLE_REPO_DIR=/path/to/uvi-time-machine/_script/A-city-never-was`.

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
default `-1.0` threshold retains every cosine candidate before ranking. For
each city pair, FAISS retrieves `30` nearest images per source image, then
keeps a global high-score pool of `200` candidates after two hard caps (at most
one pair per source image and per unordered H3-cell pair). It uses maximal
marginal relevance (MMR) to select the final `10`: `70%` cosine relevance and
`30%` penalty for looking like an already selected pair, where a pair is
represented by the normalized mean of its two DINOv3 embeddings. This helps
avoid galleries made entirely of visually repetitive scenes such as tunnels or
cars, without requiring predefined scene labels. The H3 script searches all
resolution-8 cells.

Override pairs with a semicolon-delimited variable, preserving spaces in city
names. For example:

```bash
CITY_PAIRS='Paris|London;New York|London' \
  DINO_THRESHOLD=0.88 PAIRS_PER_CITY_PAIR=50 \
  MMR_CANDIDATE_POOL=500 MMR_RELEVANCE_WEIGHT=0.7 \
  sbatch slurm/dinov3_sample_image_pairs.cmd
```

Set `OUTPUT` to redirect either Parquet output. Each output has a sibling JSON
file that records input sample sizes, retrieved candidates, threshold hits, and
accepted pair counts for every city pair.
