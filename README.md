# Tree cover of Germany at 20 cm

Reference implementation for the manuscript submitted to *Remote Sensing of
Environment*.

A SegFormer trained on orthophotos from a **single** federal state maps tree
cover for all of Germany at 20 cm. The mechanism is season-aware weighted
sampling: leaf-on, transition and leaf-off are drawn equally often during
training. Tested against LiDAR in three other states the model reaches
tree-class IoU 0.844 and F1 0.892. Germany's tree cover, trees outside
forests included, is 32.3 % (≈ 115,500 km²).

---

## Install

```bash
git clone <repo-url> && cd tree-cover-germany
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # core pipeline
pip install -e ".[dl,lidar,viz]"  # + training/inference, LiDAR, figures
```

GDAL command-line tools (`gdalbuildvrt`, `gdalwarp`, `gdaladdo`) must be on
`PATH` for the tiling and merge steps. For the `dl` extra, install the
[PyTorch build matching your CUDA version](https://pytorch.org) first.

Then point the code at your data:

```bash
cp configs/paths.example.yaml configs/paths.yaml
$EDITOR configs/paths.yaml       # at minimum: training_data.root
```

`paths.yaml` is gitignored. Any single value can be overridden from the
environment, which is handy on a cluster:

```bash
TREECOVER_PREDICTIONS_ROOT=/mnt/scratch/Germany python scripts/06_tile_statistics.py --all
```

## Pipeline

Each stage is a standalone CLI. Run `--help` on any of them.

| Stage | Script | What it does |
|------:|--------|--------------|
| 1 | `01_sample_tiles.py` | Stratified selection of training tiles (TCD × settlement × season) |
| 2 | `02_download_dataset.py` | Fetch orthophotos and height models, build the nDSM |
| 3 | `03_prepare_patches.py` | Resolution alignment and sliding-window patch extraction |
| 4 | `04_train.py` | SegFormer training with season-aware weighted sampling |
| 5 | `05_inference.py` | Moving-window inference across a state |
| 6 | `06_tile_statistics.py` | Tree cover area per tile, masked to land and state borders |
| 7 | `07_validate.py` | Validation against LiDAR — four subcommands |
| 8 | `08_merge_reproject.py` | 1 km tiles → 10 km GeoTIFFs in EPSG:3857, for display |
| 9 | `09_compare_products.py` | Comparison against other products; the Results tables |
| 10 | `10_extract_tile_products.py` | Per-tile cover of every comparison product, on the common grid |
| 11 | `11_export_mosaic.py` | 1 km tiles → 10 km GeoTIFFs in native UTM, lossless; the archival product |
| 12 | `12_coverage_polygons.py` | One polygon per acquisition date, shipped beside the mosaic |

Two helpers are not stages. `prepare_tile_table.py` sits between 10 and 9
and attaches the centroids, areas and land areas the aggregation weights
by. `collect_release.py` checks the published data directory against the
inventory in [`treecover.release`](src/treecover/release.py) and writes its
manifest.

**Stage 8 is for display, stage 11 for numbers.** Stage 8 warps to Web
Mercator, which is what the online viewer serves; areas computed from its
output are inflated ~2.5× at Germany's latitude. Stage 11 copies tiles into
their windows without resampling — the predictions already sit on the 20 cm
UTM grid — so a pixel of its output is a pixel of the model. It needs no
GDAL binaries, only rasterio.

## What you have to get right

**Stage 1 is not bit-reproducible.** The draw walks the candidate table
under a fixed numpy seed, and that table has grown since 2024. The
published run asked for 200 tiles and got 152; the 2 km separation
exhausted the strata first. `--compare` prints the overlap with the
published set and, second, how many of its tiles are still eligible here.
The second number is the check — all 152 still pass. A reference tile that
the filters *reject* would be the real signal.

```bash
python scripts/01_sample_tiles.py --out results/sampling \
    --exclude .../gadm41_DEU.gpkg --exclude-layer ADM_ADM_1 \
    --exclude-where "NAME_1 == 'Bremen'" \
    --save-attributes results/sampling/attributes.gpkg
```

Reading the two auxiliary rasters is the slow part. `--save-attributes`
caches the per-tile table, `--attributes` reuses it.

**Stage 2 needs `--source index` to rebuild the training set.**
[easygeodata.de](https://easygeodata.de) indexes all sixteen states behind
one API and needs no local index, which makes it the right route for a
fresh area — but it serves the *current* acquisition of a tile only. The
training set is built from pairs: one summer image and one outside it,
sharing a label.

```bash
# easygeodata.de: one bbox query per tile, all sixteen states, nothing local
python scripts/02_download_dataset.py --out data/Sampling

# the state's own index: every flight over a tile, not just the newest
python scripts/02_download_dataset.py --source index --out data/Sampling \
    --index-dop  .../lgln-opengeodata-dop20.geojson \
    --index-bdom .../lgln-opengeodata-bdom20.geojson \
    --index-dgm  .../lgln-opengeodata-dgm1.geojson
```

An orthophoto is kept only when the surface model of that tile carries the
**same date** — the two products are published separately, and pairing a
2026 image with a 2025 height model makes the label wrong in both. In
states publishing on a 2 km grid (Lower Saxony) the download is cropped to
the 1 km tile. Label masks are not downloadable; they ship with the
training-data package.

**Stage 3 with no arguments rebuilds the published training set** — 117
tiles, 245 observations, 19,845 patches (15,957 train / 3,888 val) — and
says whether it did. Two wrong values would otherwise pass unnoticed: a 256
stride quadruples the set, and the GeoPackage's `split` column moves
published training tiles into validation.

**Stage 7 has a manual step in the middle.**

```bash
python scripts/07_validate.py sample    --state BB --candidates candidates.gpkg
python scripts/07_validate.py reference --state BB
#   ← open tree_mask_footprints.geojson in QGIS, set exclude = 1 where the
#     ground changed between the LiDAR and the orthophoto
python scripts/07_validate.py score     --all-states
```

`sample` must run **once** — re-drawing changes every reported number, so
it refuses to overwrite without `--force`. `reference` runs for hours
against flaky state servers, so it caches and resumes. `summarise` rebuilds
the accuracy table from metrics already on disk, no rasters needed:

```bash
python scripts/07_validate.py summarise --metrics-dir publication/validation
```

A third of the validation boxes hold no reference tree, where IoU is
undefined, and the rule for them decides the headline. `--empty score`
(default, as published) counts an empty prediction as correct and invented
canopy as wrong: IoU 0.844, F1 0.892. `--empty drop` measures only where
trees are: 0.771 / 0.846. The rule lives in
[`validation.metrics.score_zero_reference`](src/treecover/validation/metrics.py).

**Class codes are 0 background, 1 tree, 255 nodata**, defined once in
[`constants.py`](src/treecover/constants.py). `validate_prediction_codes()`
rejects anything else. Codes 3–6 belong to the trees-outside-forests
classification of a different paper, which is not part of this
repository — reading such a raster here counts those classes as background.

**Merging overlaps: newest acquisition date wins.** About 2.5 % of 1 km
cells — 9,679 of 370,515 — are covered by more than one prediction tile,
almost always where two states flew the same ground at a border. The
manuscript prescribes no rule, and three earlier revisions sorted the file
list three different ways, disagreeing on every contested cell. The choice
is now made in [`merge.py`](src/treecover/merge.py) **before GDAL is
invoked**, with a deterministic tiebreak, so VRT source ordering decides
nothing. `--report` writes a CSV of every contested cell.

**Never quote a percentage without its reference area.** Three figures, one
measurement:

| route | reference area | cover | tree |
|---|---|---|---|
| pixel count, all tiles, masked | 356,381 km² mapped land | 32.33 % | 115,202 km² |
| Table 1, common product baseline | 350,435 km² | 32.31 % | 113,231 km² |
| the paper's headline | 357,596 km², all of Germany | 32.3 % | ≈ 115,500 km² |

Mapped land is 99.66 % of Germany; the missing 1,215 km² are absent or
corrupt tiles in SL, HB, SH and MV. Table 1 is narrower still — a product
only enters over tiles where our map is valid too.

Aggregation weights by `land_area_km2`, the land measured inside each tile;
`prepare_tile_table.py --land-areas` attaches it. **Do not fall back to
`tile_area_km2`**, which is the lon/lat envelope of a UTM square and a mean
4.5 % too large. Percentages survive that, areas do not.

Stage 6 masks each tile to the OSM land polygons ∩ its own GADM state
border before counting. Without it, border tiles are counted twice and
coastal tiles count open water as treeless land.

## Pinned to the manuscript

Tests hold these to what was published, so a refactor cannot drift:

| Setting | Value | Where |
|---|---|---|
| Sampling strata / separation | 4 TCD bins × settlement, 2 km apart | `tile_sampling.PUBLISHED_BIN_TARGETS`, `PUBLISHED_SETTINGS` |
| Training patch size / stride | 512 / 512 px — non-overlapping | `03_prepare_patches.PUBLISHED_STRIDE` |
| Training split | from `observations.csv`, not the GeoPackage column | `03_prepare_patches.apply_published_splits` |
| Inference patch / stride / margin | 512 / 360 / 76 px | `inference.tiling` |
| Neighbourhood context per tile | 256 px | `inference.sources.CONTEXT_PX` |
| Common comparison grid | 1 km | `comparison.GRID_DLON/DLAT` |

The inference stride equals the kept inner region, so patches tile a tile
exactly. The 256 px halo is read from neighbouring imagery and cropped away
before writing; without it the merged map shows seams along the tile grid.

## Season-aware sampling

The paper's central mechanism, in
[`data/seasons.py`](src/treecover/data/seasons.py). Aerial surveys are
dominated by summer, so a uniform draw teaches the model "tree = green
blob" and it fails on bare winter canopy. Weighting each patch by the
inverse frequency of its phenological stage equalises the three stages per
epoch without discarding data:

```
counts  = {leaf_on: 36, transition: 27, leaf_off: 18}
weights = {leaf_on: 0.75, transition: 1.0, leaf_off: 1.5}
        → 27 expected draws per season
```

The published counts — leaf-off 3,159, transition 5,184, leaf-on 7,614 —
give the weights 1.68, 1.03 and 0.70 that the manuscript quotes.
`04_train.py --no-season-weighting` reproduces the ablation.

## Layout

```
configs/          states.yaml (checked in) · paths.yaml (yours, gitignored)
src/treecover/    the library — no absolute paths anywhere in here
  constants.py      class codes; the single source of truth
  config.py         path + state configuration loading
  io/               tile discovery and naming
  masking.py        land + state-border masking for area totals
  statistics.py     per-tile tree cover area
  data/ models/     dataset, augmentation, SegFormer construction
  inference/        tiling, stitching, imagery sources
  validation/       sampling, CHM generation, metrics
scripts/          thin CLIs, one per pipeline stage
figures/          one script per paper figure
tests/            synthetic-data tests, no download required
```

Everything that differs between federal states — URL templates, GADM keys,
CRS, the `NRW`/`NW` naming split — lives in
[`configs/states.yaml`](configs/states.yaml) and nowhere else.

## Data

### The training-data package

128 MB, published alongside the paper. Point `training_data.root` in
`paths.yaml` at it; stage 3 and figure 3 then run with no further
arguments.

```
sampled_tiles_100.gpkg    152 tiles, EPSG:25832, with the sampling strata
labels/                   152 label masks, 5000 x 5000 px at 20 cm, 0/1/255
patches/                  observations.csv, patches_metadata.csv, split_info.json,
                          region_vrts.json, experiment_config.json
logs/                     which tile-date came from which URL, and when
```

The label masks cannot be obtained anywhere else. The orthophotos and
height models they were drawn on are **not** included: 71 GB of openly
licensed LGLN data, re-downloadable per tile, every URL recorded in
`logs/`.

> **The `split` column in the GeoPackage is not the published split.** It
> is an earlier three-way draw (train 73 / val 23 / test 23) whose classes
> cut across the published ones. Using it puts published *training* tiles
> into validation and silently inflates every metric measured afterwards.
> The real assignment is `patches/observations.csv`.

Not in the repository:

- **Model weights** — Zenodo (DOI pending). The B5 checkpoint is 323 MB.
- **Tile indices** — per-state orthophoto and LiDAR footprints, ~172 MB.
- **Orthophotos** — openly licensed, from the state surveying authorities;
  endpoints for all 16 states in `configs/states.yaml`.
- **Auxiliary layers** — CORINE Land Cover 2018, Copernicus HRL Tree Cover
  Density, Naturräume (BfN), SRTM, GADM 4.1, OSM land polygons and
  building footprints.

### The published data directory

```bash
python scripts/collect_release.py --root /path/to/publication
```

Walks the inventory, sizes directories, hashes files under the limit,
writes `MANIFEST.csv`, and reports what is missing, what is present but
empty, and what is there without being listed. `--add SRC=DEST` copies
stragglers in, `--prune` clears `.ipynb_checkpoints` and interrupted
downloads. It fixes nothing else — a figure that was never rendered has to
be rendered by its own script.

## Figures

One script per paper figure. Each runs on its own and writes PNG (600 dpi)
plus PDF. Script numbers match the manuscript's figure numbers.

| Figure | Script | Input |
|---:|---|---|
| 1 | `fig01_acquisition_coverage.py` | the prediction archive (filenames only) |
| 2 | — | QGIS composition, not scripted |
| 3 | `fig03_training_examples.py` | training DOP + nDSM + label masks |
| 4 | `fig04_scatter_lidar_vs_model.py` | the per-state validation metrics directory |
| 5 | `fig05_stratified_iou.py` | the per-state validation metrics directory |
| 6 | `fig06_example_tiles.py` | LiDAR masks + orthophotos + predictions |
| 7 | `fig07_scatter_products.py` | per-tile product CSV |
| 8 | `fig08_product_comparison.py` | per-tile product CSV |
| 9 | `fig09_relative_difference.py` | per-tile product CSV |
| 10 | `fig10_local_comparison.py` | orthophotos + predictions + product rasters |

```bash
python figures/fig01_acquisition_coverage.py   --predictions-root /path/to/Germany --save-coverage coverage.csv
python figures/fig03_training_examples.py      --tiles .../sampled_tiles.gpkg --images .../DOP --masks .../predictions --ndsm .../nDSM
python figures/fig04_scatter_lidar_vs_model.py --metrics-dir publication/validation
python figures/fig05_stratified_iou.py         --metrics-dir publication/validation
python figures/fig06_example_tiles.py          --masks-root .../lidar_masks --predictions-root .../Germany
python figures/fig07_scatter_products.py       --tiles tiles_with_treecover.csv
python figures/fig08_product_comparison.py     --tiles tiles_with_treecover.csv
python figures/fig09_relative_difference.py    --tiles tiles_with_treecover.csv
python figures/fig10_local_comparison.py       --products-root .../Other_Tree_Products --predictions-root .../Germany
```

**Figure 1** reads tile position and date from the filename rather than
opening 380,000 rasters — minutes instead of an hour. It uses the same
per-cell selection as the merge, so the date map and the merged raster
cannot disagree.

**Figures 3, 6 and 10 have their tiles pinned** in `PUBLISHED_TILES` and
`PUBLISHED_SAMPLES`. Their automatic selection depends on the whole
candidate set, so one tile added to the archive would silently republish a
different figure. `--auto-select`, `--pick-by-iou`, `--tile-ids` and
`--samples` re-open the choice; `--list-candidates` prints the 72 tiles
eligible for figure 3.

Figure 3 shows what a training tile is: not one image but a *pair* of
acquisitions sharing one label. Its nDSM is image-based (`dom1` − `dgm1`)
and the labels were digitised by hand with it and the near-infrared band —
not thresholded out of it. That is why the height ramp is purple and
carries no 3 m contour. The thresholded LiDAR CHM is figure 6. Figure 10
shows each product at its **native** resolution, which is what makes the
resolution dependence visible.

Styling is centralised in
[`figures/style.py`](src/treecover/figures/style.py). `PRODUCT_COLORS` and
`PRODUCT_LABELS` are keyed by *column*, so a renamed caption cannot change
a colour. Categorical hues are Okabe–Ito. The diverging default is
red–blue, matching the submitted manuscript; `DIVERGING_CVD`
(purple–orange) is the better choice under protanopia and is available for
new figures.

## Tests

```bash
pytest
```

Synthetic data with known properties, asserting the invariants that matter:
patches tile a raster exactly once at any size, a tile never appears in two
splits, season weighting equalises the three stages whatever the input
imbalance, and a canopy 10 m above a 100 m hill measures 10 m. No download
required.

## License

Code is Apache-2.0 — see `LICENSE`. The map product is CC-BY-4.0 and is
published on Zenodo, not here. Third-party layers keep their own terms:
CORINE and the Copernicus HRL Tree Cover Density are Copernicus data, the
OSM land polygons are ODbL, and GADM 4.1 permits academic use but not
redistribution — so it is referenced, never shipped.

## Citation

`CITATION.cff` will be added once the DOI is issued.
