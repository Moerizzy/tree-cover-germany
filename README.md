# Tree cover of Germany at 20 cm

Reference implementation for

> Lucas, M., Brandt, M., Waske, B. *Overcoming seasonal heterogeneity in
> national aerial surveys: 20 cm resolution tree cover mapping of Germany.*
> Submitted to Remote Sensing of Environment.

A SegFormer model trained on openly licensed orthophotos from a **single**
federal state produces a wall-to-wall tree cover map of Germany at 20 cm.
The key ingredient is a season-aware weighted sampling strategy that covers
leaf-on, transition and leaf-off conditions during training. Tested against
LiDAR reference data in three other states the model reaches a tree-class
IoU of 0.844 and F1 of 0.892. Germany's total tree cover, including trees
outside forests, is estimated at 32.3 % (≈ 115,500 km²).

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

`paths.yaml` is gitignored. Any single value can also be overridden from the
environment, which is handy on a cluster:

```bash
TREECOVER_PREDICTIONS_ROOT=/mnt/scratch/Germany python scripts/06_tile_statistics.py --all
```

## Pipeline

Each stage is a standalone CLI; run `--help` on any of them.

| Stage | Script | What it does |
|------:|--------|--------------|
| 1 | `01_sample_tiles.py` | Stratified selection of training tiles (TCD × settlement × season) |
| 2 | `02_download_dataset.py` | Fetch the orthophotos and height models, and build the nDSM |
| 3 | `03_prepare_patches.py` | Resolution alignment and sliding-window patch extraction |
| 4 | `04_train.py` | SegFormer training with season-aware weighted sampling |
| 5 | `05_inference.py` | Moving-window inference across a state |
| 6 | `06_tile_statistics.py` | Tree cover area per tile, masked to land and state borders |
| 7 | `07_validate.py` | Validation against LiDAR — three subcommands, below |
| 8 | `08_merge_reproject.py` | 1 km tiles → 10 km GeoTIFFs in EPSG:3857, for display |
| 9 | `09_compare_products.py` | Comparison against other products; the Results tables |
| 10 | `10_extract_tile_products.py` | Per-tile cover of every comparison product, on the common grid |
| 11 | `11_export_mosaic.py` | 1 km tiles → 10 km GeoTIFFs in native UTM, lossless; the archival product |
| 12 | `12_coverage_polygons.py` | One polygon per acquisition date, shipped beside the mosaic |

Two scripts are not stages. `prepare_tile_table.py` is the converter
between 10 and 9: it attaches the centroids, areas and land areas the
aggregation weights by. `collect_release.py` checks the published data
directory against the inventory in
[`treecover.release`](src/treecover/release.py) and writes its manifest.

Stages 8 and 11 both merge, and the difference matters. Stage 8 warps to
Web Mercator for display; areas must never be computed from its output,
which inflates them ~2.5× at Germany's latitude. Stage 11 copies tiles
into their windows without resampling — the predictions already sit on the
20 cm UTM grid — so a pixel of its output is a pixel of the model, and
counting them gives real areas. It needs no GDAL binaries, only rasterio.

### Tile sampling

Stage 1 draws the tiles to label. Eligibility is decided by four hard
constraints — both seasons flown, a non-summer flight *before or after* the
summer one rather than a second summer flight, 2 km minimum separation, and
at most five tiles per flight date per stratum — and the draw then fills a
4 × 2 grid of density bins × settlement type, taking one tile at a time from
whichever stratum is furthest below its target.

```bash
python scripts/01_sample_tiles.py --out results/sampling \
    --exclude .../gadm41_DEU.gpkg --exclude-layer ADM_ADM_1 \
    --exclude-where "NAME_1 == 'Bremen'" \
    --save-attributes results/sampling/attributes.gpkg
```

Reading the two auxiliary rasters is the slow part, so `--save-attributes`
caches the per-tile table and `--attributes` reuses it: re-drawing with
different targets then takes seconds instead of minutes.

The published run asked for 200 tiles and got 152 — the separation
constraint exhausted the strata first, which is why the training package
holds 152 and not 200. **The draw is not bit-reproducible**: it walks the
candidate table in order under a fixed numpy seed, so a tile index that has
gained flights since 2024 yields a different, equally valid set. `--compare`
therefore reports two numbers, and the second is the one that matters:

```
Against sampled_tiles_100.gpkg: 23 of 152 tiles in common (15 %), 152 of them eligible here
```

All 152 published tiles still pass every filter, so the selection rules
reproduce; only the random draw lands elsewhere, on a pool that has grown
from what it was. A reference tile that the filters *reject* would be the
real signal — that would mean a constraint here is stricter than the one
that produced the published set — and the run says so explicitly.

### Fetching the imagery

Stage 2 turns tile ids into rasters on disk, named `<tile_id>_<date>.tif`,
plus the nDSM the labels were digitised with — the image-based surface
model minus the LiDAR ground model. URLs come from one of two sources, and
the difference decides what you can build:

```bash
# easygeodata.de: one bbox query per tile, all sixteen states, nothing local
python scripts/02_download_dataset.py --out data/Sampling

# the state's own index: every flight over a tile, not just the newest
python scripts/02_download_dataset.py --source index --out data/Sampling \
    --index-dop  .../lgln-opengeodata-dop20.geojson \
    --index-bdom .../lgln-opengeodata-bdom20.geojson \
    --index-dgm  .../lgln-opengeodata-dgm1.geojson
```

[easygeodata.de](https://easygeodata.de) indexes the open geodata of every
state behind one API and needs no local index, which makes it the right
route for a fresh area. **It serves the current acquisition of a tile
only.** The training set is built from *pairs* — one summer image and one
outside it, sharing a label — so reproducing it needs `--source index`.
A run that cannot reach the dates the tile table records says so at the
end rather than leaving it to surface in stage 3.

Two things the stage insists on. An orthophoto is only kept when the
surface model of that tile carries the **same date**: the two products are
published separately, so asking for the newest of each can pair a 2026
image with a 2025 height model, and a label drawn from that pair is wrong
in both. And in states publishing orthophotos on a 2 km grid — Lower
Saxony is one — the download is cropped to the 1 km tile, so the patch
extractor cannot read imagery its label mask does not cover.

Label masks are not downloadable: they were drawn by hand and are
published with the paper's training-data package.

### Validation

`07_validate.py` has four subcommands rather than one run, because two
hard boundaries sit between them:

```bash
python scripts/07_validate.py sample    --state BB --candidates candidates.gpkg
python scripts/07_validate.py reference --state BB
#   ← open tree_mask_footprints.geojson in QGIS, set exclude = 1 where the
#     ground changed between the LiDAR and the orthophoto
python scripts/07_validate.py score     --all-states
```

`sample` runs in minutes and must run **once**: re-drawing changes the
sample set and with it every reported number, so it refuses to overwrite an
existing set without `--force`. `reference` runs for hours against state
servers that drop connections, so it caches, resumes, and is expected to be
re-run. Between the two lies a manual inspection pass that no script can do.

The fourth, `summarise`, rebuilds the accuracy table from per-sample
metrics already on disk — no rasters, so the published numbers can be
checked without the terabytes behind them:

```bash
python scripts/07_validate.py summarise --metrics-dir publication/validation
```

A third of the validation boxes hold no reference tree at all, where IoU is
undefined, and what happens to them decides the headline. `--empty score`
(the default, and what was published) counts a box the model also leaves
empty as correct and one where it invents canopy as wrong: mean IoU 0.844,
F1 0.892. `--empty drop` measures the model only where trees are: 0.771 /
0.846. The rule lives in
[`validation.metrics.score_zero_reference`](src/treecover/validation/metrics.py)
so the table and the figures cannot disagree about it.

Paper figures are reproduced by the scripts in `figures/`.

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

Everything that differs between federal states — download URL templates,
GADM keys, CRS, the `NRW`/`NW` naming split — lives in
[`configs/states.yaml`](configs/states.yaml) and nowhere else.

## Class codes

The map is binary: `0` background, `1` tree, `255` nodata. Defined once in
[`src/treecover/constants.py`](src/treecover/constants.py); import from
there rather than re-declaring the numbers.

`validate_prediction_codes()` rejects rasters holding anything else. Codes
3–6 belong to the trees-outside-forests classification of Lucas et al.
(2025), which is **not part of this repository** — reading such a raster
here would count those classes as background and quietly deflate every
tree-cover figure.

## Overlap resolution when merging

About 2.5 % of 1 km cells — 9,679 of 370,515 in the archive — are covered by
more than one prediction tile, almost always because two states flew the
same ground at a border.

The original code left the choice to `gdalbuildvrt` source ordering, and
three revisions sorted the file list three different ways: alphabetically by
path, by file modification time, and by acquisition date. Applied to the
real archive the three disagree about every one of the 9,679 contested
cells.

The manuscript prescribes no rule — its *Nationwide mapping* section says
nothing about merging. Only one of the three is a property of the imagery
though, so there is now one rule: **newest acquisition date wins**, with a
deterministic tiebreak. The choice is made in
[`src/treecover/merge.py`](src/treecover/merge.py) **before GDAL is
invoked**, so VRT source ordering decides nothing. `--report` writes a CSV
of every contested cell and what was dropped.

## Matching the manuscript

Three settings come straight from the *Nationwide mapping* and *Comparison
to reference products* sections and are pinned by tests, so a refactor
cannot drift away from what was published:

| Setting | Value | Where |
|---|---|---|
| Sampling strata / separation | 4 TCD bins × settlement, 2 km apart | `tile_sampling.PUBLISHED_BIN_TARGETS`, `PUBLISHED_SETTINGS` |
| Training patch size / stride | 512 / 512 px — non-overlapping | `03_prepare_patches.PUBLISHED_STRIDE` |
| Training split | from `observations.csv`, not the GeoPackage column | `03_prepare_patches.apply_published_splits` |
| Patch size / stride / margin | 512 / 360 / 76 px | `inference.tiling` |
| Neighbourhood context per tile | 256 px | `inference.sources.CONTEXT_PX` |
| Common comparison grid | 1 km | `comparison.GRID_DLON/DLAT` |

Stage 3 run with no arguments rebuilds the published training set exactly —
117 tiles, 245 observations, 19,845 patches (15,957 train / 3,888 val) — and
says whether it did. Both training settings above have a plausible wrong
value that produces a perfectly valid patch table nobody would question: a
256 stride quadruples the training set, and the GeoPackage's `split` column
moves published training tiles into validation. Neither is detectable
downstream, which is why the run checks itself.

Those counts also give the season weights the manuscript quotes: leaf-off
3,159, transition 5,184 and leaf-on 7,614 training patches weight to 1.68,
1.03 and 0.70.

The stride equals the kept inner region, so patches tile a tile exactly —
no gaps, no overlaps. The 256 px halo is read from neighbouring imagery,
passed to the model, and cropped away before the prediction is written;
without it, patches at a tile border have context on one side only and the
merged map shows seams along the tile grid.

## Area statistics

`06_tile_statistics.py` produces the per-tile table the comparison figures
and the national totals are built from. It masks each tile to the
intersection of the OSM land polygons and the tile's own GADM state border
before counting.

That masking is not cosmetic. Without it, tiles straddling a state border
are counted twice — once for each state — and coastal tiles count open
water as treeless land. Both inflate the denominator. The paper's figures
are computed with masking on.

**Which number, over which area.** Three figures, one measurement:

| route | reference area | cover | tree |
|---|---|---|---|
| pixel count, all tiles, masked | 356,381 km² mapped land | 32.33 % | 115,202 km² |
| Table 1, common product baseline | 350,435 km² | 32.31 % | 113,231 km² |
| the paper's headline | 357,596 km², all of Germany | 32.3 % | ≈ 115,500 km² |

Mapped land is 99.66 % of Germany; the 1,215 km² missing are absent or
corrupt tiles, concentrated in SL, HB, SH and MV. Table 1 is narrower
still, because a product only enters over tiles where our map is valid
too. The headline applies the measured fraction to Germany's total area,
and the caption of Table 1 says so.

Quote a percentage without its reference area and the two stop matching —
that is exactly how an earlier draft came to pair 32.21 % with
120,943 km², two figures implying reference areas 5 % apart.

**Weights.** Aggregation prefers `land_area_km2`, the land measured inside
each tile; `prepare_tile_table.py --land-areas` attaches it from the
per-tile statistics. Do not fall back to a `tile_area_km2` derived from
the lon/lat bounding box: that is the envelope of a UTM square, inflated a
mean 4.5 % by meridian convergence. Percentages survive it — they are
ratios of the same weights — but any area computed as percentage × weight
absorbs all of it.

## Data

### The training-data package

128 MB, published alongside the paper. Point `training_data.root` in
`paths.yaml` at it and stage 3 and figure 3 run with no further arguments.

```
sampled_tiles_100.gpkg    152 tiles, EPSG:25832, with the sampling strata
labels/                   152 label masks, 5000 x 5000 px at 20 cm, 0/1/255
patches/                  observations.csv, patches_metadata.csv, split_info.json,
                          region_vrts.json, experiment_config.json
logs/                     which tile-date came from which URL, and when
```

The label masks are the part that cannot be obtained anywhere else. The
orthophotos and height models they were drawn on are **not** in the package:
71 GB of openly licensed LGLN data, re-downloadable per tile, with every URL
recorded in `logs/`.

> **The `split` column in the GeoPackage is not the published split.** It is
> an earlier three-way draw — train 73 / val 23 / test 23 — and its classes
> cut across both published ones. Using it puts published *training* tiles
> into validation and inflates every metric measured afterwards, silently.
> The real assignment is in `patches/observations.csv`, which
> `03_prepare_patches.py` reads by default.

Not included in the repository:

- **Model weights** — Zenodo (DOI pending). The B5 checkpoint is 323 MB.
- **Tile indices** — per-state orthophoto and LiDAR footprints, ~172 MB.
  Stage 2 works without them through the easygeodata API, at the cost of
  the historical acquisitions.
- **Orthophotos** — openly licensed, from the state surveying authorities.
  Endpoints for all 16 states are recorded in `configs/states.yaml`.
- **Auxiliary layers** — CORINE Land Cover 2018 and Copernicus HRL Tree
  Cover Density (Copernicus Land Monitoring Service), Naturräume (BfN),
  SRTM, GADM 4.1, OSM land polygons and building footprints.

### The published data directory

Everything the paper releases sits in one directory — the map, the
checkpoint, the training and validation data, the per-tile tables and the
figures. What belongs in it is an inventory in
[`treecover.release`](src/treecover/release.py) rather than a paragraph
somewhere, because a paragraph drifts: the release README claimed the
weights were on another machine for two days after they had been copied in,
and nothing noticed.

```bash
python scripts/collect_release.py --root /path/to/publication
```

It walks the inventory, sizes directories, hashes files under the limit,
writes `MANIFEST.csv`, and reports three things: entries that are missing,
entries that are present but empty, and files that are there without being
listed. `--add SRC=DEST` copies stragglers in, `--prune` clears
`.ipynb_checkpoints` and interrupted downloads. It fixes nothing else — a
figure that was never rendered has to be rendered by its own script, and a
placeholder here would hide that.

## Tests

```bash
pytest
```

The tests build synthetic data with known properties and assert the
invariants that matter: patches tile a raster exactly once at any size, a
tile never appears in two splits, season weighting equalises the three
phenological stages whatever the input imbalance, and a canopy 10 m above a
100 m hill measures 10 m rather than 110. No data download required.

## License

Not yet chosen — see the open questions below. Apache-2.0 is suggested for
the code and CC-BY-4.0 for the map product.

## Citation

`CITATION.cff` will be added once the DOI is issued.

---

## Figures

`figures/` holds one script per paper figure; each runs on its own and
writes PNG (600 dpi) plus PDF. Script numbers match the manuscript's
figure numbers.

| Figure | Script | Input |
|---:|---|---|
| 1 | `fig01_acquisition_coverage.py` | the prediction archive (filenames only) |
| 2 | — | QGIS composition |
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
python figures/fig06_example_tiles.py          --masks-root .../lidar_masks --predictions-root /tf/Germany
python figures/fig07_scatter_products.py       --tiles tiles_with_treecover.csv
python figures/fig08_product_comparison.py     --tiles tiles_with_treecover.csv
python figures/fig09_relative_difference.py    --tiles tiles_with_treecover.csv
python figures/fig10_local_comparison.py       --products-root /tf/Other_Tree_Products --predictions-root /tf/Germany
```

Figure 2 is a QGIS composition and is not scripted.

**Figure 1** derives tile position and date from the **filename**, so it
scans 380,000 names rather than opening 380,000 rasters — minutes instead of
an hour. `--save-coverage` caches the table. It uses the same per-cell
selection as the merge, so the date map and the merged raster cannot
disagree about which acquisition covers a place.

**Figure 3** shows what a training tile actually is: not one image but a
*pair* of acquisitions of the same ground sharing one label. Columns are the
summer orthophoto the label was drawn on, the nDSM that supported the
drawing, the label itself, and the non-summer acquisition that inherits it.

Two rules decide which tiles are drawn, both in
[`treecover.figures.training_examples`](src/treecover/figures/training_examples.py).
The label source comes from the same `label_source_index()` the training set
was built with, so the figure cannot caption an image as the label source
that the pipeline never labelled from. The partner is the acquisition at the
greatest *phenological* distance, not the nearest in time — a leaf-off
partner six months away says more than a transition one six weeks away.
Tiles whose two acquisitions fall in the same season are skipped, since they
cannot illustrate the pairing; `--allow-same-season` keeps them.

The three published tiles are **pinned** in `PUBLISHED_TILES`, for the
reason figure 10 pins its scenes: the automatic selection depends on the
whole candidate set, so one tile added to or removed from the training table
would silently republish a different figure. `--auto-select` re-runs the
selection, `--list-candidates` prints all 72 eligible tiles with their cover,
stratum and acquisition pair, and `--tile-ids` overrides the choice. The
published rows are an urban village (18 % canopy), hedgerows around a field
copse (31 %) and closed forest (63 %).

The selection behind `--auto-select` is stratified the way the sample was —
tree cover density **and** settlement type. `--urban-rows` (1 by default)
reserves rows for the urban stratum; ranking on cover alone returns only
rural tiles, because urban tiles are a low-cover minority of the sample and
never reach a bin centre of the whole set. A shortfall of urban candidates
goes back to the rural rows, so reserving a row never costs one.

Within each stratum, rows are sampled at bin centres of the cover gradient
rather than its endpoints. The endpoints of this training set are a 2 %
heath tile whose nDSM and label panels are both blank and a closed-canopy
tile whose label is a solid green square; neither shows a label being made.
Within two ranks of each target, three tiebreaks apply in order: a flight
date no other row uses, then the widest seasonal contrast, then closeness to
the target. Neighbouring tiles were often sampled from the same two
acquisitions, so without the first the figure can show one flight pair twice
and read as though the archive held a handful of dates. All three move
canopy share by a fraction of a point, so the gradient survives them.

The nDSM is **image-based** (`dom1` − `dgm1`) and the labels were digitised
by hand with it and the near-infrared band as aids — not thresholded out of
it. That is why no 3 m contour is drawn on the height panel, and why the
height ramp is purple: an nDSM is just as high over a roof as over a crown,
and a green ramp would read as canopy already found. The thresholded LiDAR
CHM is figure 6, not this one.

**Figure 6** draws the four boxes of the published figure, named in
`PUBLISHED_SAMPLES` — the notebook chose them by hand and no ranking
reproduces them. The figure itself carries no sample ids, only acquisition
dates, and those dates are what identifies the boxes: BY #82, BY #50,
NW #166 and NW #84 resolve to 2025-08-13, 2025-05-11, 2025-04-06 and
2025-04-07, in that order. Rendered against the archive this reproduces the
published PNG pixel for pixel. `--samples` names others and `--pick-by-iou 3`
falls back to a best/median/worst draw per state. Imagery is looked up
beside the prediction it produced — the folder is `RGB` in the five states
publishing three bands and `RGBI` in the other eleven, and both are
searched.

**Figure 10** shows each product at its *native* resolution rather than on
the common grid, which is what makes the resolution dependence visible.
A scene is the centre 500 m of a 1 km tile. The three published scenes are
named in `PUBLISHED_TILES`; the notebook picked them at random under seed
42, and `--draw-seed 42` reproduces that draw — but only while the archive
holds exactly the same 380,213 tiles, which is why the names are pinned
rather than the seed. Unlike the published version, every layer here is
**binary**: the two height products at 3 m, TCD at `--density-threshold`
(50 % by default).

Styling is centralised in
[`src/treecover/figures/style.py`](src/treecover/figures/style.py).
`PRODUCT_COLORS` and `PRODUCT_LABELS` are keyed by *column*, so every
figure and the stage-9 tables use the same hue and the same name for a
product — a renamed caption cannot silently change a colour. Categorical
hues are Okabe–Ito in fixed order; the three comparison products use the
orange / blue / sky-blue triple and deliberately avoid the vermillion +
orange pair, which is the one weak combination in that palette. Tree cover
uses a single sequential green ramp; differences use a diverging ramp with
a neutral midpoint pinned at zero.

The diverging default is red–blue, matching the submitted manuscript.
`DIVERGING_CVD` (purple–orange) is the better choice under protanopia and
is available for new figures.

## Season-aware sampling

The paper's central mechanism, in
[`src/treecover/data/seasons.py`](src/treecover/data/seasons.py). Aerial
surveys are dominated by summer acquisitions, so a uniformly drawn training
set teaches the model "tree = green blob" and it fails on bare winter
canopy. Weighting each patch by the inverse frequency of its phenological
stage makes all three equally likely per epoch without discarding data:

```
counts  = {leaf_on: 36, transition: 27, leaf_off: 18}
weights = {leaf_on: 0.75, transition: 1.0, leaf_off: 1.5}
        → 27 expected draws per season
```

Reproduce the ablation that motivates it with
`04_train.py --no-season-weighting`.

## Status

Every stage of the pipeline is ported and tested end to end, along with the
configuration, class codes, tile discovery, model construction and figures
1 and 3–10. Figure 2 is a QGIS composition and is the one thing no script
reproduces.

Verified on the inference container: 493 tests pass, stage 3 → stage 4 runs
as a chain, tile statistics run on real Bavarian tiles, and a checkpoint
written by stage 4 loads into stage 5's predictor. Figure 3 was rendered
from the real Lower Saxony training set — 152 tiles, 351 acquisitions, 152
label masks. Stage 1 ran against the real bDOM index (103,071 acquisitions
over 52,544 tiles) in 95 seconds, and every one of the 152 published
training tiles passes its filters. Stage 2 fetched a tile's orthophoto,
surface model and ground model and built its nDSM — canopy heights of
−1.9 to 32.9 m over a 20 cm grid — and resolved, from the state index,
exactly the acquisition pairs the published training set is built from.

### Fixed during the port

**Inference left unpredicted strips.** `plan_patches` strode to the end of a
tile and dropped any final patch shorter than half a patch. Where
`size % stride` fell between the stride and half a patch, that left a strip
with no prediction at all, which came out as background — a 920-pixel tile
lost pixels 796–920. The last patch is now anchored against the far edge and
inner regions are chained, so any size is covered exactly once. The standard
5000-pixel tile was never affected; VRT windows at a mosaic edge were.

**Training could not run on CPU.** `torch.autocast` validates its dtype on
`__enter__` even when `enabled=False`, and CPU autocast rejects float16, so
`--device cpu` crashed regardless of `--no-amp`. That made smoke-testing the
pipeline without a GPU impossible.

**Hole filling could not be switched off.** `chm_to_tree_mask` floored the
threshold at one pixel, so `max_hole_m2=0` still filled single-pixel holes
and the raw thresholded mask was unobtainable. The 10 m² default is
unchanged, so published results are unaffected.

**Tile geometry did not match the archive.** The grid-cell derivation
assumed `32660_5261`; real names are
`dop20rgb_32_660_5261_by_file_20240730`, and cells are named in units of
100 m. Everything would have landed in one `misc` directory.

**GeoPandas 0.13 + Fiona 1.10 cannot read any vector file** — the pairing on
the inference container. Centralised in `treecover.io.vector`, which prefers
`pyogrio` and reports the actual fix instead of an `AttributeError` from
inside GeoPandas.
