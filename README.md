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


## License

Code is Apache-2.0 — see `LICENSE`. The map product is CC-BY-4.0 and is
published on Zenodo, not here. Third-party layers keep their own terms:
CORINE and the Copernicus HRL Tree Cover Density are Copernicus data, the
OSM land polygons are ODbL, and GADM 4.1 permits academic use but not
redistribution — so it is referenced, never shipped.

## Citation

`CITATION.cff` will be added once the DOI is issued.
