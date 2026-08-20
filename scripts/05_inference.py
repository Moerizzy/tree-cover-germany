#!/usr/bin/env python3
"""Run tree cover inference over a state.

Replaces ``04_run_inference.py``, ``_bavaria.py``, ``_bavaria_server.py``
and ``_nrw.py``. Those differed only in where the imagery came from, which
is now the ``--source`` option:

``local``
    Orthophotos already on disk, one raster per tile. Formats and
    resolutions may vary; everything is resampled to ``--resolution``.

``vrt``
    A single VRT mosaic, cut into ``--tile-px`` windows on the fly.

``index``
    A GeoJSON tile index with a download-URL column. Tiles are fetched over
    HTTP as needed, optionally with a matching nDOM height model.

Examples::

    # Bavaria from bulk-downloaded rasters
    python scripts/05_inference.py --state BY --source local \\
        --input /data/orthos/BY --checkpoint weights/segformer_b5_....pth

    # NRW from its tile index, 5-channel RGBI+nDSM model
    python scripts/05_inference.py --state NW --source index \\
        --input nrw_ortho.geojson --url-column "asset.dop10rgbi.href" \\
        --ndsm-url-column "asset.ndom50.href" --channels 5 \\
        --checkpoint weights/segformer_b5_rgbi_ndsm.pth

    # Smoke test: three tiles, CPU, no AMP
    python scripts/05_inference.py --state BY --source local --input /data/orthos/BY \\
        --checkpoint weights/model.pth --limit 3 --device cpu --no-amp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from treecover.config import load_paths, load_states
from treecover.inference import (
    HttpTileIndexSource,
    InferenceConfig,
    InferenceRun,
    LocalRasterSource,
    VrtSource,
    run_inference,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    src = p.add_argument_group("input")
    src.add_argument("--state", required=True,
                     help="State code, e.g. BY or NW (NRW is accepted as an alias).")
    src.add_argument("--source", choices=("local", "vrt", "index"), default="local",
                     help="Where imagery comes from.")
    src.add_argument("--input", type=Path, required=True,
                     help="Directory (local), VRT file (vrt), or tile index (index).")
    src.add_argument("--url-column", default=None,
                     help="index: column holding the orthophoto URL. Defaults to the "
                          "state's ortho.url_column from states.yaml.")
    src.add_argument("--id-column", default="id", help="index: tile id column.")
    src.add_argument("--date-column", default="datetime", help="index: date column.")
    src.add_argument("--ndsm-url-column", default=None,
                     help="index: nDOM URL column, for models with a height channel.")
    src.add_argument("--tile-px", type=int, default=5000, help="vrt: window size in pixels.")
    src.add_argument("--limit", type=int, default=None, help="Stop after N tiles.")

    model = p.add_argument_group("model")
    model.add_argument("--checkpoint", type=Path, required=True, help=".pth weights.")
    model.add_argument("--backbone", default=None,
                       help="Override the backbone inferred from the checkpoint name.")
    model.add_argument("--channels", type=int, default=3,
                       help="Input bands: 3 RGB (published model), 4 RGBI, 5 RGBI+nDSM.")
    model.add_argument("--resolution", type=float, default=0.20,
                       help="Target ground resolution in metres.")

    run_g = p.add_argument_group("execution")
    run_g.add_argument("--out", type=Path, default=None,
                       help="Output root (default: predictions_root from paths.yaml).")
    run_g.add_argument("--subdir", default="predictions",
                       help="Level between year and tile. Use e.g. predictions_test "
                            "to try settings without touching real output.")
    run_g.add_argument("--patch-size", type=int, default=512)
    run_g.add_argument("--inner-fraction", type=float, default=0.7,
                       help="Fraction of each patch kept when stitching.")
    run_g.add_argument("--batch-size", type=int, default=16,
                       help="Patches per forward pass. 16 is the original default; the "
                            "model runs in eval mode with LayerNorm, so this changes "
                            "throughput, not results.")
    run_g.add_argument("--workers", type=int, default=0,
                       help="DataLoader worker processes prefetching patches. 0 keeps "
                            "everything in the main process — the original default, and "
                            "the only one that works in the inference container, whose "
                            "/dev/shm is Docker's 64 MB. Workers pass tensors through "
                            "shared memory and die with 'No space left on device'.")
    run_g.add_argument("--device", default="cuda")
    run_g.add_argument("--no-amp", dest="amp", action="store_false",
                       help="Disable float16 autocast.")
    run_g.add_argument("--overwrite", action="store_true",
                       help="Redo tiles whose prediction already exists.")
    run_g.add_argument("--write-uncertainty", action="store_true",
                       help="Also write a float32 uncertainty raster per tile.")
    run_g.add_argument("--no-progress", action="store_true")
    run_g.add_argument("-v", "--verbose", action="store_true")
    return p


def make_source(args, state_code: str):
    """Build the TileSource matching --source."""
    common = {"state": state_code, "n_channels": args.channels, "target_res_m": args.resolution}

    if args.source == "local":
        if not args.input.is_dir():
            raise SystemExit(f"error: --input must be a directory for --source local: {args.input}")
        return LocalRasterSource(args.input, **common)

    if args.source == "vrt":
        if not args.input.is_file():
            raise SystemExit(f"error: --input must be a VRT file: {args.input}")
        return VrtSource(args.input, tile_px=args.tile_px, **common)

    if not args.input.is_file():
        raise SystemExit(f"error: --input must be a tile index file: {args.input}")

    url_column = args.url_column or load_states()[state_code].ortho.get("url_column")
    if not url_column:
        raise SystemExit(
            f"error: no URL column for {state_code}. Pass --url-column, or add "
            "ortho.url_column to configs/states.yaml."
        )
    return HttpTileIndexSource(
        args.input,
        url_column=url_column,
        id_column=args.id_column,
        date_column=args.date_column,
        ndsm_url_column=args.ndsm_url_column,
        **common,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if not args.checkpoint.exists():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    state_code = load_states().resolve(args.state)
    out_root = args.out or load_paths().get_path("predictions_root")

    config = InferenceConfig(
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        n_channels=args.channels,
        patch_size=args.patch_size,
        inner_fraction=args.inner_fraction,
        batch_size=args.batch_size,
        dataloader_workers=args.workers,
        device=args.device,
        amp=args.amp,
        target_resolution_m=args.resolution,
    )
    run = InferenceRun(
        out_root=out_root,
        config=config,
        subdir=args.subdir,
        overwrite=args.overwrite,
        write_uncertainty=args.write_uncertainty,
    )

    source = make_source(args, state_code)

    print(f"State      : {state_code}   Source: {args.source}   Input: {args.input}")
    print(f"Model      : {args.checkpoint.name}  ({args.channels} ch @ {args.resolution} m)")
    print(f"Output     : {out_root}/{state_code}/<year>/{args.subdir}/")

    reports = run_inference(source, run, limit=args.limit, progress=not args.no_progress)
    if not reports:
        print("No tiles matched.", file=sys.stderr)
        return 1

    ok = [r for r in reports if r.status == "ok"]
    by_status: dict[str, int] = {}
    for r in reports:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    print(f"\n{len(ok)}/{len(reports)} tiles predicted "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(by_status.items()))})")
    if ok:
        mean_cover = sum(r.tree_cover_pct or 0 for r in ok) / len(ok)
        print(f"Mean tree cover across tiles: {mean_cover:.2f} %")

    return 0 if by_status.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
