#!/usr/bin/env python3
"""Create an offline tiled COCO instance segmentation dataset.

This is intended for high-resolution dense-object datasets where resizing the
full image makes small instances too tiny for the detector. It preserves the
existing train/val split and clips polygon annotations into each tile.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-root",
        default="/home/nikhileswara/Datasets/oweed_structure_v0_100_images_coco",
        help="Source COCO root containing train2017, val2017 and annotations/.",
    )
    parser.add_argument(
        "--out-root",
        default="/home/nikhileswara/Datasets/oweed_structure_v0_100_images_coco_tiled_1024_s512",
        help="Output tiled COCO root.",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--min-area",
        type=float,
        default=16.0,
        help="Discard clipped annotation fragments smaller than this tile-local area.",
    )
    parser.add_argument(
        "--min-visible-ratio",
        type=float,
        default=0.05,
        help="Discard clipped fragments with less than this fraction of original area.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved tiles.",
    )
    parser.add_argument(
        "--include-empty-train-tiles",
        action="store_true",
        help="Keep train tiles with no annotations. Val tiles are always kept.",
    )
    return parser.parse_args()


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_polygons(geom) -> Iterable[Polygon]:
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for part in geom.geoms:
            yield from iter_polygons(part)


def make_valid_polygon(points: list[tuple[float, float]]):
    if len(points) < 3:
        return None
    geom = Polygon(points)
    if geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    return geom


def segmentation_to_geometry(segmentation):
    if not isinstance(segmentation, list):
        raise ValueError("Only polygon COCO segmentations are supported by this tiler.")

    polygons = []
    for poly in segmentation:
        if len(poly) < 6:
            continue
        points = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
        geom = make_valid_polygon(points)
        if geom is not None:
            polygons.append(geom)

    if not polygons:
        return GeometryCollection()
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons).buffer(0)


def polygon_to_coco(poly: Polygon, tile_x: int, tile_y: int, tile_size: int) -> list[float] | None:
    coords = []
    for x, y in list(poly.exterior.coords)[:-1]:
        lx = min(max(float(x) - tile_x, 0.0), float(tile_size))
        ly = min(max(float(y) - tile_y, 0.0), float(tile_size))
        coords.extend([lx, ly])

    if len(coords) < 6:
        return None
    unique_points = {(round(coords[i], 3), round(coords[i + 1], 3)) for i in range(0, len(coords), 2)}
    if len(unique_points) < 3:
        return None
    return coords


def bbox_intersects_tile(bbox_xywh, tile_x: int, tile_y: int, tile_size: int) -> bool:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return not (
        x + w <= tile_x
        or y + h <= tile_y
        or x >= tile_x + tile_size
        or y >= tile_y + tile_size
    )


def process_split(
    split: str,
    src_root: Path,
    out_root: Path,
    tile_size: int,
    stride: int,
    min_area: float,
    min_visible_ratio: float,
    jpeg_quality: int,
    include_empty_train_tiles: bool,
) -> dict:
    ann_path = src_root / "annotations" / f"instances_{split}2017.json"
    image_dir = src_root / f"{split}2017"
    out_image_dir = out_root / f"{split}2017"
    out_ann_path = out_root / "annotations" / f"instances_{split}2017.json"

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)

    with ann_path.open() as f:
        coco = json.load(f)

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    out_images = []
    out_annotations = []
    next_image_id = 1
    next_ann_id = 1
    generated_tiles = 0
    kept_tiles = 0
    skipped_empty_train_tiles = 0
    skipped_fragments = 0

    for image_info in coco["images"]:
        image_path = image_dir / image_info["file_name"]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            x_starts = tile_starts(width, tile_size, stride)
            y_starts = tile_starts(height, tile_size, stride)

            image_annotations = anns_by_image[image_info["id"]]
            geom_cache = {}

            for tile_y in y_starts:
                for tile_x in x_starts:
                    generated_tiles += 1
                    tile_geom = box(tile_x, tile_y, tile_x + tile_size, tile_y + tile_size)
                    tile_annotations = []

                    for ann in image_annotations:
                        if not bbox_intersects_tile(ann["bbox"], tile_x, tile_y, tile_size):
                            continue

                        if ann["id"] not in geom_cache:
                            geom_cache[ann["id"]] = segmentation_to_geometry(ann["segmentation"])

                        clipped = geom_cache[ann["id"]].intersection(tile_geom)
                        clipped_area = float(clipped.area)
                        original_area = max(float(ann.get("area", clipped_area)), 1e-6)
                        if clipped_area < min_area or clipped_area / original_area < min_visible_ratio:
                            skipped_fragments += 1
                            continue

                        segmentations = []
                        for poly in iter_polygons(clipped):
                            if float(poly.area) < min_area:
                                continue
                            coco_poly = polygon_to_coco(poly, tile_x, tile_y, tile_size)
                            if coco_poly is not None:
                                segmentations.append(coco_poly)

                        if not segmentations:
                            skipped_fragments += 1
                            continue

                        minx, miny, maxx, maxy = clipped.bounds
                        minx = min(max(float(minx) - tile_x, 0.0), float(tile_size))
                        miny = min(max(float(miny) - tile_y, 0.0), float(tile_size))
                        maxx = min(max(float(maxx) - tile_x, 0.0), float(tile_size))
                        maxy = min(max(float(maxy) - tile_y, 0.0), float(tile_size))
                        if maxx <= minx or maxy <= miny:
                            skipped_fragments += 1
                            continue

                        tile_annotations.append(
                            {
                                "id": next_ann_id,
                                "image_id": next_image_id,
                                "category_id": ann["category_id"],
                                "segmentation": segmentations,
                                "bbox": [minx, miny, maxx - minx, maxy - miny],
                                "area": clipped_area,
                                "iscrowd": ann.get("iscrowd", 0),
                                "original_annotation_id": ann["id"],
                                "original_image_id": image_info["id"],
                                "tile_x": tile_x,
                                "tile_y": tile_y,
                            }
                        )
                        next_ann_id += 1

                    if split == "train" and not include_empty_train_tiles and not tile_annotations:
                        skipped_empty_train_tiles += 1
                        continue

                    stem = Path(image_info["file_name"]).stem
                    tile_file_name = f"{stem}_x{tile_x:04d}_y{tile_y:04d}.jpg"
                    tile = img.crop((tile_x, tile_y, tile_x + tile_size, tile_y + tile_size))
                    tile.save(out_image_dir / tile_file_name, quality=jpeg_quality)

                    out_images.append(
                        {
                            "id": next_image_id,
                            "file_name": tile_file_name,
                            "width": tile_size,
                            "height": tile_size,
                            "original_image_id": image_info["id"],
                            "original_file_name": image_info["file_name"],
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "tile_size": tile_size,
                            "stride": stride,
                        }
                    )
                    out_annotations.extend(tile_annotations)
                    next_image_id += 1
                    kept_tiles += 1

    out_coco = {
        "images": out_images,
        "annotations": out_annotations,
        "categories": coco["categories"],
        "tiled_dataset": {
            "source_root": str(src_root),
            "split": split,
            "tile_size": tile_size,
            "stride": stride,
            "min_area": min_area,
            "min_visible_ratio": min_visible_ratio,
        },
    }

    with out_ann_path.open("w") as f:
        json.dump(out_coco, f)

    return {
        "split": split,
        "generated_tiles": generated_tiles,
        "kept_tiles": kept_tiles,
        "skipped_empty_train_tiles": skipped_empty_train_tiles,
        "annotations": len(out_annotations),
        "skipped_fragments": skipped_fragments,
        "annotation_path": str(out_ann_path),
    }


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    summaries = []
    for split in ("train", "val"):
        summaries.append(
            process_split(
                split=split,
                src_root=src_root,
                out_root=out_root,
                tile_size=args.tile_size,
                stride=args.stride,
                min_area=args.min_area,
                min_visible_ratio=args.min_visible_ratio,
                jpeg_quality=args.jpeg_quality,
                include_empty_train_tiles=args.include_empty_train_tiles,
            )
        )

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
