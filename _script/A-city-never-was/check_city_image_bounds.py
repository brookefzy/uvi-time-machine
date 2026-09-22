#!/usr/bin/env python3
"""Report coordinate bounds of on-disk GSV images, across all years.

Uses only the Python standard library. Reads files without modifying them.
Coordinates come from gsv_pano.csv; images are enumerated under img_rgb.
Exit status: 0 = complete, 2 = incomplete coverage or input error.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys


def check_bounds(city_dir):
    metadata = city_dir / "gsvmeta" / "gsv_pano.csv"
    image_dir = city_dir / "img_rgb"
    if not image_dir.is_dir():
        raise ValueError(f"Image directory not found: {image_dir}")

    coordinates = {}
    seen_ids = set()
    conflicts = set()
    invalid_rows = 0
    with metadata.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"panoid", "lat", "lon"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{metadata}: expected columns {sorted(required)}; "
                             f"found {reader.fieldnames}")
        for row in reader:
            panoid = (row["panoid"] or "").strip()
            seen_ids.add(panoid)
            try:
                lat, lon = float(row["lat"]), float(row["lon"])
                if not panoid or not (math.isfinite(lat) and math.isfinite(lon)
                                      and -90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError("Invalid coordinates")
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
            pair = (lat, lon)
            if panoid in coordinates and coordinates[panoid] != pair:
                conflicts.add(panoid)
            coordinates[panoid] = pair

    counts = {"image_files": 0, "matched_images": 0,
              "missing_metadata_images": 0, "invalid_coordinate_images": 0,
              "conflicting_coordinate_images": 0}
    matched_ids = set()
    unresolved_examples = []
    extrema = {}

    def walk_error(error):
        # Never silently report complete coverage after skipping a directory.
        raise error

    for folder, _, filenames in os.walk(image_dir, onerror=walk_error):
        for name in filenames:
            if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
                continue
            path = Path(folder) / name
            if not path.is_file():
                raise ValueError(f"Image disappeared or is not a regular file: {path}")
            counts["image_files"] += 1
            # Same convention as the repository's download and embedding scripts.
            panoid = name[:22]
            reason = None
            if panoid in conflicts:
                reason = "conflicting_coordinate_images"
            elif panoid not in seen_ids:
                reason = "missing_metadata_images"
            elif panoid not in coordinates:
                reason = "invalid_coordinate_images"
            if reason:
                counts[reason] += 1
                if len(unresolved_examples) < 10:
                    unresolved_examples.append({"path": str(path), "reason": reason})
                continue
            lat, lon = coordinates[panoid]
            counts["matched_images"] += 1
            matched_ids.add(panoid)
            record = {"lat": lat, "lon": lon, "panoid": panoid, "path": str(path)}
            for key, value, axis, minimum in (
                ("south", lat, "lat", True), ("north", lat, "lat", False),
                ("west", lon, "lon", True), ("east", lon, "lon", False),
            ):
                if key not in extrema or (value < extrema[key][axis] if minimum
                                          else value > extrema[key][axis]):
                    extrema[key] = record

    bounds = None
    if extrema:
        bounds = {"latitude_min": extrema["south"]["lat"],
                  "latitude_max": extrema["north"]["lat"],
                  "longitude_min": extrema["west"]["lon"],
                  "longitude_max": extrema["east"]["lon"]}
    complete = counts["image_files"] > 0 and counts["matched_images"] == counts["image_files"]
    return {"city_directory": str(city_dir), "metadata": str(metadata),
            "scope": "All image files under img_rgb, all years; metadata coordinates",
            "complete": complete, **counts, "matched_unique_panoramas": len(matched_ids),
            "invalid_metadata_rows": invalid_rows,
            "conflicting_metadata_panoramas": len(conflicts),
            "bounds_of_matched_images": bounds, "extreme_image_examples": extrema,
            "unresolved_examples": unresolved_examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/lustre1/g/geog_pyloo/05_timemachine"))
    parser.add_argument("--city", default="Malegaon")
    args = parser.parse_args()
    city_dir = args.root / "GSV" / "gsv_rgb" / args.city.lower().replace(" ", "")
    try:
        report = check_bounds(city_dir)
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["complete"]:
        print("INCOMPLETE: bounds cover only matched images; inspect the counts and examples.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
