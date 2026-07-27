#!/usr/bin/env python3
"""Copy sampled images and render a side-by-side HTML preview gallery."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinov3_utils import resolve_city_file_stem


REQUIRED_COLUMNS = {
    "city_1", "name_1", "panoid_1", "lat_1", "lon_1",
    "city_2", "name_2", "panoid_2", "lat_2", "lon_2",
    "cosine_similarity",
}


def _load_image_paths(image_index_root: Path, cities: set[str]) -> dict[tuple[str, str], Path]:
    resolved: dict[tuple[str, str], Path] = {}
    for city in sorted(cities):
        index_path = image_index_root / f"{resolve_city_file_stem(city)}.parquet"
        if not index_path.exists():
            raise FileNotFoundError(f"image index is missing for {city!r}: {index_path}")
        frame = pd.read_parquet(index_path)
        if not {"name", "path"}.issubset(frame.columns):
            raise ValueError(f"{index_path} must contain name and path columns")
        for row in frame[["name", "path"]].drop_duplicates("name").itertuples(index=False):
            resolved[(city, str(row.name))] = Path(str(row.path))
    return resolved


def _copied_name(city: str, row_number: int, side: int, source: Path) -> str:
    return f"{row_number:04d}_{side}_{resolve_city_file_stem(city)}_{source.name}"


def _render_html(rows: list[dict[str, object]]) -> str:
    cards: list[str] = []
    maps: list[str] = []
    for row in rows:
        identifier = int(row["pair_number"])
        city_1, city_2 = html.escape(str(row["city_1"])), html.escape(str(row["city_2"]))
        image_1, image_2 = quote(str(row["image_1"])), quote(str(row["image_2"]))
        cards.append(f'''<article class="pair"><h2>{city_1} ↔ {city_2}</h2><p>Cosine similarity: {float(row["cosine_similarity"]):.4f}</p><div class="columns"><section><h3>{city_1}</h3><img src="images/{image_1}" alt="{city_1} sample"><p>{html.escape(str(row["panoid_1"]))}</p><div id="map-{identifier}-1" class="map"></div></section><section><h3>{city_2}</h3><img src="images/{image_2}" alt="{city_2} sample"><p>{html.escape(str(row["panoid_2"]))}</p><div id="map-{identifier}-2" class="map"></div></section></div></article>''')
        maps.append(
            f'''addMap("map-{identifier}-1", {float(row["lat_1"])}, {float(row["lon_1"])}, {json.dumps(str(row["city_1"]))});
addMap("map-{identifier}-2", {float(row["lat_2"])}, {float(row["lon_2"])}, {json.dumps(str(row["city_2"]))});'''
        )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DINOv3 similar-image samples</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{{font-family:system-ui,sans-serif;margin:2rem;background:#f5f6f8;color:#17202a}}.pair{{background:#fff;border-radius:12px;padding:1.25rem;margin:1.5rem auto;max-width:1200px;box-shadow:0 1px 4px #0002}}h2{{margin:0}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem}}img{{width:100%;height:360px;object-fit:contain;background:#eef1f5}}.map{{height:260px}}@media(max-width:700px){{.columns{{grid-template-columns:1fr}}}}</style></head><body><header><h1>DINOv3 cross-city similar-image samples</h1><p>{len(rows)} sampled pairs. Maps require an internet connection for OpenStreetMap tiles.</p></header>{''.join(cards)}<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>function addMap(id,lat,lon,label){{const map=L.map(id).setView([lat,lon],13);L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'© OpenStreetMap contributors'}}).addTo(map);L.marker([lat,lon]).addTo(map).bindPopup(label);}}{''.join(maps)}</script></body></html>'''


def build_gallery(pairs_path: Path | str, image_index_root: Path | str, output_dir: Path | str) -> pd.DataFrame:
    pairs = pd.read_parquet(pairs_path)
    missing = REQUIRED_COLUMNS - set(pairs.columns)
    if missing:
        raise ValueError(f"sampled-pairs Parquet is missing columns: {sorted(missing)}")
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cities = set(pairs["city_1"].dropna()) | set(pairs["city_2"].dropna())
    source_paths = _load_image_paths(Path(image_index_root), cities)
    manifest_rows: list[dict[str, object]] = []
    for row_number, pair in enumerate(pairs.itertuples(index=False), start=1):
        record = pair._asdict()
        images: list[str] = []
        for side in (1, 2):
            city, name = str(record[f"city_{side}"]), str(record[f"name_{side}"])
            source = source_paths.get((city, name))
            if source is None:
                raise FileNotFoundError(f"selected image {name!r} is absent from {city!r} image index")
            if not source.is_file():
                raise FileNotFoundError(f"selected image file does not exist: {source}")
            destination_name = _copied_name(city, row_number, side, source)
            shutil.copy2(source, image_dir / destination_name)
            images.append(destination_name)
        manifest_rows.append({**record, "pair_number": row_number, "image_1": images[0], "image_2": images[1]})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(output_dir / "manifest.parquet", index=False)
    (output_dir / "manifest.json").write_text(manifest.to_json(orient="records", indent=2))
    (output_dir / "index.html").write_text(_render_html(manifest_rows))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="Output from sample_image_pairs_faiss.py")
    parser.add_argument("--image-index-root", type=Path, required=True, help="Directory of city image-index Parquets")
    parser.add_argument("--output-dir", type=Path, required=True, help="Portable gallery package folder")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = build_gallery(args.pairs, args.image_index_root, args.output_dir)
    print(f"Wrote {len(manifest)} pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
