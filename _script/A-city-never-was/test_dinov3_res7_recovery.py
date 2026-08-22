import json
from pathlib import Path

import h3
import pandas as pd
import pytest

from dinov3_res7_recovery import (
    RecoveryPaths,
    audit_city,
    build_h3_overlay,
    build_pairwise_overlay,
    build_affected_pair_manifest,
    discover_city_index,
    recover_missing_indices,
    validate_h3_recovery,
    validate_pair_manifest_shards,
    write_recovery_manifests,
    validate_final_export,
)


def _cell(lat: float, lon: float, resolution: int = 7) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, resolution)
    return h3.geo_to_h3(lat, lon, resolution)


def _panoid(label: str) -> str:
    return f"{label:0<22}"[:22]


def _embedding_rows(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": names,
            "panoid": [name[:22] for name in names],
            "model_name": ["fake-dinov3"] * len(names),
            "embedding_dim": [2] * len(names),
            "e_0000": [1.0] * len(names),
            "e_0001": [0.0] * len(names),
        }
    )


def test_discover_city_index_handles_unicode_suffix_and_known_aliases(tmp_path):
    index_root = tmp_path / "indices"
    index_root.mkdir()
    for stem in ["AmsTeRdAm Netherlands", "kozhikode-kerala", "calicut"]:
        pd.DataFrame({"path": []}).to_parquet(index_root / f"{stem}.parquet")

    amsterdam = discover_city_index(
        "Amsterdam",
        [index_root],
        metadata_aliases=["Amsterdam, Netherlands"],
    )
    assert amsterdam.status == "resolved"
    assert amsterdam.path.name == "AmsTeRdAm Netherlands.parquet"

    kozhikode = discover_city_index("Kozhikode", [index_root])
    assert kozhikode.status == "ambiguous"
    assert {path.name for path in kozhikode.candidates} == {
        "calicut.parquet",
        "kozhikode-kerala.parquet",
    }

    forced = discover_city_index("Kozhikode", [index_root], preferred_stem="calicut")
    assert forced.status == "resolved"
    assert forced.path.name == "calicut.parquet"


def test_discover_city_index_records_every_checked_empty_root(tmp_path):
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()

    result = discover_city_index("Gombe", [missing, empty])

    assert result.status == "absent"
    assert result.path is None
    assert result.checked_roots == [
        {"path": str(missing), "exists": False, "parquet_count": 0},
        {"path": str(empty), "exists": True, "parquet_count": 0},
    ]


def test_absent_city_audit_lists_checked_index_gsv_and_embedding_paths(tmp_path):
    paths = RecoveryPaths(
        root=tmp_path / "root",
        index_roots=(tmp_path / "indices-a", tmp_path / "indices-b"),
        embed_root=tmp_path / "embed",
        h3_root=tmp_path / "h3",
        pairwise_root=tmp_path / "pairs",
        aggregate_root=tmp_path / "aggregate",
    )

    report = audit_city("Gombe", paths)

    checked = {item["path"] for item in report["source_index"]["checked_city_paths"]}
    assert str(tmp_path / "indices-a/gombe.parquet") in checked
    assert str(tmp_path / "indices-b/gombe.parquet") in checked
    assert str(tmp_path / "embed/gombe") in checked
    assert str(tmp_path / "root/GSV/gsv_rgb/gombe") in checked
    assert report["recoverability"] == "source_imagery_absent"


def test_audit_city_distinguishes_stale_h3_alignment_from_missing_embeddings(tmp_path):
    city = "Sitapur"
    stem = "sitapur"
    root = tmp_path / "root"
    index_root = tmp_path / "index"
    embed_root = tmp_path / "embed"
    h3_root = tmp_path / "h3"
    pairwise_root = tmp_path / "pairwise"
    aggregate_root = tmp_path / "aggregate"
    required_root = tmp_path / "required"
    core_root = tmp_path / "core"
    for path in [index_root, embed_root / stem, h3_root, required_root, core_root]:
        path.mkdir(parents=True)

    panoids = [_panoid("old-a"), _panoid("old-b"), _panoid("current-core")]
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_paths = []
    for panoid in panoids:
        image_path = image_root / f"{panoid}_000.jpg"
        image_path.write_bytes(b"image")
        image_paths.append(str(image_path))
    pd.DataFrame({"path": image_paths}).to_parquet(index_root / "sitapur.parquet")
    _embedding_rows([Path(path).name for path in image_paths]).to_parquet(
        embed_root / stem / "part.parquet"
    )

    meta_dir = root / "GSV" / "gsv_rgb" / stem / "gsvmeta"
    meta_dir.mkdir(parents=True)
    old_a = (27.0, 80.0)
    old_b = (27.1, 80.1)
    current = (28.0, 81.0)
    pd.DataFrame(
        {
            "panoid": panoids,
            "lat": [old_a[0], old_b[0], current[0]],
            "lon": [old_a[1], old_b[1], current[1]],
            "year": [2018, 2018, 2018],
        }
    ).to_csv(meta_dir / "gsv_pano.csv", index=False)
    pd.DataFrame({"panoid": panoids}).to_csv(meta_dir / "gsv_path.csv", index=False)

    old_cells = [_cell(*old_a), _cell(*old_b)]
    current_cell = _cell(*current)
    pd.DataFrame(
        {"hex_id": old_cells, "res": [7, 7], "img_count": [1, 1]}
    ).to_parquet(h3_root / f"dinov3_city={city}_res_exclude=None.parquet")
    pd.DataFrame({"hex_id": [current_cell]}).to_parquet(required_root / "sitapur.parquet")
    pd.DataFrame({"hex_id": [current_cell]}).to_parquet(core_root / "sitapur.parquet")

    report = audit_city(
        city,
        RecoveryPaths(
            root=root,
            index_roots=(index_root,),
            embed_root=embed_root,
            h3_root=h3_root,
            pairwise_root=pairwise_root,
            aggregate_root=aggregate_root,
            required_h3_root=required_root,
            core_h3_root=core_root,
        ),
    )

    assert report["source_index"]["existing_image_count"] == 3
    assert report["embeddings"]["finished_expected_count"] == 3
    assert report["embeddings"]["missing_count"] == 0
    assert report["current_membership"]["embedded_res7_h3_count"] == 3
    assert report["current_membership"]["required_overlap_count"] == 1
    assert report["current_membership"]["core_overlap_count"] == 1
    assert report["h3_summary"]["core_overlap_count"] == 0
    assert report["first_broken_boundary"] == "res7_h3_vector_summary"


def test_audit_city_does_not_call_empty_index_absent_when_embeddings_and_metadata_exist(tmp_path):
    city = "Sitapur"
    stem = "sitapur"
    root = tmp_path / "root"
    index_root = tmp_path / "index"
    embed_root = tmp_path / "embed"
    for path in [index_root, embed_root / stem]:
        path.mkdir(parents=True)
    panoid = _panoid("embedded")
    _embedding_rows([f"{panoid}_000.jpg"]).to_parquet(embed_root / stem / "part.parquet")
    meta_dir = root / "GSV" / "gsv_rgb" / stem / "gsvmeta"
    meta_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"panoid": panoid, "lat": 27.5, "lon": 80.5, "year": 2018}]
    ).to_csv(meta_dir / "gsv_pano.csv", index=False)

    report = audit_city(
        city,
        RecoveryPaths(
            root=root,
            index_roots=(index_root,),
            embed_root=embed_root,
            h3_root=tmp_path / "h3",
            pairwise_root=tmp_path / "pairwise",
            aggregate_root=tmp_path / "aggregate",
        ),
    )

    assert report["source_index"]["status"] == "absent"
    assert report["embeddings"]["unique_name_count"] == 1
    assert report["current_membership"]["embedded_res7_h3_count"] == 1
    assert report["recoverability"] == "recoverable_from_existing_embeddings"
    assert report["first_broken_boundary"] == "source_image_index"


def test_recover_missing_index_uses_only_real_images_under_selected_gsv_stem(tmp_path):
    root = tmp_path / "root"
    city_dir = root / "GSV" / "gsv_rgb" / "gombe"
    (city_dir / "nested").mkdir(parents=True)
    (city_dir / "nested" / "a.jpg").write_bytes(b"a")
    (city_dir / "nested" / "b.JPEG").write_bytes(b"b")
    (city_dir / "nested" / "not-an-image.txt").write_text("x")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "cities": [
                    {
                        "city": "Gombe",
                        "source_index": {
                            "status": "absent",
                            "selected_stem": "gombe",
                            "gsv_image_file_count": 2,
                        },
                    },
                    {
                        "city": "Kampala",
                        "source_index": {"status": "absent"},
                    },
                ]
            }
        )
    )
    output = tmp_path / "indices"

    with pytest.raises(ValueError, match="explicit opt-in"):
        recover_missing_indices(audit_path, root, output)

    report = recover_missing_indices(audit_path, root, output, allow_gsv_rebuild=True)

    assert report == {"Gombe": 2}
    recovered = pd.read_parquet(output / "gombe.parquet")
    assert sorted(Path(path).name for path in recovered["path"]) == ["a.jpg", "b.JPEG"]
    assert recovered["name"].tolist() == [Path(path).name for path in recovered["path"]]


def test_write_recovery_manifests_separates_embed_h3_and_proven_absent(tmp_path):
    audit = {
        "cities": [
            {
                "city": "Amsterdam",
                "source_index": {"status": "resolved", "index_stem": "amsterdam-index", "selected_stem": "amsterdam", "existing_image_count": 2, "path": "/indices/amsterdam-index.parquet"},
                "embeddings": {"unique_name_count": 1},
                "current_membership": {"embedded_panoid_join_count": 1},
                "recoverability": "recoverable",
            },
            {
                "city": "Sitapur",
                "source_index": {"status": "absent", "selected_stem": "sitapur", "existing_image_count": 0},
                "embeddings": {"unique_name_count": 3},
                "current_membership": {"embedded_panoid_join_count": 3},
                "recoverability": "recoverable_from_existing_embeddings",
            },
            {
                "city": "Gombe",
                "source_index": {"status": "absent", "selected_stem": "gombe", "existing_image_count": 0},
                "embeddings": {"unique_name_count": 0},
                "current_membership": {"embedded_panoid_join_count": 0},
                "recoverability": "source_imagery_absent",
            },
        ]
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    embed_manifest = tmp_path / "embed.txt"
    h3_manifest = tmp_path / "h3.txt"
    absent = tmp_path / "absent.txt"

    result = write_recovery_manifests(audit_path, embed_manifest, h3_manifest, absent)

    assert embed_manifest.read_text().splitlines() == [
        "Amsterdam|amsterdam-index|/indices|amsterdam"
    ]
    assert h3_manifest.read_text().splitlines() == [
        "Amsterdam|amsterdam",
        "Sitapur|sitapur",
    ]
    assert absent.read_text().splitlines() == ["Gombe"]
    assert result == {"embed": 1, "h3": 2, "absent": 1}


def test_build_affected_pair_manifest_excludes_unaffected_pairs_and_empty_inputs(tmp_path):
    h3_root = tmp_path / "h3"
    h3_root.mkdir()
    for city, resolutions in {
        "Amsterdam": [7],
        "Paris": [6, 7],
        "London": [7],
        "Gombe": [6],
    }.items():
        pd.DataFrame(
            {"hex_id": [f"{city}-{res}" for res in resolutions], "res": resolutions}
        ).to_parquet(h3_root / f"dinov3_city={city}_res_exclude=None.parquet")

    output = tmp_path / "affected.txt"
    pairs = build_affected_pair_manifest(
        cities=["Paris", "Gombe", "Amsterdam", "London"],
        affected_cities=["Amsterdam", "Gombe"],
        h3_root=h3_root,
        output_path=output,
        resolution=7,
    )

    assert pairs == [
        ("Amsterdam", "London"),
        ("Amsterdam", "Paris"),
    ]
    assert output.read_text().splitlines() == [
        "Amsterdam|London",
        "Amsterdam|Paris",
    ]


def test_overlay_prefers_recovered_affected_artifacts_and_links_unaffected(tmp_path):
    original_h3 = tmp_path / "original-h3"
    recovered_h3 = tmp_path / "recovered-h3"
    overlay_h3 = tmp_path / "overlay-h3"
    for root in [original_h3, recovered_h3]:
        root.mkdir()
    for city, marker in [("Amsterdam", "old"), ("Paris", "keep")]:
        pd.DataFrame({"hex_id": [marker], "res": [7]}).to_parquet(
            original_h3 / f"dinov3_city={city}_res_exclude=None.parquet"
        )
    pd.DataFrame({"hex_id": ["new"], "res": [7]}).to_parquet(
        recovered_h3 / "dinov3_city=Amsterdam_res_exclude=None.parquet"
    )

    h3_counts = build_h3_overlay(
        original_h3, recovered_h3, overlay_h3, ["Amsterdam"], resolution=7
    )

    assert h3_counts == {"original": 1, "recovered": 1}
    assert pd.read_parquet(
        overlay_h3 / "dinov3_city=Amsterdam_res_exclude=None.parquet"
    )["hex_id"].tolist() == ["new"]
    assert pd.read_parquet(
        overlay_h3 / "dinov3_city=Paris_res_exclude=None.parquet"
    )["hex_id"].tolist() == ["keep"]
    assert all(path.is_symlink() for path in overlay_h3.glob("*.parquet"))

    original_pairs = tmp_path / "original-pairs"
    recovered_pairs = tmp_path / "recovered-pairs"
    overlay_pairs = tmp_path / "overlay-pairs"

    def pair(root, city1, city2, similarity):
        path = root / "optimized" / "temp" / f"city1={city1}" / f"city2={city2}" / "part_res=7.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"hex_id1": "a", "hex_id2": "b", "similarity": similarity, "city1": city1, "city2": city2}]
        ).to_parquet(path)

    pair(original_pairs, "Amsterdam", "Paris", 0.1)
    pair(original_pairs, "London", "Paris", 0.2)
    pair(recovered_pairs, "Amsterdam", "Paris", 0.9)

    pair_counts = build_pairwise_overlay(
        original_pairs, recovered_pairs, overlay_pairs, ["Amsterdam"], resolution=7
    )

    assert pair_counts == {"original": 1, "recovered": 1}
    assert pd.read_parquet(
        overlay_pairs / "optimized/temp/city1=Amsterdam/city2=Paris/part_res=7.parquet"
    )["similarity"].tolist() == [0.9]
    assert pd.read_parquet(
        overlay_pairs / "optimized/temp/city1=London/city2=Paris/part_res=7.parquet"
    )["similarity"].tolist() == [0.2]


def test_h3_recovery_gate_requires_source_backed_core_cells_in_summary():
    report = {
        "cities": [
            {
                "city": "Sitapur",
                "recoverability": "recoverable_from_existing_embeddings",
                "current_membership": {
                    "embedded_res7_h3_count": 3,
                    "core_source_overlap_count": 1,
                    "core_overlap_count": 1,
                },
                "h3_summary": {
                    "res7_h3_count": 2,
                    "current_mapping_overlap_count": 2,
                    "core_overlap_count": 0,
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="source-backed core"):
        validate_h3_recovery(report)

    report["cities"][0]["h3_summary"]["core_overlap_count"] = 1
    result = validate_h3_recovery(report)
    assert result["cities"]["Sitapur"]["res7_h3_count"] == 2


def test_pair_manifest_gate_rejects_missing_or_invalid_shards(tmp_path):
    manifest = tmp_path / "pairs.txt"
    manifest.write_text("Amsterdam|Paris\nAmsterdam|London\n")
    pair_root = tmp_path / "pairs"

    def write(city1, city2, columns):
        path = pair_root / "optimized/temp" / f"city1={city1}" / f"city2={city2}" / "part_res=7.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns).to_parquet(path)

    write(
        "Amsterdam",
        "Paris",
        [{"hex_id1": "a", "hex_id2": "b", "similarity": 0.2}],
    )
    with pytest.raises(ValueError, match="London"):
        validate_pair_manifest_shards(manifest, pair_root, resolution=7)

    write(
        "Amsterdam",
        "London",
        [{"hex_id1": "a", "hex_id2": "c", "similarity": 0.3}],
    )
    result = validate_pair_manifest_shards(manifest, pair_root, resolution=7)
    assert result == {"expected_pair_count": 2, "valid_shard_count": 2}


def _write_membership(root: Path, city: str, cells: list[str]):
    pd.DataFrame({"hex_id": cells, "res": [7] * len(cells)}).to_parquet(
        root / f"dinov3_city={city}_res_exclude=None.parquet"
    )


def test_validate_final_export_enforces_schema_orientation_and_coverage(tmp_path):
    membership_root = tmp_path / "membership"
    export_root = tmp_path / "export"
    membership_root.mkdir()
    export_root.mkdir()
    _write_membership(membership_root, "Amsterdam", ["ams-a"])
    _write_membership(membership_root, "Paris", ["par-a"])
    pd.DataFrame(
        [
            {
                "hex_id1": "ams-a",
                "hex_id2": "par-a",
                "similarity": 0.25,
                "city_1": "Amsterdam",
                "city_2": "Paris",
            }
        ]
    ).to_parquet(export_root / "part.parquet")

    report = validate_final_export(
        export_root,
        membership_root,
        required_cities=["Amsterdam", "Paris", "Gombe"],
        allowed_missing_cities=["Gombe"],
        resolution=7,
    )

    assert report["status"] == "valid"
    assert report["row_count"] == 1
    assert report["city_h3_counts"] == {"Amsterdam": 1, "Paris": 1}
    assert report["missing_cities"] == ["Gombe"]
    assert report["missing_required_pairs"] == []


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda row: row.update(similarity=0.0), "exact-zero"),
        (lambda row: row.update(hex_id1=None), "null"),
        (lambda row: row.update(hex_id1="par-a"), "membership"),
        (lambda row: row.update(similarity=1.5), "range"),
    ],
)
def test_validate_final_export_rejects_invalid_rows(tmp_path, mutation, message):
    membership_root = tmp_path / "membership"
    export_root = tmp_path / "export"
    membership_root.mkdir()
    export_root.mkdir()
    _write_membership(membership_root, "Amsterdam", ["ams-a"])
    _write_membership(membership_root, "Paris", ["par-a"])
    row = {
        "hex_id1": "ams-a",
        "hex_id2": "par-a",
        "similarity": 0.25,
        "city_1": "Amsterdam",
        "city_2": "Paris",
    }
    mutation(row)
    pd.DataFrame([row]).to_parquet(export_root / "part.parquet")

    with pytest.raises(ValueError, match=message):
        validate_final_export(
            export_root,
            membership_root,
            required_cities=["Amsterdam", "Paris"],
            resolution=7,
        )


def test_validate_final_export_rejects_duplicate_canonical_rows(tmp_path):
    membership_root = tmp_path / "membership"
    export_root = tmp_path / "export"
    membership_root.mkdir()
    export_root.mkdir()
    _write_membership(membership_root, "Amsterdam", ["ams-a"])
    _write_membership(membership_root, "Paris", ["par-a"])
    pd.DataFrame(
        [
            {
                "hex_id1": "ams-a",
                "hex_id2": "par-a",
                "similarity": 0.25,
                "city_1": "Amsterdam",
                "city_2": "Paris",
            },
            {
                "hex_id1": "par-a",
                "hex_id2": "ams-a",
                "similarity": 0.25,
                "city_1": "Paris",
                "city_2": "Amsterdam",
            },
        ]
    ).to_parquet(export_root / "part.parquet")

    with pytest.raises(ValueError, match="duplicate canonical"):
        validate_final_export(
            export_root,
            membership_root,
            required_cities=["Amsterdam", "Paris"],
            resolution=7,
        )


def test_validate_final_export_does_not_materialize_export_with_pandas(tmp_path, monkeypatch):
    membership_root = tmp_path / "membership"
    export_root = tmp_path / "export"
    membership_root.mkdir()
    export_root.mkdir()
    _write_membership(membership_root, "Amsterdam", ["ams-a"])
    _write_membership(membership_root, "Paris", ["par-a"])
    pd.DataFrame(
        [{
            "hex_id1": "ams-a",
            "hex_id2": "par-a",
            "similarity": 0.2,
            "city_1": "Amsterdam",
            "city_2": "Paris",
        }]
    ).to_parquet(export_root / "part.parquet")
    real_read_parquet = pd.read_parquet

    def guarded(path, *args, **kwargs):
        if Path(path).parent == export_root and kwargs.get("columns") is None:
            raise AssertionError("export must be streamed by DuckDB")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)

    report = validate_final_export(
        export_root,
        membership_root,
        required_cities=["Amsterdam", "Paris"],
        resolution=7,
    )
    assert report["row_count"] == 1


def test_report_is_json_serializable(tmp_path):
    index_root = tmp_path / "indices"
    index_root.mkdir()
    result = discover_city_index("Vijayawada", [index_root])
    json.dumps(result.as_dict())


def test_slurm_recovery_scripts_have_resume_and_terminal_state_contracts():
    root = Path(__file__).resolve().parent
    pipeline = (root / "pipeline/run_dinov3_res7_recovery.bash").read_text()
    embed = (root / "slurm/dinov3_res7_embed.cmd").read_text()
    h3_summary = (root / "slurm/dinov3_res7_h3.cmd").read_text()

    assert "sacct" in pipeline
    assert "!seen || bad" in pipeline
    assert "squeue" in pipeline
    assert "READY" in pipeline
    assert pipeline.index("validate --export-root") < pipeline.index("READY")
    assert '--duckdb-temp-dir "${RUN_ROOT}/validator_duckdb"' in pipeline
    assert "build-overlays" in pipeline
    assert "--city-file-stem" in embed
    assert "--city-file-stem" in h3_summary
    assert "SLURM_ARRAY_TASK_ID" in embed
    assert "SLURM_ARRAY_TASK_ID" in h3_summary
