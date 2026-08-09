import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import duckdb as real_duckdb


MODULE_PATH = Path(__file__).resolve().parent / "B5c_pairwise_agg_optimized.py"


def load_module():
    previous = {name: sys.modules.get(name) for name in ("duckdb", "pandas", "tqdm")}
    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, **_kwargs: iterable
    sys.modules["duckdb"] = real_duckdb
    sys.modules["pandas"] = pd
    sys.modules["tqdm"] = fake_tqdm
    try:
        spec = importlib.util.spec_from_file_location("b5c_membership", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def write_membership(root: Path, city: str, rows: list[str]) -> None:
    pd.DataFrame({"hex_id": rows, "res": [7] * len(rows)}).to_parquet(
        root / f"dinov3_city={city}_res_exclude=None.parquet", index=False
    )


def write_shard(root: Path, city1: str, city2: str, rows: list[dict]) -> None:
    path = (
        root
        / "optimized"
        / "temp"
        / f"city1={city1}"
        / f"city2={city2}"
        / "part_res=7.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def read_dataset(path: Path) -> pd.DataFrame:
    parts = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    return pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)


def test_b5c_validates_orients_quarantines_and_audits_every_row(tmp_path):
    module = load_module()
    membership_root = tmp_path / "membership"
    pairwise_root = tmp_path / "pairwise"
    export_root = tmp_path / "output"
    membership_root.mkdir()

    write_membership(membership_root, "Sao Paulo, SP", ["sp_a", "sp_z"])
    write_membership(membership_root, "St. John's", ["st_b", "st_c"])
    write_membership(membership_root, "Gainesville, FL", ["fl_g"])

    write_shard(
        pairwise_root,
        "Sao Paulo, SP",
        "St. John's",
        [
            {
                "hex_id1": "sp_a",
                "hex_id2": "st_b",
                "city1": "SÃO PAULO",
                "city2": "St Johns",
                "similarity": 0.9,
            },
            {
                "hex_id1": "sp_z",
                "hex_id2": "st_c",
                "city1": "St. John's",
                "city2": "Sao Paulo",
                "similarity": 0.8,
            },
            {
                "hex_id1": "missing_hex",
                "hex_id2": "st_b",
                "city1": "São Paulo",
                "city2": "St. John's",
                "similarity": 0.7,
            },
            {
                "hex_id1": "sp_a",
                "hex_id2": "st_b",
                "city1": "Unknown City",
                "city2": "St. John's",
                "similarity": 0.6,
            },
            {
                "hex_id1": "sp_z",
                "hex_id2": "st_b",
                "city1": "Gainesville Florida",
                "city2": "St. John's",
                "similarity": 0.5,
            },
        ],
    )
    write_shard(
        pairwise_root,
        "Gainesville, FL",
        "St. John's",
        [
            {
                "hex_id1": "fl_g",
                "hex_id2": "st_c",
                "city1": "Gainesville Florida",
                "city2": "St. Johns",
                "similarity": 0.4,
            }
        ],
    )

    city_meta = tmp_path / "city_meta.csv"
    pd.DataFrame(
        {"City": ["São Paulo", "St. John's", "Gainesville, FL"]}
    ).to_csv(city_meta, index=False)
    audit_path = export_root / "audit.json"
    unresolved_root = export_root / "unresolved"
    processor = module.OptimizedUrbanSimilarityProcessor(
        {
            "CURATE_FOLDER_EXPORT2": str(pairwise_root),
            "EXPORT_FOLDER": str(export_root),
            "RES_SEL": 7,
            "RESUME": False,
            "PARQUET_FILE_SIZE_BYTES": "0",
            "H3_MEMBERSHIP_ROOT": str(membership_root),
            "H3_INPUT_TEMPLATE": "dinov3_city={city}_res_exclude=None.parquet",
            "AUDIT_REPORT_PATH": str(audit_path),
            "UNRESOLVED_FOLDER": str(unresolved_root),
        },
        log_level="WARNING",
    )

    processor.run(str(city_meta))

    output_parts = sorted(export_root.glob("similarity_*_res=7.parquet"))
    resolved = pd.concat([read_dataset(path) for path in output_parts], ignore_index=True)
    assert len(resolved) == 3

    membership = {
        "sp_a": "saopaulo",
        "sp_z": "saopaulo",
        "st_b": "stjohns",
        "st_c": "stjohns",
        "fl_g": "gainesville",
    }
    assert all(
        membership[row.hex_id1] == module.normalize_city_name(row.city_1)
        and membership[row.hex_id2] == module.normalize_city_name(row.city_2)
        for row in resolved.itertuples()
    )

    reversed_row = resolved.loc[resolved["similarity"] == 0.8].iloc[0]
    assert reversed_row["hex_id1"] == "sp_z"
    assert reversed_row["city_1"] == "Sao Paulo"
    assert reversed_row["hex_id2"] == "st_c"
    assert reversed_row["city_2"] == "St. John's"

    unresolved = pd.concat(
        [pd.read_parquet(path) for path in sorted(unresolved_root.glob("*.parquet"))],
        ignore_index=True,
    )
    assert len(unresolved) == 3
    assert set(unresolved["validation_status"]) == {"unresolved", "missing_city_h3"}

    audit = json.loads(audit_path.read_text())
    assert audit["totals"] == {
        "direct": 2,
        "reversed_fixed": 1,
        "unresolved": 1,
        "missing_city_h3": 2,
        "input_rows": 6,
        "emitted_rows": 3,
    }
    assert audit["examples"]["reversed_fixed"]
    assert audit["examples"]["unresolved"]
    assert audit["examples"]["missing_city_h3"]
