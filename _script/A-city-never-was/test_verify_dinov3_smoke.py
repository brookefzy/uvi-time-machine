from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from verify_dinov3_smoke import verify_smoke_output


def write_smoke_shard(root: Path, city_stem: str = "hongkong") -> Path:
    out_dir = root / city_stem
    out_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "name": ["panoid_a_abcdefghijkl.jpg", "panoid_b_abcdefghijkl.jpg"],
            "panoid": ["panoid_a_abcdefghijkl", "panoid_b_abcdefghijkl"],
            "model_name": ["fake-dinov3", "fake-dinov3"],
            "embedding_dim": [2, 2],
            "e_0000": [1.0, 0.0],
            "e_0001": [0.0, 1.0],
        }
    )
    path = out_dir / "hongkong_smoke.parquet"
    df.to_parquet(path, index=False)
    return path


def test_verify_smoke_output_prints_distribution_and_passes(tmp_path, capsys):
    write_smoke_shard(tmp_path)

    stats = verify_smoke_output(
        city="Hong Kong",
        smoke_root=tmp_path,
        expected_model_name="fake-dinov3",
        min_rows=2,
    )

    captured = capsys.readouterr().out
    assert stats["row_count"] == 2
    assert stats["embedding_dim"] == 2
    assert "SMOKE CHECK PASSED" in captured
    assert "rows_by_file" in captured
    assert "model_name_counts" in captured
    assert "norm_distribution" in captured
    assert "duplicate_name_count: 0" in captured


def test_verify_smoke_output_rejects_non_unit_vectors(tmp_path):
    path = write_smoke_shard(tmp_path)
    df = pd.read_parquet(path)
    df.loc[0, "e_0000"] = 2.0
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="unit-normalized"):
        verify_smoke_output(
            city="Hong Kong",
            smoke_root=tmp_path,
            expected_model_name="fake-dinov3",
            min_rows=2,
        )


def test_verify_smoke_output_rejects_missing_rows(tmp_path):
    with pytest.raises(FileNotFoundError, match="No smoke parquet files"):
        verify_smoke_output(city="Hong Kong", smoke_root=tmp_path, min_rows=1)
