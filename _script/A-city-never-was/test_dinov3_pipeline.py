from pathlib import Path

import pandas as pd
import pytest

from dinov3_pipeline import (
    DEFAULT_MODEL_NAME,
    DINOv3PipelineConfig,
    build_stage_commands,
    load_cities,
    render_slurm_array_script,
)


def test_load_cities_supports_city_column_and_one_based_array_index(tmp_path):
    city_meta = tmp_path / "city_meta.csv"
    pd.DataFrame({"City": ["Alpha", "Beta", "Alpha", None]}).to_csv(
        city_meta, index=False
    )

    assert load_cities(city_meta) == ["Alpha", "Beta"]
    assert load_cities(city_meta, city_index=2) == ["Beta"]

    with pytest.raises(IndexError):
        load_cities(city_meta, city_index=3)


def test_build_embed_and_aggregate_commands_for_selected_city(tmp_path):
    config = DINOv3PipelineConfig(
        repo_dir=Path("/repo"),
        city_meta=Path("/data/city_meta.csv"),
        model_name=DEFAULT_MODEL_NAME,
    )

    embed = build_stage_commands("embed", config, ["Hong Kong"])
    aggregate = build_stage_commands("aggregate", config, ["Hong Kong"])

    assert len(embed) == 1
    assert "B5d_dinov3_embed_city.py" in embed[0]
    assert "--city 'Hong Kong'" in embed[0]
    assert "--model-name facebook/dinov3-vitb16-pretrain-lvd1689m" in embed[0]
    assert "--local-files-only" in embed[0]

    assert len(aggregate) == 1
    assert "B5e_dinov3_vector_summary.py" in aggregate[0]
    assert "--city 'Hong Kong'" in aggregate[0]
    assert "--res-exclude 11" in aggregate[0]


def test_build_all_commands_orders_city_stages_before_global_stages(tmp_path):
    config = DINOv3PipelineConfig(repo_dir=Path("/repo"), city_meta=Path("/meta.csv"))

    commands = build_stage_commands("all", config, ["Alpha", "Beta"])

    assert len(commands) == 7
    assert "B5d_dinov3_embed_city.py" in commands[0]
    assert "B5d_dinov3_embed_city.py" in commands[1]
    assert "B5e_dinov3_vector_summary.py" in commands[2]
    assert "B5e_dinov3_vector_summary.py" in commands[3]
    assert "B5b_compute_similarity_pairwise-optimized.py" in commands[4]
    assert "B5c_pairwise_agg_optimized.py" in commands[5]
    assert "B5h_summarize_dinov3_citypair_similarity.py" in commands[6]
    assert "--threshold -1.0" in commands[4]


def test_render_slurm_array_script_uses_city_count_and_task_id():
    config = DINOv3PipelineConfig(
        repo_dir=Path("/repo"),
        city_meta=Path("/data/city_meta.csv"),
    )

    script = render_slurm_array_script(
        job_name="dinov3_embed",
        stage="embed",
        config=config,
        city_count=127,
        array_concurrency=12,
        time_limit="24:00:00",
        partition="gpu",
        gres="gpu:1",
        cpus_per_task=8,
        mem="64G",
    )

    assert "#SBATCH --array=1-127%12" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "--stage embed" in script
    assert "--city-index ${SLURM_ARRAY_TASK_ID}" in script
    assert "--execute" in script
