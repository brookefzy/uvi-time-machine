import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).parent / "slurm/submit_dinov3_b5c_batches.bash"


@pytest.mark.parametrize("failure", [False, True])
def test_batch_boundaries_and_failure_stop(tmp_path, failure):
    repo = tmp_path / "repo"
    repo.mkdir()
    meta = tmp_path / "cities.csv"
    meta.write_text("City\nAmsterdam\nHouston\nSao Paulo\n")
    sbatch = tmp_path / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s|%s|%s\\n" "$CITY_OFFSET" "$PAIRWISE_ROOT" "$*" >> "$CAPTURE"\nexit "${FAIL_CODE:-0}"\n')
    sbatch.chmod(0o755)
    capture = tmp_path / "calls"
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               UVI_SAMPLE_REPO_DIR=str(repo), ROOTFOLDER=str(tmp_path),
               VENV_PYTHON=sys.executable, CITY_META=str(meta), RESOLUTION="8",
               PAIRWISE_ROOT=str(tmp_path / "pairs"),
               SIMILARITY_EXPORT_FOLDER=str(tmp_path / "agg"),
               RUN_TAG="test", BATCH_SIZE="2", ARRAY_CONCURRENCY="1",
               CAPTURE=str(capture), FAIL_CODE="7" if failure else "0")
    result = subprocess.run(["bash", str(SCRIPT.resolve())], env=env, capture_output=True, text=True)
    calls = capture.read_text().splitlines()
    assert result.returncode == (7 if failure else 0)
    assert len(calls) == (1 if failure else 2)
    assert calls[0].startswith(f"0|{tmp_path / 'pairs'}|")
    assert '--wait --export=ALL --array=1-2%1' in calls[0]
    if not failure:
        assert calls[1].startswith('2|')
        assert '--array=1-1%1' in calls[1]
        manifest = tmp_path / 'agg/_batches/test/cities.txt'
        assert manifest.read_text() == 'Amsterdam\nHouston\nSao Paulo\n'
