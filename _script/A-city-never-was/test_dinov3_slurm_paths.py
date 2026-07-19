from pathlib import Path
import os
import subprocess


REPO_ROOT = Path(__file__).resolve().parent
H3_ARRAY_SCRIPT = REPO_ROOT / "slurm" / "dinov3_02_h3_array.cmd"


def test_h3_array_rewrites_legacy_city_meta_path(tmp_path: Path) -> None:
    repo_dir = tmp_path / "_script" / "A-city-never-was"
    repo_dir.mkdir(parents=True)
    captured_args = tmp_path / "python-args.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${CAPTURED_ARGS}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CAPTURED_ARGS": str(captured_args),
            "CITY_META": str(repo_dir / "city_meta.csv"),
            "PYTHON": str(fake_python),
            "REPO_DIR": str(repo_dir),
            "SLURM_ARRAY_TASK_ID": "1",
        }
    )

    subprocess.run(["bash", str(H3_ARRAY_SCRIPT)], env=env, check=True)

    args = captured_args.read_text(encoding="utf-8").splitlines()
    city_meta_index = args.index("--city-meta") + 1
    assert args[city_meta_index] == str(repo_dir.parent / "city_meta.csv")
