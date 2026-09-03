"""Translate one TOML preset into isolated experiment stage processes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import load_run_config


REPOSITORY_ROOT = Path(__file__).parents[2]
STAGE_DIR = Path("src/stages")


def stage_commands(preset: Path) -> list[list[str]]:
    run = load_run_config(preset)
    preset_arg = str(preset.relative_to(REPOSITORY_ROOT))
    commands = []
    if run.run_segmentation:
        commands.append(
            [str(STAGE_DIR / "ablation_segment.py"), "--preset", preset_arg]
        )
    commands.append(
        [str(STAGE_DIR / "ablation_retrieval.py"), "--preset", preset_arg]
    )
    commands.extend(
        [
            str(STAGE_DIR / "ablation_answer.py"),
            "--preset",
            preset_arg,
            "--answer-stage",
            stage.name,
        ]
        for stage in run.answers
    )
    if run.run_evaluation:
        commands.append(
            [str(STAGE_DIR / "ablation_evaluate.py"), "--preset", preset_arg]
        )
    commands.append(
        ["src/tools/summarize_ablations.py", "--preset", preset_arg]
    )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--use-srun", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = (REPOSITORY_ROOT / args.preset).resolve()
    run = load_run_config(preset)
    if args.describe:
        print(
            json.dumps(
                {
                    "name": run.name,
                    "output_root": run.output_root.as_posix(),
                    "data_dir": run.data_dir.as_posix(),
                    "lumber_dir": run.lumber_dir.as_posix(),
                    "meeting_ids": run.meeting_ids(REPOSITORY_ROOT),
                    "run_evaluation": run.run_evaluation,
                }
            )
        )
        return

    for stage in stage_commands(preset):
        command = (
            ["srun", "bash", "src/wormulon/run_python.sh", *stage]
            if args.use_srun
            else [sys.executable, *stage]
        )
        print("+", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


if __name__ == "__main__":
    main()
