"""Collect stage summaries into one small result file."""

import argparse
import json
from pathlib import Path

from meeting_qa_chunking.artifacts import (
    EXPERIMENT_VERSION,
    sha256_file,
    write_json,
)
from meeting_qa_chunking.config import load_run_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    args = parser.parse_args()

    run = load_run_config(args.preset)
    output_path = run.output_root / "summary.json"
    paths = {"retrieval": run.retrieval_dir / "summary.json"}
    paths.update(
        {
            f"answers/{stage.name}": run.answers_dir / stage.name / "summary.json"
            for stage in run.answers
        }
    )
    if run.run_evaluation:
        paths["evaluation"] = run.evaluation_dir / "summary.json"

    summaries = {}
    hashes = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        summaries[name] = json.loads(path.read_text(encoding="utf-8"))
        hashes[name] = sha256_file(path)

    write_json(
        output_path,
        {
            "experiment_version": EXPERIMENT_VERSION,
            "stages": summaries,
            "summary_hashes": hashes,
        },
    )
    print(f"Combined summary: {output_path}")


if __name__ == "__main__":
    main()
