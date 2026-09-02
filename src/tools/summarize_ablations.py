"""Collect stage summaries into one small result file."""

import argparse
import json
from pathlib import Path

from meeting_qa_chunking.experiment import EXPERIMENT_VERSION, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    output_path = args.root / "summary.json"
    summaries = {}
    for path in sorted(args.root.rglob("summary.json")):
        if path == output_path:
            continue
        name = path.parent.relative_to(args.root).as_posix()
        summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    if not summaries:
        raise ValueError(f"No stage summaries found under {args.root}")

    write_json(
        output_path,
        {
            "experiment_version": EXPERIMENT_VERSION,
            "stages": summaries,
        },
    )
    print(f"Combined summary: {output_path}")


if __name__ == "__main__":
    main()
