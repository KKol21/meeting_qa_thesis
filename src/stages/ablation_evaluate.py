"""Stage 4: evaluate saved answers with BERTScore and a cached LLM judge."""

import argparse
import hashlib
import json
import gc
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from bert_score import BERTScorer
from huggingface_hub import snapshot_download

from meeting_qa_chunking.artifacts import read_answers, read_answer_summary
from meeting_qa_chunking.config import BERTSCORE_MODEL, JUDGE_MODEL as JUDGE_SPEC
from meeting_qa_chunking.evidence import render_gold_evidence
from meeting_qa_chunking.experiment import EXPERIMENT_VERSION, write_json
from meeting_qa_chunking.judging import (
    JUDGE_INSTRUCTION,
    build_judge_prompt,
    parse_judgment,
)
from meeting_qa_chunking.local_model import LocalChatModel
from meeting_qa_chunking.qmsum import load_meeting


EVALUATION_VERSION = 3
BERTSCORE_PACKAGE_VERSION = "0.3.13"
BERT_MODEL = BERTSCORE_MODEL.name
BERT_REVISION = BERTSCORE_MODEL.revision
BERT_LAYERS = 17
JUDGE_MODEL = JUDGE_SPEC.name
JUDGE_REVISION = JUDGE_SPEC.revision
JUDGE_MAX_NEW_TOKENS = 192
DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")


def input_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluation_config() -> dict[str, object]:
    return {
        "evaluation_version": EVALUATION_VERSION,
        "bertscore": {
            "package_version": BERTSCORE_PACKAGE_VERSION,
            "model": BERT_MODEL,
            "revision": BERT_REVISION,
            "layers": BERT_LAYERS,
            "rescale_with_baseline": False,
        },
        "judge": {
            "model": JUDGE_MODEL,
            "revision": JUDGE_REVISION,
            "prequantized": True,
            "max_new_tokens": JUDGE_MAX_NEW_TOKENS,
            "temperature": 0.0,
            "instruction": JUDGE_INSTRUCTION,
        },
    }


def load_stage(
    answer_dir: Path,
    data_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    answer_summary = read_answer_summary(answer_dir / "summary.json")
    answer_paths = [
        answer_dir / f"{meeting_id}.json"
        for meeting_id in answer_summary["meeting_ids"]
    ]
    meetings = {
        meeting_id: load_meeting(data_dir / f"{meeting_id}.json")
        for meeting_id in answer_summary["meeting_ids"]
    }
    records = []
    for answer_path in answer_paths:
        saved = read_answers(answer_path)
        meeting = meetings[saved["meeting_id"]]
        for question_result in saved["questions"]:
            question_index = question_result["question_index"]
            question = meeting.questions[question_index]
            gold_evidence, _turn_ids = render_gold_evidence(question, meeting)
            for condition, result in question_result["results"].items():
                records.append(
                    {
                        "meeting_id": meeting.id,
                        "question_index": question_index,
                        "condition": condition,
                        "question": question.text,
                        "reference_answer": question.reference_answer,
                        "gold_evidence": gold_evidence,
                        "candidate_answer": result["answer"],
                    }
                )
    return answer_summary, records, input_hash(answer_paths)


def add_bertscore(
    scorer: BERTScorer,
    records: list[dict[str, object]],
    batch_size: int,
) -> None:
    precision, recall, f1 = scorer.score(
        [record["candidate_answer"] for record in records],
        [record["reference_answer"] for record in records],
        batch_size=batch_size,
    )
    for record, p, r, f in zip(
        records, precision.tolist(), recall.tolist(), f1.tolist()
    ):
        record["bertscore"] = {"precision": p, "recall": r, "f1": f}


def summarize_stage(result: dict[str, object]) -> dict[str, object]:
    records = result["records"]
    conditions = list(dict.fromkeys(record["condition"] for record in records))
    metrics = {}
    for condition in conditions:
        selected = [record for record in records if record["condition"] == condition]
        distribution = Counter(record["judge"]["score"] for record in selected)
        metrics[condition] = {
            "answer_count": len(selected),
            "bertscore": {
                metric: mean(record["bertscore"][metric] for record in selected)
                for metric in ("precision", "recall", "f1")
            },
            "judge_mean": mean(record["judge"]["score"] for record in selected),
            "judge_distribution": {
                str(score): distribution[score] for score in (1, 2, 3)
            },
        }
    return {
        "source": result["answer_summary"]["source"],
        "answer_model": result["answer_summary"]["answer_model"],
        "answer_count": len(records),
        "conditions": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--answers-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    args = parser.parse_args()

    answer_dirs = sorted(
        path.parent for path in args.answers_root.glob("*/summary.json")
    )
    if not answer_dirs:
        raise ValueError(f"No answer summaries found under {args.answers_root}")
    config = evaluation_config()
    pending = []
    completed = {}
    for answer_dir in answer_dirs:
        answer_summary, records, stage_hash = load_stage(answer_dir, args.data_dir)
        output_path = args.output_dir / f"{answer_dir.name}.json"
        if output_path.exists():
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                saved.get("experiment_version") == EXPERIMENT_VERSION
                and saved.get("evaluation_config") == config
                and saved.get("input_hash") == stage_hash
            ):
                completed[answer_dir.name] = saved
                print(f"Evaluation {answer_dir.name}: existing", flush=True)
                continue
        pending.append(
            {
                "name": answer_dir.name,
                "output_path": output_path,
                "answer_summary": answer_summary,
                "records": records,
                "input_hash": stage_hash,
            }
        )

    if pending:
        # BERTScore and the 70B judge do not fit comfortably at the same time.
        model_path = snapshot_download(
            repo_id=BERT_MODEL,
            revision=BERT_REVISION,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
        scorer = BERTScorer(
            model_type=model_path,
            num_layers=BERT_LAYERS,
            device="cuda",
            rescale_with_baseline=False,
        )
        for stage in pending:
            add_bertscore(scorer, stage["records"], args.bertscore_batch_size)
            print(f"BERTScore {stage['name']}: done", flush=True)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()

        # Load the judge only after explicitly releasing BERTScore's GPU memory.
        judge = LocalChatModel(
            model_name=JUDGE_MODEL,
            revision=JUDGE_REVISION,
            max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            temperature=0.0,
            cache_dir=Path(".cache/judgments"),
            prequantized=True,
        )
        for stage in pending:
            calls_before = judge.model_calls
            hits_before = judge.cache_hits
            output_records = []
            for index, record in enumerate(stage["records"], start=1):
                response = judge(
                    build_judge_prompt(
                        record["question"],
                        record["reference_answer"],
                        record["gold_evidence"],
                        record["candidate_answer"],
                    )
                )
                score, reason = parse_judgment(response)
                output_records.append(
                    {
                        "meeting_id": record["meeting_id"],
                        "question_index": record["question_index"],
                        "condition": record["condition"],
                        "bertscore": record["bertscore"],
                        "judge": {
                            "score": score,
                            "reason": reason,
                            "raw_response": response,
                            "cache_hit": judge.last_cache_hit,
                        },
                    }
                )
                if index % 10 == 0 or index == len(stage["records"]):
                    print(
                        f"Judge {stage['name']}: {index}/{len(stage['records'])}",
                        flush=True,
                    )

            result = {
                "experiment_version": EXPERIMENT_VERSION,
                "evaluation_config": config,
                "input_hash": stage["input_hash"],
                "answer_summary": stage["answer_summary"],
                "judge_model_calls": judge.model_calls - calls_before,
                "judge_cache_hits": judge.cache_hits - hits_before,
                "records": output_records,
            }
            write_json(stage["output_path"], result)
            completed[stage["name"]] = result
            print(f"Evaluation {stage['name']}: saved", flush=True)

    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "evaluation_config": config,
        "stages": {
            name: summarize_stage(result) for name, result in sorted(completed.items())
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"Evaluation summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
