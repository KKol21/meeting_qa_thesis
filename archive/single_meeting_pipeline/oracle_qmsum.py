"""Answer one QMSum meeting using its annotated evidence spans."""

import argparse
import json
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer

from meeting_qa_chunking.answering import (
    ANSWER_INSTRUCTION,
    ROUGE_TYPES,
    build_answer_prompt,
    score_answer,
)
from meeting_qa_chunking.chunking import turn_word_count
from meeting_qa_chunking.evidence import render_gold_evidence
from meeting_qa_chunking.local_model import LocalChatModel
from meeting_qa_chunking.qmsum import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_OUTPUT = Path("runs/oracle/qmsum/Bed002-14b.json")
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    revision = args.revision
    if args.model == DEFAULT_MODEL and revision is None:
        revision = DEFAULT_REVISION

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    prepared = [render_gold_evidence(question, meeting) for question in meeting.questions]
    model = LocalChatModel(
        model_name=args.model,
        revision=revision,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        cache_dir=Path(".cache/answers"),
    )
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    questions = []
    for index, (question, (evidence, turn_ids)) in enumerate(
        zip(meeting.questions, prepared)
    ):
        answer = model(build_answer_prompt(question.text, evidence))
        questions.append(
            {
                "question_index": index,
                "question": question.text,
                "reference_answer": question.reference_answer,
                "answer": answer,
                "rouge_f1": score_answer(
                    scorer, question.reference_answer, answer
                ),
                "gold_turn_ranges": question.relevant_turn_ranges,
                "gold_words": sum(
                    turn_word_count(meeting.turns[turn_id]) for turn_id in turn_ids
                ),
                "cache_hit": model.last_cache_hit,
            }
        )
        source = "cache" if model.last_cache_hit else "model"
        print(f"Question {index} ({source})", flush=True)

    macro_average = {
        rouge_type: mean(question["rouge_f1"][rouge_type] for question in questions)
        for rouge_type in ROUGE_TYPES
    }
    result = {
        "meeting_id": meeting.id,
        "question_count": len(questions),
        "evidence_source": "QMSum annotated relevant_text_span",
        "answer_model": {
            "model": args.model,
            "revision": revision,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "instruction": ANSWER_INSTRUCTION,
        },
        "model_calls": model.model_calls,
        "cache_hits": model.cache_hits,
        "macro_average_rouge_f1": macro_average,
        "questions": questions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Oracle  ROUGE-1  ROUGE-2  ROUGE-L")
    print(
        "        "
        f"{macro_average['rouge1']:.3f}    "
        f"{macro_average['rouge2']:.3f}    "
        f"{macro_average['rougeL']:.3f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
