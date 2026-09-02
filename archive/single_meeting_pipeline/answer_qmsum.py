"""Answer one QMSum meeting from a saved retrieval result."""

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
from meeting_qa_chunking.chunking import chunk_by_word_budget
from meeting_qa_chunking.evidence import reconstruct_evidence, render_evidence
from meeting_qa_chunking.local_model import LocalChatModel
from meeting_qa_chunking.lumber import load_lumber_chunks
from meeting_qa_chunking.qmsum import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_LUMBER_RESULT = Path("runs/lumber/qmsum/Bed002.json")
DEFAULT_RETRIEVAL_RESULT = Path("runs/retrieval/qmsum/Bed002.json")
DEFAULT_OUTPUT = Path("runs/answers/qmsum/Bed002.json")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--lumber-result", type=Path, default=DEFAULT_LUMBER_RESULT)
    parser.add_argument(
        "--retrieval-result", type=Path, default=DEFAULT_RETRIEVAL_RESULT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    revision = args.revision
    if args.model == DEFAULT_MODEL and revision is None:
        revision = DEFAULT_REVISION

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    retrieval = json.loads(args.retrieval_result.read_text(encoding="utf-8"))
    if retrieval["meeting_id"] != meeting.id:
        raise ValueError("Retrieval result belongs to a different meeting")
    if len(retrieval["questions"]) != len(meeting.questions):
        raise ValueError("Retrieval result has the wrong number of questions")

    fixed_chunks = chunk_by_word_budget(
        meeting.turns, retrieval["fixed"]["chunk_words"]
    )
    lumber_chunks = load_lumber_chunks(args.lumber_result, meeting)
    chunk_sets = {"fixed": fixed_chunks, "lumber": lumber_chunks}
    evidence_words = retrieval["evidence_words"]

    prepared = []
    for index, (question, saved_question) in enumerate(
        zip(meeting.questions, retrieval["questions"])
    ):
        if saved_question["question_index"] != index:
            raise ValueError("Retrieval question indices are not contiguous")
        if saved_question["question"] != question.text:
            raise ValueError("Retrieval result contains a different question")
        prepared.append(
            {
                method: reconstruct_evidence(
                    saved_question[method], chunks, evidence_words
                )
                for method, chunks in chunk_sets.items()
            }
        )

    model = LocalChatModel(
        model_name=args.model,
        revision=revision,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        cache_dir=Path(".cache/answers"),
    )
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    questions = []
    for index, (question, evidences) in enumerate(zip(meeting.questions, prepared)):
        item: dict[str, object] = {
            "question_index": index,
            "question": question.text,
            "reference_answer": question.reference_answer,
        }
        for method, evidence in evidences.items():
            answer = model(
                build_answer_prompt(
                    question.text,
                    render_evidence(evidence, meeting),
                )
            )
            item[method] = {
                "answer": answer,
                "rouge_f1": score_answer(scorer, question.reference_answer, answer),
                "evidence_words": evidence.word_count,
                "selected_chunk_indices": evidence.chunk_indices,
                "evidence_turn_ids": [part.turn_id for part in evidence.parts],
                "cache_hit": model.last_cache_hit,
            }
            source = "cache" if model.last_cache_hit else "model"
            print(f"Question {index} {method} ({source})", flush=True)
        questions.append(item)

    macro_average = {
        method: {
            rouge_type: mean(
                question[method]["rouge_f1"][rouge_type] for question in questions
            )
            for rouge_type in ROUGE_TYPES
        }
        for method in chunk_sets
    }
    result = {
        "meeting_id": meeting.id,
        "question_count": len(questions),
        "retrieval_result": str(args.retrieval_result),
        "lumber_result": str(args.lumber_result),
        "evidence_words": evidence_words,
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

    print("Method  ROUGE-1  ROUGE-2  ROUGE-L")
    for method, scores in macro_average.items():
        print(
            f"{method:<7} "
            f"{scores['rouge1']:.3f}    "
            f"{scores['rouge2']:.3f}    "
            f"{scores['rougeL']:.3f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
