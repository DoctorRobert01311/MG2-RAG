"""Build and query an MG2-RAG graph from the bundled multimodal input."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.mmgraphrag.MMGraphRAG import MMGraphRAG
from src.mmgraphrag.utils.config_utils import BaseConfig

DEFAULT_DATA = PROJECT_ROOT / "examples" / "data" / "impala_demo" / "input.json"


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("index", "retrieve", "all"), default=os.getenv("DEMO_STAGE", "all"))
    parser.add_argument("--save-dir", default=os.getenv("MG2RAG_SAVE_DIR", "outputs/impala_demo"))
    parser.add_argument(
        "--qa",
        action=argparse.BooleanOptionalAction,
        default=env_bool("DEMO_RUN_QA", False),
        help="Generate an answer with the configured LLM/MLLM service",
    )
    return parser.parse_args()


def resolve_asset(path: str | None) -> str | None:
    if path is None:
        return None
    asset = Path(path)
    return str(asset if asset.is_absolute() else PROJECT_ROOT / asset)


def load_fixed_data(path: Path) -> dict[str, Any]:
    data_path = path if path.is_absolute() else PROJECT_ROOT / path
    with data_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not data.get("documents") or not data.get("queries"):
        raise ValueError(f"Demo input must contain documents and queries: {data_path}")
    return data


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    data = load_fixed_data(DEFAULT_DATA)
    save_dir = Path(args.save_dir)

    documents = data["documents"]
    queries = data["queries"]
    docs = [item["text"] for item in documents]
    doc_ids = [item["id"] for item in documents]
    image_pairs = [
        (index, resolve_asset(item.get("image")))
        for index, item in enumerate(documents)
        if item.get("image")
    ]
    image_doc_indices = [index for index, _ in image_pairs]
    knowledge_images = [path for _, path in image_pairs]
    questions = [item["question"] for item in queries]
    query_images = [resolve_asset(item.get("image")) for item in queries]

    config = BaseConfig.from_env(save_dir=str(save_dir))
    rag = MMGraphRAG(global_config=config)

    if args.stage in {"index", "all"}:
        rag.index(docs=docs, imgs=knowledge_images, match_arr=image_doc_indices)
        graph_summary = rag.get_graph_info()
        graph_summary.update({"document_count": len(docs), "query_count": len(queries)})
        dump_json(save_dir / "graph_summary.json", graph_summary)
        print(json.dumps(graph_summary, indent=2))

    if args.stage in {"retrieve", "all"}:
        solutions = rag.retrieve(question_queries=questions, image_queries=query_images)
        content_to_id = dict(zip(docs, doc_ids))
        rows = []
        for query, solution in zip(queries, solutions):
            retrieved_ids = [content_to_id.get(content, "<unknown>") for content in solution.docs]
            gold_ids = set(query["gold_document_ids"])
            gold_rank = next((rank for rank, doc_id in enumerate(retrieved_ids, 1) if doc_id in gold_ids), None)
            rows.append(
                {
                    "query_id": query["id"],
                    "question": query["question"],
                    "gold_document_ids": query["gold_document_ids"],
                    "retrieved_document_ids": retrieved_ids,
                    "scores": [float(score) for score in solution.doc_scores],
                    "gold_rank": gold_rank,
                }
            )
        dump_json(save_dir / "retrieval_output.json", rows)
        print(json.dumps(rows, ensure_ascii=False, indent=2))

        if args.qa:
            mode = os.getenv("DEMO_QA_MODE", "llm").lower()
            if mode not in {"llm", "mllm"}:
                raise ValueError("DEMO_QA_MODE must be 'llm' or 'mllm'")
            _, predictions, metadata = rag.qa_infoseek(solutions, mode=mode)
            qa_rows = [
                {
                    "query_id": query["id"],
                    "question": query["question"],
                    "prediction": prediction,
                    "gold_answers": query["answers"],
                    "usage": usage,
                }
                for query, prediction, usage in zip(queries, predictions, metadata)
            ]
            dump_json(save_dir / "qa_output.json", qa_rows)
            print(json.dumps(qa_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
