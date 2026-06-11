"""Q&A dataset management: loading, validation, and synthetic generation.

Two dataset sources:
1. ``gold_qa.json`` — 50 hand-curated pairs; the primary benchmark.
2. Synthetic generation — Llama 3.2 1B generates questions from document chunks.
   Synthetic data supplements the gold set for scale but has known biases
   (the generator tends to produce literal questions that are easy to answer
   from the exact same chunk — useful for recall, not recall under paraphrase).

Usage::

    from experiments.eval.dataset import load_gold_dataset, generate_synthetic_qa

    gold = load_gold_dataset("experiments/data/gold_qa.json")
    synthetic = generate_synthetic_qa(chunks, n_per_chunk=2)
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.utils.ollama_client import chat, is_available, parse_json_response

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """A single evaluation example."""

    id: str
    question: str
    ground_truth: str
    category: Literal["standard", "out_of_scope", "ambiguous", "multi_hop"]
    relevant_sections: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    source: Literal["gold", "synthetic"] = "gold"
    ambiguity_note: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalDataset:
    """Collection of QA pairs with filtering utilities."""

    pairs: list[QAPair]
    version: str = "1.0"
    description: str = ""

    def __len__(self) -> int:
        return len(self.pairs)

    def filter_by_category(self, category: str) -> "EvalDataset":
        return EvalDataset(
            pairs=[p for p in self.pairs if p.category == category],
            version=self.version,
        )

    def filter_by_difficulty(self, difficulty: str) -> "EvalDataset":
        return EvalDataset(
            pairs=[p for p in self.pairs if p.difficulty == difficulty],
            version=self.version,
        )

    def filter_standard(self) -> "EvalDataset":
        return self.filter_by_category("standard")

    def filter_adversarial(self) -> "EvalDataset":
        return EvalDataset(
            pairs=[p for p in self.pairs if p.category != "standard"],
            version=self.version,
        )

    @property
    def questions(self) -> list[str]:
        return [p.question for p in self.pairs]

    @property
    def ground_truths(self) -> list[str]:
        return [p.ground_truth for p in self.pairs]

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.pairs]

    def stats(self) -> dict:
        from collections import Counter
        cats = Counter(p.category for p in self.pairs)
        diffs = Counter(p.difficulty for p in self.pairs)
        return {
            "total": len(self.pairs),
            "by_category": dict(cats),
            "by_difficulty": dict(diffs),
        }


def load_gold_dataset(path: str | Path = "experiments/data/gold_qa.json") -> EvalDataset:
    """Load the hand-curated gold Q&A dataset.

    Args:
        path: Path to ``gold_qa.json``.

    Returns:
        An ``EvalDataset`` containing all hand-curated pairs.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the JSON schema is missing required fields.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gold QA dataset not found at {p.resolve()}")

    with p.open() as f:
        raw = json.load(f)

    pairs = []
    for entry in raw["qa_pairs"]:
        pairs.append(
            QAPair(
                id=entry["id"],
                question=entry["question"],
                ground_truth=entry["ground_truth"],
                category=entry["category"],
                relevant_sections=entry.get("relevant_sections", []),
                difficulty=entry.get("difficulty", "medium"),
                source="gold",
                ambiguity_note=entry.get("ambiguity_note", ""),
            )
        )

    dataset = EvalDataset(
        pairs=pairs,
        version=raw.get("version", "1.0"),
        description=raw.get("description", ""),
    )
    logger.info(f"Loaded gold dataset: {dataset.stats()}")
    return dataset


def generate_synthetic_qa(
    chunk_texts: list[str],
    chunk_ids: list[str],
    n_per_chunk: int = 2,
    model: str = "llama3.2:latest",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.4,
    max_tokens: int = 300,
    seed: int = 42,
) -> EvalDataset:
    """Generate synthetic Q&A pairs from document chunks via Llama 3.2 1B.

    For each chunk, the LLM is prompted to generate ``n_per_chunk`` questions
    that can be answered from the chunk content. The question is paired with
    a short extracted answer.

    Known limitations:
    - Synthetic questions are lexically similar to the source chunk, which
      inflates retrieval recall (the generating chunk is almost always retrieved).
    - Do NOT use synthetic data as the sole benchmark — use it to augment
      the gold set for scale, not to replace it.
    - Generation is slow on CPU (~5-10s per chunk for Llama 3.2 1B).

    Args:
        chunk_texts: Raw text of each document chunk.
        chunk_ids: Stable IDs for each chunk (used in QAPair.id).
        n_per_chunk: Number of Q&A pairs to generate per chunk.
        model: Ollama model tag.
        base_url: Ollama server URL.
        temperature: Sampling temperature; higher = more diverse questions.
        max_tokens: Maximum tokens for generation response.
        seed: Base random seed (passed via Ollama options).

    Returns:
        An ``EvalDataset`` of synthetic pairs (source="synthetic").
    """
    if not is_available(base_url):
        raise RuntimeError(
            f"Ollama server not reachable at {base_url}. "
            "Start it with `ollama serve` before generating synthetic data."
        )

    system_prompt = (
        "You are a Q&A pair generator. Given a passage, generate exactly "
        f"{n_per_chunk} question-answer pairs that can be answered from the passage. "
        "Return ONLY a JSON array of objects with keys 'question' and 'answer'. "
        "Questions should be natural, varied, and not copy the passage verbatim. "
        "Answers should be concise and directly supported by the passage text."
    )

    pairs: list[QAPair] = []
    n_failed = 0

    for i, (text, cid) in enumerate(zip(chunk_texts, chunk_ids)):
        prompt = f"Passage:\n{text}\n\nGenerate {n_per_chunk} Q&A pairs as JSON:"
        try:
            response = chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            qa_list = parse_json_response(response)
            if not isinstance(qa_list, list):
                qa_list = [qa_list]

            for j, qa in enumerate(qa_list[:n_per_chunk]):
                if "question" not in qa or "answer" not in qa:
                    continue
                pairs.append(
                    QAPair(
                        id=f"syn_{cid}_{j}",
                        question=str(qa["question"]).strip(),
                        ground_truth=str(qa["answer"]).strip(),
                        category="standard",
                        relevant_sections=[cid],
                        difficulty="medium",
                        source="synthetic",
                        metadata={"source_chunk_id": cid},
                    )
                )

        except Exception as exc:
            logger.warning(f"Failed to generate QA for chunk {cid}: {exc}")
            n_failed += 1

        if (i + 1) % 10 == 0:
            logger.info(f"Generated QA for {i+1}/{len(chunk_texts)} chunks ({n_failed} failures)")

    logger.info(
        f"Synthetic QA generation complete: {len(pairs)} pairs from "
        f"{len(chunk_texts)} chunks ({n_failed} failures)"
    )
    return EvalDataset(
        pairs=pairs,
        version="synthetic",
        description=f"Synthetic QA generated from {len(chunk_texts)} chunks by {model}",
    )


def save_dataset(dataset: EvalDataset, path: str | Path) -> None:
    """Persist an EvalDataset to JSON.

    Args:
        dataset: Dataset to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": dataset.version,
        "description": dataset.description,
        "stats": dataset.stats(),
        "qa_pairs": [
            {
                "id": p.id,
                "question": p.question,
                "ground_truth": p.ground_truth,
                "category": p.category,
                "relevant_sections": p.relevant_sections,
                "difficulty": p.difficulty,
                "source": p.source,
                "ambiguity_note": p.ambiguity_note,
                **p.metadata,
            }
            for p in dataset.pairs
        ],
    }
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(dataset)} pairs to {path}")


def merge_datasets(*datasets: EvalDataset) -> EvalDataset:
    """Concatenate multiple EvalDatasets, deduplicating by question text.

    Args:
        *datasets: Datasets to merge.

    Returns:
        Merged dataset with unique questions.
    """
    seen_questions: set[str] = set()
    merged: list[QAPair] = []

    for ds in datasets:
        for pair in ds.pairs:
            key = pair.question.lower().strip()
            if key not in seen_questions:
                seen_questions.add(key)
                merged.append(pair)

    return EvalDataset(pairs=merged, description="Merged dataset")
