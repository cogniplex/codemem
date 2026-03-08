#!/usr/bin/env python3
"""LongMemEval benchmark harness for codemem.

LongMemEval (ICLR 2025) evaluates long-term conversational memory across 6 question
types testing 5 core abilities:
- Single-session user: Recall user-stated info
- Single-session assistant: Recall assistant-stated info
- Single-session preference: Extract implicit preferences
- Multi-session: Synthesize across sessions
- Knowledge-update: Handle info that changed over time
- Temporal-reasoning: Time-based reasoning

Dataset: 500 questions, each with ~40 haystack sessions (~115K tokens).
Each question is evaluated independently: ingest → recall → generate → score → cleanup.

Requires OPENAI_API_KEY for answer generation.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import OpenAI
from tqdm import tqdm

CODEMEM_BASE = "http://localhost:3179/api"
DEFAULT_DATASET = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"


@dataclass
class Config:
    base_url: str = CODEMEM_BASE
    dataset_path: Path = DEFAULT_DATASET
    recall_limit: int = 10
    graph_depth: int = 2
    mode: str = "codemem"  # baseline | codemem | codemem-graph
    llm_model: str = "gpt-4o"
    eval_model: str = "gpt-4o"
    use_llm_eval: bool = False
    max_questions: int = 0  # 0 = all
    skip_ingest: bool = False
    output: Path | None = None


# ---------------------------------------------------------------------------
# Codemem API client
# ---------------------------------------------------------------------------

class CodememClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> bool:
        try:
            r = self.http.get("/health")
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    def store_memory(
        self,
        content: str,
        namespace: str,
        memory_type: str = "Context",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> str | None:
        r = self.http.post(
            "/memories",
            json={
                "content": content,
                "namespace": namespace,
                "memory_type": memory_type,
                "importance": importance,
                "tags": tags or [],
            },
        )
        if r.status_code == 201:
            return r.json().get("id")
        print(f"  [warn] store failed ({r.status_code}): {r.text[:200]}", file=sys.stderr)
        return None

    def recall(self, query: str, namespace: str, limit: int = 10) -> list[dict]:
        r = self.http.get(
            "/search",
            params={"q": query, "namespace": namespace, "k": limit},
        )
        if r.status_code == 200:
            return r.json().get("results", [])
        return []

    def graph_neighbors(self, node_id: str, depth: int = 2) -> list[dict]:
        r = self.http.get(f"/graph/neighbors/{node_id}", params={"depth": depth})
        if r.status_code == 200:
            return r.json().get("nodes", [])
        return []

    def get_memory(self, memory_id: str) -> dict | None:
        r = self.http.get(f"/memories/{memory_id}")
        if r.status_code == 200:
            return r.json()
        return None

    def delete_namespace(self, namespace: str) -> bool:
        r = self.http.delete(f"/namespaces/{namespace}")
        return r.status_code in (200, 404)

    def consolidate(self, cycle: str) -> bool:
        r = self.http.post(f"/consolidation/{cycle}")
        return r.status_code == 200


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "samples", "questions"):
            if key in data:
                return data[key]
        return [data]
    return data


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_sessions(
    client: CodememClient,
    namespace: str,
    item: dict,
) -> int:
    """Ingest haystack sessions for one question into codemem.

    Stores one memory per session (all turns concatenated).
    """
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])
    session_dates = item.get("haystack_dates", [])
    stored = 0

    for i, (turns, sid, date_str) in enumerate(
        zip(sessions, session_ids, session_dates, strict=True)
    ):
        if not turns:
            continue

        lines = []
        for turn in turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")

        session_content = f"[Session {sid}] [{date_str}]\n" + "\n".join(lines)

        tags = [
            f"question:{item['question_id']}",
            f"session:{sid}",
        ]

        mid = client.store_memory(
            content=session_content,
            namespace=namespace,
            memory_type="Context",
            importance=0.5,
            tags=tags,
        )
        if mid:
            stored += 1

    return stored


# ---------------------------------------------------------------------------
# Recall strategies
# ---------------------------------------------------------------------------

def recall_codemem(
    client: CodememClient,
    question: str,
    namespace: str,
    limit: int,
) -> list[str]:
    results = client.recall(question, namespace, limit=limit)
    return [r.get("content", "") for r in results if r.get("content")]


def recall_codemem_graph(
    client: CodememClient,
    question: str,
    namespace: str,
    limit: int,
    graph_depth: int = 2,
) -> list[str]:
    results = client.recall(question, namespace, limit=limit)

    contents = []
    seen_ids = set()
    for r in results:
        content = r.get("content", "")
        node_id = r.get("id", r.get("node_id", ""))
        if content:
            contents.append(content)
        if node_id:
            seen_ids.add(node_id)

    seed_ids = [
        r.get("id", r.get("node_id", ""))
        for r in results[:10]
        if r.get("id") or r.get("node_id")
    ]
    for seed_id in seed_ids:
        if not seed_id:
            continue
        neighbors = client.graph_neighbors(seed_id, depth=graph_depth)
        for node in neighbors:
            memory_id = node.get("memory_id", node.get("id", ""))
            if not memory_id or memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)
            if node.get("kind") == "Memory":
                mem = client.get_memory(memory_id)
                if mem and mem.get("content"):
                    contents.append(mem["content"])

    return contents[:limit * 2]


def recall_baseline(item: dict) -> list[str]:
    """Baseline: return all session text."""
    contents = []
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])
    session_dates = item.get("haystack_dates", [])

    for turns, sid, date_str in zip(sessions, session_ids, session_dates, strict=True):
        if not turns:
            continue
        lines = [f"[Session {sid}] [{date_str}]"]
        for turn in turns:
            role = turn.get("role", "unknown")
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.get('content', '')}")
        contents.append("\n".join(lines))
    return contents


# ---------------------------------------------------------------------------
# Answer generation (LLM required)
# ---------------------------------------------------------------------------

def generate_answer(
    openai_client: OpenAI,
    question: str,
    memories: list[str],
    question_date: str,
    model: str = "gpt-4o",
) -> str:
    """Generate an answer using LLM with recalled memories as context."""
    if not memories:
        return "I don't know."

    context_parts = []
    for i, mem in enumerate(memories, 1):
        context_parts.append(f"[Memory {i}]\n{mem}")
    context = "\n\n".join(context_parts)

    # Truncate context to ~100K chars to stay within token limits
    if len(context) > 100_000:
        context = context[:100_000] + "\n\n[...truncated...]"

    prompt = f"""You are answering a question based on recalled conversation memories.

First, extract relevant information from each memory excerpt.
Then, reason about the answer based on the extracted information.
Finally, provide a concise answer.

If the information is not available in the provided memories, respond with "I don't know."

Memories:
{context}

Question (asked on {question_date}): {question}

Step 1 - Extract relevant information:
Step 2 - Reasoning:
Step 3 - Answer:"""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        full_answer = response.choices[0].message.content.strip()

        # Extract Step 3 answer
        step3_match = re.search(
            r"Step\s*3(?:\s*[\.\-:]|\s)*\s*(?:Answer|Final Answer)?\s*[:\-]?\s*",
            full_answer,
            flags=re.IGNORECASE,
        )
        if step3_match:
            trailing = full_answer[step3_match.end():]
            next_step = re.search(r"\n\s*Step\s*\d+\s*[\.\-:]", trailing, flags=re.IGNORECASE)
            answer_part = trailing[:next_step.start()] if next_step else trailing
            answer_part = re.sub(
                r"^\s*(?:Answer|Final Answer)\s*[:\-]?\s*", "", answer_part, flags=re.IGNORECASE
            ).strip()
            if answer_part:
                return answer_part

        return full_answer

    except Exception as e:
        print(f"  [warn] LLM error: {e}", file=sys.stderr)
        return "I don't know."


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

ABSTENTION_PHRASES = [
    "i don't know", "i do not know", "cannot determine",
    "not enough information", "no information", "not mentioned",
    "unable to determine", "no relevant", "cannot be determined",
    "isn't mentioned", "not available", "don't have enough",
    "do not have enough", "no memory", "no record",
]


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an|is|was|were|are|am)\b", " ", s)
    for word, digit in NUMBER_WORDS.items():
        s = re.sub(rf"\b{word}\b", digit, s)
    return re.sub(r"\s+", " ", s).strip()


def f1_token_overlap(hypothesis: str, reference: str) -> float:
    h_tokens = normalize_text(hypothesis).split()
    r_tokens = normalize_text(reference).split()
    if not h_tokens or not r_tokens:
        return 0.0
    common = sum((Counter(h_tokens) & Counter(r_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(h_tokens)
    recall = common / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def is_abstention_question(question_id: str) -> bool:
    return question_id.endswith("_abs")


def quick_score(hypothesis: str, reference: str, question_id: str) -> dict:
    """Local scoring: exact match, substring, F1 token overlap."""
    if is_abstention_question(question_id):
        h_lower = normalize_text(hypothesis)
        abstained = any(phrase in h_lower for phrase in ABSTENTION_PHRASES)
        return {"is_correct": abstained, "f1": 0.0, "explanation": f"abstention={'correct' if abstained else 'missed'}"}

    h_norm = normalize_text(hypothesis)
    r_norm = normalize_text(reference)

    # Exact match
    if h_norm == r_norm:
        return {"is_correct": True, "f1": 1.0, "explanation": "exact match"}

    # Substring match
    if r_norm in h_norm or h_norm in r_norm:
        return {"is_correct": True, "f1": 1.0, "explanation": "substring match"}

    # F1 token overlap
    f1 = f1_token_overlap(hypothesis, reference)
    is_correct = f1 >= 0.5
    return {"is_correct": is_correct, "f1": round(f1, 4), "explanation": f"f1={f1:.3f}"}


def llm_evaluate(
    openai_client: OpenAI,
    question: str,
    hypothesis: str,
    reference: str,
    question_id: str,
    model: str = "gpt-4o",
    is_preference: bool = False,
) -> dict:
    """GPT-4o binary judge matching LongMemEval paper methodology."""
    if is_abstention_question(question_id):
        h_lower = normalize_text(hypothesis)
        abstained = any(phrase in h_lower for phrase in ABSTENTION_PHRASES)
        return {"is_correct": abstained, "confidence": 0.9 if abstained else 0.1, "explanation": "abstention check"}

    if is_preference:
        prompt = f"""You are evaluating whether an AI assistant's answer aligns with a user's known preferences.

Question: {question}

User preference rubric: {reference}

Assistant's answer: {hypothesis}

Judge whether the assistant's answer is consistent with the user's known preferences described in the rubric. The answer is correct if it reflects the user's preferences (e.g., mentioning the right tools, topics, or approaches). It does NOT need to match the rubric text — it should demonstrate awareness of the user's preferences.

Respond with ONLY a JSON object:
{{"correct": true/false, "confidence": 0.0-1.0, "explanation": "brief reason"}}"""
    else:
        prompt = f"""You are evaluating whether an AI assistant's answer to a question is correct.

Question: {question}

Reference answer: {reference}

Assistant's answer: {hypothesis}

Judge whether the assistant's answer is correct. The answer doesn't need to be word-for-word identical, but it should convey the same key information as the reference answer. Minor variations in phrasing, additional context, or slightly different formatting are acceptable as long as the core answer is correct.

Respond with ONLY a JSON object:
{{"correct": true/false, "confidence": 0.0-1.0, "explanation": "brief reason"}}"""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown fences
        fence = re.match(r"^\s*```[a-zA-Z]*\s*\n(?P<body>.*)\n\s*```\s*$", content, re.S)
        if fence:
            content = fence.group("body").strip()
        result = json.loads(content)
        return {
            "is_correct": result.get("correct", False),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", ""),
        }
    except Exception as e:
        # Fall back to quick score
        qs = quick_score(hypothesis, reference, question_id)
        return {"is_correct": qs["is_correct"], "confidence": 0.5, "explanation": f"LLM eval failed ({e}), quick score fallback"}


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

QUESTION_TYPE_NAMES = {
    "single-session-user": "Single-Session (User)",
    "single-session-assistant": "Single-Session (Asst)",
    "single-session-preference": "Single-Session (Pref)",
    "multi-session": "Multi-Session",
    "knowledge-update": "Knowledge Update",
    "temporal-reasoning": "Temporal Reasoning",
}


def run_benchmark(config: Config) -> dict:
    client = CodememClient(config.base_url)
    openai_client = OpenAI()

    # Health check
    if config.mode != "baseline" and not client.health():
        print("ERROR: codemem server not reachable at", config.base_url, file=sys.stderr)
        print("Start it with: codemem serve --api --port 3179", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    if not config.dataset_path.exists():
        print(f"ERROR: Dataset not found at {config.dataset_path}", file=sys.stderr)
        print("Download from: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset(config.dataset_path)
    print(f"Loaded {len(dataset)} questions")

    if config.max_questions > 0:
        dataset = dataset[:config.max_questions]
        print(f"Limited to {len(dataset)} questions")

    all_results = []
    type_stats: dict[str, dict] = {}

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        question_id = item["question_id"]
        question = item["question"]
        reference = item.get("answer", "")
        question_type = item.get("question_type", "unknown")
        question_date = item.get("question_date", "")
        namespace = f"longmemeval-{question_id}"

        # 1. Ingest
        if config.mode != "baseline" and not config.skip_ingest:
            client.delete_namespace(namespace)
            time.sleep(0.3)
            stored = ingest_sessions(client, namespace, item)
            # Build graph edges between related sessions
            if config.mode == "codemem-graph":
                client.consolidate("creative")
            time.sleep(0.5)
        else:
            stored = 0

        # 2. Recall — use higher limits for multi-session and temporal questions
        recall_limit = config.recall_limit
        if question_type in ("multi-session", "knowledge-update", "temporal-reasoning"):
            recall_limit = max(recall_limit, 15)

        if config.mode == "baseline":
            memories = recall_baseline(item)
        elif config.mode == "codemem-graph":
            memories = recall_codemem_graph(
                client, question, namespace, recall_limit, config.graph_depth,
            )
        else:
            memories = recall_codemem(client, question, namespace, recall_limit)

        # 3. Generate answer
        hypothesis = generate_answer(
            openai_client, question, memories, question_date, config.llm_model,
        )

        # 4. Score — preference questions always use LLM eval since their
        # references are qualitative rubrics, not factual answers.
        use_llm = config.use_llm_eval or question_type == "single-session-preference"
        if use_llm:
            score_result = llm_evaluate(
                openai_client, question, hypothesis, reference, question_id, config.eval_model,
                is_preference=(question_type == "single-session-preference"),
            )
        else:
            score_result = quick_score(hypothesis, reference, question_id)

        result = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "reference": reference,
            "hypothesis": hypothesis,
            "is_correct": score_result["is_correct"],
            "confidence": score_result.get("confidence", score_result.get("f1", 0)),
            "explanation": score_result["explanation"],
            "memories_recalled": len(memories),
            "memories_stored": stored,
            "mode": config.mode,
        }
        all_results.append(result)

        # Track per-type
        if question_type not in type_stats:
            type_stats[question_type] = {"correct": 0, "total": 0}
        type_stats[question_type]["total"] += 1
        if score_result["is_correct"]:
            type_stats[question_type]["correct"] += 1

        # 5. Cleanup (per-question, like LongMemEval expects)
        if config.mode != "baseline" and not config.skip_ingest:
            client.delete_namespace(namespace)

    # Summary
    total_correct = sum(v["correct"] for v in type_stats.values())
    total_questions = sum(v["total"] for v in type_stats.values())
    overall = total_correct / total_questions * 100 if total_questions else 0

    print(f"\n{'='*60}")
    print(f"LongMemEval Results — Mode: {config.mode}")
    print(f"{'='*60}")
    print(f"\n{'Type':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*10}")
    for qt in sorted(type_stats.keys()):
        s = type_stats[qt]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        name = QUESTION_TYPE_NAMES.get(qt, qt)
        print(f"{name:<30} {s['correct']:>8} {s['total']:>8} {acc:>9.1f}%")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*10}")
    print(f"{'OVERALL':<30} {total_correct:>8} {total_questions:>8} {overall:>9.1f}%")

    print(f"\n--- Landscape ---")
    print(f"  Oracle gpt-4o:     82.4%")
    print(f"  Zep:               71.2%")
    print(f"  Naive RAG:         52.0%")
    print(f"  Best Guess:        18.8%")

    output = {
        "mode": config.mode,
        "metric": "longmemeval",
        "llm_model": config.llm_model,
        "recall_limit": config.recall_limit,
        "total_questions": len(all_results),
        "type_stats": type_stats,
        "overall_accuracy": round(overall, 2),
        "results": all_results,
    }

    # Save
    out_path = config.output
    if not out_path:
        out_path = Path(__file__).parent / "results" / f"{config.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


# ---------------------------------------------------------------------------
# Rescore existing results
# ---------------------------------------------------------------------------

def rescore_results(results_path: Path, args):
    """Re-score an existing results file with different scoring settings.

    Useful for re-evaluating with LLM judge or preference-aware scoring
    without re-running the full ingest/recall/generation pipeline.
    """
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    type_stats: dict[str, dict] = {}
    new_results = []

    for r in tqdm(results, desc="Re-scoring"):
        question_type = r["question_type"]
        hypothesis = r["hypothesis"]
        reference = r["reference"]
        question_id = r["question_id"]
        question = r["question"]

        use_llm = args.llm_eval or question_type == "single-session-preference"
        if use_llm:
            score_result = llm_evaluate(
                openai_client, question, hypothesis, reference, question_id,
                args.eval_model,
                is_preference=(question_type == "single-session-preference"),
            )
        else:
            score_result = quick_score(hypothesis, reference, question_id)

        r["is_correct"] = score_result["is_correct"]
        r["confidence"] = score_result.get("confidence", score_result.get("f1", 0))
        r["explanation"] = score_result["explanation"]
        new_results.append(r)

        if question_type not in type_stats:
            type_stats[question_type] = {"correct": 0, "total": 0}
        type_stats[question_type]["total"] += 1
        if score_result["is_correct"]:
            type_stats[question_type]["correct"] += 1

    total_correct = sum(v["correct"] for v in type_stats.values())
    total_questions = sum(v["total"] for v in type_stats.values())
    overall = total_correct / total_questions * 100 if total_questions else 0

    print(f"\n{'='*60}")
    print(f"Re-scored: {results_path.name}")
    print(f"{'='*60}")
    print(f"\n{'Type':<35s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*10}")

    for qt in sorted(type_stats.keys()):
        name = QUESTION_TYPE_NAMES.get(qt, qt)
        s = type_stats[qt]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"{name:<35s} {s['correct']:>8d} {s['total']:>8d} {acc:>9.1f}%")

    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*10}")
    print(f"{'OVERALL':<35s} {total_correct:>8d} {total_questions:>8d} {overall:>9.1f}%")

    # Save rescored results
    data["results"] = new_results
    data["type_stats"] = type_stats
    data["overall_accuracy"] = overall
    out_path = args.output or results_path.with_suffix(".rescored.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nRe-scored results saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark harness for codemem")
    parser.add_argument("--mode", default="codemem",
                        choices=["baseline", "codemem", "codemem-graph"],
                        help="Recall strategy to benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--base-url", default=CODEMEM_BASE,
                        help="Codemem API base URL")
    parser.add_argument("--recall-limit", type=int, default=10,
                        help="Number of memories to recall per question")
    parser.add_argument("--graph-depth", type=int, default=2,
                        help="Graph expansion depth for codemem-graph mode")
    parser.add_argument("--llm-model", default="gpt-4o",
                        help="LLM model for answer generation")
    parser.add_argument("--eval-model", default="gpt-4o",
                        help="LLM model for evaluation judge")
    parser.add_argument("--llm-eval", action="store_true",
                        help="Use GPT-4o judge instead of quick_score")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Limit number of questions (0 = all)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion, reuse existing memories")
    parser.add_argument("--output", type=Path,
                        help="Output file path for results JSON")
    parser.add_argument("--rescore", type=Path, metavar="RESULTS_JSON",
                        help="Re-score existing results file (skip ingest/recall/generation)")
    args = parser.parse_args()

    if args.rescore:
        rescore_results(args.rescore, args)
    else:
        config = Config(
            base_url=args.base_url,
            dataset_path=args.dataset,
            recall_limit=args.recall_limit,
            graph_depth=args.graph_depth,
            mode=args.mode,
            llm_model=args.llm_model,
            eval_model=args.eval_model,
            use_llm_eval=args.llm_eval,
            max_questions=args.max_questions,
            skip_ingest=args.skip_ingest,
            output=args.output,
        )
        run_benchmark(config)


if __name__ == "__main__":
    main()
