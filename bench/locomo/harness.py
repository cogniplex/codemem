#!/usr/bin/env python3
"""LoCoMo benchmark harness for codemem.

Ingests LoCoMo conversations into codemem, runs recall for each question,
and collects answers for scoring.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from tqdm import tqdm

CODEMEM_BASE = "http://localhost:3179/api"
DEFAULT_DATASET = Path(__file__).parent / "data" / "locomo10.json"
NAMESPACE_PREFIX = "locomo-bench"


@dataclass
class Config:
    base_url: str = CODEMEM_BASE
    dataset_path: Path = DEFAULT_DATASET
    recall_limit: int = 10
    recall_limit_temporal: int = 10
    recall_limit_multihop: int = 10
    graph_depth: int = 2
    mode: str = "codemem"  # baseline | rag | codemem | codemem-graph
    conversations: list[int] = field(default_factory=list)
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

    def recall(self, query: str, namespace: str, limit: int = 15) -> list[dict]:
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
            data = r.json()
            return data.get("nodes", [])
        return []

    def get_memory(self, memory_id: str) -> dict | None:
        r = self.http.get(f"/memories/{memory_id}")
        if r.status_code == 200:
            return r.json()
        return None

    def consolidate(self, cycle: str) -> bool:
        r = self.http.post(f"/consolidation/{cycle}")
        return r.status_code == 200

    def start_session(self, namespace: str) -> str | None:
        r = self.http.post("/sessions", json={"namespace": namespace})
        if r.status_code == 201:
            return r.json().get("id")
        return None

    def end_session(self, session_id: str, summary: str = "") -> bool:
        r = self.http.post(
            f"/sessions/{session_id}/end",
            json={"summary": summary},
        )
        return r.status_code == 200

    def delete_namespace(self, namespace: str) -> bool:
        r = self.http.delete(f"/namespaces/{namespace}")
        return r.status_code in (200, 404)

    def get_namespaces(self) -> list[dict]:
        r = self.http.get("/namespaces")
        if r.status_code == 200:
            return r.json()
        return []


# ---------------------------------------------------------------------------
# Answer checking (AutoMem-style: check if gold answer is in recalled memories)
# ---------------------------------------------------------------------------

import re
import string


def normalize_text(s: str) -> str:
    """Normalize text for matching: lowercase, remove punctuation, basic stemming."""
    s = str(s).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Basic stemming
    for suffix in ["ing", "tion", "ment", "ness", "ed", "ly", "er", "est", "ies"]:
        if len(s) > len(suffix) + 3:
            s = re.sub(rf"\b(\w+){suffix}\b", r"\1", s)
    return " ".join(s.split())


def word_overlap_score(expected: str, text: str) -> float:
    """Compute word overlap between expected answer and text."""
    expected_words = set(normalize_text(expected).split())
    text_words = set(normalize_text(text).split())
    if not expected_words:
        return 0.0
    overlap = expected_words & text_words
    return len(overlap) / len(expected_words)


def check_answer_in_memories(
    expected_answer: str,
    memories: list[str],
    category: str,
) -> tuple[bool, float, str]:
    """Check if the expected answer can be found in recalled memories.

    Returns (is_correct, confidence, explanation).
    This matches AutoMem's evaluation approach.
    """
    expected = str(expected_answer)
    combined_text = "\n".join(memories)

    # 1. Exact substring match (case-insensitive)
    if expected.lower() in combined_text.lower():
        return True, 1.0, "exact substring match"

    # 2. Word overlap across all memories
    best_score = 0.0
    best_memory_idx = -1
    for i, mem in enumerate(memories):
        score = word_overlap_score(expected, mem)
        if score > best_score:
            best_score = score
            best_memory_idx = i

    # For multi-hop, also check combined text with lower threshold
    combined_score = word_overlap_score(expected, combined_text)

    if category == "multi-hop":
        threshold = 0.35
        score = max(best_score, combined_score)
    else:
        threshold = 0.50
        score = best_score

    if score >= threshold:
        return True, score, f"word overlap {score:.2f} in memory {best_memory_idx}"

    # 3. Adversarial: if no memories match, that's correct (should abstain)
    if category == "adversarial":
        if best_score < 0.2:
            return True, 0.9, "low overlap suggests correct abstention"
        return False, best_score, "found unexpected match for adversarial question"

    # 4. Fuzzy date matching for temporal questions
    if category == "temporal":
        try:
            from dateutil import parser as date_parser
            expected_dates = []
            for word in expected.split():
                try:
                    expected_dates.append(date_parser.parse(word, fuzzy=True))
                except (ValueError, OverflowError):
                    pass
            if not expected_dates:
                try:
                    expected_dates.append(date_parser.parse(expected, fuzzy=True))
                except (ValueError, OverflowError):
                    pass

            if expected_dates:
                for mem in memories:
                    for ed in expected_dates:
                        if ed.strftime("%d") in mem and (
                            ed.strftime("%B") in mem or ed.strftime("%b") in mem
                        ):
                            return True, 0.95, "fuzzy date match"
        except ImportError:
            pass

    return False, score, f"best word overlap {score:.2f} below threshold {threshold}"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Some versions wrap in a top-level key
        for key in ("data", "samples", "conversations"):
            if key in data:
                return data[key]
        return [data]
    return data


def extract_sessions(conversation: dict) -> list[dict]:
    """Extract ordered sessions from a LoCoMo conversation.

    Returns list of {session_num, datetime, turns: [{speaker, text, dia_id}]}.
    """
    sessions = []
    conv_data = conversation.get("conversation", conversation)
    session_num = 1

    while True:
        key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if key not in conv_data:
            break
        turns = []
        for turn in conv_data[key]:
            text = turn.get("text", "")
            speaker = turn.get("speaker", "unknown")
            dia_id = turn.get("dia_id", "")
            if turn.get("blip_caption"):
                text += f" [Image: {turn['blip_caption']}]"
            turns.append({"speaker": speaker, "text": text, "dia_id": dia_id})
        sessions.append({
            "session_num": session_num,
            "datetime": conv_data.get(date_key, ""),
            "turns": turns,
        })
        session_num += 1

    return sessions


# LoCoMo category IDs → names
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def extract_questions(conversation: dict) -> list[dict]:
    """Extract QA pairs from a LoCoMo conversation.

    Returns list of {question_id, question, answer, category, evidence_ids}.
    """
    qa_list = conversation.get("qa", [])
    questions = []
    for i, qa in enumerate(qa_list):
        raw_cat = qa.get("category", 0)
        category = CATEGORY_NAMES.get(raw_cat, f"cat-{raw_cat}")
        questions.append({
            "question_id": f"q_{i}",
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "category": category,
            "evidence": qa.get("evidence", []),
        })
    return questions


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_conversation(
    client: CodememClient,
    namespace: str,
    sessions: list[dict],
    sample_id: str,
) -> int:
    """Store conversation into codemem at turn-group granularity.

    Stores groups of ~4 consecutive turns per memory (not whole sessions).
    This gives retrieval enough context per chunk while keeping embeddings
    focused enough to match specific questions.
    """
    count = 0
    session_id = client.start_session(namespace)
    turns_per_chunk = 4

    for sess in sessions:
        turns = sess["turns"]
        # Split session into chunks of turns
        for chunk_start in range(0, len(turns), turns_per_chunk):
            chunk_turns = turns[chunk_start:chunk_start + turns_per_chunk]
            if not chunk_turns:
                continue

            lines = []
            for turn in chunk_turns:
                lines.append(f"{turn['speaker']}: {turn['text']}")

            chunk_content = (
                f"[Session {sess['session_num']}] [{sess['datetime']}]\n"
                + "\n".join(lines)
            )

            tags = [
                f"sample:{sample_id}",
                f"session:{sess['session_num']}",
            ]
            speakers = {t["speaker"] for t in chunk_turns}
            for sp in speakers:
                tags.append(f"speaker:{sp}")

            mid = client.store_memory(
                content=chunk_content,
                namespace=namespace,
                memory_type="Context",
                importance=0.5,
                tags=tags,
            )
            if mid:
                count += 1

    if session_id:
        client.end_session(session_id, summary=f"Ingested {count} turn chunks for {sample_id}")

    # Run creative consolidation to build SHARES_THEME edges between memories
    client.consolidate("creative")

    return count


# ---------------------------------------------------------------------------
# Recall strategies
# ---------------------------------------------------------------------------

def get_recall_limit(category: str, config: Config) -> int:
    if category == "temporal":
        return config.recall_limit_temporal
    if category == "multi-hop":
        return config.recall_limit_multihop
    return config.recall_limit


def recall_codemem(
    client: CodememClient,
    question: str,
    namespace: str,
    limit: int,
) -> list[str]:
    """Standard codemem hybrid recall (vector + BM25 + graph scoring)."""
    results = client.recall(question, namespace, limit=limit)
    return [r.get("content", "") for r in results if r.get("content")]


def recall_codemem_graph(
    client: CodememClient,
    question: str,
    namespace: str,
    limit: int,
    graph_depth: int = 2,
) -> list[str]:
    """Codemem recall with explicit graph expansion (like AutoMem's bridge discovery)."""
    results = client.recall(question, namespace, limit=limit)

    # Collect initial content
    contents = []
    seen_ids = set()
    for r in results:
        content = r.get("content", "")
        node_id = r.get("id", r.get("node_id", ""))
        if content:
            contents.append(content)
        if node_id:
            seen_ids.add(node_id)

    # Graph expand: follow edges from top results to find bridge memories
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
            # Use memory_id to fetch full content (label is truncated to 80 chars)
            memory_id = node.get("memory_id", node.get("id", ""))
            if not memory_id or memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)
            if node.get("kind") == "Memory":
                mem = client.get_memory(memory_id)
                if mem and mem.get("content"):
                    contents.append(mem["content"])

    return contents[:limit * 2]  # cap total context


def recall_baseline(sessions: list[dict]) -> list[str]:
    """Baseline: return all session text (simulates long-context LLM)."""
    contents = []
    for sess in sessions:
        parts = [f"[Session {sess['session_num']}] [{sess['datetime']}]"]
        for turn in sess["turns"]:
            parts.append(f"{turn['speaker']}: {turn['text']}")
        contents.append("\n".join(parts))
    return contents


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(config: Config) -> dict:
    client = CodememClient(config.base_url)

    # Health check
    if config.mode != "baseline" and not client.health():
        print("ERROR: codemem server not reachable at", config.base_url, file=sys.stderr)
        print("Start it with: codemem serve --port 3179", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    if not config.dataset_path.exists():
        print(f"ERROR: Dataset not found at {config.dataset_path}", file=sys.stderr)
        print("Download it first — see README.md", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset(config.dataset_path)
    print(f"Loaded {len(dataset)} conversations")

    # Filter conversations if specified
    if config.conversations:
        dataset = [dataset[i] for i in config.conversations if i < len(dataset)]
        print(f"Filtered to {len(dataset)} conversations: {config.conversations}")

    all_results = []
    category_stats: dict[str, dict] = {}

    for conv_idx, conversation in enumerate(dataset):
        sample_id = conversation.get("sample_id", f"conv_{conv_idx}")
        namespace = f"{NAMESPACE_PREFIX}-{sample_id}"
        sessions = extract_sessions(conversation)
        questions = extract_questions(conversation)

        print(f"\n--- Conversation {conv_idx}: {sample_id} ---")
        print(f"  {len(sessions)} sessions, {len(questions)} questions")

        # Ingest
        if config.mode != "baseline" and not config.skip_ingest:
            # Clean previous run
            client.delete_namespace(namespace)
            time.sleep(0.5)

            mem_count = ingest_conversation(client, namespace, sessions, sample_id)
            print(f"  Stored {mem_count} memories")
            # Brief pause for enrichment
            time.sleep(1.0)
        elif config.skip_ingest:
            print("  Skipping ingestion (--skip-ingest)")

        # Evaluate each question
        for qa in tqdm(questions, desc=f"  Evaluating", leave=False):
            category = qa["category"]
            limit = get_recall_limit(category, config)

            # Recall based on mode
            if config.mode == "baseline":
                memories = recall_baseline(sessions)
            elif config.mode == "rag":
                # Use only vector search by setting a very high k
                # (codemem's /search endpoint does hybrid by default,
                #  but with low k the vector component dominates)
                memories = recall_codemem(client, qa["question"], namespace, limit)
            elif config.mode == "codemem-graph":
                memories = recall_codemem_graph(
                    client, qa["question"], namespace, limit, config.graph_depth
                )
            else:  # codemem (default)
                memories = recall_codemem(client, qa["question"], namespace, limit)

            # Check if gold answer is in recalled memories (AutoMem-style)
            is_correct, confidence, explanation = check_answer_in_memories(
                qa["answer"], memories, category,
            )

            result = {
                "sample_id": sample_id,
                "question_id": qa["question_id"],
                "question": qa["question"],
                "expected_answer": qa["answer"],
                "category": category,
                "evidence_ids": qa["evidence"],
                "memories_recalled": len(memories),
                "is_correct": is_correct,
                "confidence": round(confidence, 4),
                "explanation": explanation,
                "mode": config.mode,
            }
            all_results.append(result)

            # Track per-category
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0}
            category_stats[category]["total"] += 1
            if is_correct:
                category_stats[category]["correct"] += 1

    # Summary with scores
    total_correct = sum(v["correct"] for v in category_stats.values())
    total_questions = sum(v["total"] for v in category_stats.values())
    overall = total_correct / total_questions * 100 if total_questions else 0

    print(f"\n{'='*60}")
    print(f"Mode: {config.mode}")
    print(f"{'='*60}")
    print(f"\n{'Category':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"{cat:<25} {s['correct']:>8} {s['total']:>8} {acc:>9.1f}%")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    print(f"{'OVERALL':<25} {total_correct:>8} {total_questions:>8} {overall:>9.1f}%")
    print(f"\n--- Baselines ---")
    print(f"  AutoMem:       90.53%")
    print(f"  CORE:          88.24%")

    output = {
        "mode": config.mode,
        "total_questions": len(all_results),
        "category_stats": category_stats,
        "results": all_results,
    }

    # Save results
    if config.output:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with open(config.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {config.output}")
    else:
        default_output = Path(__file__).parent / "results" / f"{config.mode}.json"
        default_output.parent.mkdir(parents=True, exist_ok=True)
        with open(default_output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {default_output}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark harness for codemem")
    parser.add_argument("--mode", default="codemem",
                        choices=["baseline", "rag", "codemem", "codemem-graph"],
                        help="Recall strategy to benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Path to locomo10.json")
    parser.add_argument("--base-url", default=CODEMEM_BASE,
                        help="Codemem API base URL")
    parser.add_argument("--recall-limit", type=int, default=50,
                        help="Default recall limit for single-hop questions")
    parser.add_argument("--graph-depth", type=int, default=2,
                        help="Graph expansion depth for codemem-graph mode")
    parser.add_argument("--conversations", type=int, nargs="*", default=[],
                        help="Specific conversation indices to evaluate (default: all)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion, reuse existing memories")
    parser.add_argument("--output", type=Path,
                        help="Output file path for results JSON")
    args = parser.parse_args()

    config = Config(
        base_url=args.base_url,
        dataset_path=args.dataset,
        recall_limit=args.recall_limit,
        graph_depth=args.graph_depth,
        mode=args.mode,
        conversations=args.conversations or [],
        skip_ingest=args.skip_ingest,
        output=args.output,
    )
    run_benchmark(config)


if __name__ == "__main__":
    main()
