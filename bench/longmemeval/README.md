# LongMemEval Benchmark for Codemem

Evaluates codemem's long-term conversational memory against the [LongMemEval benchmark](https://arxiv.org/abs/2410.10813) (ICLR 2025).

## Results

| System | Accuracy | Recall Limit | LLM Judge | Notes |
|--------|----------|-------------|-----------|-------|
| **Codemem** | **70%** | **10** | GPT-4o | Graph-vector hybrid, OpenAI text-embedding-3-small, no evidence oracle |
| Oracle gpt-4o | 82.4% | — | GPT-4o | All sessions provided as context (upper bound) |
| Zep | 71.2% | — | GPT-4o | |
| Naive RAG | 52.0% | — | GPT-4o | Vector-only retrieval |
| Best Guess | 18.8% | — | — | No memory at all |

### Breakdown by question type

| Type | Count | Accuracy |
|------|-------|----------|
| Single-Session (User) | 43 | 81.4% |
| Single-Session (Assistant) | 55 | 78.2% |
| Single-Session (Preference) | 37 | 56.8% |
| Multi-Session | 87 | 58.6% |
| Knowledge Update | 63 | 71.4% |
| Temporal Reasoning | 68 | 52.9% |
| Abstention | 147 | 80.3% |

### Why this matters

LongMemEval is significantly harder than LoCoMo — it tests whether retrieved memories actually enable correct answer generation, not just whether the right text was recalled. Each question involves ~40 sessions (~115K tokens), requiring precise retrieval from a large haystack.

All published leaderboard scores use LLM evaluation (GPT-4o judge), making results directly comparable. Codemem achieves near-Zep accuracy with a recall limit of just 10 memories per query, compared to systems that retrieve far more context.

## How it differs from LoCoMo

| | LoCoMo | LongMemEval |
|---|---|---|
| **Evaluation** | Retrieval accuracy (is answer in memories?) | Generative QA (LLM produces answer, judge scores) |
| **Dataset** | 10 conversations, 1,986 questions | 500 independent questions, ~40 sessions each |
| **Token scale** | ~10K tokens per conversation | ~115K tokens per question |
| **LLM required** | No | Yes (answer generation + judge) |
| **Question types** | 5 categories (single-hop, temporal, multi-hop, open-domain, adversarial) | 6 types + abstention |

## Our approach

- **No evidence oracle** — retrieval finds memories on its own
- **Recall limit of 10** — forces precise retrieval over brute-force
- **Per-session storage** — one memory per conversation session
- **Per-question lifecycle** — ingest, recall, generate, score, cleanup for each question independently
- **LLM eval** with rubric-aware prompts for preference questions, binary correctness for factual questions
- **Rescore mode** — re-evaluate saved results without re-running the full pipeline

## Setup

```bash
cd bench/longmemeval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Download Dataset

From the [LongMemEval GitHub repo](https://github.com/xiaowu0162/LongMemEval):

```bash
mkdir -p data
git clone https://github.com/xiaowu0162/LongMemEval /tmp/LongMemEval
cd /tmp/LongMemEval && bash download_data.sh
cp /tmp/LongMemEval/data/longmemeval_s_cleaned.json bench/longmemeval/data/
```

The file needed is `longmemeval_s_cleaned.json` (500 questions, LongMemEval-S variant).

## Run

```bash
# Start codemem server
codemem serve --api --port 3179

# Quick test (5 questions)
python3 harness.py --max-questions 5

# Full benchmark (500 questions)
python3 harness.py --llm-eval

# Different modes
python3 harness.py --mode baseline        # Full context upper bound
python3 harness.py --mode codemem         # Hybrid recall (default)
python3 harness.py --mode codemem-graph   # Hybrid + graph expansion

# Re-score existing results without re-running pipeline
python3 harness.py --rescore results/latest.json

# Custom recall limit
python3 harness.py --recall-limit 20
```

## Scoring

Two scoring modes:

1. **Local quick_score** (default, free): exact match, substring match, or F1 token overlap >= 0.5
2. **GPT-4o judge** (`--llm-eval`): binary correctness judgment matching the LongMemEval paper methodology

Preference questions automatically use LLM eval with a rubric-aware prompt, since their references are qualitative rubrics rather than factual answers.

Abstention questions (ID ending in `_abs`) are scored separately — the model should respond with "I don't know" or equivalent.

## Question Types

| Type | Description |
|------|-------------|
| Single-Session (User) | Recall info stated by the user |
| Single-Session (Assistant) | Recall info stated by the assistant |
| Single-Session (Preference) | Extract implicit user preferences |
| Multi-Session | Synthesize information across sessions |
| Knowledge Update | Handle information that changed over time |
| Temporal Reasoning | Time-based reasoning about events |
