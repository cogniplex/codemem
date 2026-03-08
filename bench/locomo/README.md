# LoCoMo Benchmark for Codemem

Evaluates codemem's long-term conversational memory against the [LoCoMo benchmark](https://snap-research.github.io/locomo/) (ACL 2024).

## Results

Full dataset: 10 conversations, 1,986 questions.

| System | Accuracy | Recall Limit | Evidence Oracle | Embedding Fallback |
|--------|----------|-------------|----------------|-------------------|
| **Codemem** (OpenAI embed) | **91.64%** | **10** | No | No |
| **Codemem** (local BERT) | **89.58%** | **10** | No | No |
| Published SOTA | 90.53% | 50-100 | Yes | Yes |
| CORE | 88.24% | -- | -- | -- |

### Embedding model comparison

| Mode | OpenAI text-embedding-3-small | Local BERT (bge-base-en-v1.5) | Delta |
|------|------|------|-------|
| codemem | 91.64% | 89.58% | +2.06% |
| codemem-graph | 91.49% | 91.49% | 0% |

Graph expansion closes the gap entirely for BERT (89.58% → 91.49%) but doesn't help OpenAI (91.64% → 91.49%). Better embeddings already retrieve what graph expansion would find, making the two modes converge.

### Breakdown by category (OpenAI codemem, 1,986 questions)

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Adversarial | 446 | 446 | 100.0% |
| Open-domain | 833 | 841 | 99.0% |
| Temporal | 270 | 321 | 84.1% |
| Single-hop | 215 | 282 | 76.2% |
| Multi-hop | 56 | 96 | 58.3% |
| **Overall** | **1820** | **1986** | **91.64%** |

### Why this matters

Most published LoCoMo benchmarks inflate their scores through techniques that bypass actual retrieval quality:

- **Evidence oracle**: Using ground-truth evidence IDs from the dataset to directly fetch the memories containing the answer, rather than relying on the retrieval system to find them.
- **High recall limits**: Retrieving 50-100 memories per question out of ~100-400 total. At that ratio, you're returning most of the conversation.
- **Embedding similarity fallback**: When word overlap fails, falling back to cosine similarity, effectively adding a second retrieval pass.
- **Low thresholds**: Word overlap thresholds as low as 0.30, which produces false positives.

Codemem's benchmark takes a stricter approach:

- **No evidence oracle** -- retrieval must find the right memories on its own
- **Recall limit of 10** -- forces the system to retrieve precisely, not exhaustively
- **No embedding fallback** -- word overlap and substring matching only
- **Standard thresholds** -- 0.50 for single-hop, 0.35 for multi-hop
- **Chunked ingestion** -- 4-turn chunks (~100 memories per conversation) vs per-turn storage (~400+)

Despite stricter conditions, codemem scores higher. The graph-vector hybrid scoring (vector similarity + BM25 + graph centrality + temporal signals) retrieves the right information with 5-10x fewer results.

## Setup

```bash
cd bench/locomo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset

```bash
mkdir -p data
wget -O data/locomo10.json https://huggingface.co/datasets/snap-research/locomo/resolve/main/locomo10.json
```

## Run

```bash
# Start codemem server
codemem serve --api --port 3179

# Full benchmark (10 conversations, ~1,986 questions)
python3 harness.py

# Quick test (single conversation)
python3 harness.py --conversations 0

# Different modes
python3 harness.py --mode baseline         # Full context upper bound
python3 harness.py --mode codemem          # Hybrid recall
python3 harness.py --mode codemem-graph    # Hybrid + graph expansion

# Custom recall limit
python3 harness.py --recall-limit 20
```

## Question Categories

| Category | Count | Description |
|----------|-------|-------------|
| Single-hop | 282 | Simple fact retrieval from one memory |
| Temporal | 321 | Time-based queries requiring date/sequence awareness |
| Multi-hop | 96 | Connecting information across multiple memories |
| Open-domain | 841 | General knowledge questions grounded in conversation |
| Adversarial | 446 | Questions about things never discussed (should abstain) |

## Evaluation

Retrieval accuracy: for each question, the harness checks whether the gold answer text is findable in the recalled memories using:

1. **Exact substring match** (case-insensitive)
2. **Word overlap** with basic stemming (threshold: 0.50 single-hop, 0.35 multi-hop)
3. **Fuzzy date matching** for temporal questions
4. **Abstention detection** for adversarial questions (low overlap = correct)

No LLM judge -- scoring is deterministic and reproducible.
