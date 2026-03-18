# How Codemem Compares

Every other AI memory tool is Python-based, requires external databases, and targets general-purpose conversation memory. Codemem is a **single Rust binary** purpose-built for **AI coding assistants** -- zero dependencies, offline-first, with structural code intelligence.

## Why Codemem

| What matters | Codemem | Everyone else |
|-------------|---------|---------------|
| **Install** | One binary. No Python, no Docker, no DB | Python + Node.js + vector DB + graph DB |
| **Startup** | <100ms | Seconds to 30s |
| **Offline** | Fully local with built-in BERT | Needs API keys or cloud |
| **Code understanding** | 14 languages via ast-grep + SCIP compiler-grade edges | Text blobs, maybe AST splitting |
| **Graph intelligence** | 25 algorithms (PageRank, Louvain, betweenness, SCC) + temporal layer | External Neo4j or none at all |
| **Temporal** | Commits as graph nodes, symbol history, drift detection, time-travel queries | Zep has bi-temporal; most have nothing |
| **Code review** | Diff-aware blast radius with risk scoring | Nobody does this |
| **Scoring** | 9-component hybrid (vector + graph + BM25 + scope + temporal + tags + importance + confidence + recency) | Vector similarity + maybe BM25 |
| **Memory lifecycle** | 5 consolidation cycles, self-editing (refine/split/merge), TTL expiration | Store and forget |
| **MCP tools** | 32 | 3-4 typical |

## Quick Comparison

| Tool | Stars | What it is | Why you'd pick it over Codemem |
|------|-------|-----------|-------------------------------|
| [claude-mem](https://github.com/thedotmack/claude-mem) | ~32k | Claude Code plugin, AI-compressed observations | Zero-config, always-on compression. No graph, no code awareness, needs Node+Bun+Python |
| [Mem0](https://github.com/mem0ai/mem0) | ~48k | General-purpose memory layer ($24M raised) | Multi-tenant cloud, 24+ embedding providers. Not code-aware, needs external DBs |
| [Supermemory](https://github.com/supermemoryai/supermemory) | ~17k | Universal memory API (#1 on benchmarks, $3M raised) | Multi-modal (PDFs, images, video), SaaS connectors. Cloud-only |
| [Zep/Graphiti](https://github.com/getzep/graphiti) | ~23k | Temporal knowledge graph | Bi-temporal event timelines, cloud-scale graph backends. Not code-aware |
| [Letta](https://github.com/letta-ai/letta) | ~21k | Stateful AI agents (MemGPT) | Full agent platform with self-editing memory. Heavier, not code-specific |
| [Cognee](https://github.com/topoteretes/cognee) | ~13k | LLM-powered knowledge graphs ($7.5M raised) | Enterprise knowledge extraction from any data source. Python + external DBs |
| [claude-context](https://github.com/zilliztech/claude-context) | ~6k | AST-aware code search (by Zilliz) | Good code search. No persistent memory, no graph, needs Zilliz Cloud |
| [AutoMem](https://automem.ai/) | ~1k | Graph-vector hybrid (HippoRAG) | Research-grade LoCoMo benchmark. Needs Python + Docker + FalkorDB + Qdrant |
| [OpenMemory MCP](https://github.com/CaviraOSS/OpenMemory) | ~4k | Multi-sector memory server | Emotional/procedural memory sectors, SaaS connectors. Needs Python + Node |

## When to Use What

| If you need... | Use |
|----------------|-----|
| Persistent memory for AI coding, zero setup, offline | **Codemem** |
| Code-aware graph intelligence + blast radius + temporal queries | **Codemem** |
| Multi-modal memory (docs, images, video) | Supermemory |
| Zero-config session memory with AI compression | claude-mem |
| Cloud-scale multi-tenant memory | Mem0 |
| Bi-temporal event timelines for conversational agents | Zep/Graphiti |
| Full autonomous agent platform | Letta |
| Enterprise knowledge extraction from diverse sources | Cognee |
| Code search only (no persistent memory) | claude-context |
