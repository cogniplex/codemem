# Comparison: Memory Tools for AI Coding Assistants

A comprehensive comparison of Codemem against other memory and context tools in the ecosystem.

## At a Glance

| | Codemem | claude-mem | Mem0 | Zep/Graphiti | Letta (MemGPT) | Cognee | claude-context | AutoMem | AgentDB |
|---|--------|------------|------|-------------|----------------|--------|----------------|---------|---------|
| **Language** | Rust | TypeScript | Python | Python | Python | Python | TypeScript | Python+TS | TypeScript |
| **Architecture** | Single binary | Plugin + worker svc | Client-server / cloud | Cloud + open-source graph | Client-server | Library + optional cloud | MCP server + Milvus | Client-server | npm package |
| **Runtime deps** | None | Node.js+Bun+uv+Python | Python + vector/graph DBs | Neo4j or FalkorDB | Python + DB | Python + backends | Node.js + Milvus/Zilliz | Python + FalkorDB + Qdrant | Node.js |
| **Startup** | <100ms | ~2s | Seconds | Seconds | Seconds | Seconds | Seconds | 10-30s | 2-5s |
| **Offline** | Yes | Partial | Partial | No | Partial | Partial | No (needs embedding API) | No | Yes |
| **Embedding** | Pluggable via env var: Candle (local, default), Ollama, OpenAI-compatible (768-dim) | Chroma (via uvx) | Pluggable (API or local) | LLM-managed | Configurable | Pluggable | Pluggable (OpenAI/Voyage) | Qdrant | ONNX (384-dim) |
| **Code-aware** | Yes (hooks + tree-sitter) | No | No | No | No | Partial (codify tool) | Yes (AST splitting) | No | No |
| **MCP tools** | 38 | 3 | Yes (official) | Yes (Graphiti MCP) | Yes | Yes (cognee-mcp) | Yes (4 tools) | Yes (4 tools) | Manual |
| **Graph** | Built-in (petgraph, 25 algos) | No | Neo4j (optional) | Neo4j/FalkorDB | No | Neo4j/FalkorDB/Kuzu | No | FalkorDB | Optional |
| **Compression** | Pluggable LLM (optional) | AI-powered (always on) | No | No | No | No | No | No | No |
| **Consolidation** | 4 cycles (decay/creative/cluster/forget) | No | Auto-extraction | Temporal updates | Self-editing memory | ECL pipeline | No | HippoRAG-inspired | NightlyLearner |
| **Open source** | Yes (Apache 2.0) | Yes (AGPL-3.0) | Yes (Apache 2.0) | Graphiti: Yes | Yes (Apache 2.0) | Yes (Apache 2.0) | Yes (MIT) | Yes | Yes |

## Detailed Comparisons

### Codemem vs Mem0

[Mem0](https://github.com/mem0ai/mem0) (47K+ GitHub stars) is a general-purpose memory layer for AI applications.

| Aspect | Codemem | Mem0 |
|--------|--------|------|
| **Install** | `curl -fsSL .../install.sh \| sh` or `brew install cogniplex/tap/codemem` | `pip install mem0ai` + vector DB + graph DB |
| **Storage** | Single SQLite file + HNSW index | Triple-store: vector DB (24+ providers) + graph DB (Neo4j/Memgraph) + relational DB |
| **Auto-capture** | PostToolUse hooks capture Read/Grep/Edit/Write automatically | Manual `add()` calls only |
| **Code awareness** | 13 language extractors, structural indexing, code-aware memory types | General-purpose fact extraction from conversations |
| **Recall scoring** | 9-component hybrid (vector + graph strength via PageRank/betweenness + BM25 + temporal + tags + importance + confidence + recency) | Vector similarity + optional graph traversal |
| **Contextual embeddings** | Yes (metadata + graph context enrichment at ingestion) | No |
| **Local-only** | Yes, fully offline | Requires external embedding API for full functionality |

**When to choose Mem0**: Multi-user AI applications, cloud-native deployments, conversation memory across many sessions, when you need managed infrastructure.

**When to choose Codemem**: Developer-local AI coding workflows, offline/air-gapped environments, code-specific memory with structural indexing, zero-dependency deployment.

### Codemem vs Zep/Graphiti

[Zep](https://www.getzep.com/) provides a temporal knowledge graph (Graphiti) for agent memory.

| Aspect | Codemem | Zep/Graphiti |
|--------|--------|-------------|
| **Graph model** | 15 relationship types, 25 algorithms (PageRank, Louvain, betweenness, SCC) | Temporal knowledge graph with bi-temporal data model |
| **Time handling** | Recency scoring component (5% weight) + temporal alignment (10%) | First-class temporal relationships (event time + ingestion time) |
| **Updates** | Explicit store/update/consolidate | Real-time incremental graph updates (no batch recomputation) |
| **Backend** | In-process (petgraph + SQLite) | Neo4j or FalkorDB (external service) |
| **Accuracy** | 9-component hybrid scoring | 18.5% improvement on LongMemEval |

**When to choose Zep**: Temporal reasoning is critical, you need event-timeline queries, building conversational agents that track how facts change over time.

**When to choose Codemem**: Code-specific memory, single-binary deployment, graph algorithms beyond traversal (PageRank, community detection, impact analysis).

### Codemem vs Letta (MemGPT)

[Letta](https://github.com/letta-ai/letta) builds stateful AI agents with OS-inspired memory hierarchy.

| Aspect | Codemem | Letta |
|--------|--------|-------|
| **Memory model** | Graph-vector hybrid with 7 typed memories | OS-inspired tiers: core memory (RAM) + archival (disk) + recall (search) |
| **Self-editing** | No (memories are stored/recalled/consolidated) | Yes (agents modify their own memory via tool calls) |
| **Architecture** | Library/binary, embedded in coding workflow | Agent platform with REST API, GUI, and SDKs |
| **Focus** | Passive capture + active recall for coding | General-purpose stateful agent deployment |

**When to choose Letta**: Building autonomous agents that need to self-manage their memory, need a full agent deployment platform.

**When to choose Codemem**: Augmenting existing AI coding assistants with persistent memory, don't need a full agent platform.

### Codemem vs Cognee

[Cognee](https://github.com/topoteretes/cognee) builds knowledge graphs via triplet extraction (subject-relation-object).

| Aspect | Codemem | Cognee |
|--------|--------|--------|
| **Knowledge extraction** | Hook-based capture + structural code indexing | LLM-based triplet extraction from any data source |
| **Code support** | 13 language extractors (Rust, TS/JS, Python, Go, C/C++, Java, Ruby, C#, Kotlin, Swift, PHP, Scala, HCL) via tree-sitter | `codify` tool for Python codebases |
| **Pipeline** | Ingest -> enrich -> embed -> store -> graph | ECL: Extract -> Cognify -> Load |
| **Accuracy** | 9-component hybrid scoring | Claims 92.5% vs 60% traditional RAG |
| **Dependencies** | None (single binary) | Python + choice of graph/vector/relational backends |

**When to choose Cognee**: Processing diverse data sources (PDFs, docs, web), need LLM-powered knowledge extraction, building general knowledge graphs.

**When to choose Codemem**: Code-focused memory, offline operation, single-binary deployment.

### Codemem vs claude-context

[claude-context](https://github.com/zilliztech/claude-context) (by Zilliz) provides AST-aware code search via MCP.

| Aspect | Codemem | claude-context |
|--------|--------|---------------|
| **Scope** | Full memory engine (capture + recall + graph + consolidation) | Code search/retrieval only |
| **Code parsing** | tree-sitter (13 languages) | AST splitting (14 languages) |
| **MCP tools** | 38 tools | 4 tools (index, search, clear, status) |
| **Memory** | Persistent across sessions with 7 memory types | Index only (no persistent learning) |
| **Embedding** | Pluggable: Candle (local, default), Ollama, OpenAI-compatible (Voyage AI, etc.) | External API required (OpenAI/Voyage/Ollama/Gemini) |
| **Search** | Hybrid (vector + graph + BM25 + temporal + tags + ...) | Hybrid (BM25 + dense vector) |
| **Vector store** | Embedded usearch HNSW | External Milvus or Zilliz Cloud |
| **Graph** | 25 algorithms, 15 relationship types | None |

**When to choose claude-context**: Need only code search (not persistent memory), already use Milvus/Zilliz, need 14+ language support.

**When to choose Codemem**: Need persistent memory across sessions, graph-based reasoning, offline operation, consolidation cycles.

### Codemem vs claude-mem

[claude-mem](https://github.com/thedotmack/claude-mem) (31K+ GitHub stars) is a Claude Code plugin that captures session observations and compresses them via the Claude Agent SDK.

| Aspect | Codemem | claude-mem |
|--------|--------|------------|
| **Language** | Rust (single binary) | TypeScript + Bun + Python (uv) |
| **Runtime deps** | None | Node.js + Bun + uv + Python + Express on port 37777 |
| **Hooks** | 4 lifecycle hooks (same events) | 5 lifecycle hooks |
| **Knowledge graph** | petgraph with 25 algorithms (PageRank, Louvain, betweenness, SCC) | None |
| **Code intelligence** | tree-sitter indexing, 13 languages, structural relationships | None (stores raw text observations) |
| **Scoring** | 9-component hybrid (vector + graph + BM25 + temporal + tags + importance + confidence + recency) | FTS5 keyword + Chroma vector (separate) |
| **Embeddings** | Pluggable: Candle BERT (Metal/CUDA, default), Ollama, OpenAI-compatible | Chroma (external process via uvx) |
| **BM25** | Okapi BM25 with code-aware tokenizer | SQLite FTS5 (not code-aware) |
| **Observation compression** | Pluggable LLM (Ollama/OpenAI/Anthropic), optional | Claude Agent SDK (always on) |
| **Contextual embeddings** | Metadata + graph neighbors enriched before embedding | Raw text embedding |
| **Consolidation** | 4 biologically-inspired cycles | None |
| **MCP tools** | 38 | 3 (search, timeline, get_observations) |
| **Session summaries** | Structured (files/decisions/searches) | AI-generated |
| **Web viewer** | PCA visualization | React UI with SSE, infinite scroll |
| **Privacy** | Namespace scoping | `<private>` tag exclusion |

**When to choose claude-mem**: Want zero-config AI compression, prefer a polished web viewer, already use Bun/Node.js, want the Claude Code plugin marketplace experience.

**When to choose Codemem**: Need code-aware structural intelligence, graph-based reasoning, single binary with no runtime deps, offline-first operation, code-aware hybrid scoring, consolidation cycles.

### Codemem vs AutoMem

[AutoMem](https://automem.ai/) implements a graph-vector hybrid inspired by HippoRAG.

| Aspect | Codemem | AutoMem |
|--------|--------|---------|
| **Benchmark** | Inspired by AutoMem's 90.53% LoCoMo | 90.53% LoCoMo accuracy |
| **Architecture** | Single binary (SQLite + usearch + petgraph) | Flask API + FalkorDB + Qdrant |
| **Scoring** | 9-component hybrid | 9-component hybrid (semantic, lexical, graph, temporal, importance) |
| **Consolidation** | 4 neuroscience-inspired cycles | Zettelkasten-inspired clustering |
| **Code awareness** | Structural indexing, passive hook capture | General-purpose |
| **Dependencies** | None | Python + Docker + FalkorDB + Qdrant |

Codemem was directly inspired by AutoMem's research. The key difference is packaging: Codemem embeds the same graph-vector hybrid approach into a single Rust binary optimized for AI coding assistants.

### Codemem vs AgentDB

[AgentDB](https://github.com/anthropics/agentdb) is a research platform for episodic agent memory.

| Aspect | Codemem | AgentDB |
|--------|--------|---------|
| **RL algorithms** | None | 9 (DQN, PPO, A3C, REINFORCE, etc.) |
| **GNN reasoning** | No | CausalMemoryGraph (proprietary @ruvector/gnn) |
| **Skill library** | No | Trajectory segmentation, typed I/O |
| **Monorepo support** | Namespace scoping, structural indexing | Score: 65/100 (critical gaps) |
| **Proprietary deps** | None | ~40% of logic tied to @ruvector/* modules |

**When to choose AgentDB**: Research on RL-based agent learning, need GNN causal reasoning, working on frontier memory research.

**When to choose Codemem**: Production use with AI coding assistants, need cross-repo support, need single-binary deployment.

### Other Notable Tools

| Tool | Description | Key Differentiator |
|------|-------------|-------------------|
| [LangMem](https://github.com/langchain-ai/langmem) | Memory SDK for LangChain/LangGraph | Deep LangGraph integration, prompt optimization |
| [Khoj](https://github.com/khoj-ai/khoj) | Personal AI "second brain" | Multi-platform (web, Obsidian, Emacs, mobile), document search |
| [OpenMemory MCP](https://github.com/CaviraOSS/OpenMemory) | Local-first MCP memory server | Cross-tool context sharing (Cursor, Claude, Windsurf) |
| [Graphlit](https://www.graphlit.com/) | Cloud knowledge platform | Real-time sync across Slack, GitHub, Jira, Google Drive |

## When to Use What

| If you need... | Use |
|----------------|-----|
| Persistent memory for AI coding with zero setup | **Codemem** |
| Single binary, offline, no runtime deps | **Codemem** |
| Code-aware structural indexing + graph reasoning | **Codemem** |
| Zero-config session memory with AI compression | **claude-mem** |
| General-purpose AI memory with cloud scale | **Mem0** |
| Temporal knowledge graph with event timelines | **Zep/Graphiti** |
| Autonomous self-editing agent memory | **Letta** |
| LLM-powered knowledge extraction from documents | **Cognee** |
| Code search with AST splitting + external vector DB | **claude-context** |
| RL-based agent research platform | **AgentDB** |

## Summary

The AI memory landscape has matured significantly. Most tools are Python-based, require external services (vector DBs, graph DBs, embedding APIs), and target general-purpose conversation memory.

Codemem occupies a unique position: a **single Rust binary** purpose-built for **AI coding assistants**, combining the best research ideas (graph-vector hybrid from AutoMem/HippoRAG, contextual embeddings from Anthropic, neuroscience-inspired consolidation) into a **zero-dependency, offline-first** package. The tradeoff is intentional: Codemem drops cloud-scale multi-tenancy, RL/GNN research features, and pluggable backend architecture in favor of simplicity, speed, and local-first operation.

Install in one line:

```bash
curl -fsSL https://raw.githubusercontent.com/cogniplex/codemem/main/install.sh | sh
# or: brew install cogniplex/tap/codemem
# or: cargo install codemem-cli
```

Then use the [code-mapper agent](../examples/agents/code-mapper.md) to index your codebase, run PageRank, detect clusters, and store architectural insights as persistent memories.
