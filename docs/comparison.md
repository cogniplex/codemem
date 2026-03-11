# Comparison: Memory Tools for AI Coding Assistants

A comprehensive comparison of Codemem against other memory and context tools in the ecosystem. Last updated March 2026.

## At a Glance

| | Codemem | claude-mem | Mem0 | Supermemory | Zep/Graphiti | Letta | Cognee | claude-context | AutoMem | OpenMemory MCP |
|---|--------|------------|------|-------------|-------------|-------|--------|----------------|---------|----------------|
| **Language** | Rust | TypeScript | Python | TypeScript | Python | Python | Python | TypeScript | Python+TS | Python+Node |
| **Architecture** | Single binary | Plugin + worker svc | Client-server / cloud | Cloud + local SDK | Cloud + open-source graph | Client-server | Library + optional cloud | MCP server + Zilliz | Client-server | Self-hosted MCP |
| **Runtime deps** | None | Node.js+Bun+uv+Python | Python + vector/graph DBs | Python or TS SDK | Neo4j/FalkorDB/Kuzu | Python + DB | Python + backends | Node.js + Zilliz Cloud | Python + FalkorDB + Qdrant | Python + Node.js |
| **GitHub stars** | -- | ~32k | ~48k | ~17k | ~23k | ~21k | ~13k | ~6k | ~1k | ~4k |
| **Startup** | <100ms | ~2s | Seconds | Seconds | Seconds | Seconds | Seconds | Seconds | 10-30s | Seconds |
| **Offline** | Yes | Partial | Partial | No | No | Partial | Partial | No | No | Partial (Ollama) |
| **Embedding** | Pluggable: Candle (local), Ollama, OpenAI-compatible (768-dim) | Chroma (via uvx) | Pluggable (24+ providers) | Built-in | Pluggable (OpenAI, etc.) | Configurable | Pluggable | OpenAI/Voyage/Ollama | Qdrant + Voyage AI | OpenAI, Gemini, Ollama |
| **Code-aware** | Yes (ast-grep + CST chunking) | No | No | Partial (code connector) | No | Yes (Letta Code) | Partial (codify) | Yes (AST splitting) | No | No |
| **MCP tools** | 32 | 3 | Yes (OpenMemory MCP) | Yes (plugin) | Yes (Graphiti MCP) | Yes | Yes (cognee-mcp) | 4 | 4 | Yes |
| **Graph** | Built-in (petgraph, 25 algos) | No | Neo4j (optional) | No | Neo4j/FalkorDB/Kuzu | No | Neo4j/FalkorDB/Kuzu | No | FalkorDB | Temporal KG |
| **Compression** | Pluggable LLM (optional) | AI-powered (always on) | No | No | No | No | No | No | No | No |
| **Consolidation** | 5 cycles (decay/creative/cluster/summarize/forget) | No | Auto-extraction | Contradiction handling | Temporal updates | Self-editing memory | ECL pipeline | No | HippoRAG-inspired | Adaptive decay |
| **Self-editing** | Yes (refine/split/merge) | No | No | No | No | Yes (core feature) | No | No | No | No |
| **Open source** | Yes (Apache 2.0) | Yes (AGPL-3.0) | Yes (Apache 2.0) | Yes (Apache 2.0) | Yes (Apache 2.0) | Yes (Apache 2.0) | Yes (Apache 2.0) | Yes (MIT) | Yes | Yes |

## Detailed Comparisons

### Codemem vs Mem0

[Mem0](https://github.com/mem0ai/mem0) (~48k GitHub stars) is a general-purpose memory layer for AI applications. Raised $24M from YC, Peak XV, and Basis Set in October 2025. Claims +26% accuracy over OpenAI Memory on LOCOMO and 91% faster retrieval.

| Aspect | Codemem | Mem0 |
|--------|--------|------|
| **Install** | `curl -fsSL .../install.sh \| sh` or `brew install cogniplex/tap/codemem` | `pip install mem0ai` + vector DB + graph DB |
| **Storage** | Single SQLite file + HNSW index | Triple-store: vector DB (24+ providers) + graph DB (Neo4j/Memgraph) + relational DB |
| **Auto-capture** | 9 lifecycle hooks capture Read/Grep/Edit/Write automatically + trigger-based auto-insights | Manual `add()` calls only |
| **Code awareness** | 14 language extractors, CST-aware chunking, structural indexing | General-purpose fact extraction from conversations |
| **Recall scoring** | 8-component hybrid (vector + graph strength via PageRank/betweenness + BM25 + temporal + tags + importance + confidence + recency) | Vector similarity + optional graph traversal + reranking (Cohere, etc.) |
| **Memory scoping** | Namespace-scoped (per project directory) | Multi-level: User, Session, Agent state |
| **Contextual embeddings** | Yes (metadata + graph context enrichment at ingestion) | No |
| **Local-only** | Yes, fully offline with Candle BERT | Requires external embedding API; local via Ollama possible |

**When to choose Mem0**: Multi-user AI applications, cloud-native deployments, conversation memory across many sessions, when you need managed infrastructure or multi-tenant support.

**When to choose Codemem**: Developer-local AI coding workflows, offline/air-gapped environments, code-specific memory with structural indexing, single-binary zero-dependency deployment.

### Codemem vs Supermemory

[Supermemory](https://github.com/supermemoryai/supermemory) (~17k GitHub stars) is a universal memory API ranked #1 on LongMemEval, LoCoMo, and ConvoMem benchmarks. Raised $3M. Ships plugins for Claude Code, OpenCode, and OpenClaw.

| Aspect | Codemem | Supermemory |
|--------|--------|-------------|
| **Architecture** | Single binary, in-process | Cloud-hosted API + local plugins |
| **Benchmarks** | Inspired by AutoMem/HippoRAG research | #1 on LongMemEval, LoCoMo, ConvoMem |
| **Modalities** | Code-focused (text + code indexing) | Multi-modal (PDFs, images/OCR, video transcription, code) |
| **Connectors** | Lifecycle hooks (automatic) | Google Drive, Gmail, Notion, OneDrive, GitHub (real-time sync) |
| **Recall** | 8-component hybrid scoring | Auto-extraction + contradiction detection + forgetting, ~50ms |
| **Offline** | Yes, fully local | No (cloud API required) |
| **Dependencies** | None | Cloud account or self-hosted setup |

**When to choose Supermemory**: Need multi-modal memory (documents, images, video), want cloud-hosted with connectors to SaaS tools, prioritize benchmark-leading accuracy.

**When to choose Codemem**: Need offline/air-gapped operation, code-specific structural intelligence (ast-grep + CST chunking), graph algorithms (PageRank, community detection), single-binary deployment with no cloud dependency.

### Codemem vs Zep/Graphiti

[Zep/Graphiti](https://github.com/getzep/graphiti) (~23k GitHub stars) provides a temporal knowledge graph for agent memory. Ships Graphiti MCP Server v1.0.

| Aspect | Codemem | Zep/Graphiti |
|--------|--------|-------------|
| **Graph model** | 24 relationship types, 25 algorithms (PageRank, Louvain, betweenness, SCC) | Temporal knowledge graph with bi-temporal data model |
| **Graph backends** | In-process (petgraph + SQLite) | Neo4j, FalkorDB, Kuzu, Amazon Neptune |
| **Time handling** | Temporal edges (valid_from/valid_to) + recency scoring (5%) + temporal alignment (10%) | First-class bi-temporal relationships (event time + ingestion time), point-in-time queries |
| **Retrieval** | 8-component hybrid scoring | Hybrid: semantic embeddings + BM25 + graph traversal |
| **Updates** | Explicit store/update/consolidate with provenance tracking | Real-time incremental graph updates (no batch recomputation) |
| **Custom entities** | 7 memory types + 13 node kinds | Custom entity definitions via Pydantic models |

**When to choose Zep**: Temporal reasoning is critical, you need bi-temporal event-timeline queries, building conversational agents that track how facts change over time, need cloud-scale graph backends.

**When to choose Codemem**: Code-specific memory, single-binary deployment, graph algorithms beyond traversal (PageRank, community detection, impact analysis), offline operation.

### Codemem vs Letta (MemGPT)

[Letta](https://github.com/letta-ai/letta) (~21k GitHub stars) builds stateful AI agents with OS-inspired memory hierarchy. Now ships Letta Code, a terminal-based memory-first coding agent.

| Aspect | Codemem | Letta |
|--------|--------|-------|
| **Memory model** | Graph-vector hybrid with 7 typed memories, self-editing (refine/split/merge) | OS-inspired tiers: core memory (RAM) + archival (disk) + recall (search) |
| **Self-editing** | Yes (refine_memory, split_memory, merge_memories with provenance) | Yes (agents modify their own memory via tool calls, core feature) |
| **Coding support** | MCP server for any AI assistant | Letta Code CLI (`npm install -g @letta-ai/letta-code`) |
| **Architecture** | Library/binary, embedded in coding workflow | Agent platform with REST API, GUI, SDKs, and Conversations API |
| **Context management** | SessionStart injection + session_checkpoint tool | Context Repositories with git-based versioning |

**When to choose Letta**: Building autonomous agents that need to self-manage their memory, need a full agent deployment platform, want the Letta Code CLI experience.

**When to choose Codemem**: Augmenting any existing AI coding assistant (Claude Code, Cursor, Windsurf) with persistent memory, need structural code intelligence, prefer a single binary with no runtime deps.

### Codemem vs Cognee

[Cognee](https://github.com/topoteretes/cognee) (~13k GitHub stars) builds knowledge graphs via triplet extraction (subject-relation-object). Raised $7.5M seed from Pebblebed, backed by OpenAI and FAIR founders. Over 1M pipeline runs across 70+ companies.

| Aspect | Codemem | Cognee |
|--------|--------|--------|
| **Knowledge extraction** | Hook-based capture + CST-aware chunking + structural code indexing | LLM-based triplet extraction from any data source |
| **Code support** | 14 language extractors (Rust, TS/JS/JSX, Python, Go, C/C++, Java, Ruby, C#, Kotlin, Swift, PHP, Scala, HCL) via ast-grep | `codify` tool for Python codebases |
| **Pipeline** | Ingest -> enrich -> embed -> store -> graph | ECL: Extract -> Cognify -> Load |
| **Multi-tenancy** | Namespace scoping per project | pgvector + Neo4j with tenant permission checks |
| **Integrations** | MCP server (any MCP client) | LangGraph, Google ADK, Claude Agent SDK, Memgraph, n8n |
| **Dependencies** | None (single binary) | Python + choice of graph/vector/relational backends |

**When to choose Cognee**: Processing diverse data sources (PDFs, docs, web), need LLM-powered knowledge extraction, building knowledge graphs at enterprise scale, need multilingual support.

**When to choose Codemem**: Code-focused memory with structural indexing (14 languages), offline operation, single-binary zero-dependency deployment, graph algorithms with cached centrality.

### Codemem vs claude-context

[claude-context](https://github.com/zilliztech/claude-context) (~6k GitHub stars, by Zilliz) provides AST-aware code search via MCP. Supports Claude Code, Cursor, and Gemini CLI.

| Aspect | Codemem | claude-context |
|--------|--------|---------------|
| **Scope** | Full memory engine (capture + recall + graph + consolidation + auto-insights) | Code search/retrieval only |
| **Code parsing** | ast-grep (14 languages) + CST-aware chunking | AST splitting (14 languages) |
| **MCP tools** | 32 tools | 4 tools (index, search, clear, status) |
| **Memory** | Persistent across sessions with 7 memory types | Index only (no persistent learning) |
| **Embedding** | Pluggable: Candle (local, default), Ollama, OpenAI-compatible | External API required (OpenAI/Voyage/Ollama/Gemini) |
| **Search** | Hybrid (vector + graph + BM25 + temporal + tags + ...) | Hybrid (BM25 + dense vector) |
| **Vector store** | Embedded usearch HNSW | Zilliz Cloud (required) |
| **Graph** | 25 algorithms, 24 relationship types | None |

**When to choose claude-context**: Need only code search (not persistent memory), already use Zilliz Cloud, want ~40% token reduction from smart retrieval.

**When to choose Codemem**: Need persistent memory across sessions, graph-based reasoning, offline operation, consolidation cycles, auto-insight generation.

### Codemem vs claude-mem

[claude-mem](https://github.com/thedotmack/claude-mem) (~32k GitHub stars) is a Claude Code plugin that captures session observations and compresses them via the Claude Agent SDK. Currently at v10.5.2.

| Aspect | Codemem | claude-mem |
|--------|--------|------------|
| **Language** | Rust (single binary) | TypeScript + Bun + Python (uv) |
| **Runtime deps** | None | Node.js + Bun + uv + Python + Express on port 37777 |
| **Hooks** | 9 lifecycle hooks + trigger-based auto-insights | 5 lifecycle hooks |
| **Knowledge graph** | petgraph with 25 algorithms (PageRank, Louvain, betweenness, SCC) | None |
| **Code intelligence** | ast-grep indexing (14 languages), CST-aware chunking, structural relationships | None (stores raw text observations) |
| **Scoring** | 8-component hybrid (vector + graph + BM25 + temporal + tags + importance + confidence + recency) | FTS5 keyword + Chroma vector (separate) |
| **Embeddings** | Pluggable: Candle BERT (Metal/CUDA, default), Ollama, OpenAI-compatible | Chroma (external process via uvx) |
| **BM25** | Okapi BM25 with code-aware tokenizer (camelCase/snake_case splitting) | SQLite FTS5 (not code-aware) |
| **Observation compression** | Pluggable LLM (Ollama/OpenAI/Anthropic), optional | Claude Agent SDK (always on) |
| **Contextual embeddings** | Metadata + graph neighbors enriched before embedding | Raw text embedding |
| **Consolidation** | 5 biologically-inspired cycles (decay/creative/cluster/summarize/forget) | None |
| **Self-editing** | Yes (refine/split/merge with provenance chains) | No |
| **MCP tools** | 32 | 3 (search, timeline, get_observations) |
| **Session insights** | Auto-generated (trigger-based) + session_checkpoint tool | None (manual only) |
| **Web viewer** | PCA visualization dashboard | React UI with SSE, infinite scroll, Endless Mode (beta) |
| **Privacy** | Namespace scoping | `<private>` tag exclusion |

**When to choose claude-mem**: Want zero-config AI compression with always-on Claude Agent SDK, already use Bun/Node.js, want the Claude Code plugin marketplace experience.

**When to choose Codemem**: Need code-aware structural intelligence, graph-based reasoning, single binary with no runtime deps, offline-first operation, code-aware hybrid scoring, consolidation cycles, self-editing memory.

### Codemem vs AutoMem

[AutoMem](https://automem.ai/) implements a graph-vector hybrid inspired by HippoRAG. Achieved 90.53% accuracy on LoCoMo benchmark (beating previous SOTA by 2.29 points). Listed on the official MCP Registry.

| Aspect | Codemem | AutoMem |
|--------|--------|---------|
| **Benchmark** | Inspired by AutoMem's LoCoMo research | 90.53% LoCoMo accuracy (SOTA at time of publication) |
| **Architecture** | Single binary (SQLite + usearch + petgraph) | Flask API + FalkorDB + Qdrant |
| **Scoring** | 8-component hybrid | 8-component hybrid (semantic, lexical, graph, temporal, importance) |
| **Consolidation** | 5 neuroscience-inspired cycles | Zettelkasten-inspired clustering |
| **Enrichment** | Just-in-time + git history + security + performance enrichment | Just-in-time enrichment on recall |
| **Embeddings** | Pluggable: Candle (local), Ollama, OpenAI-compatible | Qdrant + Voyage AI |
| **Code awareness** | Structural indexing, CST-aware chunking, passive hook capture | General-purpose |
| **Dependencies** | None | Python + Docker + FalkorDB + Qdrant |

Codemem was directly inspired by AutoMem's research. The key difference is packaging: Codemem embeds the same graph-vector hybrid approach into a single Rust binary optimized for AI coding assistants.

**When to choose AutoMem**: Research benchmark reproduction, already running FalkorDB + Qdrant, want the original HippoRAG implementation.

**When to choose Codemem**: Code-specific memory, single binary with no Docker/DB dependencies, offline operation, 14-language structural indexing.

### Codemem vs OpenMemory MCP

[OpenMemory MCP](https://github.com/CaviraOSS/OpenMemory) (~4k GitHub stars) is a self-hosted MCP memory server with multi-sector memory. Distinct from Mem0's built-in OpenMemory.

| Aspect | Codemem | OpenMemory MCP |
|--------|--------|----------------|
| **Memory model** | 7 typed memories + 24 relationship types | 5 sectors: Episodic, Semantic, Procedural, Emotional, Reflective |
| **Graph** | petgraph with 25 algorithms | Temporal knowledge graph with valid_from/valid_to |
| **Scoring** | 8-component hybrid | Composite: salience + recency + coactivation |
| **Decay** | Power-law consolidation cycle (configurable) | Adaptive decay per sector |
| **Explainability** | Score breakdown per component | "Waypoint" traces for recall |
| **Migration** | N/A | Import from Mem0, Zep, Supermemory |
| **Connectors** | Lifecycle hooks (automatic) | GitHub, Notion, Google Drive, OneDrive |
| **Dependencies** | None (single binary) | Python + Node.js |

**When to choose OpenMemory MCP**: Want multi-sector memory model (emotional, procedural), need data connectors to SaaS tools, migrating from Mem0/Zep.

**When to choose Codemem**: Need code-aware structural intelligence, offline operation, single binary, graph algorithms with PageRank/Louvain/betweenness.

### Other Notable Tools

| Tool | Stars | Description | Key Differentiator |
|------|-------|-------------|-------------------|
| [MemOS](https://github.com/MemTensor/MemOS) | ~5k | Memory Operating System unifying plaintext, activation, and parameter-level memory | Multi-modal (images/charts), tool memory for agent planning, 159% improvement in temporal reasoning over OpenAI Memory on LoCoMo |
| [LangMem](https://github.com/langchain-ai/langmem) | ~1k | Memory SDK for LangChain/LangGraph | Deep LangGraph integration, background memory manager, multiple namespace support |
| [Khoj](https://github.com/khoj-ai/khoj) | ~33k | Personal AI "second brain" | Multi-platform (web, Obsidian, Emacs, Desktop, Phone, WhatsApp), deep research mode, document ingestion |
| [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) | ~1k | Python MCP memory server | 24 MCP tools, natural language time-based recall, HTTP dashboard |
| [Hindsight](https://github.com/vectorize-io/hindsight) | -- | Memory as four logical networks (facts, experiences, summaries, beliefs) | NeurIPS 2025 paper, Fortune 500 production use, retain/recall/reflect operations |
| [Graphlit](https://www.graphlit.com/) | -- | Cloud knowledge platform | Real-time sync across Slack, GitHub, Jira, Google Drive |

**Note on AgentDB**: The standalone AgentDB project (formerly at `github.com/ruvnet/agentdb`) has been absorbed into [agentic-flow](https://github.com/ruvnet/agentic-flow) as a sub-package. It is no longer maintained as an independent tool.

## When to Use What

| If you need... | Use |
|----------------|-----|
| Persistent memory for AI coding with zero setup | **Codemem** |
| Single binary, offline, no runtime deps | **Codemem** |
| Code-aware structural indexing + CST chunking + graph reasoning | **Codemem** |
| Multi-modal memory (docs, images, video) + cloud-hosted API | **Supermemory** |
| Zero-config session memory with AI compression | **claude-mem** |
| General-purpose AI memory with cloud scale and multi-tenancy | **Mem0** |
| Temporal knowledge graph with event timelines | **Zep/Graphiti** |
| Autonomous self-editing agent memory + deployment platform | **Letta** |
| LLM-powered knowledge extraction from diverse documents | **Cognee** |
| Code search only (no persistent memory) + managed Zilliz Cloud | **claude-context** |
| Multi-sector memory with adaptive decay + SaaS connectors | **OpenMemory MCP** |
| Memory OS with multi-modal + parameter-level memory | **MemOS** |

## Summary

The AI memory landscape has matured significantly since early 2025. Multiple tools have raised venture funding (Mem0 $24M, Cognee $7.5M, Supermemory $3M), benchmark competition has intensified (Supermemory leads LongMemEval/LoCoMo/ConvoMem, AutoMem's 90.53% LoCoMo SOTA, MemOS's 159% temporal reasoning improvement), and the MCP protocol has become the standard integration layer.

Most tools remain Python-based, require external services (vector DBs, graph DBs, embedding APIs), and target general-purpose conversation memory. The trend is toward multi-modal memory (images, video, documents), cloud-hosted APIs with SaaS connectors, and increasingly sophisticated temporal reasoning.

Codemem occupies a unique position: a **single Rust binary** purpose-built for **AI coding assistants**, combining the best research ideas (graph-vector hybrid from AutoMem/HippoRAG, CST-aware chunking inspired by the cAST paper, contextual embeddings, neuroscience-inspired consolidation) into a **zero-dependency, offline-first** package with **32 MCP tools**, **self-editing memory**, **trigger-based auto-insights**, **14 enrichment types**, and **14-language structural code intelligence**. The tradeoff is intentional: Codemem drops cloud-scale multi-tenancy, multi-modal support, and pluggable backend architecture in favor of simplicity, speed, and local-first operation.

Install in one line:

```bash
curl -fsSL https://raw.githubusercontent.com/cogniplex/codemem/main/install.sh | sh
# or: brew install cogniplex/tap/codemem
# or: cargo install codemem
```

Then use the [code-mapper agent](../examples/agents/code-mapper.md) to index your codebase, run PageRank, detect clusters, and store architectural insights as persistent memories.
