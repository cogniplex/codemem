# codemem-mcp

MCP server exposing 38 tools over JSON-RPC 2.0 (stdio transport).

## Overview

The central integration point for AI coding assistants. Handles memory CRUD, self-editing (refine/split/merge), graph traversal, code search, consolidation (including LLM-powered summarization), impact analysis, pattern detection, namespace management, and operational metrics.

## Modules

- `tools_memory.rs` — store, update, delete, recall, associate, refine, split, merge
- `tools_graph.rs` — graph traversal, index_codebase, search_symbols, get_dependencies, get_impact, get_clusters, get_pagerank, search_code
- `tools_recall.rs` — recall_with_expansion, recall_with_impact, namespace management
- `tools_consolidation.rs` — decay (power-law), creative (semantic KNN), cluster (cosine + union-find), summarize (LLM), forget, consolidation status
- `scoring.rs` — 9-component hybrid scorer
- `types.rs` — JSON-RPC protocol types, tool result wrappers
- `compress.rs` — LLM-powered observation compression (Ollama/OpenAI/Anthropic)
- `patterns.rs` — Cross-session pattern detection with log-scaled confidence
- `metrics.rs` — Operational metrics (per-tool latency percentiles, counters, gauges)

## Key Features

- BM25 scoring with code-aware tokenizer (camelCase/snake_case splitting)
- Contextual embedding enrichment (metadata + graph context prepended before embedding)
- Cross-session pattern detection (repeated searches, file hotspots, decision chains)
- Self-editing memory with provenance tracking (EVOLVED_INTO, PART_OF, SUMMARIZES edges)
- Temporal edge support (valid_from/valid_to)
- RwLock-based scoring weights with typed lock helpers
- Persistent config loaded from `~/.codemem/config.toml` at startup

## Tool Categories

| Category | Count | Tools |
|----------|-------|-------|
| Core Memory | 8 | store, recall, update, delete, associate, traverse, stats, health |
| Self-Editing | 3 | refine_memory, split_memory, merge_memories |
| Structural Index | 10 | index_codebase, search_symbols, get_symbol_info, get_dependencies, get_impact, get_clusters, get_cross_repo, get_pagerank, search_code, set_scoring_weights |
| Export/Import | 2 | export_memories, import_memories |
| Recall & Namespace | 4 | recall_with_expansion, list_namespaces, namespace_stats, delete_namespace |
| Consolidation | 6 | consolidate_decay/creative/cluster/forget/summarize, consolidation_status |
| Impact & Patterns | 4 | recall_with_impact, get_decision_chain, detect_patterns, pattern_insights |
| Observability | 1 | codemem_metrics |

See [MCP Tools Reference](../../docs/mcp-tools.md) for full API documentation.
