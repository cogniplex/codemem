# codemem-mcp

MCP server exposing 33 tools over JSON-RPC 2.0 (stdio transport).

## Overview

The central integration point for AI coding assistants. Handles memory CRUD, graph traversal, code search, consolidation, impact analysis, pattern detection, and namespace management.

## Modules

- `tools_memory.rs` — store, update, delete, recall, associate
- `tools_graph.rs` — graph traversal, index_codebase, search_symbols, get_dependencies, get_impact, get_clusters, get_pagerank, search_code
- `tools_recall.rs` — recall_with_expansion, recall_with_impact, namespace management
- `tools_consolidation.rs` — decay, creative, cluster, forget cycles, consolidation status
- `scoring.rs` — 9-component hybrid scorer
- `types.rs` — JSON-RPC protocol types, tool result wrappers

## Key Features

- BM25 scoring with code-aware tokenizer (camelCase/snake_case splitting)
- Contextual embedding enrichment (metadata + graph context prepended before embedding)
- Cross-session pattern detection (repeated searches, file hotspots, decision chains)
- RwLock-based scoring weights with typed lock helpers
- Persistent config loaded from `~/.codemem/config.toml` at startup

## Tool Categories

| Category | Count | Tools |
|----------|-------|-------|
| Core Memory | 8 | store, recall, update, delete, associate, traverse, stats, health |
| Structural Index | 10 | index_codebase, search_symbols, get_symbol_info, get_dependencies, get_impact, get_clusters, get_cross_repo, get_pagerank, search_code, set_scoring_weights |
| Export/Import | 2 | export_memories, import_memories |
| Recall & Namespace | 4 | recall_with_expansion, list_namespaces, namespace_stats, delete_namespace |
| Consolidation | 5 | consolidate_decay/creative/cluster/forget, consolidation_status |
| Impact & Patterns | 4 | recall_with_impact, get_decision_chain, detect_patterns, pattern_insights |

See [MCP Tools Reference](../../docs/mcp-tools.md) for full API documentation.
