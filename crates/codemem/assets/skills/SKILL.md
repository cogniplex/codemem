---
name: codemem
description: >
  Quick reference for all 32 codemem MCP tools. Use when working with the codemem
  knowledge graph — finding code, traversing relationships, storing memories,
  or running analysis.
user-invocable: true
argument-hint: "[query or topic]"
---

# Codemem Tool Guide

Quick reference for codemem's 32 MCP tools. If arguments were provided, use them as context for the most relevant tool below.

$ARGUMENTS

## Finding Code & Symbols

| Scenario | Tool | Key Params |
|----------|------|------------|
| Find function by name | `search_code` | `query`, `mode: "text"` |
| Find code by concept | `search_code` | `query`, `mode: "semantic"` |
| Find code by name + concept | `search_code` | `query`, `mode: "hybrid"` |
| Get full symbol details | `get_symbol_info` | `qualified_name` |
| Browse file/package tree | `summary_tree` | `start_id: "pkg:src/"`, `max_depth` |

`search_code` also accepts: `k` (result count, default 10), `kind` (filter by symbol kind).

## Graph Traversal

| Scenario | Tool | Key Params |
|----------|------|------------|
| What calls this function? | `get_symbol_graph` | `qualified_name`, `direction: "incoming"`, `depth: 1` |
| What does this function call? | `get_symbol_graph` | `qualified_name`, `direction: "outgoing"`, `depth: 1` |
| Full blast radius of a change | `get_symbol_graph` | `qualified_name`, `direction: "incoming"`, `depth: 2` |
| Walk relationships from any node | `graph_traverse` | `start_id`, `max_depth`, `include_relationships`, `exclude_kinds` |
| Find most critical symbols | `find_important_nodes` | `top_k: 20` (PageRank, `damping: 0.85`) |
| Find related symbol clusters | `find_related_groups` | `resolution: 1.0` (Louvain communities) |

`graph_traverse` also accepts: `algorithm` ("bfs"/"dfs"), `exclude_kinds`, `include_relationships`.

`get_symbol_graph` directions: `"incoming"`, `"outgoing"`, `"both"` (default).

## Memories (Stored Knowledge)

| Scenario | Tool | Key Params |
|----------|------|------------|
| Ask a question about the codebase | `recall` | `query` |
| Ask with graph context expansion | `recall` | `query`, `expand: true`, `expansion_depth: 2` |
| Ask with architectural impact data | `recall` | `query`, `include_impact: true` |
| Get memories linked to a symbol/file | `get_node_memories` | `node_id: "sym:Name"` or `"file:path"` |
| Check if symbols have documentation | `node_coverage` | `node_ids: ["sym:X", "sym:Y"]` |
| Follow decision evolution over time | `get_decision_chain` | `topic` and/or `file_path` |
| Store a finding | `store_memory` | `content`, `memory_type`, `importance`, `tags`, `links` |
| Update a finding (new version) | `refine_memory` | `id`, `content` (creates EVOLVED_INTO edge) |
| Update in-place (destructive) | `refine_memory` | `id`, `content`, `destructive: true` |
| Split a large memory | `split_memory` | `id`, `parts` (creates PART_OF edges) |
| Merge related memories | `merge_memories` | `ids`, `content` (creates SUMMARIZES edges) |
| Delete a memory | `delete_memory` | `id` |
| Link two memories | `associate_memories` | `source_id`, `target_id`, `relationship` |

`recall` also accepts: `k`, `memory_type`, `namespace`, `exclude_tags`, `min_importance`, `min_confidence`.

`store_memory` types: `decision`, `pattern`, `preference`, `style`, `habit`, `insight`, `context` (default).

## Analysis & Health

| Scenario | Tool | Key Params |
|----------|------|------------|
| Check graph size & health | `codemem_status` | `include: ["stats", "health", "metrics"]` |
| Detect recurring patterns | `detect_patterns` | `min_frequency: 3` |
| Enrich with git/security/perf | `enrich_codebase` | `path`, `analyses: ["git", "security", "performance"]` |
| Git history enrichment only | `enrich_git_history` | `path`, `days: 90` |
| Full pipeline (index+enrich+rank) | `analyze_codebase` | `path` |
| Deduplicate similar memories | `consolidate` | `mode: "cluster"`, `similarity_threshold: 0.85` |
| Clean up low-value memories | `consolidate` | `mode: "forget"`, `importance_threshold: 0.3` |
| Find creative cross-connections | `consolidate` | `mode: "creative"` |
| Decay old memories | `consolidate` | `mode: "decay"`, `threshold_days: 30` |
| Summarize memory clusters | `consolidate` | `mode: "summarize"`, `cluster_size: 5` |
| Auto-select consolidation | `consolidate` | `mode: "auto"` |

## Namespace & Session Management

| Scenario | Tool | Key Params |
|----------|------|------------|
| List all namespaces | `list_namespaces` | (no params) |
| Namespace stats | `namespace_stats` | `namespace` |
| Delete a namespace | `delete_namespace` | `namespace` |
| Session progress snapshot | `session_checkpoint` | `session_id` |
| Get session context | `session_context` | `namespace`, `k` |
| Cross-repo memories | `get_cross_repo` | `namespace` |

## Quick Decision Tree

- **"What calls X / what does X depend on?"** -> `get_symbol_graph`
- **"How does feature Y work?"** -> `recall` + `search_code` mode=semantic
- **"Is it safe to change X?"** -> `get_symbol_graph` direction=incoming depth=2
- **"Do we already have something like Z?"** -> `search_code` mode=semantic
- **"Why was X designed this way?"** -> `recall` or `get_node_memories`
- **"Which files matter most?"** -> `find_important_nodes`
- **"What's related to what?"** -> `find_related_groups`
- **"Has this been documented?"** -> `node_coverage`
- **"What changed recently?"** -> `recall` query="pending analysis"

## Relationship Types (for associate_memories)

RELATES_TO, LEADS_TO, PART_OF, REINFORCES, CONTRADICTS, EVOLVED_INTO, DERIVED_FROM,
INVALIDATED_BY, DEPENDS_ON, IMPORTS, EXTENDS, CALLS, CONTAINS, SUPERSEDES, BLOCKS,
IMPLEMENTS, INHERITS, SIMILAR_TO, PRECEDED_BY, EXEMPLIFIES, EXPLAINS, SHARES_THEME,
SUMMARIZES, CO_CHANGED
