---
name: code-mapper
description: >
  Orchestrates team-based codebase analysis: indexing, enrichment, priority
  computation, baseline coverage, tiered deep analysis via parallel agents,
  consolidation, and cleanup.
disable-model-invocation: true
---

# Code Mapper Skill

Maps a codebase's structural relationships using Codemem's indexing and graph tools, then orchestrates a **team of analysis agents** for deep, LLM-powered understanding. Static analysis signals (PageRank, churn, cluster size) determine WHERE agents focus — expensive deep analysis goes to high-impact code first.

## When to Use

- After `codemem init` to build a comprehensive knowledge graph
- When "Pending Analysis" appears in session context (files changed since last analysis)
- Periodically to keep the memory graph fresh as the codebase evolves

## Memory Types Guide

Use the right memory type for each finding — don't default everything to "insight":

| Type | When to Use |
|------|------------|
| **decision** | Architectural choices, trade-offs, WHY something was designed a certain way |
| **pattern** | Recurring code structures, naming conventions, repeated approaches |
| **preference** | Team/project conventions, preferred libraries, style choices |
| **style** | Coding style norms, formatting, naming patterns |
| **insight** | Cross-cutting architectural observations, system-level findings |
| **context** | File contents, structural context from exploration |
| **habit** | Workflow patterns, testing approaches, development practices |

> At least 50% of stored memories should be Decision or Pattern type. See [memory-types.md](memory-types.md) for detailed examples and guidelines.

## Workflow Overview

The analysis runs in 4 phases. Each phase has a dedicated reference file with step-by-step instructions and error recovery.

### Phase 1: Foundation
Index the codebase, run static enrichment, browse the directory structure, compute priority scores, and check for pending changes. The code-mapper runs this directly.
→ See [phase-1-foundation.md](phase-1-foundation.md)

### Phase 2: Baseline Coverage
Establish baseline coverage so every file and package has at least one memory. Package summaries first, then file summaries (parallelizable for large codebases).
→ See [phase-2-baseline.md](phase-2-baseline.md)

### Phase 3: Tiered Deep Analysis
Assign symbols to tiers (Critical/Important/Standard), check coverage, spawn 2-4 parallel analysis agents with tier-aware work packets, then store cluster summaries.
→ See [phase-3-deep-analysis.md](phase-3-deep-analysis.md)

### Phase 4: Consolidation
Run coverage reports, clean up static-analysis noise, consolidate findings, store an architectural summary, clean up pending-analysis memories, and shut down the team.
→ See [phase-4-consolidation.md](phase-4-consolidation.md)

## Team Lifecycle Management

### Sizing
- <50 source files → 2 agents
- 50-200 files → 3 agents
- 200+ files → 4 agents

### Monitoring
- Check TaskList after spawning for stuck/completed tasks
- If agent idle with incomplete tasks for 2+ turns → reassign remaining work
- Use session_checkpoint mid-run for progress tracking

### Shutdown (after Phase 4)
1. Send shutdown_request to each teammate via SendMessage
2. Wait for shutdown_response from each
3. If rejected (still working) → wait, then retry
4. Call TeamDelete to remove team resources

### Completion Checklist
- All tasks status: completed
- All teammates shut down
- TeamDelete called
- pending-analysis memories deleted
- Coverage report generated and stored

## Evaluation Criteria

### Coverage thresholds (MUST PASS)
- **Critical symbols (top 2%)**: ≥90% with at least 1 memory
- **Important symbols (top 15%)**: ≥80% with at least 1 memory
- **All source files**: 100% baseline summary
- **All top-level packages**: 100% package summary

### Quality targets (SHOULD PASS)
- ≥50% of memories are Decision or Pattern type
- ≥80% of memories have at least 1 symbol link
- <5% orphan memories (zero graph connections)

### Cleanup (MUST PASS)
- 0 remaining pending-analysis memories
- All agents terminated, TeamDelete called
- Unverified low-importance static-analysis memories removed

## What Gets Created

| Artifact | Storage | Description |
|----------|---------|-------------|
| Symbol nodes | `graph_nodes` table | One per function/struct/class/method. ID: `sym:qualified_name` |
| File nodes | `graph_nodes` table | One per source file. ID: `file:path` |
| Package nodes | `graph_nodes` table | One per directory. ID: `pkg:dir/` |
| Reference edges | `graph_edges` table | CALLS, IMPORTS, IMPLEMENTS, INHERITS, DEPENDS_ON |
| Symbol embeddings | `memory_embeddings` + HNSW index | 768-dim contextual embeddings |
| File hash cache | `file_hashes` table | SHA-256 per file for incremental re-indexing |
| Baseline summaries | `memories` table | 1 context memory per package and per file |
| Diverse memories | `memories` table | Decisions, patterns, preferences, styles, insights, habits |
| Cluster summaries | `memories` table | 1 insight per Louvain cluster with 3+ nodes |
| Static analysis tags | `static-analysis` tag | All enrichment outputs tagged for agent review |
| Analysis checkpoints | `agent-analyzed` tag | Per-file records with file hashes |

## Supported Languages

Rust (.rs), TypeScript (.ts/.tsx), Python (.py), Go (.go), C/C++ (.c/.h/.cpp/.hpp), Java (.java)

## Tips

- Start with `summary_tree { "start_id": "pkg:src/" }` to see the module hierarchy
- Run `find_important_nodes` to identify architectural weight — top 10 symbols tell you what matters
- Use `graph_traverse` with `"exclude_kinds": ["chunk"]` and `"include_relationships": ["CALLS", "IMPORTS"]` for clean call graphs
- Use `recall` with `"exclude_tags": ["static-analysis"]` to skip enrichment noise
- Use `node_coverage` to batch-check many nodes at once — faster than `get_node_memories` one by one

See [error-handling.md](error-handling.md) for recovery strategies, [human-input.md](human-input.md) for ambiguity detection, and [incremental.md](incremental.md) for re-analysis after file changes.
