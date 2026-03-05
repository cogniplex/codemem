---
name: code-mapper
description: >
  Orchestrates large-scale codebase analysis: indexing, enrichment, planning,
  and dispatching up to 30 specialized worker agents for comprehensive coverage.
disable-model-invocation: true
---

# Code Mapper Skill

Maps a codebase using Codemem's indexing and graph tools, then orchestrates a **swarm of specialized worker agents** (up to 30) for deep, parallel analysis. A planning phase creates precise work packets so every file, symbol, API endpoint, and architectural pattern gets covered — even in repos with thousands of files.

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

## Memory Content Size Limits

Every memory MUST follow these size constraints:

| Memory Purpose | Max Length | Guideline |
|---------------|-----------|-----------|
| Baseline/context | 150 chars | 1-2 sentences. What it is, not how it works |
| Decision/pattern | 300 chars | 2-4 sentences. The WHAT and WHY |
| Cluster/API summary | 400 chars | 3-5 sentences. Central nodes + relationships |
| Architectural summary | 800 chars | 5-10 sentences. High-level only |

If a finding needs more than 4 sentences, **split it** into multiple linked memories using `associate_memories`. Never store a wall of text as a single memory — it degrades recall quality.

## Workflow Overview

### Phase 1: Foundation (code-mapper direct)
Index, enrich (all relevant analyses), browse structure, compute priorities at symbol level, check existing coverage.
→ See [phase-1-foundation.md](phase-1-foundation.md)

### Phase 2: Planning (code-mapper direct)
Build a comprehensive work plan: enumerate all work units (files, symbols, API endpoints, clusters), size them, and assign to agent roles. This is the KEY phase — a good plan means complete coverage.
→ See [phase-2-planning.md](phase-2-planning.md)

### Phase 3: Parallel Execution (up to 30 agents)
Dispatch specialized worker agents in waves. Each agent gets a small, focused work packet with clear scope and completion criteria.
→ See [phase-3-execution.md](phase-3-execution.md)

### Phase 4: Consolidation (code-mapper direct)
Coverage audit, dedup, cross-cutting pattern extraction, architectural summary, cleanup.
→ See [phase-4-consolidation.md](phase-4-consolidation.md)

## Agent Scaling Formula

| Repo Size | Total Agents | Breakdown |
|-----------|-------------|-----------|
| <30 files | 3-5 | 2 baseline + 1-3 deep |
| 30-100 files | 6-10 | 3 baseline + 3-7 deep |
| 100-300 files | 10-18 | 5 baseline + 5-13 deep |
| 300-1000 files | 18-25 | 8 baseline + 10-17 deep |
| 1000+ files | 25-30 | 10 baseline + 15-20 deep |

Agents are spawned in **waves** (not all at once) to manage coordination:
- **Wave 1**: Baseline agents (file/package summaries) — run first, complete before Wave 2
- **Wave 2**: Deep analysis agents (symbol analysis, API surface, patterns) — run in parallel
- **Wave 3**: Cross-cutting agents (architecture, data flow, security review) — run after Wave 2

## Agent Roles (Specialized, Not Generalist)

Each agent has ONE role with a narrow scope:

| Role | Scope | What It Produces |
|------|-------|-----------------|
| **baseline-scanner** | Batch of 20-50 files | 1 context memory per file, 1 per package |
| **symbol-analyst** | 10-30 critical/important symbols | Purpose, decision, pattern memories per symbol |
| **api-mapper** | All endpoints in a module/router | 1 decision memory per endpoint (route, method, auth, shape) |
| **pattern-hunter** | 1 cluster or module group | Cross-file patterns, conventions, recurring structures |
| **architecture-reviewer** | Full module dependency graph | Layering decisions, dependency patterns, boundaries |
| **security-reviewer** | Auth, validation, trust boundaries | Security decisions, trust model, input validation patterns |
| **data-flow-tracer** | Key data pipelines/workflows | Data lifecycle, transformation chains, storage patterns |
| **test-mapper** | Test files and coverage | Testing patterns, coverage gaps, test organization |

## Task Tracking

- Create one task per agent work packet using TaskCreate
- Set to `in_progress` when agent starts, `completed` when done
- Monitor via TaskList — reassign if agent idle 2+ turns
- Use `session_checkpoint` every 10 completed tasks for progress tracking

## Shutdown Protocol

After Phase 4:
1. Send `shutdown_request` to each teammate via SendMessage
2. Wait for `shutdown_response`; retry once if rejected
3. Call TeamDelete
4. Verify 0 pending-analysis memories remain

## Evaluation Criteria

### Coverage (MUST PASS)
- **Critical symbols (top 5%)**: ≥95% with at least 1 memory
- **Important symbols (top 20%)**: ≥85% with at least 1 memory
- **API endpoints**: 100% with at least 1 decision memory
- **All source files**: 100% baseline summary
- **All packages**: 100% package summary

### Quality (SHOULD PASS)
- ≥50% of memories are Decision or Pattern type
- ≥80% of memories have at least 1 symbol link
- <5% orphan memories (zero connections)
- Average memory content length: 80-300 chars (not too short, not too long)

### Cleanup (MUST PASS)
- 0 remaining pending-analysis memories
- All agents terminated, TeamDelete called
- Unverified static-analysis noise removed

## Tips

- `summary_tree { "start_id": "pkg:src/" }` for module hierarchy
- `find_important_nodes { "top_k": 100 }` for architectural weight (use higher k for large repos)
- `graph_traverse` with `"exclude_kinds": ["chunk"]` and `"include_relationships": ["CALLS", "IMPORTS"]` for clean call graphs
- `node_coverage` to batch-check many nodes at once
- `search_code { "query": "handler route endpoint", "mode": "hybrid" }` for API discovery
- `recall { "exclude_tags": ["static-analysis"] }` to skip enrichment noise

See [error-handling.md](error-handling.md) for recovery strategies, [human-input.md](human-input.md) for ambiguity detection, and [incremental.md](incremental.md) for re-analysis after file changes.
