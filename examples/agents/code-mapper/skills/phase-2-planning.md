# Phase 2: Planning

The code-mapper runs this phase directly. This is the MOST IMPORTANT phase — a precise plan ensures nothing is missed.

**Task tracking**: Create a Phase 2 task. Set to `in_progress` when starting, `completed` when the work plan is finalized.

## Goal

Produce a concrete work plan: a list of **work packets**, each assigned to a specific agent role, with explicit scope, file lists, symbol lists, and completion criteria. Every source file and every important symbol must appear in exactly one work packet.

## Step 1: Determine agent count and roles

Use the scaling formula from SKILL.md. Then allocate roles:

### Wave 1: Baseline Agents (run first)

**Role: baseline-scanner**

Split ALL source files into batches of 20-50 files. Each batch = 1 baseline-scanner agent.

```
Example for 150-file repo:
  baseline-agent-1: files 1-50
  baseline-agent-2: files 51-100
  baseline-agent-3: files 101-150
```

Each baseline-scanner produces:
- 1 context memory per file (using graph symbols, not just first 50 lines)
- 1 context memory per package it encounters
- Marks each file as baseline-covered

### Wave 2: Deep Analysis Agents (run after Wave 1 completes)

Assign agents by ROLE, not by cluster:

**Role: symbol-analyst** (1 agent per 10-30 uncovered critical/important symbols)
- Gets a list of specific symbol IDs to analyze
- Must READ the actual source code for each symbol
- Produces purpose, decision, pattern memories

**Role: api-mapper** (1 agent per module/router group with endpoints)
- Gets a list of endpoint-containing files
- Must document every route: method, path, auth, request/response shape, error handling
- Tags all memories with `api-surface`

**Role: pattern-hunter** (1 agent per 2-3 Louvain clusters)
- Gets cluster membership lists
- Looks for cross-file patterns WITHIN the cluster
- Produces pattern and style memories

### Wave 3: Cross-Cutting Agents (run after Wave 2 completes)

**Role: architecture-reviewer** (1-2 agents)
- Gets the full module dependency graph
- Analyzes layering, boundaries, dependency directions
- Produces decision and insight memories about module relationships

**Role: security-reviewer** (1 agent, if security-relevant code exists)
- Gets files flagged by security enrichment + auth/validation modules
- Analyzes trust model, auth flow, input validation strategy

**Role: data-flow-tracer** (1 agent, if data pipelines exist)
- Gets key data entry/exit points (API endpoints → storage, or ETL pipelines)
- Traces data transformation chains

**Role: test-mapper** (1 agent, if test files exist)
- Gets all test files and test-mapping enrichment results
- Documents testing patterns, coverage gaps, test organization

## Step 2: Build work packets

For each agent, create a structured work packet:

```
Work Packet #<N>:
  agent_role: <role>
  wave: <1|2|3>
  scope:
    files: [<explicit file list>]
    symbols: [<explicit symbol ID list with tiers>]
    clusters: [<cluster IDs if pattern-hunter>]
  context:
    priority_scores: { <symbol>: <score>, ... }
    existing_coverage: [<already-covered symbol IDs to skip>]
    large_files: [<files needing chunked reading>]
    static_analysis_findings: [<relevant enrichment memory IDs to review>]
  completion_criteria:
    - <specific measurable criteria>
  memory_budget:
    max_memories: <N>  # prevent runaway storage
    target_types: { decision: 40%, pattern: 30%, context: 20%, other: 10% }
```

### Memory budget per agent

Prevent bloat by capping memory count per agent:

| Role | Max Memories | Rationale |
|------|-------------|-----------|
| baseline-scanner | 1 per file + 1 per package | Strict 1:1 |
| symbol-analyst | 3 per critical + 1.5 per important symbol | Depth where it matters |
| api-mapper | 2 per endpoint | Route + auth/shape |
| pattern-hunter | 5-10 per cluster | Cross-file patterns only |
| architecture-reviewer | 15-25 total | System-level only |
| security-reviewer | 10-20 total | Focused findings |
| data-flow-tracer | 10-15 total | Pipeline-level only |
| test-mapper | 10-15 total | Patterns, not per-test |

## Step 3: Verify complete coverage

Before dispatching, verify:

1. **Every source file** appears in exactly one baseline-scanner packet
2. **Every critical symbol** appears in exactly one symbol-analyst packet
3. **Every important symbol** appears in at least one packet (symbol-analyst or pattern-hunter)
4. **Every API endpoint** appears in an api-mapper packet
5. **No file appears in two baseline-scanner packets** (no duplicates)
6. **No symbol appears in two symbol-analyst packets** (no duplicates)

If gaps exist, create additional work packets or expand existing ones.

## Step 4: Estimate total work and adjust

Calculate total expected memories:
```
total = sum(agent.max_memories for all agents)
```

If total > 500 for a small repo (<100 files), you're over-allocating — reduce agent count.
If total < 2 × file_count for a large repo, you're under-allocating — add more symbol-analyst agents.

## Output

Phase 2 produces a finalized work plan:
- List of all work packets with assigned roles and explicit scope
- Wave ordering (1 → 2 → 3)
- Expected memory count per agent
- Coverage verification checklist

This feeds directly into Phase 3 execution.
