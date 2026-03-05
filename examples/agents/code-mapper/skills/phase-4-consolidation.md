# Phase 4: Consolidation

The code-mapper runs this phase directly after all waves complete.

**Task tracking**: Create a Phase 4 task. Set to `in_progress` when starting, `completed` when done.

## Step 1: Coverage audit

### 1a: Symbol coverage

```
node_coverage { "node_ids": [<all critical + important symbol IDs>] }
```

Check against targets:
- Critical (top 5%): ≥95% covered
- Important (top 20%): ≥85% covered
- API endpoints: 100% covered

### 1b: File coverage

```
node_coverage { "node_ids": [<all file node IDs>] }
```

Target: 100% of source files have at least a baseline summary.

### 1c: Gap filling (max 1 round)

If coverage falls short:
1. List uncovered symbols/files
2. Create mini work packets (5-10 symbols each)
3. Spawn 1-3 targeted agents to fill gaps
4. Wait for completion
5. Re-check coverage

If still short after 1 retry round, store remaining gaps as `needs-review` memories:
```
store_memory {
  "content": "COVERAGE GAP: <N> symbols still uncovered: <list>. Needs manual review or next analysis run.",
  "memory_type": "context",
  "importance": 0.4,
  "tags": ["needs-review", "coverage-gap"],
  "namespace": "project"
}
```

## Step 2: Quality audit

Check the quality of stored memories:

### 2a: Type distribution
```
recall { "query": "decision pattern architecture design", "k": 100, "exclude_tags": ["static-analysis", "baseline"] }
```

Count by type. Target: ≥50% decision + pattern. If mostly context/insight, the analysis was too shallow — note this for the coverage gap memory.

### 2b: Link rate
Check a sample of 20 recent memories via `get_node_memories` on high-importance symbols. Target: ≥80% have symbol links.

### 2c: Content length
Spot-check 10 memories. Flag any over 500 chars for potential splitting.

## Step 3: Clean up static-analysis noise

Bulk-remove low-value enrichment memories that agents didn't touch:

```
consolidate { "mode": "forget", "importance_threshold": 0.4, "target_tags": ["static-analysis"], "max_access_count": 1 }
```

This removes `static-analysis` memories with importance < 0.4 that were only accessed once (never recalled by an agent). Agent-refined or agent-verified memories survive.

## Step 4: Deduplicate across agents

Agents working in parallel may have stored similar findings:

```
consolidate { "mode": "cluster", "similarity_threshold": 0.85 }
```

Merges near-duplicate memories. Then:

```
consolidate { "mode": "creative" }
```

Finds cross-cutting connections between memories that individual agents missed.

### Error recovery
- If consolidation fails: log and continue. Memories are already stored.

## Step 5: Cluster summaries

Store 1 insight memory per Louvain cluster with 3+ nodes (if not already done by pattern-hunter agents):

```
store_memory {
  "content": "Cluster <N> (<theme>): <what these symbols do together>. Central: <2-3 top symbols>.",
  "memory_type": "insight",
  "importance": 0.6,
  "tags": ["cluster-summary", "architecture"],
  "links": ["sym:<central_1>", "sym:<central_2>"],
  "namespace": "project"
}
```

Check if pattern-hunter agents already covered this cluster — skip if so.

## Step 6: Architectural summary

Store a high-level summary as a high-importance Decision memory:

```
store_memory {
  "content": "<module structure>, <key design decisions>, <primary patterns>, <critical dependencies>, <API surface summary>. Coverage: <X>% critical, <Y>% important, <Z> endpoints documented.",
  "memory_type": "decision",
  "importance": 0.9,
  "tags": ["architecture", "summary", "agent-analyzed"],
  "namespace": "project"
}
```

**Max 800 chars.** This is the single most important memory — it should give a new session everything it needs to orient.

## Step 7: Clean up pending-analysis memories

Delete all `pending-analysis` tagged memories processed during this run:

```
delete_memory { "id": "<pending-analysis-memory-id>" }
```

## Step 8: Team shutdown

1. Mark Phase 4 task as `completed`
2. Verify ALL tasks are `completed` via TaskList
3. Send `shutdown_request` to each teammate via SendMessage
4. Wait for `shutdown_response`; retry once if rejected
5. Call TeamDelete

## Step 9: Final report

Output a summary to the user:

```
Analysis Complete:
  Files analyzed: <N>/<total> (baseline), <M> (deep)
  Symbols covered: <critical>/<total_critical> critical, <important>/<total_important> important
  API endpoints documented: <N>/<total>
  Memories stored: <N> total (<breakdown by type>)
  Agents used: <N> across <waves> waves
  Gaps remaining: <list or "none">
  Quality: <type distribution>, <link rate>%
```

### Error recovery
- Teammate unresponsive to shutdown: retry once, then TeamDelete anyway
- Coverage targets unmet: store `needs-review` memory with uncovered symbol list
- Quality targets unmet: log for next run, don't block completion
