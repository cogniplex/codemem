# Phase 3: Tiered Deep Analysis

This phase spawns parallel analysis agents for deep, LLM-powered code understanding.

## Step 7: Tier assignment

Divide all symbols into tiers based on PageRank and git churn:

| Tier | Criteria | Analysis Depth |
|------|----------|---------------|
| **Critical** (top 2%) | Highest PageRank × git_churn | 2-4 memories each (purpose, design decision, pattern, dependencies) |
| **Important** (top 15%) | PageRank above median | 1 purpose memory each |
| **Standard** (rest) | Below median PageRank | Already covered by baseline (Phase 2) |

## Step 8: Coverage check

Before spawning agents, check which critical/important nodes already have fresh memories:

```
node_coverage { "node_ids": ["sym:module::ImportantStruct", "sym:module::key_function", ...] }
```

Pass all top-15% symbol IDs. The response shows which nodes have attached memories and which are uncovered. Skip nodes that already have fresh, relevant memories.

## Step 9: Spawn analysis agents with tier-aware work packets

Create the analysis team:

```
Use TeamCreate to create a team for coordinating analysis agents.
```

Spawn 2-4 analysis agents (via the Agent tool with `team_name`), each assigned a work packet organized by cluster and tier:

```
Work Packet for each agent:
  cluster_id: <cluster number>
  critical_symbols: [
    { qualified_name: "module::CriticalStruct", file: "src/module.rs", pagerank: 0.089, churn: 47 },
    ...
  ]
  important_symbols: [
    { qualified_name: "module::helper_fn", file: "src/module.rs", pagerank: 0.045 },
    ...
  ]
  files: [
    { path: "src/module.rs", priority: 0.92 },
    ...
  ]
```

### Agent rules

Each analysis agent MUST follow these rules:

1. **Read assigned files** using the Read tool — actually read the code
2. **Explore graph context** for each file using filtered traversal:
   ```
   graph_traverse { "start_id": "file:<path>", "max_depth": 2, "exclude_kinds": ["chunk"], "include_relationships": ["CALLS", "IMPORTS", "IMPLEMENTS"] }
   ```
   This skips chunk dead-ends and structural CONTAINS edges, showing only meaningful code relationships.
3. **Check existing coverage** before storing to avoid duplicates:
   ```
   get_node_memories { "node_id": "sym:<qualified_name>" }
   ```
   If a node already has relevant, fresh memories, skip it or refine the existing memory rather than duplicating.
4. **Understand WHY** the code is structured this way, not just WHAT it does
5. **Mandatory symbol links**: Every `store_memory` about a function or type MUST include `links: ["sym:<qualified_name>"]`:
   ```
   store_memory {
     "content": "CodememEngine::recall uses 9-component hybrid scoring to rank memories. Vector similarity dominates (25%) but graph centrality (20%) ensures structurally important memories surface even with weaker semantic matches.",
     "memory_type": "decision",
     "importance": 0.8,
     "tags": ["scoring", "architecture"],
     "links": ["sym:codemem_engine::CodememEngine::recall"],
     "namespace": "/path/to/project"
   }
   ```
6. **Semantic relationships**: After storing a decision or insight memory, use `associate_memories` with `EXPLAINS` to link it to the symbol it explains:
   ```
   associate_memories { "source_id": "<new_memory_id>", "target_id": "sym:<qualified_name>", "relationship": "EXPLAINS" }
   ```
7. **Structured template per critical function** — store up to 3 memories per critical symbol:
   - **Purpose insight** (what it does + why it matters in the system)
   - **Design decision** (why it's this way, what alternatives were considered) — only if non-obvious
   - **Pattern** (recurring structure it participates in) — only if part of a recognizable pattern
8. **Important symbols** get 1 memory each: a purpose insight with `links: ["sym:<qualified_name>"]`
9. **Store diverse memory types** — use Decision, Pattern, Preference, Style, Insight, Context, Habit as appropriate
10. **Review existing `static-analysis` tagged memories** for assigned files:
    ```
    recall { "query": "static analysis for <file>", "exclude_tags": [], "min_confidence": 0.3 }
    ```
    For each result tagged `static-analysis`:
    - If it's noise (e.g., "Complex file: X — 48 symbols"): `delete_memory { "id": "..." }`
    - If it's useful but shallow: `refine_memory { "id": "...", "content": "deeper analysis..." }`
    - If it's accurate: leave it, or add `agent-verified` tag via `refine_memory { "id": "...", "tags": ["agent-verified", "static-analysis"], "destructive": true }`
11. **Mark files as analyzed** after processing:
    ```
    store_memory {
      "content": "Agent analysis complete for <file_path>",
      "memory_type": "context",
      "importance": 0.2,
      "tags": ["agent-analyzed"],
      "links": ["file:<file_path>"],
      "metadata": { "file_hash": "<current_sha256>", "analyzed_at": "<ISO timestamp>" }
    }
    ```

### What each agent should look for

| Analysis Layer | Memory Types | What To Identify |
|---------------|-------------|-----------------|
| **Architecture** | Decision, Insight | Why modules are structured this way, key abstractions, layering decisions |
| **Patterns** | Pattern, Style | Recurring code structures, error handling approaches, naming conventions |
| **Conventions** | Preference, Habit | Library choices, testing patterns, workflow norms, CI practices |
| **Security** | Decision, Insight | Auth patterns, input validation strategy, trust boundaries (with WHY) |
| **Performance** | Decision, Pattern | Caching strategy, query patterns, concurrency approach (with trade-offs) |
| **Technical Debt** | Insight, Context | TODOs with context, stale code, missing tests — but only with LLM understanding of severity |

### Error recovery
- **Agent spawn failure**: If an agent fails to spawn, fall back to sequential processing — the code-mapper handles that agent's work packet directly.
- **Agent crash/timeout**: If an agent becomes unresponsive (idle for 3+ turns with incomplete tasks), reassign its remaining work to another agent or handle sequentially.
- **Coverage gaps after retry**: After one round of gap-filling (max 1 retry round), store remaining gaps as `needs-review` memories and move to Phase 4.

## Step 10: Cluster summaries

After agents complete their work, store 1 insight memory per Louvain cluster that has 3+ nodes:

```
store_memory {
  "content": "Cluster <N> (<theme>): <description of what these symbols do together and how they relate>. Central nodes: <2-3 most connected symbols>.",
  "memory_type": "insight",
  "importance": 0.6,
  "tags": ["cluster-summary", "architecture"],
  "links": ["sym:<central_node_1>", "sym:<central_node_2>"],
  "namespace": "/path/to/project"
}
```

Link to the 2-3 most central nodes in each cluster (highest PageRank within the cluster).
