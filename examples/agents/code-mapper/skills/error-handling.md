# Error Handling & Recovery

Comprehensive error recovery strategies organized by phase and error source.

## Phase 1: Foundation

| Error | Recovery |
|-------|----------|
| `enrich_codebase` fails | Log and continue. Enrichment is optional. Redistribute git churn weight (0.25) to PageRank (+0.10) and Unanalyzed (+0.15). |
| Empty PageRank (no edges) | Proceed with equal priority for all files. The graph may only have CONTAINS edges. |
| Single Louvain cluster | All files are interconnected. Skip cluster_size weighting (set weight to 0, redistribute to others). |
| `summary_tree` returns empty | The namespace may be wrong or indexing produced no packages. Check `codemem_status` for node counts. |

## Phase 2: Baseline Coverage

| Error | Recovery |
|-------|----------|
| File read failure (binary, deleted, permissions) | Skip the file. Do not store a baseline memory for unreadable files. |
| `store_memory` fails for one file | Retry once. If it fails again, continue with next file. One missing baseline is not critical. |
| Duplicate baseline detected | Use `refine_memory` to update existing baseline rather than creating a duplicate. Check via `get_node_memories` before storing. |
| Agent spawn fails (for parallelized baseline) | Fall back to sequential processing — the code-mapper handles file summaries directly. |

## Phase 3: Deep Analysis

| Error | Recovery |
|-------|----------|
| Agent spawn failure | Fall back to sequential processing — the code-mapper handles that agent's work packet directly. |
| Agent crash/timeout (idle 3+ turns) | Reassign the agent's remaining work to another agent, or handle sequentially. |
| Coverage gaps after agents finish | Run one retry round: create targeted work packets for uncovered critical/important symbols. Max 1 retry round. |
| Coverage gaps after retry | Store remaining gaps as `needs-review` memories and proceed to Phase 4. |
| `associate_memories` fails | Log and continue. The memory is stored; the EXPLAINS edge is supplementary. |
| `get_node_memories` timeout on large graph | Batch node IDs and check in smaller groups. Use `node_coverage` for bulk checks. |

## Phase 4: Consolidation

| Error | Recovery |
|-------|----------|
| `consolidate` (any mode) fails | Log and continue. Consolidation is non-critical — memories are already stored. |
| Teammate doesn't respond to shutdown | Wait 1 turn, retry. After 2 failed attempts, proceed with TeamDelete. |
| TeamDelete fails | Log the error. Team resources are ephemeral and will be cleaned up. |
| Coverage targets unmet | Store a `needs-review` memory listing uncovered symbols for the next run. |
| `delete_memory` fails for pending-analysis | Retry once. If it fails, the memory will surface in the next session — not critical. |

## General

| Error | Recovery |
|-------|----------|
| MCP server unreachable | All codemem tools will fail. Abort and inform the user to check `codemem serve` status. |
| Namespace mismatch | Verify namespace with `list_namespaces`. Ensure all tool calls use the same namespace (project path). |
| Lock poisoned / concurrent access | Retry the operation once. If persistent, inform the user — another process may be using the database. |
