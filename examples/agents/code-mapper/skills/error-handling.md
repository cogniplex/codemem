# Error Handling & Recovery

## Phase 1: Foundation

| Error | Recovery |
|-------|----------|
| `enrich_codebase` fails | Log and continue. Redistribute missing signal weights to remaining signals. |
| Empty PageRank (no edges) | All symbols get equal priority. Graph may only have CONTAINS edges. |
| Single Louvain cluster | Skip cluster weighting, redistribute to other signals. |
| `summary_tree` empty | Check namespace with `list_namespaces`. Verify indexing produced nodes via `codemem_status`. |
| `search_code` for endpoints returns nothing | Project may not have REST/RPC endpoints. Skip api-mapper agents in planning. |

## Phase 2: Planning

| Error | Recovery |
|-------|----------|
| Too few symbols for tiering | Use flat priority — all symbols get equal analysis depth. Reduce agent count. |
| Can't determine API endpoints | Skip api-mapper role. Note in plan that API coverage is manual. |
| File count exceeds batch limits | Increase baseline-scanner count. Max 50 files per agent. |

## Phase 3: Execution

| Error | Recovery |
|-------|----------|
| Agent spawn failure | Merge packet into adjacent agent's work, or code-mapper handles directly. |
| Agent crash/timeout (3+ min idle) | Reassign remaining work to new agent or handle directly. |
| Agent exceeds memory budget | Send stop signal. Keep stored memories. Proceed with next agent. |
| Agent stores oversized memories (>500 chars) | Flag in Phase 4 quality audit. Consider splitting in consolidation. |
| Wave timeout (all agents stuck) | Proceed to next wave with partial results. |
| `store_memory` fails | Retry once. If persistent, skip that memory and continue. |
| `associate_memories` fails | Log and continue. Memory exists; the edge is supplementary. |
| `get_node_memories` timeout | Batch node IDs into smaller groups. Use `node_coverage` for bulk checks. |
| Read tool fails (binary/deleted file) | Skip file. Remove from work packet. Don't store baseline for unreadable files. |
| Duplicate memory detected during recall check | Use `refine_memory` on existing instead of creating new. |

## Phase 4: Consolidation

| Error | Recovery |
|-------|----------|
| `consolidate` (any mode) fails | Log and continue. Consolidation is non-critical. |
| Coverage targets unmet after gap-fill | Store `needs-review` memory with uncovered symbol list for next run. |
| Quality targets unmet | Log metrics. Don't block completion. Note in final report. |
| Teammate unresponsive to shutdown | Retry once, then TeamDelete anyway. Team resources are ephemeral. |
| `delete_memory` fails for pending-analysis | Retry once. Memory will surface next session — not critical. |

## General

| Error | Recovery |
|-------|----------|
| MCP server unreachable | All codemem tools fail. Abort and inform user to check `codemem serve`. |
| Namespace mismatch | Verify with `list_namespaces`. Ensure all calls use same namespace. |
| Lock poisoned / concurrent access | Retry once. If persistent, another process may hold the database. |
| Context window pressure (agent overloaded) | Agent should store what it has and report remaining work as incomplete. |
