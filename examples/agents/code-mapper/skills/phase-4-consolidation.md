# Phase 4: Consolidation

The code-mapper runs this phase directly after agents complete their work.

## Step 11: Coverage report

Run coverage checks to verify analysis completeness:

```
node_coverage { "node_ids": [<all critical + important symbol IDs>] }
```

Target coverage:
- **Critical symbols (top 2%)**: 90% should have at least one attached memory
- **Important symbols (top 15%)**: 80% should have at least one attached memory
- **Files**: 100% should have at least a baseline summary (from Phase 2)

If coverage falls short, create targeted follow-up work packets for the gaps (max 1 retry round — see Phase 3 error recovery).

## Step 12: Clean up low-quality static-analysis noise

Before consolidating, use tag-aware forget to bulk-remove low-value enrichment memories that agents didn't refine or verify:

```
consolidate { "mode": "forget", "importance_threshold": 0.4, "target_tags": ["static-analysis"], "max_access_count": 1 }
```

This removes `static-analysis` tagged memories with importance < 0.4 that were only accessed once (never recalled by an agent). Memories that agents refined or verified will have higher access counts and survive.

## Step 13: Consolidate findings

```
consolidate { "mode": "cluster", "similarity_threshold": 0.85 }
```

Merge duplicate or near-duplicate findings across agents.

```
consolidate { "mode": "creative" }
```

Find cross-cutting connections between memories that individual agents may have missed.

### Error recovery
- If consolidation fails: log the error but continue. Consolidation is non-critical — the memories are already stored. Skip to Step 14.

## Step 14: Store architectural summary

Store a high-level architectural summary as a high-importance Decision memory:

```
store_memory {
  "content": "<comprehensive architectural summary covering key design decisions, module structure, primary patterns, and critical dependencies>",
  "memory_type": "decision",
  "importance": 0.9,
  "tags": ["architecture", "summary", "agent-analyzed"],
  "namespace": "/path/to/project"
}
```

## Step 15: Clean up pending-analysis memories

Delete any `pending-analysis` tagged memories that were processed during this run:

```
delete_memory { "id": "<pending-analysis-memory-id>" }
```

## Team Shutdown

After all consolidation steps are complete:

1. **Verify all tasks completed**: Run TaskList and confirm every task has status `completed`
2. **Send shutdown requests**: For each teammate, use SendMessage with `type: "shutdown_request"`
3. **Wait for responses**: Each teammate should respond with `type: "shutdown_response"` approving shutdown
4. **Handle rejections**: If a teammate rejects (still working), wait for them to finish, then retry
5. **Clean up team**: Call TeamDelete to remove team resources
6. **Final verification**: Confirm 0 remaining pending-analysis memories, all agents terminated

### Error recovery
- If a teammate doesn't respond to shutdown: wait 1 turn, then retry. If still unresponsive after 2 attempts, proceed with TeamDelete anyway.
- If coverage targets are unmet after gap-filling: store a `needs-review` memory listing the uncovered symbols so the next run can address them.
