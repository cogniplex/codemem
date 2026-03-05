# Incremental Analysis (File Changes)

For re-analysis after file changes — this is the primary use case for active repos. Full analysis runs once; incremental runs frequently.

## Step 1: Re-index from CLI

```bash
codemem index /path/to/project
```

This is incremental — only changed/new files are re-indexed via SHA-256 hash comparison.

## Step 2: Identify what changed

### 2a: Check pending-analysis memories (from Stop hook)

```
recall { "query": "pending analysis file changes", "k": 50 }
```

These contain lists of files edited since last analysis, stored by the Stop lifecycle hook.

### 2b: Diff the graph

Compare current graph state against stored analysis records:

```
recall { "query": "agent analysis complete", "k": 100, "exclude_tags": ["static-analysis"] }
```

For each `agent-analyzed` memory, compare the stored `file_hash` in metadata with the current file hash from indexing. Files with stale hashes need re-analysis.

### 2c: Detect structural changes

```
find_important_nodes { "top_k": 50 }
```

Compare with stored PageRank data. If a symbol's rank changed significantly (>20% shift), it may need re-analysis even if its file didn't change (its callers changed).

## Step 3: Classify changes

| Change Type | Action |
|-------------|--------|
| **New file** | Add to baseline-scanner batch + assign to symbol-analyst if has critical symbols |
| **Modified file** | Re-read, update baseline, re-analyze changed symbols |
| **Deleted file** | Delete associated memories, clean up orphaned edges |
| **Renamed/moved file** | Update file node ID, update baseline memory, preserve symbol memories |
| **New symbols in existing file** | Add to symbol-analyst batch for the file |
| **Removed symbols** | Delete purpose memories, check if cluster summary needs update |
| **Changed imports** | Re-run architecture-reviewer on affected module boundaries |
| **Changed API endpoints** | Re-run api-mapper on affected router file |

## Step 4: Cascade analysis

When a critical symbol changes, its dependents may need re-analysis:

```
graph_traverse {
  "start_id": "sym:<changed_symbol>",
  "max_depth": 2,
  "direction": "incoming",
  "include_relationships": ["CALLS", "IMPORTS", "IMPLEMENTS"]
}
```

For each dependent:
- If it has a decision memory referencing the changed symbol → mark for review
- If it's a critical symbol itself → add to re-analysis batch

## Step 5: Execute incremental analysis

Use the same wave model as full analysis, but with much smaller batches:

1. **Wave 1**: Update baselines for changed files only (1-2 baseline-scanner agents)
2. **Wave 2**: Re-analyze changed symbols + cascade targets (1-3 targeted agents)
3. **Wave 3**: Update architecture/cluster summaries if module boundaries changed (1 agent)

Typically 3-6 agents total for incremental analysis.

## Step 6: Update cluster summaries

If cluster membership changed (symbols added/removed):

```
find_related_groups { "resolution": 1.0 }
```

Compare with stored cluster summaries. Update any that have new members or lost key members.

## Step 7: Clean up

- Delete `pending-analysis` memories that were processed
- Delete orphaned memories for deleted files/symbols
- Update `agent-analyzed` markers with new file hashes
