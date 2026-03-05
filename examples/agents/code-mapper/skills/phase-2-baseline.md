# Phase 2: Baseline Coverage

The code-mapper runs this phase directly. Before deep analysis, establish baseline coverage so every file and package has at least one memory.

**Task tracking**: Set the Phase 2 task to `in_progress` via `TaskUpdate` when starting. When all package and file summaries are stored, set it to `completed`.

## Step 6a: Package summaries

For each top-level package visible in the `summary_tree` output:

1. Read the package node from `summary_tree` to get file count, symbol count, and structure
2. Store 1 context memory per package:

```
store_memory {
  "content": "Package <pkg> contains <N> files. Purpose: <inferred from file names and structure>. Key exports: <top symbols>.",
  "memory_type": "context",
  "importance": 0.4,
  "tags": ["baseline", "package-summary"],
  "links": ["pkg:dir/"],
  "namespace": "/path/to/project"
}
```

## Step 6b: File summaries

For each source file in the codebase:

1. Read the first ~50 lines of the file using the Read tool
2. Explore immediate graph context:
   ```
   graph_traverse { "start_id": "file:<path>", "max_depth": 1, "exclude_kinds": ["chunk"] }
   ```
3. Store 1 context memory per file:

```
store_memory {
  "content": "File <path>: <purpose inferred from imports, exports, and first ~50 lines>. Key functions/exports: <list>. Approximate size: <line count>.",
  "memory_type": "context",
  "importance": 0.3,
  "tags": ["baseline", "file-summary"],
  "links": ["file:<path>"],
  "namespace": "/path/to/project"
}
```

This step is parallelizable: for large codebases (50-100+ files), spawn 2-4 agents to split the file list and process them concurrently. Each agent gets a disjoint file list.

### Duplicate handling
Before storing, check `get_node_memories` for the file/package node. If a baseline summary already exists with a matching file hash, skip it. If it exists with a stale hash, use `refine_memory` to update the content rather than creating a duplicate.

### Error recovery
- If a file cannot be read (deleted, binary, permissions): skip it and log the path. Do not store a baseline memory for unreadable files.
- If `store_memory` fails for a specific file: retry once. If it fails again, continue with the next file — one missing baseline is not critical.
