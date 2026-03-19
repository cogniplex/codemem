# Post-Review Fixes: Temporal Graph, Gemini Provider, Migration Safety

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 2 critical bugs, 4 important issues, and 2 low-severity issues discovered during code review of PRs #50, #52, and #54.

**Architecture:** Fixes are scoped to three subsystems: temporal graph ingestion/queries (engine crate), Gemini embedding provider (embeddings crate), and migration safety (storage crate). Each task is independent and can be worked in any order.

**Tech Stack:** Rust, SQLite, mockito (HTTP mocking), chrono, reqwest, petgraph

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/codemem-engine/src/enrichment/temporal.rs` | Modify | Fix edge insertion silencing (Critical #1), fix expire_deleted_symbols resurrection bug (Critical #2), fix parse_git_log date fallback (Low #2) |
| `crates/codemem-engine/src/graph_ops.rs` | Modify | Add from < to validation (Medium #1), optimize symbol_history with get_edges_for_node (Medium #2) |
| `crates/codemem/src/mcp/tools_temporal.rs` | No change needed | stale_days clamped at engine level in graph_ops.rs |
| `crates/codemem-embeddings/src/tests/gemini_tests.rs` | Modify | Add mockito-based tests (Medium #3) |
| `crates/codemem-embeddings/src/tests/lib_tests.rs` | Modify | Add from_env Gemini test (Medium #4) |
| `crates/codemem-storage/src/migrations/015_temporal_graph_nodes.sql` | Modify | Make ALTER TABLE defensive (Low #1) |
| `crates/codemem-engine/src/tests/enrichment_tests.rs` | Modify | Add temporal ingestion integration tests |
| `crates/codemem-engine/src/tests/graph_ops_tests.rs` | Create | Temporal query integration tests |
| `crates/codemem-engine/src/lib.rs` | Modify | Register new test module via `#[path]` attribute |

---

### Task 1: Fix silent edge insertion in temporal ingestion (Critical)

**Problem:** `ingest_git_temporal` persists edges to SQLite via `insert_graph_edges_batch` (line 295), then silently discards `add_edge` failures when adding to the in-memory graph (lines 300-304). When `file:` nodes don't exist in the graph, edges are stored in SQLite but missing from memory. This divergence persists across restarts.

**Fix:** Before adding edges to the in-memory graph, ensure placeholder file nodes exist. Log warnings for any remaining failures.

**Files:**
- Modify: `crates/codemem-engine/src/enrichment/temporal.rs:293-305`
- Test: `crates/codemem-engine/src/tests/enrichment_tests.rs`

- [ ] **Step 1: Write the failing test**

In `crates/codemem-engine/src/tests/enrichment_tests.rs`, add:

```rust
#[test]
fn temporal_edge_insertion_creates_placeholder_nodes() {
    // When a ModifiedBy edge references a file:src/main.rs node that doesn't
    // exist in the graph, the ingestion should create a placeholder File node
    // so the in-memory graph stays consistent with SQLite.
    let engine = CodememEngine::for_testing();

    // Manually insert a commit node and an edge referencing a non-existent file
    let commit_node = GraphNode {
        id: "commit:abc123".into(),
        kind: NodeKind::Commit,
        label: "abc123 test commit".into(),
        payload: Default::default(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".into()),
        valid_from: Some(chrono::Utc::now()),
        valid_to: None,
    };

    let edge = Edge {
        id: "modby:file:src/main.rs:abc123".into(),
        src: "file:src/main.rs".into(),
        dst: "commit:abc123".into(),
        relationship: RelationshipType::ModifiedBy,
        weight: 0.4,
        properties: Default::default(),
        created_at: chrono::Utc::now(),
        valid_from: Some(chrono::Utc::now()),
        valid_to: None,
    };

    // Persist to storage
    engine.storage.insert_graph_nodes_batch(&[commit_node.clone()]).unwrap();
    engine.storage.insert_graph_edges_batch(&[edge.clone()]).unwrap();

    // Simulate the fixed Step 9 logic
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(commit_node).unwrap();
    }

    // The edge's source file:src/main.rs doesn't exist yet.
    // After calling ensure_edge_endpoints + add_edge, it should exist.
    engine.ensure_edge_endpoints_and_add(&[edge]).unwrap();

    let graph = engine.lock_graph().unwrap();
    let file_node = graph.get_node("file:src/main.rs").unwrap();
    assert!(file_node.is_some(), "Placeholder file node should be created");
    assert_eq!(file_node.unwrap().kind, NodeKind::File);

    let edges = graph.get_edges("file:src/main.rs").unwrap();
    assert!(!edges.is_empty(), "Edge should exist in in-memory graph");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p codemem-engine temporal_edge_insertion_creates_placeholder_nodes -- --nocapture`
Expected: FAIL — `ensure_edge_endpoints_and_add` does not exist yet.

- [ ] **Step 3: Write the implementation**

In `crates/codemem-engine/src/enrichment/temporal.rs`, add a helper method and update Step 9:

```rust
impl CodememEngine {
    /// Add edges to the in-memory graph, creating placeholder nodes for any
    /// missing endpoints. This prevents storage/memory divergence when edges
    /// reference file: nodes that haven't been indexed yet.
    pub(crate) fn ensure_edge_endpoints_and_add(
        &self,
        edges: &[Edge],
    ) -> Result<(), CodememError> {
        let mut graph = self.lock_graph()?;
        let mut warned = 0usize;

        for edge in edges {
            // Ensure source node exists
            if graph.get_node(&edge.src)?.is_none() {
                // Use File for file: prefixes, Function as default for sym: prefixes
                // (symbol-level edges point to functions/methods most often),
                // External for anything else (safe catch-all, already in the enum).
                let kind = if edge.src.starts_with("file:") {
                    NodeKind::File
                } else if edge.src.starts_with("sym:") {
                    NodeKind::Function
                } else {
                    NodeKind::External
                };
                let placeholder = GraphNode {
                    id: edge.src.clone(),
                    kind,
                    label: edge.src.clone(),
                    payload: Default::default(),
                    centrality: 0.0,
                    memory_id: None,
                    namespace: None,
                    valid_from: None,
                    valid_to: None,
                };
                graph.add_node(placeholder)?;
            }

            // Ensure destination node exists
            if graph.get_node(&edge.dst)?.is_none() {
                let kind = if edge.dst.starts_with("commit:") {
                    NodeKind::Commit
                } else if edge.dst.starts_with("pr:") {
                    NodeKind::PullRequest
                } else {
                    NodeKind::External
                };
                let placeholder = GraphNode {
                    id: edge.dst.clone(),
                    kind,
                    label: edge.dst.clone(),
                    payload: Default::default(),
                    centrality: 0.0,
                    memory_id: None,
                    namespace: None,
                    valid_from: None,
                    valid_to: None,
                };
                graph.add_node(placeholder)?;
            }

            if let Err(e) = graph.add_edge(edge.clone()) {
                warned += 1;
                if warned <= 5 {
                    tracing::warn!("Failed to add edge {} to in-memory graph: {e}", edge.id);
                }
            }
        }

        if warned > 5 {
            tracing::warn!("... and {} more edge insertion warnings", warned - 5);
        }

        Ok(())
    }
}
```

Then replace the Step 9 block in `ingest_git_temporal` (lines 293-305):

**Old:**
```rust
        // ── Step 9: Persist to storage and in-memory graph ──────────────
        self.storage.insert_graph_nodes_batch(&commit_nodes)?;
        self.storage.insert_graph_edges_batch(&edges)?;

        {
            let mut graph = self.lock_graph()?;
            for node in commit_nodes {
                let _ = graph.add_node(node);
            }
            for edge in edges {
                let _ = graph.add_edge(edge);
            }
        }
```

**New:**
```rust
        // ── Step 9: Persist to storage and in-memory graph ──────────────
        self.storage.insert_graph_nodes_batch(&commit_nodes)?;
        self.storage.insert_graph_edges_batch(&edges)?;

        {
            let mut graph = self.lock_graph()?;
            for node in commit_nodes {
                let _ = graph.add_node(node);
            }
        }
        self.ensure_edge_endpoints_and_add(&edges)?;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p codemem-engine temporal_edge_insertion_creates_placeholder_nodes -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/codemem-engine/src/enrichment/temporal.rs crates/codemem-engine/src/tests/enrichment_tests.rs
git commit -m "fix: create placeholder nodes for temporal edge endpoints

Edges persisted to SQLite could reference file: nodes not yet in the
in-memory graph, causing storage/memory divergence. Now creates
placeholder File/Symbol nodes before adding edges, with warnings
for any remaining failures."
```

---

### Task 2: Fix delete-then-recreate file expiry bug (Critical)

**Problem:** `expire_deleted_symbols` runs `git log --diff-filter=D` independently of the incremental sentinel. If a file was deleted then re-created, the old deletion event is re-detected and `valid_to` is set on the now-alive node. The `valid_to.is_some()` guard only prevents double-expiry, not expiry of a resurrected node that has `valid_to: None`.

**Fix:** After collecting deletion events, check if the file currently exists in the working tree. If it does, skip expiry. Also track the deletion processing boundary in the sentinel.

**Files:**
- Modify: `crates/codemem-engine/src/enrichment/temporal.rs:484-593`
- Test: `crates/codemem-engine/src/tests/enrichment_tests.rs`

- [ ] **Step 1: Write a unit test for the filtering logic**

The `expire_deleted_symbols` method calls `git log` and operates on a real repo, making it hard to unit-test directly. Instead, test the fix indirectly: the new filtering code uses `Path::exists()` to skip resurrected files. We can't easily mock the filesystem, but we CAN write an integration-style test that verifies the overall behavior by setting up a temp git repo.

In `crates/codemem-engine/src/tests/enrichment_tests.rs`, add:

```rust
#[test]
fn expire_deleted_symbols_skips_files_that_exist_on_disk() {
    // The fix filters out deleted-file paths that currently exist on disk.
    // We test this by creating a temp dir with a file, setting up deletion
    // events that include that file, and verifying it's NOT expired.
    use tempfile::TempDir;
    use std::fs;

    let engine = CodememEngine::for_testing();
    let tmp = TempDir::new().unwrap();
    let tmp_path = tmp.path();

    // Create a file that "exists on disk" (simulating resurrection)
    let resurrected = tmp_path.join("src").join("resurrected.rs");
    fs::create_dir_all(resurrected.parent().unwrap()).unwrap();
    fs::write(&resurrected, "fn main() {}").unwrap();

    // Initialize a git repo so git log works
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(tmp_path)
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["add", "."])
        .current_dir(tmp_path)
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["commit", "-m", "initial", "--allow-empty"])
        .current_dir(tmp_path)
        .output()
        .unwrap();

    // Add file node to graph
    let file_node = GraphNode {
        id: "file:src/resurrected.rs".into(),
        kind: NodeKind::File,
        label: "src/resurrected.rs".into(),
        payload: Default::default(),
        centrality: 0.5,
        memory_id: None,
        namespace: Some("test".into()),
        valid_from: Some(chrono::Utc::now() - chrono::Duration::days(10)),
        valid_to: None,
    };
    engine.storage.insert_graph_node(&file_node).unwrap();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node).unwrap();
    }

    // expire_deleted_symbols should find no deletions (file exists on disk)
    // even if called — git log --diff-filter=D won't find anything in this
    // fresh repo, AND the new filter would catch it if it did.
    let result = engine.expire_deleted_symbols(
        tmp_path.to_str().unwrap(),
        &[], // empty commits list
        "test",
    );
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);

    // Verify file is still alive
    let graph = engine.lock_graph().unwrap();
    let node = graph.get_node("file:src/resurrected.rs").unwrap().unwrap();
    assert!(node.valid_to.is_none(), "Resurrected file should not be expired");
}
```

Note: `expire_deleted_symbols` is a private method. If it's `fn expire_deleted_symbols` (not `pub`), the test needs to be in the same crate or the method needs `pub(crate)` visibility. Check and adjust. The `#[path]` test modules in this crate are compiled as part of the crate, so private method access should work.

- [ ] **Step 2: Run test to verify baseline**

Run: `cargo test -p codemem-engine expire_deleted_symbols_skips_files -- --nocapture`

- [ ] **Step 3: Write the implementation**

In `expire_deleted_symbols`, after parsing deletion events (line 539) and before collecting expired nodes (line 545), add a working-tree existence check:

```rust
        // Filter out files that currently exist in the working tree
        // (they were deleted then re-created, so should not be expired)
        for (_date, deleted_files) in &mut deletions {
            deleted_files.retain(|f| {
                let full_path = std::path::Path::new(path).join(f);
                !full_path.exists()
            });
        }
        deletions.retain(|(_, files)| !files.is_empty());

        if deletions.is_empty() {
            return Ok(0);
        }
```

Insert this block between line 539 (`if deletions.is_empty() { return Ok(0); }`) and line 545 (`// Phase 1: collect expired nodes`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p codemem-engine expire_deleted_symbols_skips_resurrected_files -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cargo test --workspace`

- [ ] **Step 6: Commit**

```bash
git add crates/codemem-engine/src/enrichment/temporal.rs crates/codemem-engine/src/tests/enrichment_tests.rs
git commit -m "fix: skip expiry for delete-then-recreate files

expire_deleted_symbols now checks if deleted files currently exist in
the working tree before setting valid_to. Files that were deleted then
re-created are not expired."
```

---

### Task 3: Add from < to date validation in temporal queries (Medium)

**Problem:** `what_changed` and `detect_drift` silently return empty/misleading results when `from > to`. The `find_stale_files` `stale_days` parameter has no upper bound, and a u64 > i64::MAX wraps to negative.

**Files:**
- Modify: `crates/codemem-engine/src/graph_ops.rs:178-183, 395-400, 312-322`
- Create: `crates/codemem-engine/src/tests/graph_ops_tests.rs`
- Modify: `crates/codemem-engine/src/tests/mod.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/codemem-engine/src/tests/graph_ops_tests.rs`:

```rust
use crate::CodememEngine;
use chrono::{Duration, Utc};

#[test]
fn what_changed_rejects_reversed_date_range() {
    let engine = CodememEngine::for_testing();
    let now = Utc::now();
    let yesterday = now - Duration::days(1);

    let result = engine.what_changed(now, yesterday, None);
    assert!(result.is_err(), "from > to should return an error");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("before"), "Error should mention date ordering");
}

#[test]
fn detect_drift_rejects_reversed_date_range() {
    let engine = CodememEngine::for_testing();
    let now = Utc::now();
    let yesterday = now - Duration::days(1);

    let result = engine.detect_drift(now, yesterday, None);
    assert!(result.is_err());
}

#[test]
fn find_stale_files_clamps_extreme_stale_days() {
    let engine = CodememEngine::for_testing();
    // Should not panic or produce an absurd cutoff
    let result = engine.find_stale_files(None, u64::MAX);
    assert!(result.is_ok());
}
```

Register the module in `crates/codemem-engine/src/lib.rs` using the crate's existing `#[path]` pattern (add alongside the other test module declarations around line 81):

```rust
#[cfg(test)]
#[path = "tests/graph_ops_tests.rs"]
mod graph_ops_tests;
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p codemem-engine what_changed_rejects_reversed -- --nocapture`
Expected: FAIL — currently returns Ok([]) instead of Err.

- [ ] **Step 3: Write the implementation**

In `crates/codemem-engine/src/graph_ops.rs`, add validation at the top of `what_changed` (after line 183):

```rust
    pub fn what_changed(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        namespace: Option<&str>,
    ) -> Result<Vec<ChangeEntry>, CodememError> {
        if from > to {
            return Err(CodememError::InvalidInput(
                "'from' must be before 'to'".into(),
            ));
        }
```

Same for `detect_drift` (after line 400):

```rust
    pub fn detect_drift(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        namespace: Option<&str>,
    ) -> Result<DriftReport, CodememError> {
        if from > to {
            return Err(CodememError::InvalidInput(
                "'from' must be before 'to'".into(),
            ));
        }
```

For `find_stale_files`, clamp `stale_days` (line 322):

```rust
        let stale_days = stale_days.min(3650); // Cap at 10 years
        let cutoff = Utc::now() - chrono::Duration::days(stale_days as i64);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p codemem-engine graph_ops_tests -- --nocapture`
Expected: All 3 pass.

- [ ] **Step 5: Run full test suite**

Run: `cargo test --workspace`

- [ ] **Step 6: Commit**

```bash
git add crates/codemem-engine/src/graph_ops.rs crates/codemem-engine/src/tests/graph_ops_tests.rs crates/codemem-engine/src/lib.rs
git commit -m "fix: validate date ranges in temporal queries

what_changed and detect_drift now reject from > to with InvalidInput.
find_stale_files clamps stale_days to 3650 to prevent i64 overflow."
```

---

### Task 4: Optimize symbol_history to reduce edge scanning (Medium)

**Problem:** `symbol_history` calls `all_graph_edges()` loading every edge from SQLite. The method has two phases: (1) find commit IDs that the queried node connects to, (2) for each commit, find ALL files/symbols that were also modified. Phase 1 can use the targeted `get_edges_for_node(node_id)`. Phase 2 needs edges for each commit — use `get_edges_for_node` per commit instead of a single full scan.

**Files:**
- Modify: `crates/codemem-engine/src/graph_ops.rs:509-590`
- Test: `crates/codemem-engine/src/tests/graph_ops_tests.rs`

- [ ] **Step 1: Write a test showing the optimization works identically**

In `crates/codemem-engine/src/tests/graph_ops_tests.rs`, add:

```rust
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::HashMap;

#[test]
fn symbol_history_returns_commits_with_all_changed_files() {
    let engine = CodememEngine::for_testing();
    let now = Utc::now();

    // Create a file node, a second file node, and a commit node
    let file1 = GraphNode {
        id: "file:src/lib.rs".into(),
        kind: NodeKind::File,
        label: "src/lib.rs".into(),
        payload: Default::default(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".into()),
        valid_from: None,
        valid_to: None,
    };
    let file2 = GraphNode {
        id: "file:src/utils.rs".into(),
        kind: NodeKind::File,
        label: "src/utils.rs".into(),
        payload: Default::default(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".into()),
        valid_from: None,
        valid_to: None,
    };
    let commit = GraphNode {
        id: "commit:aaa".into(),
        kind: NodeKind::Commit,
        label: "aaa fix bug".into(),
        payload: {
            let mut p = HashMap::new();
            p.insert("hash".into(), serde_json::json!("aaa"));
            p.insert("author".into(), serde_json::json!("alice"));
            p.insert("date".into(), serde_json::json!(now.to_rfc3339()));
            p.insert("subject".into(), serde_json::json!("fix bug"));
            p
        },
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".into()),
        valid_from: Some(now),
        valid_to: None,
    };

    engine.storage.insert_graph_nodes_batch(&[file1.clone(), file2.clone(), commit.clone()]).unwrap();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file1).unwrap();
        graph.add_node(file2).unwrap();
        graph.add_node(commit).unwrap();
    }

    // Commit aaa modified BOTH files
    let edge1 = Edge {
        id: "modby:file:src/lib.rs:aaa".into(),
        src: "file:src/lib.rs".into(),
        dst: "commit:aaa".into(),
        relationship: RelationshipType::ModifiedBy,
        weight: 0.4,
        properties: Default::default(),
        created_at: now,
        valid_from: Some(now),
        valid_to: None,
    };
    let edge2 = Edge {
        id: "modby:file:src/utils.rs:aaa".into(),
        src: "file:src/utils.rs".into(),
        dst: "commit:aaa".into(),
        relationship: RelationshipType::ModifiedBy,
        weight: 0.4,
        properties: Default::default(),
        created_at: now,
        valid_from: Some(now),
        valid_to: None,
    };
    engine.storage.insert_graph_edges_batch(&[edge1, edge2]).unwrap();

    // Query history for lib.rs — should show commit aaa with BOTH files in changed_files
    let history = engine.symbol_history("file:src/lib.rs").unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].hash, "aaa");
    // Critical: changed_files must include utils.rs too, not just the queried file
    assert!(history[0].changed_files.contains(&"src/lib.rs".to_string()));
    assert!(history[0].changed_files.contains(&"src/utils.rs".to_string()));
}
```

- [ ] **Step 2: Run test to verify baseline**

Run: `cargo test -p codemem-engine symbol_history_returns_commits_with_all -- --nocapture`
Expected: PASS (current implementation loads all edges so this works).

- [ ] **Step 3: Write the optimized implementation**

In `graph_ops.rs`, replace `symbol_history` (lines 509-590). The optimization:
- Phase 1: use `get_edges_for_node(node_id)` to find commit IDs (avoids full scan)
- Phase 2: use `get_edges_for_node(commit_id)` per commit to find sibling files/symbols

**Old (line 509-511):**
```rust
    pub fn symbol_history(&self, node_id: &str) -> Result<Vec<ChangeEntry>, CodememError> {
        let all_edges = self.storage.all_graph_edges()?;

        // Find all ModifiedBy edges from this node to commit nodes
        let commit_ids: HashSet<&str> = all_edges
            .iter()
            .filter(|e| e.src == node_id && e.relationship == RelationshipType::ModifiedBy)
            .map(|e| e.dst.as_str())
            .collect();
```

**New:**
```rust
    pub fn symbol_history(&self, node_id: &str) -> Result<Vec<ChangeEntry>, CodememError> {
        // Phase 1: find commit IDs connected to this node (targeted query)
        let node_edges = self.storage.get_edges_for_node(node_id)?;

        let commit_ids: HashSet<String> = node_edges
            .iter()
            .filter(|e| e.src == node_id && e.relationship == RelationshipType::ModifiedBy)
            .map(|e| e.dst.clone())
            .collect();
```

**Old (line 533-541):**
```rust
        // Index ModifiedBy edges by dst to populate changed_files/symbols
        let mut edges_by_dst: HashMap<&str, Vec<&Edge>> = HashMap::new();
        for edge in &all_edges {
            if edge.relationship == RelationshipType::ModifiedBy
                && commit_ids.contains(edge.dst.as_str())
            {
                edges_by_dst.entry(&edge.dst).or_default().push(edge);
            }
        }
```

**New:**
```rust
        // Phase 2: for each commit, load its edges to find sibling files/symbols
        let mut edges_by_commit: HashMap<String, Vec<Edge>> = HashMap::new();
        for cid in &commit_ids {
            let commit_edges = self.storage.get_edges_for_node(cid)?;
            let modby: Vec<Edge> = commit_edges
                .into_iter()
                .filter(|e| e.relationship == RelationshipType::ModifiedBy && e.dst == *cid)
                .collect();
            edges_by_commit.insert(cid.clone(), modby);
        }
```

Then update the lookup below to use `edges_by_commit`:

**Old (line 548):**
```rust
            if let Some(commit_edges) = edges_by_dst.get(node.id.as_str()) {
```

**New:**
```rust
            if let Some(commit_edges) = edges_by_commit.get(&node.id) {
```

And the edge iteration now references owned `Edge` values:
```rust
                for edge in commit_edges {
```

Also update the `commit_ids.contains` check on line 529 — it now holds `String` not `&str`:
```rust
                .filter(|n| commit_ids.contains(&n.id))
```

- [ ] **Step 4: Run test to verify it still passes**

Run: `cargo test -p codemem-engine symbol_history_returns_commits_with_all -- --nocapture`
Expected: PASS — both files appear in changed_files.

- [ ] **Step 5: Run full test suite**

Run: `cargo test --workspace`

- [ ] **Step 6: Commit**

```bash
git add crates/codemem-engine/src/graph_ops.rs crates/codemem-engine/src/tests/graph_ops_tests.rs
git commit -m "perf: use targeted edge queries in symbol_history

Replaces all_graph_edges() full table scan with get_edges_for_node()
for both the queried node and each commit. Still correctly reports
all sibling files modified in each commit."
```

---

### Task 5: Add mockito-based tests for Gemini embedding provider (Medium)

**Problem:** Gemini has 66 lines of tests vs 558 for OpenAI. Zero CI coverage for HTTP response parsing, errors, batch chunking. Follow the OpenAI test patterns.

**Files:**
- Modify: `crates/codemem-embeddings/src/tests/gemini_tests.rs`

- [ ] **Step 1: Read the existing OpenAI tests for reference**

Read: `crates/codemem-embeddings/src/tests/openai_tests.rs`
Understand the mockito server setup pattern, response formats, and error scenarios.

- [ ] **Step 2: Write mockito tests for single embed**

Add to `crates/codemem-embeddings/src/tests/gemini_tests.rs`:

```rust
use crate::gemini::GeminiProvider;
use crate::EmbeddingProvider;

#[test]
fn embed_single_success() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .match_header("x-goog-api-key", "test-key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding": {"values": [0.1, 0.2, 0.3]}}"#)
        .create();

    let provider = GeminiProvider::new("test-key", "text-embedding-004", 3, Some(&server.url()));
    let result = provider.embed("hello world").unwrap();
    assert_eq!(result, vec![0.1f32, 0.2, 0.3]);
    mock.assert();
}

#[test]
fn embed_single_401_unauthorized() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(401)
        .with_body(r#"{"error": {"message": "API key not valid"}}"#)
        .create();

    let provider = GeminiProvider::new("bad-key", "text-embedding-004", 3, Some(&server.url()));
    let result = provider.embed("test");
    assert!(result.is_err());
    mock.assert();
}

#[test]
fn embed_single_500_server_error() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(500)
        .with_body("Internal Server Error")
        .create();

    let provider = GeminiProvider::new("test-key", "text-embedding-004", 3, Some(&server.url()));
    let result = provider.embed("test");
    assert!(result.is_err());
    mock.assert();
}

#[test]
fn embed_single_429_rate_limited() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(429)
        .with_body(r#"{"error": {"message": "Rate limit exceeded"}}"#)
        .create();

    let provider = GeminiProvider::new("test-key", "text-embedding-004", 3, Some(&server.url()));
    let result = provider.embed("test");
    assert!(result.is_err());
    mock.assert();
}

#[test]
fn embed_single_malformed_json() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(200)
        .with_body("not json at all")
        .create();

    let provider = GeminiProvider::new("test-key", "text-embedding-004", 3, Some(&server.url()));
    let result = provider.embed("test");
    assert!(result.is_err());
    mock.assert();
}

#[test]
fn embed_single_missing_embedding_field() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(200)
        .with_body(r#"{"something_else": true}"#)
        .create();

    let provider = GeminiProvider::new("test-key", "text-embedding-004", 3, Some(&server.url()));
    let result = provider.embed("test");
    assert!(result.is_err());
    mock.assert();
}

#[test]
fn embed_single_dimension_mismatch() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:embedContent")
        .with_status(200)
        .with_body(r#"{"embedding": {"values": [0.1, 0.2, 0.3]}}"#)
        .create();

    // Request 768 dims but server returns 3
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 768, Some(&server.url()));
    let result = provider.embed("test");
    assert!(result.is_err(), "Should detect dimension mismatch");
    mock.assert();
}
```

- [ ] **Step 3: Run single-embed tests**

Run: `cargo test -p codemem-embeddings gemini_tests -- --nocapture`
Expected: All new tests pass (they test the existing Gemini implementation against mocked HTTP).

- [ ] **Step 4: Write mockito tests for batch embed**

Add to the same file:

```rust
#[test]
fn embed_batch_success() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:batchEmbedContents")
        .match_header("x-goog-api-key", "test-key")
        .with_status(200)
        .with_body(r#"{"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]}"#)
        .create();

    let provider = GeminiProvider::new("test-key", "text-embedding-004", 2, Some(&server.url()));
    let result = provider.embed_batch(&["hello", "world"]).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], vec![0.1f32, 0.2]);
    assert_eq!(result[1], vec![0.3f32, 0.4]);
    mock.assert();
}

#[test]
fn embed_batch_empty_input() {
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 3, None);
    let result = provider.embed_batch(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn embed_batch_count_mismatch() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:batchEmbedContents")
        .with_status(200)
        .with_body(r#"{"embeddings": [{"values": [0.1, 0.2]}]}"#)
        .create();

    // Send 2 texts but mock returns only 1 embedding
    let provider = GeminiProvider::new("test-key", "text-embedding-004", 2, Some(&server.url()));
    let result = provider.embed_batch(&["hello", "world"]);
    assert!(result.is_err(), "Should detect count mismatch");
    mock.assert();
}

#[test]
fn embed_batch_500_error() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:batchEmbedContents")
        .with_status(500)
        .with_body("Internal Server Error")
        .create();

    let provider = GeminiProvider::new("test-key", "text-embedding-004", 2, Some(&server.url()));
    let result = provider.embed_batch(&["hello", "world"]);
    assert!(result.is_err());
    mock.assert();
}

#[test]
fn embed_batch_malformed_response() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/models/text-embedding-004:batchEmbedContents")
        .with_status(200)
        .with_body(r#"{"embeddings": "not an array"}"#)
        .create();

    let provider = GeminiProvider::new("test-key", "text-embedding-004", 2, Some(&server.url()));
    let result = provider.embed_batch(&["hello", "world"]);
    assert!(result.is_err());
    mock.assert();
}
```

- [ ] **Step 5: Run batch tests**

Run: `cargo test -p codemem-embeddings gemini_tests -- --nocapture`
Expected: All tests pass.

- [ ] **Step 6: Run full test suite**

Run: `cargo test --workspace`

- [ ] **Step 7: Commit**

```bash
git add crates/codemem-embeddings/src/tests/gemini_tests.rs
git commit -m "test: add mockito-based tests for Gemini embedding provider

Adds 12 mock HTTP tests covering: single embed success/errors,
dimension validation, malformed responses, batch success/errors,
count mismatch. Brings Gemini closer to OpenAI test parity."
```

---

### Task 6: Add from_env integration test for Gemini provider (Medium)

**Problem:** The factory wiring for `"gemini"`/`"google"` in `from_env()` (lib.rs lines 688-714) has no test. All other providers have from_env tests.

**Files:**
- Modify: `crates/codemem-embeddings/src/tests/lib_tests.rs`

- [ ] **Step 1: Write the tests**

Add to `crates/codemem-embeddings/src/tests/lib_tests.rs`, near the existing `from_env_*` tests:

```rust
#[test]
fn from_env_gemini_provider() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "gemini") };
    unsafe { std::env::set_var("CODEMEM_EMBED_API_KEY", "test-gemini-key") };
    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };
    unsafe { std::env::remove_var("CODEMEM_EMBED_API_KEY") };

    let provider = result.expect("from_env should succeed for gemini");
    assert_eq!(provider.name(), "gemini");
}

#[test]
fn from_env_google_alias() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "google") };
    unsafe { std::env::set_var("CODEMEM_EMBED_API_KEY", "test-google-key") };
    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };
    unsafe { std::env::remove_var("CODEMEM_EMBED_API_KEY") };

    let provider = result.expect("'google' alias should create gemini provider");
    assert_eq!(provider.name(), "gemini");
}

#[test]
fn from_env_gemini_missing_api_key() {
    let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("CODEMEM_EMBED_PROVIDER", "gemini") };
    unsafe { std::env::remove_var("CODEMEM_EMBED_API_KEY") };
    unsafe { std::env::remove_var("GEMINI_API_KEY") };
    unsafe { std::env::remove_var("GOOGLE_API_KEY") };
    let result = from_env(None);
    unsafe { std::env::remove_var("CODEMEM_EMBED_PROVIDER") };

    assert!(result.is_err(), "gemini without API key should fail");
}
```

Note: Uses the same `ENV_MUTEX` + `unsafe { set_var/remove_var }` pattern as all other `from_env_*` tests in this file. `provider.name()` returns `"gemini"` (not `"cached(gemini)"`) because `CachedProvider` delegates to the inner provider's name.

- [ ] **Step 2: Run the tests**

Run: `cargo test -p codemem-embeddings from_env_gemini -- --nocapture`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace`

- [ ] **Step 4: Commit**

```bash
git add crates/codemem-embeddings/src/tests/lib_tests.rs
git commit -m "test: add from_env integration tests for gemini/google provider

Covers gemini provider, google alias, and missing API key error.
Matches existing from_env test patterns for ollama/openai."
```

---

### Task 7: Make migration 015 defensive against column re-add (Low)

**Problem:** `ALTER TABLE ADD COLUMN` in SQLite fails if the column already exists. While the migration runner tracks versions, a corrupted `schema_version` or manual schema edit could cause a hard failure. Migration 003 has the same pattern — this is a pre-existing issue, but 015 is new and can be fixed.

**Fix:** Wrap ALTER TABLE in a conditional check. SQLite doesn't support `IF NOT EXISTS` for ADD COLUMN, so use `PRAGMA table_info()` check in the migration runner, or just use a TRY pattern in SQL.

**Files:**
- Modify: `crates/codemem-storage/src/migrations/015_temporal_graph_nodes.sql`

- [ ] **Step 1: Write the defensive SQL**

SQLite doesn't have `TRY` in SQL, but we can detect existing columns using a pattern that's already safe: if `ALTER TABLE ADD COLUMN` fails because the column exists, the entire migration transaction rolls back and the version isn't recorded — so it will retry next time. The simplest defensive approach is to not change the SQL but instead handle it at the runner level.

However, the lightest fix that's consistent with the codebase is to create a temp table approach. Actually, the cleanest approach for SQLite is:

Replace `crates/codemem-storage/src/migrations/015_temporal_graph_nodes.sql`:

```sql
-- Add temporal validity columns to graph_nodes, matching graph_edges.
-- NULL means "always valid" (backward compatible with existing nodes).
-- Use a CREATE TABLE + copy pattern if columns already exist (e.g., partial upgrade).

-- These ALTER TABLE statements will fail if columns already exist.
-- The migration runner wraps this in a transaction, so partial failure
-- is safe (rolls back and retries on next startup).
-- If this becomes a problem, switch to the CREATE TABLE AS pattern.
ALTER TABLE graph_nodes ADD COLUMN valid_from INTEGER;
ALTER TABLE graph_nodes ADD COLUMN valid_to INTEGER;

CREATE INDEX IF NOT EXISTS idx_graph_nodes_temporal ON graph_nodes(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_kind_temporal ON graph_nodes(kind, valid_from);
```

Actually, since this matches the existing pattern (migration 003 is identical) and the migration runner's transaction-based approach handles it correctly, the real fix is to document it and optionally add a comment. The transaction rollback on failure + version tracking is the correct defense.

**Revised approach:** Add a comment documenting the behavior, consistent with migration 003. No code change needed — the existing transaction wrapping is the safety net.

- [ ] **Step 1: Add documentation comment to migration 015**

```sql
-- Add temporal validity columns to graph_nodes, matching graph_edges.
-- NULL means "always valid" (backward compatible with existing nodes).
--
-- NOTE: ALTER TABLE ADD COLUMN has no IF NOT EXISTS in SQLite.
-- If this migration fails (e.g., columns already exist from partial upgrade),
-- the migration runner's transaction will roll back and retry on next startup.
ALTER TABLE graph_nodes ADD COLUMN valid_from INTEGER;
ALTER TABLE graph_nodes ADD COLUMN valid_to INTEGER;

CREATE INDEX IF NOT EXISTS idx_graph_nodes_temporal ON graph_nodes(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_kind_temporal ON graph_nodes(kind, valid_from);
```

- [ ] **Step 2: Run migration tests**

Run: `cargo test -p codemem-storage migrations -- --nocapture`
Expected: PASS (existing tests cover fresh DB + idempotent re-run)

- [ ] **Step 3: Commit**

```bash
git add crates/codemem-storage/src/migrations/015_temporal_graph_nodes.sql
git commit -m "docs: document ALTER TABLE non-idempotency in migration 015

Adds a comment explaining that ALTER TABLE ADD COLUMN lacks IF NOT EXISTS
in SQLite. The migration runner's transaction wrapping handles failure
safely."
```

---

### Task 8: Fix parse_git_log silent date fallback (Low)

**Problem:** When RFC3339 date parsing fails, `parse_git_log` silently falls back to `Utc::now()` (line 350), which creates commits with wrong timestamps that skew temporal queries.

**Fix:** Log a warning and skip the commit instead of silently using current time.

**Files:**
- Modify: `crates/codemem-engine/src/enrichment/temporal.rs:348-351`

- [ ] **Step 1: Write the fix**

Replace lines 348-350 in `parse_git_log`:

**Old:**
```rust
                    let date = chrono::DateTime::parse_from_rfc3339(parts[3])
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now());
```

**New:**
```rust
                    let date = match chrono::DateTime::parse_from_rfc3339(parts[3]) {
                        Ok(dt) => dt.with_timezone(&chrono::Utc),
                        Err(e) => {
                            tracing::warn!(
                                "Skipping commit {}: unparseable date {:?}: {e}",
                                &parts[0][..parts[0].len().min(7)],
                                parts[3]
                            );
                            continue;
                        }
                    };
```

- [ ] **Step 2: Run related tests**

Run: `cargo test -p codemem-engine enrichment -- --nocapture`
Expected: PASS (no existing test depends on the fallback behavior)

- [ ] **Step 3: Run full test suite and clippy**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings`

- [ ] **Step 4: Commit**

```bash
git add crates/codemem-engine/src/enrichment/temporal.rs
git commit -m "fix: skip commits with unparseable dates instead of using Utc::now

parse_git_log was silently falling back to current time on date parse
failures, which created commits with wrong timestamps. Now logs a
warning and skips the commit."
```

---

## Execution Order

Tasks are independent. Recommended order by priority:

1. **Task 1** — Critical: edge insertion divergence
2. **Task 2** — Critical: delete-then-recreate expiry
3. **Task 3** — Medium: date validation
4. **Task 4** — Medium: symbol_history optimization
5. **Task 5** — Medium: Gemini mockito tests
6. **Task 6** — Medium: Gemini from_env tests
7. **Task 8** — Low: date fallback fix
8. **Task 7** — Low: migration documentation

Tasks 1+2 share the same file (temporal.rs) so should run sequentially. Tasks 3+4 share graph_ops.rs — also sequential. Tasks 5+6 are independent of everything else and can run in parallel with 1-4. Tasks 7+8 are fully independent.

**Parallel groups:**
- Group A: Tasks 1 → 2 → 8 (temporal.rs)
- Group B: Tasks 3 → 4 (graph_ops.rs)
- Group C: Tasks 5 → 6 (embeddings tests)
- Group D: Task 7 (migration docs)

Groups A, B, C, D can all run in parallel.
