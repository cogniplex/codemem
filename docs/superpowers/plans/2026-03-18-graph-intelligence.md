# Graph Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six graph intelligence features: missing co-change detection in review_diff, test impact analysis MCP tool, cycle detection MCP tool, weighted community detection, community auto-labeling, and confidence tags in tool output.

**Architecture:** All features build on existing infrastructure — graph traversal, Louvain, Tarjan's SCC, CoChanged edges, and the MCP tool framework. No new crates or major architectural changes. Each feature is independent and can be implemented/tested in isolation.

**Tech Stack:** Rust, petgraph (SCC already used), existing MCP tool framework, existing graph traversal/BFS

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/codemem-engine/src/review.rs` | Modify | Add `missing_co_changes` field to `BlastRadiusReport`, query CoChanged edges |
| `crates/codemem-engine/src/tests/review_tests.rs` | Modify | Tests for missing co-change detection |
| `crates/codemem-engine/src/graph_ops.rs` | Modify | Add `test_impact()`, `detect_cycles()`, `find_call_path()` methods |
| `crates/codemem-engine/src/tests/graph_ops_tests.rs` | Modify | Tests for new graph ops |
| `crates/codemem/src/mcp/definitions.rs` | Modify | Register `test_impact`, `cycles` tool schemas |
| `crates/codemem/src/mcp/tools_graph.rs` | Modify | Add `tool_test_impact()`, `tool_cycles()` handlers |
| `crates/codemem/src/mcp/mod.rs` | Modify | Route new tool names to handlers |
| `crates/codemem-storage/src/graph/algorithms.rs` | Modify | Add `louvain_communities_weighted()` with relationship-aware weights, add `label_community()` |
| `crates/codemem-storage/src/tests/graph_tests.rs` | Modify | Tests for weighted Louvain + auto-labeling |

---

## Task 1: Missing Co-Change Detection in review_diff

Enhance `BlastRadiusReport` to include files that historically change together with the diff's changed files but are absent from the diff.

**Files:**
- Modify: `crates/codemem-engine/src/review.rs`
- Modify: `crates/codemem-engine/src/tests/review_tests.rs`

**Context:** `blast_radius()` in `review.rs` already parses diffs, maps changed lines to symbols, and computes dependents. `CoChanged` edges exist between files that frequently co-change in git history. The `Edge` struct uses `src`/`dst` fields (not `source`/`target`). Access edges via `self.storage.all_graph_edges()` or `graph.get_edges(node_id)`.

- [ ] **Step 1: Add `MissingCoChange` struct and field to `BlastRadiusReport`**

In `crates/codemem-engine/src/review.rs`, add after `MissingChange`:

```rust
/// A file historically coupled with changed files but absent from the diff.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MissingCoChange {
    /// The file missing from the diff.
    pub file_path: String,
    /// Which changed file(s) it's coupled with.
    pub coupled_with: Vec<String>,
    /// Average coupling strength (0.0–1.0).
    pub strength: f64,
}
```

Add to `BlastRadiusReport`:

```rust
pub missing_co_changes: Vec<MissingCoChange>,
```

- [ ] **Step 2: Fix compilation — initialize new field**

Find where `BlastRadiusReport` is constructed in `blast_radius()` and add `missing_co_changes: Vec::new()`. Run `cargo check -p codemem-engine` to verify it compiles.

- [ ] **Step 3: Write failing test**

Add to `crates/codemem-engine/src/tests/review_tests.rs`:

```rust
#[test]
fn missing_co_changes_detected() {
    let engine = CodememEngine::for_testing();

    // Create file nodes
    {
        let mut graph = engine.lock_graph().unwrap();
        graph.add_node(file_node("src/auth.rs")).unwrap();
        graph.add_node(file_node("src/auth_test.rs")).unwrap();
        graph.add_node(file_node("src/middleware.rs")).unwrap();
    }

    // Add CoChanged edge: auth.rs <-> auth_test.rs (strength 0.8)
    engine.storage.upsert_graph_edge(&Edge {
        id: "cochanged:auth-authtest".into(),
        src: "file:src/auth.rs".into(),
        dst: "file:src/auth_test.rs".into(),
        relationship: RelationshipType::CoChanged,
        weight: 0.8,
        properties: HashMap::new(),
        created_at: chrono::Utc::now(),
        valid_from: None,
        valid_to: None,
    }).unwrap();

    // Diff only touches auth.rs — auth_test.rs should be flagged as missing
    let diff = "\
--- a/src/auth.rs
+++ b/src/auth.rs
@@ -10,3 +10,4 @@
 fn validate() {
+    check_token();
 }
";

    let report = engine.blast_radius(diff, 2).unwrap();
    assert!(!report.missing_co_changes.is_empty(),
        "auth_test.rs should be flagged as missing co-change");
    assert!(report.missing_co_changes.iter().any(|m| m.file_path.contains("auth_test")));
}
```

Adapt the test helper names (`file_node`, `Edge` construction) to match existing test patterns in the file. Read `review_tests.rs` first to see what helpers exist.

- [ ] **Step 4: Run test to verify it fails**

Run: `cargo test -p codemem-engine missing_co_changes -- --nocapture`
Expected: FAIL — `missing_co_changes` is always empty

- [ ] **Step 5: Implement co-change detection in `blast_radius()`**

In `review.rs`, inside `blast_radius()`, after computing `changed_files` but before building the final report:

```rust
// Detect missing co-changes: files coupled with changed files but absent from diff.
let changed_file_set: HashSet<&str> = diff_mapping.changed_files
    .iter()
    .map(|f| f.as_str())
    .collect();

let mut co_change_map: HashMap<String, (Vec<String>, f64, usize)> = HashMap::new();

for changed_file_id in &diff_mapping.changed_files {
    if let Ok(edges) = graph.get_edges(changed_file_id) {
        for edge in &edges {
            if edge.relationship != RelationshipType::CoChanged {
                continue;
            }
            // Find the other end of the edge
            let other = if edge.src == *changed_file_id {
                &edge.dst
            } else if edge.dst == *changed_file_id {
                &edge.src
            } else {
                continue;
            };
            // Skip if the coupled file is already in the diff
            if changed_file_set.contains(other.as_str()) {
                continue;
            }
            let entry = co_change_map.entry(other.clone()).or_insert_with(|| {
                (Vec::new(), 0.0, 0)
            });
            // Extract file path from node ID (strip "file:" prefix)
            let coupled_path = changed_file_id.strip_prefix("file:").unwrap_or(changed_file_id);
            entry.0.push(coupled_path.to_string());
            entry.1 += edge.weight;
            entry.2 += 1;
        }
    }
}

let missing_co_changes: Vec<MissingCoChange> = co_change_map
    .into_iter()
    .map(|(file_id, (coupled_with, total_weight, count))| {
        let file_path = file_id.strip_prefix("file:").unwrap_or(&file_id).to_string();
        MissingCoChange {
            file_path,
            coupled_with,
            strength: total_weight / count as f64,
        }
    })
    .collect();
```

Set `missing_co_changes` on the report.

- [ ] **Step 6: Run test**

Run: `cargo test -p codemem-engine missing_co_changes -- --nocapture`
Expected: PASS

- [ ] **Step 7: Run full tests + clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`

- [ ] **Step 8: Commit**

```bash
git add crates/codemem-engine/src/review.rs crates/codemem-engine/src/tests/review_tests.rs
git commit -m "feat: detect missing co-changes in review_diff blast radius

Files historically coupled (CoChanged edges) with changed files but
absent from the diff are now surfaced in BlastRadiusReport. Helps
catch forgotten test files, config files, and tightly-coupled modules."
```

---

## Task 2: Test Impact Analysis MCP Tool

New `test_impact` MCP tool: given a diff or symbol names, BFS callers up to depth 4, filter to test files, split into direct (depth ≤ 2) and transitive (depth 3+).

**Files:**
- Modify: `crates/codemem-engine/src/graph_ops.rs`
- Modify: `crates/codemem-engine/src/tests/graph_ops_tests.rs`
- Modify: `crates/codemem/src/mcp/definitions.rs`
- Modify: `crates/codemem/src/mcp/tools_graph.rs`
- Modify: `crates/codemem/src/mcp/mod.rs`

**Context:** `graph_ops.rs` has `graph_traverse()` with BFS support. `GraphBackend::bfs_filtered()` in `traversal.rs` supports `max_depth`, `include_relationships`, and `exclude_kinds`. Test files have nodes with `is_test: true` in payload or file paths matching `test_*`, `*_test.*`, `tests/`, `__tests__/`.

- [ ] **Step 1: Define `TestImpactResult` in `graph_ops.rs`**

```rust
/// Result of test impact analysis.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TestImpactResult {
    /// Tests directly calling changed symbols (depth ≤ 2).
    pub direct_tests: Vec<TestHit>,
    /// Tests transitively affected (depth 3+).
    pub transitive_tests: Vec<TestHit>,
    /// Changed symbols that were analyzed.
    pub analyzed_symbols: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TestHit {
    pub test_symbol: String,
    pub file_path: String,
    pub via: String,  // intermediate symbol that connects to the changed symbol
    pub depth: usize,
}
```

- [ ] **Step 2: Write failing test**

Add to `crates/codemem-engine/src/tests/graph_ops_tests.rs`:

```rust
#[test]
fn test_impact_finds_direct_and_transitive_tests() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        // Source symbols
        graph.add_node(sym_node("validate", "function", "src/auth.rs")).unwrap();
        graph.add_node(sym_node("login", "function", "src/auth.rs")).unwrap();
        graph.add_node(sym_node("middleware", "function", "src/middleware.rs")).unwrap();
        // Test symbols
        graph.add_node(test_node("test_validate", "tests/test_auth.rs")).unwrap();
        graph.add_node(test_node("test_login", "tests/test_auth.rs")).unwrap();
        graph.add_node(test_node("test_middleware", "tests/test_middleware.rs")).unwrap();

        // test_validate calls validate (depth 1 — direct)
        graph.add_edge(calls_edge("test_validate", "validate")).unwrap();
        // test_login calls login, login calls validate (depth 2 via login — direct)
        graph.add_edge(calls_edge("test_login", "login")).unwrap();
        graph.add_edge(calls_edge("login", "validate")).unwrap();
        // test_middleware calls middleware calls login calls validate (depth 3 — transitive)
        graph.add_edge(calls_edge("test_middleware", "middleware")).unwrap();
        graph.add_edge(calls_edge("middleware", "login")).unwrap();
    }

    let result = engine.test_impact(&["sym:validate"], 4).unwrap();
    assert!(!result.direct_tests.is_empty());
    assert!(result.direct_tests.iter().any(|t| t.test_symbol.contains("test_validate")));
    assert!(!result.transitive_tests.is_empty());
    assert!(result.transitive_tests.iter().any(|t| t.test_symbol.contains("test_middleware")));
}
```

Create helper functions `sym_node`, `test_node`, `calls_edge` matching existing test helpers in the file. Read the file first to reuse existing patterns.

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p codemem-engine test_impact -- --nocapture`
Expected: FAIL — method not found

- [ ] **Step 4: Implement `test_impact()` on `CodememEngine`**

In `crates/codemem-engine/src/graph_ops.rs`:

```rust
impl CodememEngine {
    /// Analyze which tests are affected by changes to the given symbols.
    /// BFS callers up to `max_depth`, split into direct (≤2) and transitive (3+).
    pub fn test_impact(
        &self,
        changed_symbol_ids: &[&str],
        max_depth: usize,
    ) -> Result<TestImpactResult, CodememError> {
        let graph = self.lock_graph()?;
        let mut direct_tests = Vec::new();
        let mut transitive_tests = Vec::new();
        let mut seen = HashSet::new();

        for &symbol_id in changed_symbol_ids {
            // BFS callers (reverse direction) up to max_depth
            // We need to manually BFS since we want depth tracking
            let mut queue: VecDeque<(String, usize, String)> = VecDeque::new();
            let mut visited: HashSet<String> = HashSet::new();
            visited.insert(symbol_id.to_string());

            // Seed with direct callers
            if let Ok(edges) = graph.get_edges(symbol_id) {
                for edge in &edges {
                    if edge.relationship != RelationshipType::Calls { continue; }
                    // We want callers OF this symbol, so dst == symbol_id
                    if edge.dst == symbol_id && !visited.contains(&edge.src) {
                        visited.insert(edge.src.clone());
                        queue.push_back((edge.src.clone(), 1, edge.src.clone()));
                    }
                }
            }

            while let Some((node_id, depth, via)) = queue.pop_front() {
                if depth > max_depth { continue; }

                // Check if this node is a test
                if let Ok(Some(node)) = graph.get_node(&node_id) {
                    let is_test = node.payload.get("is_test")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                        || node.payload.get("kind")
                            .and_then(|v| v.as_str())
                            .is_some_and(|k| k == "test")
                        || is_test_file(
                            node.payload.get("file_path")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                        );

                    if is_test && seen.insert(node_id.clone()) {
                        let file_path = node.payload.get("file_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let hit = TestHit {
                            test_symbol: node.label.clone(),
                            file_path,
                            via: via.clone(),
                            depth,
                        };
                        if depth <= 2 {
                            direct_tests.push(hit);
                        } else {
                            transitive_tests.push(hit);
                        }
                    }
                }

                // Continue BFS — find callers of this node
                if depth < max_depth {
                    if let Ok(edges) = graph.get_edges(&node_id) {
                        for edge in &edges {
                            if edge.relationship != RelationshipType::Calls { continue; }
                            if edge.dst == node_id && !visited.contains(&edge.src) {
                                visited.insert(edge.src.clone());
                                queue.push_back((edge.src.clone(), depth + 1, via.clone()));
                            }
                        }
                    }
                }
            }
        }

        Ok(TestImpactResult {
            direct_tests,
            transitive_tests,
            analyzed_symbols: changed_symbol_ids.iter().map(|s| s.to_string()).collect(),
        })
    }
}

fn is_test_file(path: &str) -> bool {
    let p = path.to_lowercase();
    p.contains("/tests/") || p.contains("/__tests__/") || p.contains("/test_")
        || p.contains("_test.") || p.contains(".test.") || p.contains(".spec.")
        || p.starts_with("test_") || p.starts_with("tests/")
}
```

- [ ] **Step 5: Run test**

Run: `cargo test -p codemem-engine test_impact -- --nocapture`
Expected: PASS

- [ ] **Step 6: Register MCP tool**

In `crates/codemem/src/mcp/definitions.rs`, add tool schema (follow existing patterns):

```rust
{
    "name": "test_impact",
    "description": "Find tests affected by changes to symbols. BFS callers up to depth 4, splits into direct (depth ≤ 2) and transitive (depth 3+).",
    "inputSchema": {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Symbol IDs or names to analyze"
            },
            "diff": {
                "type": "string",
                "description": "Unified diff to parse for changed symbols (alternative to symbols)"
            },
            "max_depth": {
                "type": "integer",
                "description": "Max BFS depth (default 4)"
            }
        }
    }
}
```

In `crates/codemem/src/mcp/tools_graph.rs`, add handler:

```rust
pub(crate) fn tool_test_impact(&self, args: &Value) -> ToolResult {
    let max_depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(4) as usize;

    // Resolve symbol IDs from explicit list or diff
    let symbol_ids: Vec<String> = if let Some(symbols) = args.get("symbols").and_then(|v| v.as_array()) {
        symbols.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
    } else if let Some(diff) = args.get("diff").and_then(|v| v.as_str()) {
        // Reuse diff parsing from review.rs
        let mapping = self.engine.map_diff_to_symbols(diff);
        mapping.changed_symbols
    } else {
        return ToolResult::tool_error("Provide 'symbols' or 'diff'");
    };

    let refs: Vec<&str> = symbol_ids.iter().map(|s| s.as_str()).collect();
    match self.engine.test_impact(&refs, max_depth) {
        Ok(result) => ToolResult::text(
            serde_json::to_string_pretty(&result).expect("JSON serialization"),
        ),
        Err(e) => ToolResult::tool_error(format!("Test impact failed: {e}")),
    }
}
```

In `crates/codemem/src/mcp/mod.rs`, add route:

```rust
"test_impact" => self.tool_test_impact(args),
```

- [ ] **Step 7: Run full tests + clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`

- [ ] **Step 8: Commit**

```bash
git commit -m "feat: add test_impact MCP tool for test-aware change analysis

BFS callers from changed symbols up to depth 4, filtering to test
files/symbols. Results split into direct (depth ≤ 2) and transitive
(depth 3+). Accepts symbol IDs or unified diff as input."
```

---

## Task 3: Cycle Detection MCP Tool

New `cycles` MCP tool using petgraph's Tarjan SCC to find circular dependencies. Groups ≥ 5 flagged as CRITICAL.

**Files:**
- Modify: `crates/codemem-engine/src/graph_ops.rs`
- Modify: `crates/codemem-engine/src/tests/graph_ops_tests.rs`
- Modify: `crates/codemem/src/mcp/definitions.rs`
- Modify: `crates/codemem/src/mcp/tools_graph.rs`
- Modify: `crates/codemem/src/mcp/mod.rs`

**Context:** `algorithms.rs:444-461` already has `strongly_connected_components()` returning `Vec<Vec<String>>`. The graph engine exposes this. The MCP tool just needs to format the output with severity levels.

- [ ] **Step 1: Define `CycleReport` in `graph_ops.rs`**

```rust
#[derive(Debug, Clone, serde::Serialize)]
pub struct CycleReport {
    pub cycles: Vec<CycleGroup>,
    pub total_cycles: usize,
    pub critical_count: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CycleGroup {
    pub nodes: Vec<String>,
    pub size: usize,
    pub severity: String,  // "critical" (≥5), "warning" (3-4), "info" (2)
}
```

- [ ] **Step 2: Write failing test**

```rust
#[test]
fn detect_cycles_finds_circular_dependencies() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        // A -> B -> C -> A (cycle of 3)
        graph.add_node(sym_node("a", "module", "src/a.rs")).unwrap();
        graph.add_node(sym_node("b", "module", "src/b.rs")).unwrap();
        graph.add_node(sym_node("c", "module", "src/c.rs")).unwrap();
        graph.add_edge(calls_edge("a", "b")).unwrap();
        graph.add_edge(calls_edge("b", "c")).unwrap();
        graph.add_edge(calls_edge("c", "a")).unwrap();
    }

    let report = engine.detect_cycles().unwrap();
    assert_eq!(report.total_cycles, 1);
    assert_eq!(report.cycles[0].size, 3);
    assert_eq!(report.cycles[0].severity, "warning");
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p codemem-engine detect_cycles -- --nocapture`

- [ ] **Step 4: Implement `detect_cycles()`**

```rust
impl CodememEngine {
    pub fn detect_cycles(&self) -> Result<CycleReport, CodememError> {
        let graph = self.lock_graph()?;
        let sccs = graph.strongly_connected_components();

        // Filter to actual cycles (size ≥ 2) and sort by size descending
        let mut cycles: Vec<CycleGroup> = sccs
            .into_iter()
            .filter(|scc| scc.len() >= 2)
            .map(|nodes| {
                let size = nodes.len();
                let severity = if size >= 5 {
                    "critical"
                } else if size >= 3 {
                    "warning"
                } else {
                    "info"
                }.to_string();
                CycleGroup { nodes, size, severity }
            })
            .collect();

        cycles.sort_by(|a, b| b.size.cmp(&a.size));
        let critical_count = cycles.iter().filter(|c| c.severity == "critical").count();
        let total_cycles = cycles.len();

        Ok(CycleReport { cycles, total_cycles, critical_count })
    }
}
```

- [ ] **Step 5: Run test**

Run: `cargo test -p codemem-engine detect_cycles -- --nocapture`
Expected: PASS

- [ ] **Step 6: Register MCP tool + handler**

Definition schema, handler calling `self.engine.detect_cycles()`, route `"cycles" => self.tool_cycles(args)`.

- [ ] **Step 7: Run full tests + clippy, commit**

```bash
git commit -m "feat: add cycles MCP tool for circular dependency detection

Uses Tarjan's SCC via petgraph. Groups sized ≥5 flagged as critical,
3-4 as warning, 2 as info. Sorted by size descending."
```

---

## Task 4: Weighted Community Detection

Modify Louvain to use relationship-aware weights: CALLS edges at 1.0, heritage edges (Extends, Implements, Inherits) at 0.5, other structural edges at their stored weight.

**Files:**
- Modify: `crates/codemem-storage/src/graph/algorithms.rs:209-342`
- Modify: `crates/codemem-storage/src/tests/graph_tests.rs`

**Context:** The Louvain already uses edge weights from the petgraph `DiGraph`. The weight is set when edges are added to the `GraphEngine`. The fix is to apply relationship-aware scaling when building the undirected adjacency in `louvain_communities()`. The edge's stored weight is in `self.edges[edge_id].weight`, but the petgraph DiGraph stores just the f64. We need to look up the relationship type.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn louvain_weighted_heritage_edges_lower_influence() {
    let mut graph = GraphEngine::new();
    // Group 1: A, B connected by CALLS (weight 1.0)
    graph.add_node(sym_node("a", "a.rs")).unwrap();
    graph.add_node(sym_node("b", "b.rs")).unwrap();
    graph.add_edge(Edge {
        id: "calls:a->b".into(),
        src: "sym:a".into(), dst: "sym:b".into(),
        relationship: RelationshipType::Calls, weight: 1.0,
        ..Default::default()
    }).unwrap();

    // Group 2: C, D connected by CALLS (weight 1.0)
    graph.add_node(sym_node("c", "c.rs")).unwrap();
    graph.add_node(sym_node("d", "d.rs")).unwrap();
    graph.add_edge(Edge {
        id: "calls:c->d".into(),
        src: "sym:c".into(), dst: "sym:d".into(),
        relationship: RelationshipType::Calls, weight: 1.0,
        ..Default::default()
    }).unwrap();

    // Cross-group: B extends C (heritage, should be 0.5 effective weight)
    graph.add_edge(Edge {
        id: "extends:b->c".into(),
        src: "sym:b".into(), dst: "sym:c".into(),
        relationship: RelationshipType::Extends, weight: 1.0,
        ..Default::default()
    }).unwrap();

    let communities = graph.louvain_communities(1.0);
    // With heritage at 0.5, the cross-group link is weaker than intra-group,
    // so we should get 2 communities
    assert!(communities.len() >= 2,
        "Heritage edges at 0.5 should not merge call-based communities; got {} communities",
        communities.len());
}
```

Adapt helpers to match existing patterns. Check if `Edge` has a `Default` impl.

- [ ] **Step 2: Run test**

Run: `cargo test -p codemem-storage louvain_weighted -- --nocapture`
Expected: FAIL (currently heritage treated same as calls)

- [ ] **Step 3: Implement relationship-aware weighting**

In `algorithms.rs`, modify `louvain_communities()` around line 225-234 where the undirected adjacency is built. Currently it reads `self.graph[edge_ref]` for the weight. Change to apply a multiplier based on relationship type:

```rust
for edge_ref in self.graph.edge_indices() {
    if let Some((src_idx, dst_idx)) = self.graph.edge_endpoints(edge_ref) {
        let base_weight = self.graph[edge_ref];
        // Look up the relationship type to apply community-detection weighting
        let w = if let Some(edge_id) = self.petgraph_to_edge_id.get(&edge_ref) {
            if let Some(edge) = self.edges.get(edge_id.as_str()) {
                let multiplier = match edge.relationship {
                    RelationshipType::Extends
                    | RelationshipType::Implements
                    | RelationshipType::Inherits => 0.5,
                    _ => 1.0,
                };
                base_weight * multiplier
            } else {
                base_weight
            }
        } else {
            base_weight
        };
        // ... rest of adjacency building
    }
}
```

**Important:** Check if `GraphEngine` has a mapping from petgraph `EdgeIndex` to the edge ID string. If not, you may need to build one, or iterate `self.edges` to find the matching edge by src/dst. Read the `GraphEngine` struct fields carefully.

- [ ] **Step 4: Run test**

Run: `cargo test -p codemem-storage louvain_weighted -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run full tests + clippy, commit**

```bash
git commit -m "feat: apply relationship-aware weights in Louvain community detection

Heritage edges (Extends, Implements, Inherits) weighted at 0.5x
during community detection, calls at 1.0x. Prevents inheritance
hierarchies from dominating community structure."
```

---

## Task 5: Community Auto-Labeling

Label communities by shared parent directory names. If all members share the same directory, use that name. Otherwise combine the two most frequent directories.

**Files:**
- Modify: `crates/codemem-storage/src/graph/algorithms.rs`
- Modify: `crates/codemem-storage/src/tests/graph_tests.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn label_community_by_directory() {
    let graph = GraphEngine::new();
    // All from same directory
    let nodes = vec![
        ("sym:auth.validate", "src/auth/validate.rs"),
        ("sym:auth.login", "src/auth/login.rs"),
        ("sym:auth.token", "src/auth/token.rs"),
    ];
    let label = graph.label_community(&nodes);
    assert_eq!(label, "auth");
}

#[test]
fn label_community_mixed_directories() {
    let graph = GraphEngine::new();
    let nodes = vec![
        ("sym:auth.validate", "src/auth/validate.rs"),
        ("sym:auth.login", "src/auth/login.rs"),
        ("sym:middleware.cors", "src/middleware/cors.rs"),
    ];
    let label = graph.label_community(&nodes);
    assert_eq!(label, "auth+middleware");
}
```

- [ ] **Step 2: Run test, verify it fails**

- [ ] **Step 3: Implement `label_community()`**

```rust
impl GraphEngine {
    /// Label a community by the most common parent directories of its members.
    pub fn label_community(&self, members: &[(&str, &str)]) -> String {
        let mut dir_counts: HashMap<&str, usize> = HashMap::new();
        for &(_, file_path) in members {
            // Extract parent directory name (last directory component)
            if let Some(dir) = file_path.rsplit('/').nth(1) {
                *dir_counts.entry(dir).or_insert(0) += 1;
            }
        }

        if dir_counts.is_empty() {
            return "unknown".to_string();
        }

        let mut sorted: Vec<_> = dir_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        if sorted.len() == 1 || sorted[0].1 > sorted.get(1).map(|s| s.1).unwrap_or(0) * 2 {
            sorted[0].0.to_string()
        } else {
            format!("{}+{}", sorted[0].0, sorted[1].0)
        }
    }
}
```

This takes `(node_id, file_path)` pairs. The caller (analysis pipeline or MCP tool) retrieves file paths from node payloads.

- [ ] **Step 4: Run test, commit**

```bash
git commit -m "feat: auto-label communities by parent directory names

Single-directory communities get that name. Mixed communities get
the two most frequent directories joined with +."
```

---

## Task 6: Confidence Tags in Tool Output

Add `(~)` for medium confidence (≥ 0.5) and `(?)` for low confidence (< 0.5) when displaying edges in MCP tool output.

**Files:**
- Modify: `crates/codemem/src/mcp/tools_graph.rs`

**Context:** Tool output currently shows `"weight": e.weight`. The change adds a `confidence_tag` field to edge objects in tool responses.

- [ ] **Step 1: Add helper function**

In `tools_graph.rs`:

```rust
/// Format a confidence tag for edge display.
fn confidence_tag(weight: f64) -> &'static str {
    if weight >= 0.9 {
        ""      // high confidence — no tag
    } else if weight >= 0.5 {
        " (~)"  // medium confidence
    } else {
        " (?)"  // low confidence
    }
}
```

- [ ] **Step 2: Apply to edge serialization in tool output**

Find all places where edges are serialized in tool responses (search for `"weight": e.weight` in tools_graph.rs). Add `"confidence": confidence_tag(e.weight)` to each JSON object. Key locations:

- `get_symbol_info` (~line 351)
- `get_symbol_graph` (~line 399)
- `get_cross_repo` (~line 583)
- `graph_traverse` output

For example:

```rust
json!({
    "src": e.src,
    "dst": e.dst,
    "relationship": e.relationship.to_string(),
    "weight": e.weight,
    "confidence": confidence_tag(e.weight),
})
```

- [ ] **Step 3: Write test**

Add a test that constructs edges with different weights and verifies the tag:

```rust
#[test]
fn confidence_tag_thresholds() {
    assert_eq!(confidence_tag(1.0), "");
    assert_eq!(confidence_tag(0.9), "");
    assert_eq!(confidence_tag(0.7), " (~)");
    assert_eq!(confidence_tag(0.5), " (~)");
    assert_eq!(confidence_tag(0.3), " (?)");
    assert_eq!(confidence_tag(0.0), " (?)");
}
```

- [ ] **Step 4: Run tests + clippy, commit**

```bash
git commit -m "feat: add confidence tags to edge output in MCP tools

Edges displayed with (~) for medium confidence (0.5-0.9) and (?)
for low confidence (<0.5). High confidence (≥0.9) has no tag."
```

---

## Task 7: Integration Verification

- [ ] **Step 1: Run full test suite**

Run: `cargo test --workspace`

- [ ] **Step 2: Run clippy with CI flags**

Run: `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets -- -D warnings`

- [ ] **Step 3: Run cargo fmt check**

Run: `cargo fmt --all -- --check`
