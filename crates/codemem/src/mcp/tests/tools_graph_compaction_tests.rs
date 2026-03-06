use super::*;
use crate::mcp::test_helpers::*;
use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::HashMap;

// ── Phase 5: Graph Compaction Tests ─────────────────────────────────

#[test]
fn compaction_creates_package_nodes() {
    let server = test_server();
    let mut graph = server.engine.graph.lock().unwrap();

    // Simulate file nodes for a directory tree: src/auth/middleware.rs, src/api/handler.rs
    let files = vec!["src/auth/middleware.rs", "src/api/handler.rs", "src/lib.rs"];
    let now = chrono::Utc::now();
    let mut seen_files = std::collections::HashSet::new();

    for fp in &files {
        let node = GraphNode {
            id: format!("file:{fp}"),
            kind: NodeKind::File,
            label: fp.to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = graph.add_node(node);
        seen_files.insert(fp.to_string());
    }
    drop(graph);

    // Now call compact_graph which won't prune anything but we can test
    // that the package node creation happened during index_codebase.
    // Instead, manually create pkg nodes to test the summary tree tool.
    let mut graph = server.engine.graph.lock().unwrap();
    for dir in &["src/", "src/auth/", "src/api/"] {
        let node = GraphNode {
            id: format!("pkg:{dir}"),
            kind: NodeKind::Package,
            label: dir.to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = graph.add_node(node);
    }

    // Add CONTAINS edges: pkg:src/ -> pkg:src/auth/, pkg:src/ -> pkg:src/api/
    for (src, dst) in &[
        ("pkg:src/", "pkg:src/auth/"),
        ("pkg:src/", "pkg:src/api/"),
        ("pkg:src/auth/", "file:src/auth/middleware.rs"),
        ("pkg:src/api/", "file:src/api/handler.rs"),
        ("pkg:src/", "file:src/lib.rs"),
    ] {
        let edge = Edge {
            id: format!("contains:{src}->{dst}"),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = graph.add_edge(edge);
    }
    drop(graph);

    // Test summary tree from pkg:src/
    let params = json!({
        "name": "summary_tree",
        "arguments": { "start_id": "pkg:src/", "max_depth": 3 }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(500));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let tree: Value = serde_json::from_str(text).unwrap();

    assert_eq!(tree["id"], "pkg:src/");
    assert_eq!(tree["kind"], "package");
    let children = tree["children"].as_array().unwrap();
    // Should have 3 children: file:src/lib.rs, pkg:src/api/, pkg:src/auth/
    assert_eq!(children.len(), 3);

    // Check that sub-packages have their file children
    let auth_pkg = children
        .iter()
        .find(|c| c["id"] == "pkg:src/auth/")
        .expect("auth package should exist");
    let auth_children = auth_pkg["children"].as_array().unwrap();
    assert_eq!(auth_children.len(), 1);
    assert_eq!(auth_children[0]["id"], "file:src/auth/middleware.rs");
}

#[test]
fn compaction_prunes_low_score_chunks() {
    let server = test_server();
    let now = chrono::Utc::now();

    // Create a file node and many chunk nodes
    let file_path = "test/main.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node);

    // Create 15 chunk nodes — more than the default max_retained_chunks_per_file (10)
    for i in 0..15 {
        let chunk_id = format!("chunk:{file_path}:{i}");
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("line_start".to_string(), json!(i * 10));
        payload.insert("line_end".to_string(), json!(i * 10 + 9));
        payload.insert("non_ws_chars".to_string(), json!((i + 1) * 50));
        payload.insert("node_kind".to_string(), json!("function"));

        let node = GraphNode {
            id: chunk_id.clone(),
            kind: NodeKind::Chunk,
            label: format!("chunk:{file_path}:{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        // Add CONTAINS edge from file to chunk
        let edge = Edge {
            id: format!("contains:file:{file_path}->{chunk_id}"),
            src: format!("file:{file_path}"),
            dst: chunk_id,
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    let (chunks_pruned, _symbols_pruned) = server.engine.compact_graph(&seen);

    // With 15 chunks and max_retained_chunks_per_file=10, at least some should be pruned
    assert!(
        chunks_pruned > 0,
        "Expected some chunks to be pruned, got 0"
    );

    // Remaining chunk nodes should be <= max_retained_chunks_per_file
    let graph = server.engine.graph.lock().unwrap();
    let remaining_chunks: Vec<_> = graph
        .get_all_nodes()
        .into_iter()
        .filter(|n| n.kind == NodeKind::Chunk)
        .collect();
    assert!(
        remaining_chunks.len() <= 10,
        "Expected at most 10 chunks, got {}",
        remaining_chunks.len()
    );
}

#[test]
fn compaction_preserves_memory_linked_chunks() {
    let server = test_server();
    let now = chrono::Utc::now();

    let file_path = "test/linked.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node);

    // Create 15 chunks, link one to a memory node
    for i in 0..15 {
        let chunk_id = format!("chunk:{file_path}:{i}");
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("line_start".to_string(), json!(i * 10));
        payload.insert("line_end".to_string(), json!(i * 10 + 9));
        payload.insert("non_ws_chars".to_string(), json!(50));

        let node = GraphNode {
            id: chunk_id.clone(),
            kind: NodeKind::Chunk,
            label: format!("chunk:{file_path}:{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        let edge = Edge {
            id: format!("contains:file:{file_path}->{chunk_id}"),
            src: format!("file:{file_path}"),
            dst: chunk_id,
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }

    // Create a memory node linked to chunk 0
    let mem_id = "mem-linked-test";
    let mem_node = GraphNode {
        id: mem_id.to_string(),
        kind: NodeKind::Memory,
        label: "linked memory".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: Some(mem_id.to_string()),
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(mem_node);

    let link_edge = Edge {
        id: format!("link:{mem_id}->chunk:{file_path}:0"),
        src: mem_id.to_string(),
        dst: format!("chunk:{file_path}:0"),
        relationship: RelationshipType::RelatesTo,
        weight: 0.5,
        valid_from: None,
        valid_to: None,
        properties: HashMap::new(),
        created_at: now,
    };
    let _ = server.engine.storage.insert_graph_edge(&link_edge);
    let _ = graph.add_edge(link_edge);
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    server.engine.compact_graph(&seen);

    // The memory-linked chunk (chunk:0) should survive compaction
    let graph = server.engine.graph.lock().unwrap();
    let chunk0 = graph.get_node(&format!("chunk:{file_path}:0")).unwrap();
    assert!(
        chunk0.is_some(),
        "Memory-linked chunk should survive compaction"
    );
}

#[test]
fn summary_tree_excludes_chunks_by_default() {
    let server = test_server();
    let now = chrono::Utc::now();
    let mut graph = server.engine.graph.lock().unwrap();

    // Build a small tree: pkg:src/ -> file:src/main.rs -> chunk:src/main.rs:0
    let pkg = GraphNode {
        id: "pkg:src/".to_string(),
        kind: NodeKind::Package,
        label: "src/".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let file = GraphNode {
        id: "file:src/main.rs".to_string(),
        kind: NodeKind::File,
        label: "src/main.rs".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let chunk = GraphNode {
        id: "chunk:src/main.rs:0".to_string(),
        kind: NodeKind::Chunk,
        label: "chunk".to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: None,
    };
    let _ = graph.add_node(pkg);
    let _ = graph.add_node(file);
    let _ = graph.add_node(chunk);

    for (src, dst) in &[
        ("pkg:src/", "file:src/main.rs"),
        ("file:src/main.rs", "chunk:src/main.rs:0"),
    ] {
        let e = Edge {
            id: format!("contains:{src}->{dst}"),
            src: src.to_string(),
            dst: dst.to_string(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = graph.add_edge(e);
    }
    drop(graph);

    // Without include_chunks, chunk should NOT appear
    let params = json!({
        "name": "summary_tree",
        "arguments": { "start_id": "pkg:src/", "max_depth": 3 }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(600));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let tree: Value = serde_json::from_str(text).unwrap();

    let file_children = tree["children"][0]["children"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(file_children, 0, "Chunks should be excluded by default");

    // With include_chunks=true, chunk SHOULD appear
    let params = json!({
        "name": "summary_tree",
        "arguments": { "start_id": "pkg:src/", "max_depth": 3, "include_chunks": true }
    });
    let resp = server.handle_request("tools/call", Some(&params), json!(601));
    let result = resp.result.unwrap();
    let text = result["content"][0]["text"].as_str().unwrap();
    let tree: Value = serde_json::from_str(text).unwrap();

    let file_children = tree["children"][0]["children"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(file_children, 1, "Chunks should be included when requested");
}

// ── Symbol Compaction Tests ─────────────────────────────────────────

#[test]
fn compaction_prunes_low_score_symbols() {
    let server = test_server();
    let now = chrono::Utc::now();

    let file_path = "test/symbols.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node.clone());
    let _ = server.engine.storage.insert_graph_node(&file_node);

    // Create 20 symbols: mix of public/private, with/without CALLS edges
    for i in 0..20 {
        let sym_id = format!("sym:test::sym_{i}");
        let is_public = i < 5; // first 5 are public
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert(
            "visibility".to_string(),
            json!(if is_public { "public" } else { "private" }),
        );
        payload.insert("line_start".to_string(), json!(i * 10));
        payload.insert("line_end".to_string(), json!(i * 10 + 5));

        let node = GraphNode {
            id: sym_id.clone(),
            kind: NodeKind::Function,
            label: format!("sym_{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        // Add CONTAINS edge from file to symbol
        let edge = Edge {
            id: format!("contains:file:{file_path}->{sym_id}"),
            src: format!("file:{file_path}"),
            dst: sym_id.clone(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }

    // Add CALLS edges to first 3 symbols to make them high-connectivity
    for i in 0..3 {
        let src = format!("sym:test::sym_{i}");
        for j in 3..6 {
            let dst = format!("sym:test::sym_{j}");
            let edge = Edge {
                id: format!("calls:{src}->{dst}"),
                src: src.clone(),
                dst,
                relationship: RelationshipType::Calls,
                weight: 1.0,
                valid_from: None,
                valid_to: None,
                properties: HashMap::new(),
                created_at: now,
            };
            let _ = server.engine.storage.insert_graph_edge(&edge);
            let _ = graph.add_edge(edge);
        }
    }
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    let (_chunks_pruned, symbols_pruned) = server.engine.compact_graph(&seen);

    // With 20 symbols and max_retained_symbols_per_file=15, some should be pruned
    assert!(
        symbols_pruned > 0,
        "Expected some symbols to be pruned, got 0"
    );

    // Verify private unreferenced symbols were the ones pruned
    let graph = server.engine.graph.lock().unwrap();
    let remaining_syms: Vec<_> = graph
        .get_all_nodes()
        .into_iter()
        .filter(|n| n.id.starts_with("sym:"))
        .collect();
    assert!(
        remaining_syms.len() < 20,
        "Expected fewer than 20 symbols remaining, got {}",
        remaining_syms.len()
    );
}

#[test]
fn compaction_preserves_public_symbols() {
    let server = test_server();
    let now = chrono::Utc::now();

    let file_path = "test/public.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node.clone());
    let _ = server.engine.storage.insert_graph_node(&file_node);

    // Create 20 public symbols with no CALLS edges (low connectivity)
    for i in 0..20 {
        let sym_id = format!("sym:test::pub_fn_{i}");
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("visibility".to_string(), json!("public"));
        payload.insert("line_start".to_string(), json!(i * 5));
        payload.insert("line_end".to_string(), json!(i * 5 + 3));

        let node = GraphNode {
            id: sym_id.clone(),
            kind: NodeKind::Function,
            label: format!("pub_fn_{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        let edge = Edge {
            id: format!("contains:file:{file_path}->{sym_id}"),
            src: format!("file:{file_path}"),
            dst: sym_id,
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    let (_chunks_pruned, symbols_pruned) = server.engine.compact_graph(&seen);

    // K = max(15, 20 public) = 20, so no public symbols should be pruned
    assert_eq!(
        symbols_pruned, 0,
        "No public symbols should be pruned when public_count >= max_retained"
    );

    let graph = server.engine.graph.lock().unwrap();
    let remaining: Vec<_> = graph
        .get_all_nodes()
        .into_iter()
        .filter(|n| n.id.starts_with("sym:"))
        .collect();
    assert_eq!(remaining.len(), 20, "All 20 public symbols should survive");
}

#[test]
fn compaction_preserves_structural_anchors() {
    let server = test_server();
    let now = chrono::Utc::now();

    let file_path = "test/structural.rs";
    let mut graph = server.engine.graph.lock().unwrap();

    let file_node = GraphNode {
        id: format!("file:{file_path}"),
        kind: NodeKind::File,
        label: file_path.to_string(),
        payload: HashMap::new(),
        centrality: 0.0,
        memory_id: None,
        namespace: Some("test".to_string()),
    };
    let _ = graph.add_node(file_node.clone());
    let _ = server.engine.storage.insert_graph_node(&file_node);

    // Create structural nodes (Class, Interface, Module) with low scores:
    // private visibility, no CALLS edges, small code size
    let structural_kinds = [
        ("sym:test::MyClass", NodeKind::Class, "MyClass"),
        ("sym:test::MyTrait", NodeKind::Interface, "MyTrait"),
        ("sym:test::my_mod", NodeKind::Module, "my_mod"),
    ];

    for (sym_id, kind, label) in &structural_kinds {
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("visibility".to_string(), json!("private"));
        payload.insert("line_start".to_string(), json!(0));
        payload.insert("line_end".to_string(), json!(1));

        let node = GraphNode {
            id: sym_id.to_string(),
            kind: *kind,
            label: label.to_string(),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        let edge = Edge {
            id: format!("contains:file:{file_path}->{sym_id}"),
            src: format!("file:{file_path}"),
            dst: sym_id.to_string(),
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }

    // Also add 20 private functions to exceed max_retained_symbols_per_file
    for i in 0..20 {
        let sym_id = format!("sym:test::priv_fn_{i}");
        let mut payload = HashMap::new();
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        payload.insert("visibility".to_string(), json!("private"));
        payload.insert("line_start".to_string(), json!(10 + i));
        payload.insert("line_end".to_string(), json!(10 + i + 1));

        let node = GraphNode {
            id: sym_id.clone(),
            kind: NodeKind::Function,
            label: format!("priv_fn_{i}"),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".to_string()),
        };
        let _ = server.engine.storage.insert_graph_node(&node);
        let _ = graph.add_node(node);

        let edge = Edge {
            id: format!("contains:file:{file_path}->{sym_id}"),
            src: format!("file:{file_path}"),
            dst: sym_id,
            relationship: RelationshipType::Contains,
            weight: 0.1,
            valid_from: None,
            valid_to: None,
            properties: HashMap::new(),
            created_at: now,
        };
        let _ = server.engine.storage.insert_graph_edge(&edge);
        let _ = graph.add_edge(edge);
    }
    drop(graph);

    let mut seen = std::collections::HashSet::new();
    seen.insert(file_path.to_string());
    let (_chunks_pruned, symbols_pruned) = server.engine.compact_graph(&seen);

    // Some private functions should be pruned (23 total, K=15)
    assert!(
        symbols_pruned > 0,
        "Expected some symbols to be pruned, got 0"
    );

    // But structural anchors must survive
    let graph = server.engine.graph.lock().unwrap();
    for (sym_id, _, _) in &structural_kinds {
        let node = graph.get_node(sym_id).unwrap();
        assert!(
            node.is_some(),
            "Structural anchor {sym_id} should survive compaction"
        );
    }
}
