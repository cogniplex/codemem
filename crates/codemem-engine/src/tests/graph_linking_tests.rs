use crate::CodememEngine;
use codemem_core::{
    Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind, RelationshipType,
};
use std::collections::HashMap;

fn make_memory(id: &str, content: &str) -> MemoryNode {
    make_memory_with_opts(id, content, MemoryType::Context, None, &[], 0.7, 0.9)
}

fn make_memory_with_opts(
    id: &str,
    content: &str,
    memory_type: MemoryType,
    namespace: Option<&str>,
    tags: &[&str],
    importance: f64,
    confidence: f64,
) -> MemoryNode {
    let mut m = MemoryNode::test_default(content);
    m.id = id.to_string();
    m.memory_type = memory_type;
    m.importance = importance;
    m.confidence = confidence;
    m.tags = tags.iter().map(|s| s.to_string()).collect();
    m.namespace = namespace.map(String::from);
    m
}

/// Helper: add a graph node to both storage and in-memory graph.
fn add_graph_node(engine: &CodememEngine, node: GraphNode) {
    engine.storage().insert_graph_node(&node).unwrap();
    let mut graph = engine.lock_graph().unwrap();
    graph.add_node(node).unwrap();
}

// ── auto_link_to_code_nodes ─────────────────────────────────────────

#[test]
fn auto_link_creates_edge_to_existing_file_node() {
    let engine = CodememEngine::for_testing();

    // Create a file node in both storage and graph
    add_graph_node(
        &engine,
        GraphNode {
            id: "file:src/main.rs".to_string(),
            kind: NodeKind::File,
            label: "src/main.rs".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        },
    );

    // Persist a memory (creates its graph node in both storage and graph)
    let mem = make_memory("mem-file-1", "I looked at src/main.rs and found something");
    engine.persist_memory(&mem).unwrap();

    // Auto-link: content references "src/main.rs" which exists as "file:src/main.rs"
    let created = engine.auto_link_to_code_nodes("mem-file-1", &mem.content, &[]);
    assert!(
        created >= 1,
        "should create at least one edge to file:src/main.rs, got {created}"
    );

    // Verify the edge exists in the graph
    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges("mem-file-1").unwrap_or_default();
    let has_file_edge = edges
        .iter()
        .any(|e| e.dst == "file:src/main.rs" && e.relationship == RelationshipType::RelatesTo);
    assert!(has_file_edge, "should have RELATES_TO edge to file node");
}

#[test]
fn auto_link_creates_edge_to_existing_symbol_node() {
    let engine = CodememEngine::for_testing();

    add_graph_node(
        &engine,
        GraphNode {
            id: "sym:std::collections::HashMap".to_string(),
            kind: NodeKind::Class,
            label: "HashMap".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        },
    );

    let mem = make_memory("mem-sym-1", "We use std::collections::HashMap for caching");
    engine.persist_memory(&mem).unwrap();

    let created = engine.auto_link_to_code_nodes("mem-sym-1", &mem.content, &[]);
    assert!(
        created >= 1,
        "should create at least one edge to sym:std::collections::HashMap, got {created}"
    );

    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges("mem-sym-1").unwrap_or_default();
    let has_sym_edge = edges.iter().any(|e| {
        e.dst == "sym:std::collections::HashMap" && e.relationship == RelationshipType::RelatesTo
    });
    assert!(has_sym_edge, "should have RELATES_TO edge to symbol node");
}

#[test]
fn auto_link_no_edges_for_nonexistent_nodes() {
    let engine = CodememEngine::for_testing();

    let mem = make_memory(
        "mem-noexist",
        "Reference to nonexistent/path.rs and fake::Module",
    );
    engine.persist_memory(&mem).unwrap();

    // Neither file:nonexistent/path.rs nor sym:fake::Module exist in the graph
    let created = engine.auto_link_to_code_nodes("mem-noexist", &mem.content, &[]);
    assert_eq!(created, 0, "should create no edges for non-existent nodes");
}

#[test]
fn auto_link_does_not_duplicate_existing_links() {
    let engine = CodememEngine::for_testing();

    add_graph_node(
        &engine,
        GraphNode {
            id: "file:src/lib.rs".to_string(),
            kind: NodeKind::File,
            label: "src/lib.rs".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        },
    );

    let mem = make_memory("mem-dup", "Check src/lib.rs for details");
    engine.persist_memory(&mem).unwrap();

    // Pass "file:src/lib.rs" as an existing link — should be skipped
    let existing_links = vec!["file:src/lib.rs".to_string()];
    let created = engine.auto_link_to_code_nodes("mem-dup", &mem.content, &existing_links);
    assert_eq!(
        created, 0,
        "should not create edges for already-linked nodes"
    );
}

#[test]
fn auto_link_return_count_matches_edges_created() {
    let engine = CodememEngine::for_testing();

    // Create two file nodes
    for path in &["src/a.rs", "src/b.rs"] {
        add_graph_node(
            &engine,
            GraphNode {
                id: format!("file:{path}"),
                kind: NodeKind::File,
                label: path.to_string(),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: None,
                namespace: None,
            },
        );
    }

    let mem = make_memory("mem-count", "Files src/a.rs and src/b.rs are related");
    engine.persist_memory(&mem).unwrap();

    let created = engine.auto_link_to_code_nodes("mem-count", &mem.content, &[]);

    // Verify returned count matches actual edges
    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges("mem-count").unwrap_or_default();
    let auto_linked_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.relationship == RelationshipType::RelatesTo
                && (e.dst.starts_with("file:src/a") || e.dst.starts_with("file:src/b"))
        })
        .collect();
    assert_eq!(
        created,
        auto_linked_edges.len(),
        "return count should match edges actually created"
    );
}

// ── auto_link_by_tags ───────────────────────────────────────────────

#[test]
fn auto_link_by_tags_session_tag_creates_preceded_by() {
    let engine = CodememEngine::for_testing();

    // Create first memory with session tag
    let m1 = make_memory_with_opts(
        "tag-s1",
        "first session memory",
        MemoryType::Context,
        None,
        &["session:abc"],
        0.7,
        0.9,
    );
    engine.persist_memory(&m1).unwrap();

    // Create second memory with same session tag
    let m2 = make_memory_with_opts(
        "tag-s2",
        "second session memory",
        MemoryType::Context,
        None,
        &["session:abc"],
        0.7,
        0.9,
    );
    engine.persist_memory(&m2).unwrap();

    // persist_memory calls auto_link_by_tags internally.
    // Verify PRECEDED_BY edge exists between them.
    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges("tag-s2").unwrap_or_default();
    let has_preceded = edges
        .iter()
        .any(|e| e.relationship == RelationshipType::PrecededBy);
    assert!(
        has_preceded,
        "session tag should create PRECEDED_BY edge; edges: {:?}",
        edges
            .iter()
            .map(|e| (&e.src, &e.dst, &e.relationship))
            .collect::<Vec<_>>()
    );

    // Verify weight is 0.8
    let preceded_edge = edges
        .iter()
        .find(|e| e.relationship == RelationshipType::PrecededBy)
        .unwrap();
    assert!(
        (preceded_edge.weight - 0.8).abs() < 0.01,
        "PRECEDED_BY weight should be 0.8, got {}",
        preceded_edge.weight
    );
}

#[test]
fn auto_link_by_tags_non_session_tag_creates_shares_theme() {
    let engine = CodememEngine::for_testing();

    let m1 = make_memory_with_opts(
        "tag-t1",
        "first thematic memory",
        MemoryType::Context,
        None,
        &["architecture"],
        0.7,
        0.9,
    );
    engine.persist_memory(&m1).unwrap();

    let m2 = make_memory_with_opts(
        "tag-t2",
        "second thematic memory",
        MemoryType::Context,
        None,
        &["architecture"],
        0.7,
        0.9,
    );
    engine.persist_memory(&m2).unwrap();

    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges("tag-t2").unwrap_or_default();
    let has_shares_theme = edges
        .iter()
        .any(|e| e.relationship == RelationshipType::SharesTheme);
    assert!(
        has_shares_theme,
        "non-session tag should create SHARES_THEME edge; edges: {:?}",
        edges
            .iter()
            .map(|e| (&e.src, &e.dst, &e.relationship))
            .collect::<Vec<_>>()
    );

    let theme_edge = edges
        .iter()
        .find(|e| e.relationship == RelationshipType::SharesTheme)
        .unwrap();
    assert!(
        (theme_edge.weight - 0.5).abs() < 0.01,
        "SHARES_THEME weight should be 0.5, got {}",
        theme_edge.weight
    );
}

#[test]
fn auto_link_by_tags_empty_tags_creates_no_edges() {
    let engine = CodememEngine::for_testing();

    let m1 = make_memory("tag-empty1", "first memory no tags");
    engine.persist_memory(&m1).unwrap();

    let m2 = make_memory("tag-empty2", "second memory no tags");
    engine.persist_memory(&m2).unwrap();

    // With no tags, auto_link_by_tags returns early. Only edges created
    // are the graph node additions from persist_memory, not inter-memory edges.
    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges("tag-empty2").unwrap_or_default();
    let inter_memory_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.relationship == RelationshipType::PrecededBy
                || e.relationship == RelationshipType::SharesTheme
        })
        .collect();
    assert!(
        inter_memory_edges.is_empty(),
        "empty tags should not create inter-memory edges"
    );
}

#[test]
fn auto_link_by_tags_duplicate_tags_linked_once() {
    let engine = CodememEngine::for_testing();

    // m1 has a tag
    let m1 = make_memory_with_opts(
        "tag-dup1",
        "first memory with tags",
        MemoryType::Context,
        None,
        &["topic-x"],
        0.7,
        0.9,
    );
    engine.persist_memory(&m1).unwrap();

    // m2 has the same tag twice — the HashSet in auto_link_by_tags should dedup
    let mut m2 = MemoryNode::test_default("second memory with dup tags");
    m2.id = "tag-dup2".to_string();
    m2.importance = 0.7;
    m2.confidence = 0.9;
    m2.tags = vec!["topic-x".to_string(), "topic-x".to_string()];
    engine.persist_memory(&m2).unwrap();

    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges("tag-dup2").unwrap_or_default();
    let theme_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.relationship == RelationshipType::SharesTheme
                && (e.src == "tag-dup1" || e.dst == "tag-dup1")
        })
        .collect();
    assert_eq!(
        theme_edges.len(),
        1,
        "duplicate tags should produce only one edge to the same sibling, got {}",
        theme_edges.len()
    );
}

#[test]
fn auto_link_by_tags_edges_created_correctly() {
    let engine = CodememEngine::for_testing();

    // Create memories with both session and non-session tags
    let m1 = make_memory_with_opts(
        "tag-mix1",
        "first mixed tag memory",
        MemoryType::Context,
        None,
        &["session:xyz", "design"],
        0.7,
        0.9,
    );
    engine.persist_memory(&m1).unwrap();

    let m2 = make_memory_with_opts(
        "tag-mix2",
        "second mixed tag memory",
        MemoryType::Context,
        None,
        &["session:xyz", "design"],
        0.7,
        0.9,
    );
    engine.persist_memory(&m2).unwrap();

    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges("tag-mix2").unwrap_or_default();

    // The sibling "tag-mix1" should be linked. Due to HashSet dedup in auto_link_by_tags,
    // only one edge is created per sibling regardless of how many shared tags exist.
    // The first matching tag determines the edge type.
    let linking_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            (e.src == "tag-mix1" || e.dst == "tag-mix1")
                && (e.relationship == RelationshipType::PrecededBy
                    || e.relationship == RelationshipType::SharesTheme)
        })
        .collect();
    assert!(
        !linking_edges.is_empty(),
        "should create at least one edge between siblings with shared tags"
    );
}

// ── get_node_memories ───────────────────────────────────────────────

#[test]
fn get_node_memories_finds_at_depth_1() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();

    // Create a code node in both storage and graph
    add_graph_node(
        &engine,
        GraphNode {
            id: "sym:MyFunc".to_string(),
            kind: NodeKind::Function,
            label: "MyFunc".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        },
    );

    // Create a memory linked to the code node
    let mem = make_memory("gnm-1", "memory about MyFunc behavior");
    engine.persist_memory(&mem).unwrap();

    // Add edge: sym:MyFunc -> gnm-1 (via engine.add_edge to satisfy FK)
    engine
        .add_edge(Edge {
            id: "sym:MyFunc-RELATES_TO-gnm-1".to_string(),
            src: "sym:MyFunc".to_string(),
            dst: "gnm-1".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .unwrap();

    let results = engine.get_node_memories("sym:MyFunc", 1, None).unwrap();
    assert!(!results.is_empty(), "should find memory at depth 1");
    assert_eq!(results[0].memory.id, "gnm-1");
    assert_eq!(results[0].depth, 1);
}

#[test]
fn get_node_memories_finds_at_depth_2() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();

    // Create nodes: sym:A -> sym:B -> mem
    for (id, kind) in &[("sym:A", NodeKind::Function), ("sym:B", NodeKind::Function)] {
        add_graph_node(
            &engine,
            GraphNode {
                id: id.to_string(),
                kind: *kind,
                label: id.to_string(),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: None,
                namespace: None,
            },
        );
    }

    let mem = make_memory("gnm-depth2", "memory at depth 2");
    engine.persist_memory(&mem).unwrap();

    // sym:A -> sym:B
    engine
        .add_edge(Edge {
            id: "A-calls-B".to_string(),
            src: "sym:A".to_string(),
            dst: "sym:B".to_string(),
            relationship: RelationshipType::Calls,
            weight: 0.8,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .unwrap();

    // sym:B -> gnm-depth2
    engine
        .add_edge(Edge {
            id: "B-relates-mem".to_string(),
            src: "sym:B".to_string(),
            dst: "gnm-depth2".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .unwrap();

    let results = engine.get_node_memories("sym:A", 2, None).unwrap();
    let found = results.iter().any(|r| r.memory.id == "gnm-depth2");
    assert!(found, "should find memory at depth 2");
}

#[test]
fn get_node_memories_depth_limit_respected() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();

    // sym:Start -> sym:Mid -> mem-deep
    for (id, kind) in &[
        ("sym:Start", NodeKind::Function),
        ("sym:Mid", NodeKind::Function),
    ] {
        add_graph_node(
            &engine,
            GraphNode {
                id: id.to_string(),
                kind: *kind,
                label: id.to_string(),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: None,
                namespace: None,
            },
        );
    }

    let mem = make_memory("gnm-deep", "deep memory");
    engine.persist_memory(&mem).unwrap();

    engine
        .add_edge(Edge {
            id: "start-mid".to_string(),
            src: "sym:Start".to_string(),
            dst: "sym:Mid".to_string(),
            relationship: RelationshipType::Calls,
            weight: 0.8,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .unwrap();

    engine
        .add_edge(Edge {
            id: "mid-deep".to_string(),
            src: "sym:Mid".to_string(),
            dst: "gnm-deep".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .unwrap();

    // Depth 1 should NOT find the memory (it's 2 hops away)
    let results = engine.get_node_memories("sym:Start", 1, None).unwrap();
    let found_deep = results.iter().any(|r| r.memory.id == "gnm-deep");
    assert!(!found_deep, "depth=1 should not find memory at depth 2");
}

#[test]
fn get_node_memories_skips_chunk_nodes() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();

    // sym:Root -> chunk:1 -> mem-via-chunk
    add_graph_node(
        &engine,
        GraphNode {
            id: "sym:Root".to_string(),
            kind: NodeKind::Function,
            label: "Root".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        },
    );

    add_graph_node(
        &engine,
        GraphNode {
            id: "chunk:1".to_string(),
            kind: NodeKind::Chunk,
            label: "chunk 1".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        },
    );

    let mem = make_memory("gnm-chunk", "memory behind a chunk node");
    engine.persist_memory(&mem).unwrap();

    engine
        .add_edge(Edge {
            id: "root-chunk".to_string(),
            src: "sym:Root".to_string(),
            dst: "chunk:1".to_string(),
            relationship: RelationshipType::Contains,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .unwrap();

    engine
        .add_edge(Edge {
            id: "chunk-mem".to_string(),
            src: "chunk:1".to_string(),
            dst: "gnm-chunk".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .unwrap();

    // BFS should skip chunk nodes, so the memory should not be found even at depth 2
    let results = engine.get_node_memories("sym:Root", 2, None).unwrap();
    let found_via_chunk = results.iter().any(|r| r.memory.id == "gnm-chunk");
    assert!(
        !found_via_chunk,
        "BFS should skip Chunk nodes so memory behind chunk is not reachable"
    );
}

#[test]
fn get_node_memories_relationship_filter() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();

    add_graph_node(
        &engine,
        GraphNode {
            id: "sym:Filtered".to_string(),
            kind: NodeKind::Function,
            label: "Filtered".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        },
    );

    let mem_relates = make_memory("gnm-rel", "related memory");
    engine.persist_memory(&mem_relates).unwrap();
    let mem_calls = make_memory("gnm-calls", "called memory");
    engine.persist_memory(&mem_calls).unwrap();

    engine
        .add_edge(Edge {
            id: "filter-relates".to_string(),
            src: "sym:Filtered".to_string(),
            dst: "gnm-rel".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .unwrap();

    engine
        .add_edge(Edge {
            id: "filter-calls".to_string(),
            src: "sym:Filtered".to_string(),
            dst: "gnm-calls".to_string(),
            relationship: RelationshipType::Calls,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        })
        .unwrap();

    // Filter to only RelatesTo — should find gnm-rel but not gnm-calls
    let results = engine
        .get_node_memories("sym:Filtered", 1, Some(&[RelationshipType::RelatesTo]))
        .unwrap();
    let found_relates = results.iter().any(|r| r.memory.id == "gnm-rel");
    let found_calls = results.iter().any(|r| r.memory.id == "gnm-calls");
    assert!(found_relates, "should find memory via RelatesTo edge");
    assert!(
        !found_calls,
        "should not find memory via Calls edge when filtered"
    );
}

#[test]
fn get_node_memories_empty_graph_returns_empty() {
    let engine = CodememEngine::for_testing();

    let results = engine.get_node_memories("nonexistent", 3, None).unwrap();
    assert!(
        results.is_empty(),
        "empty graph should return empty results"
    );
}
