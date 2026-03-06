use crate::CodememEngine;
use codemem_core::{
    Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind, RelationshipType,
};
use codemem_storage::Storage;
use std::collections::HashMap;

fn make_memory_typed(
    id: &str,
    content: &str,
    memory_type: MemoryType,
    namespace: Option<&str>,
) -> MemoryNode {
    let now = chrono::Utc::now();
    MemoryNode {
        id: id.to_string(),
        content: content.to_string(),
        memory_type,
        importance: 0.7,
        confidence: 0.9,
        access_count: 0,
        content_hash: Storage::content_hash(content),
        tags: vec![],
        metadata: HashMap::new(),
        namespace: namespace.map(String::from),
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

// ── recall_with_impact ──────────────────────────────────────────────

#[test]
fn recall_with_impact_empty() {
    let engine = CodememEngine::for_testing();
    let results = engine
        .recall_with_impact("nonexistent query xyz", 5, None)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn recall_with_impact_returns_impact_data() {
    let engine = CodememEngine::for_testing();

    let m = make_memory_typed(
        "imp1",
        "architecture decision about modules",
        MemoryType::Context,
        None,
    );
    engine.persist_memory(&m).unwrap();

    let results = engine
        .recall_with_impact("architecture modules", 5, None)
        .unwrap();
    assert!(!results.is_empty());

    let first = &results[0];
    assert_eq!(first.search_result.memory.id, "imp1");
    // PageRank/centrality will be 0.0 for isolated nodes, but the fields should exist
    assert!(first.pagerank >= 0.0);
    assert!(first.centrality >= 0.0);
}

#[test]
fn recall_with_impact_finds_connected_decisions() {
    let engine = CodememEngine::for_testing();

    // Create a context memory and a decision memory, link them
    let m1 = make_memory_typed(
        "ctx1",
        "context about database schema design",
        MemoryType::Context,
        None,
    );
    let m2 = make_memory_typed(
        "dec1",
        "decision about database schema design",
        MemoryType::Decision,
        None,
    );
    engine.persist_memory(&m1).unwrap();
    engine.persist_memory(&m2).unwrap();

    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        let edge = Edge {
            id: "ctx1-dec1".to_string(),
            src: "ctx1".to_string(),
            dst: "dec1".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.8,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        };
        let _ = graph.add_edge(edge);
    }

    let results = engine
        .recall_with_impact("database schema", 5, None)
        .unwrap();
    // The context result should list the decision as a connected decision
    let ctx_result = results.iter().find(|r| r.search_result.memory.id == "ctx1");
    if let Some(r) = ctx_result {
        assert!(
            r.connected_decisions.contains(&"dec1".to_string()),
            "should find connected decision: {:?}",
            r.connected_decisions
        );
    }
}

#[test]
fn recall_with_impact_finds_dependent_files() {
    let engine = CodememEngine::for_testing();

    let m = make_memory_typed(
        "dep1",
        "memory about file dependency tracking",
        MemoryType::Context,
        None,
    );
    engine.persist_memory(&m).unwrap();

    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        // Add a file node
        let file_node = GraphNode {
            id: "file:src/main.rs".to_string(),
            kind: NodeKind::File,
            label: "src/main.rs".to_string(),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        };
        graph.add_node(file_node).unwrap();

        // Link memory to file
        let edge = Edge {
            id: "dep1-file".to_string(),
            src: "dep1".to_string(),
            dst: "file:src/main.rs".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        };
        let _ = graph.add_edge(edge);
    }

    let results = engine
        .recall_with_impact("file dependency", 5, None)
        .unwrap();
    let result = results.iter().find(|r| r.search_result.memory.id == "dep1");
    if let Some(r) = result {
        assert!(
            r.dependent_files.contains(&"src/main.rs".to_string()),
            "should find dependent file: {:?}",
            r.dependent_files
        );
    }
}

// ── get_decision_chain ──────────────────────────────────────────────

#[test]
fn decision_chain_requires_filter() {
    let engine = CodememEngine::for_testing();
    let result = engine.get_decision_chain(None, None);
    assert!(
        result.is_err(),
        "should error when neither file_path nor topic provided"
    );
}

#[test]
fn decision_chain_empty_for_no_match() {
    let engine = CodememEngine::for_testing();
    let chain = engine
        .get_decision_chain(Some("nonexistent_file.rs"), None)
        .unwrap();
    assert_eq!(chain.chain_length, 0);
    assert!(chain.decisions.is_empty());
}

#[test]
fn decision_chain_finds_decisions_by_file_path() {
    let engine = CodememEngine::for_testing();

    let d1 = make_memory_typed(
        "chain-d1",
        "decided to use PostgreSQL for src/db.rs",
        MemoryType::Decision,
        None,
    );
    let d2 = make_memory_typed(
        "chain-d2",
        "revised database choice for src/db.rs to use SQLite",
        MemoryType::Decision,
        None,
    );
    engine.persist_memory(&d1).unwrap();
    engine.persist_memory(&d2).unwrap();

    // Link them with EvolvedInto
    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        let edge = Edge {
            id: "d1-d2".to_string(),
            src: "chain-d1".to_string(),
            dst: "chain-d2".to_string(),
            relationship: RelationshipType::EvolvedInto,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        };
        let _ = graph.add_edge(edge);
    }

    let chain = engine.get_decision_chain(Some("src/db.rs"), None).unwrap();
    assert_eq!(chain.chain_length, 2, "should find both decisions");
    assert_eq!(chain.file_path.as_deref(), Some("src/db.rs"));
}

#[test]
fn decision_chain_finds_decisions_by_topic() {
    let engine = CodememEngine::for_testing();

    let d1 = make_memory_typed(
        "topic-d1",
        "decided to use REST API for the authentication system",
        MemoryType::Decision,
        None,
    );
    engine.persist_memory(&d1).unwrap();

    let chain = engine
        .get_decision_chain(None, Some("authentication"))
        .unwrap();
    assert!(chain.chain_length >= 1, "should find decision by topic");
    assert_eq!(chain.topic.as_deref(), Some("authentication"));
}

#[test]
fn decision_chain_follows_evolved_into_edges() {
    let engine = CodememEngine::for_testing();

    // Create a chain of 3 decisions linked by EvolvedInto
    let d1 = make_memory_typed(
        "evo-d1",
        "initial architecture approach for config module",
        MemoryType::Decision,
        None,
    );
    let d2 = make_memory_typed(
        "evo-d2",
        "revised architecture for config module to use TOML",
        MemoryType::Decision,
        None,
    );
    let d3 = make_memory_typed(
        "evo-d3",
        "final architecture for config module with env var overrides",
        MemoryType::Decision,
        None,
    );
    engine.persist_memory(&d1).unwrap();
    engine.persist_memory(&d2).unwrap();
    engine.persist_memory(&d3).unwrap();

    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        for (id, src, dst) in [
            ("evo-e1", "evo-d1", "evo-d2"),
            ("evo-e2", "evo-d2", "evo-d3"),
        ] {
            let edge = Edge {
                id: id.to_string(),
                src: src.to_string(),
                dst: dst.to_string(),
                relationship: RelationshipType::EvolvedInto,
                weight: 1.0,
                properties: HashMap::new(),
                created_at: now,
                valid_from: None,
                valid_to: None,
            };
            let _ = graph.add_edge(edge);
        }
    }

    let chain = engine
        .get_decision_chain(None, Some("config module"))
        .unwrap();
    assert_eq!(
        chain.chain_length, 3,
        "should follow EvolvedInto edges to find all 3 decisions"
    );
}

// ── node_coverage ───────────────────────────────────────────────────

#[test]
fn node_coverage_no_memories() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(GraphNode {
                id: "file:test.rs".to_string(),
                kind: NodeKind::File,
                label: "test.rs".to_string(),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: None,
                namespace: None,
            })
            .unwrap();
    }

    let coverage = engine.node_coverage(&["file:test.rs"]).unwrap();
    assert_eq!(coverage.len(), 1);
    assert_eq!(coverage[0].memory_count, 0);
    assert!(!coverage[0].has_coverage);
}

#[test]
fn node_coverage_with_memory_link() {
    let engine = CodememEngine::for_testing();

    let m = make_memory_typed("cov-mem", "coverage test memory", MemoryType::Context, None);
    engine.persist_memory(&m).unwrap();

    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        graph
            .add_node(GraphNode {
                id: "file:covered.rs".to_string(),
                kind: NodeKind::File,
                label: "covered.rs".to_string(),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: None,
                namespace: None,
            })
            .unwrap();

        let edge = Edge {
            id: "cov-edge".to_string(),
            src: "file:covered.rs".to_string(),
            dst: "cov-mem".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        };
        let _ = graph.add_edge(edge);
    }

    let coverage = engine.node_coverage(&["file:covered.rs"]).unwrap();
    assert_eq!(coverage.len(), 1);
    assert_eq!(coverage[0].memory_count, 1);
    assert!(coverage[0].has_coverage);
}

#[test]
fn node_coverage_multiple_nodes() {
    let engine = CodememEngine::for_testing();
    {
        let mut graph = engine.lock_graph().unwrap();
        for id in ["file:a.rs", "file:b.rs", "file:c.rs"] {
            graph
                .add_node(GraphNode {
                    id: id.to_string(),
                    kind: NodeKind::File,
                    label: id.to_string(),
                    payload: HashMap::new(),
                    centrality: 0.0,
                    memory_id: None,
                    namespace: None,
                })
                .unwrap();
        }
    }

    let coverage = engine
        .node_coverage(&["file:a.rs", "file:b.rs", "file:c.rs"])
        .unwrap();
    assert_eq!(coverage.len(), 3);
}
