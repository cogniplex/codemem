use crate::CodememEngine;
use codemem_core::{Edge, GraphBackend, MemoryNode, MemoryType, RelationshipType};
use codemem_storage::Storage;
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
    let now = chrono::Utc::now();
    MemoryNode {
        id: id.to_string(),
        content: content.to_string(),
        memory_type,
        importance,
        confidence,
        access_count: 0,
        content_hash: Storage::content_hash(content),
        tags: tags.iter().map(|s| s.to_string()).collect(),
        metadata: HashMap::new(),
        namespace: namespace.map(String::from),
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

// ── Basic recall ────────────────────────────────────────────────────

#[test]
fn recall_returns_matching_memories() {
    let engine = CodememEngine::for_testing();
    engine
        .persist_memory(&make_memory("r1", "Rust ownership and borrowing rules"))
        .unwrap();
    engine
        .persist_memory(&make_memory("r2", "Python list comprehensions"))
        .unwrap();

    let results = engine
        .recall("ownership borrowing", 5, None, None, &[], None, None)
        .unwrap();
    assert!(!results.is_empty(), "should find at least one result");
    assert_eq!(results[0].memory.id, "r1");
}

#[test]
fn recall_respects_k_limit() {
    let engine = CodememEngine::for_testing();
    for i in 0..10 {
        engine
            .persist_memory(&make_memory(
                &format!("k{i}"),
                &format!("memory about testing topic number {i}"),
            ))
            .unwrap();
    }

    let results = engine
        .recall("testing topic", 3, None, None, &[], None, None)
        .unwrap();
    assert!(results.len() <= 3, "should return at most k=3 results");
}

#[test]
fn recall_sorted_by_score_descending() {
    let engine = CodememEngine::for_testing();
    engine
        .persist_memory(&make_memory("s1", "alpha beta gamma"))
        .unwrap();
    engine
        .persist_memory(&make_memory("s2", "alpha beta"))
        .unwrap();

    let results = engine
        .recall("alpha beta gamma", 5, None, None, &[], None, None)
        .unwrap();
    if results.len() >= 2 {
        assert!(
            results[0].score >= results[1].score,
            "results should be sorted by score descending"
        );
    }
}

// ── Memory type filter ──────────────────────────────────────────────

#[test]
fn recall_filters_by_memory_type() {
    let engine = CodememEngine::for_testing();
    engine
        .persist_memory(&make_memory_with_opts(
            "t1",
            "decision about database schema",
            MemoryType::Decision,
            None,
            &[],
            0.7,
            0.9,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "t2",
            "context about database schema",
            MemoryType::Context,
            None,
            &[],
            0.7,
            0.9,
        ))
        .unwrap();

    let results = engine
        .recall(
            "database schema",
            5,
            Some(MemoryType::Decision),
            None,
            &[],
            None,
            None,
        )
        .unwrap();
    for r in &results {
        assert_eq!(
            r.memory.memory_type,
            MemoryType::Decision,
            "should only return Decision type"
        );
    }
}

// ── Namespace filter ────────────────────────────────────────────────

#[test]
fn recall_filters_by_namespace() {
    let engine = CodememEngine::for_testing();
    engine
        .persist_memory(&make_memory_with_opts(
            "n1",
            "project alpha architecture notes",
            MemoryType::Context,
            Some("alpha"),
            &[],
            0.7,
            0.9,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "n2",
            "project beta architecture notes",
            MemoryType::Context,
            Some("beta"),
            &[],
            0.7,
            0.9,
        ))
        .unwrap();

    let results = engine
        .recall(
            "architecture notes",
            5,
            None,
            Some("alpha"),
            &[],
            None,
            None,
        )
        .unwrap();
    for r in &results {
        assert_eq!(
            r.memory.namespace.as_deref(),
            Some("alpha"),
            "should only return alpha namespace"
        );
    }
}

// ── Tag exclusion ───────────────────────────────────────────────────

#[test]
fn recall_excludes_tags() {
    let engine = CodememEngine::for_testing();
    engine
        .persist_memory(&make_memory_with_opts(
            "ex1",
            "temporary draft notes about architecture",
            MemoryType::Context,
            None,
            &["draft"],
            0.7,
            0.9,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "ex2",
            "finalized notes about architecture",
            MemoryType::Context,
            None,
            &["final"],
            0.7,
            0.9,
        ))
        .unwrap();

    let exclude = vec!["draft".to_string()];
    let results = engine
        .recall("architecture", 5, None, None, &exclude, None, None)
        .unwrap();
    for r in &results {
        assert!(
            !r.memory.tags.contains(&"draft".to_string()),
            "should not contain excluded tag"
        );
    }
}

// ── Importance threshold ────────────────────────────────────────────

#[test]
fn recall_filters_by_min_importance() {
    let engine = CodememEngine::for_testing();
    engine
        .persist_memory(&make_memory_with_opts(
            "imp1",
            "low importance note about testing",
            MemoryType::Context,
            None,
            &[],
            0.1,
            0.9,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "imp2",
            "high importance note about testing",
            MemoryType::Context,
            None,
            &[],
            0.9,
            0.9,
        ))
        .unwrap();

    let results = engine
        .recall("testing", 5, None, None, &[], Some(0.5), None)
        .unwrap();
    for r in &results {
        assert!(
            r.memory.importance >= 0.5,
            "should only return memories with importance >= 0.5, got {}",
            r.memory.importance
        );
    }
}

// ── Confidence threshold ────────────────────────────────────────────

#[test]
fn recall_filters_by_min_confidence() {
    let engine = CodememEngine::for_testing();
    engine
        .persist_memory(&make_memory_with_opts(
            "conf1",
            "uncertain note about deployment",
            MemoryType::Context,
            None,
            &[],
            0.7,
            0.2,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "conf2",
            "confident note about deployment",
            MemoryType::Context,
            None,
            &[],
            0.7,
            0.95,
        ))
        .unwrap();

    let results = engine
        .recall("deployment", 5, None, None, &[], None, Some(0.8))
        .unwrap();
    for r in &results {
        assert!(
            r.memory.confidence >= 0.8,
            "should only return memories with confidence >= 0.8, got {}",
            r.memory.confidence
        );
    }
}

// ── Empty recall ────────────────────────────────────────────────────

#[test]
fn recall_scores_low_for_no_token_overlap() {
    let engine = CodememEngine::for_testing();
    // Store a memory with specific content
    engine
        .persist_memory(&make_memory("no1", "cats dogs pets animals"))
        .unwrap();
    // Store a matching memory
    engine
        .persist_memory(&make_memory(
            "match1",
            "quantum chromodynamics gluon plasma",
        ))
        .unwrap();

    let results = engine
        .recall(
            "quantum_chromodynamics_gluon_plasma",
            5,
            None,
            None,
            &[],
            None,
            None,
        )
        .unwrap();
    // BM25 fallback may still return results due to importance/confidence/recency
    // scoring components, but the matching memory should score higher
    if results.len() >= 2 {
        let match_score = results
            .iter()
            .find(|r| r.memory.id == "match1")
            .map(|r| r.score);
        let no_match_score = results
            .iter()
            .find(|r| r.memory.id == "no1")
            .map(|r| r.score);
        if let (Some(ms), Some(nms)) = (match_score, no_match_score) {
            assert!(ms >= nms, "matching memory should score >= non-matching");
        }
    }
}

// ── Entity expansion in default recall ──────────────────────────────

#[test]
fn recall_finds_entity_connected_memories() {
    let engine = CodememEngine::for_testing();

    // Create a memory that is semantically unrelated to "AuthService"
    let m = make_memory("entity-m1", "database connection pool tuning parameters");
    engine.persist_memory(&m).unwrap();

    // Create a code entity node in the graph and link the memory to it
    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        let code_node = codemem_core::GraphNode {
            id: "sym:AuthService".to_string(),
            kind: codemem_core::NodeKind::Class,
            label: "AuthService".to_string(),
            payload: std::collections::HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        };
        graph.add_node(code_node).unwrap();

        let edge = Edge {
            id: "sym:AuthService-RELATES_TO-entity-m1".to_string(),
            src: "sym:AuthService".to_string(),
            dst: "entity-m1".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        };
        graph.add_edge(edge).unwrap();
    }

    // Query mentions the entity name — should find the connected memory
    // even though "AuthService" has zero token overlap with "database connection pool"
    let results = engine
        .recall("AuthService", 10, None, None, &[], None, None)
        .unwrap();

    let found = results.iter().any(|r| r.memory.id == "entity-m1");
    assert!(
        found,
        "entity expansion should surface memory connected to AuthService node; got: {:?}",
        results.iter().map(|r| &r.memory.id).collect::<Vec<_>>()
    );
}

#[test]
fn resolve_entity_memories_skips_expired_edges() {
    let engine = CodememEngine::for_testing();

    let m = make_memory("entity-exp1", "expired edge memory content");
    engine.persist_memory(&m).unwrap();

    let now = chrono::Utc::now();
    let past = now - chrono::Duration::hours(1);
    {
        let mut graph = engine.lock_graph().unwrap();
        let code_node = codemem_core::GraphNode {
            id: "sym:ExpiredService".to_string(),
            kind: codemem_core::NodeKind::Class,
            label: "ExpiredService".to_string(),
            payload: std::collections::HashMap::new(),
            centrality: 0.0,
            memory_id: None,
            namespace: None,
        };
        graph.add_node(code_node).unwrap();

        // Create an expired edge (valid_to in the past)
        let edge = Edge {
            id: "sym:ExpiredService-RELATES_TO-entity-exp1".to_string(),
            src: "sym:ExpiredService".to_string(),
            dst: "entity-exp1".to_string(),
            relationship: RelationshipType::RelatesTo,
            weight: 0.5,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: Some(past),
        };
        graph.add_edge(edge).unwrap();
    }

    // Directly test resolve_entity_memories — expired edge should be skipped
    let graph = engine.lock_graph().unwrap();
    let entity_ids = engine.resolve_entity_memories("ExpiredService", &graph, now);
    assert!(
        !entity_ids.contains("entity-exp1"),
        "resolve_entity_memories should not return memories connected via expired edges"
    );
}

// ── Temporal edge filtering in expansion ─────────────────────────────

#[test]
fn recall_expansion_excludes_expired_edge_memories() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();
    let past = now - chrono::Duration::hours(1);

    // Create seed memory and a second memory connected by an expired edge
    let m1 = make_memory("temp-exp-seed", "temporal test seed about architecture");
    let m2 = make_memory("temp-exp-target", "expired edge target about architecture");
    engine.persist_memory(&m1).unwrap();
    engine.persist_memory(&m2).unwrap();

    // Create an expired edge between them
    {
        let mut graph = engine.lock_graph().unwrap();
        let edge = Edge {
            id: "temp-expired-edge".to_string(),
            src: "temp-exp-seed".to_string(),
            dst: "temp-exp-target".to_string(),
            relationship: RelationshipType::LeadsTo,
            weight: 0.8,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: Some(past), // expired
        };
        let _ = graph.add_edge(edge);
    }

    // recall_with_expansion filters expired edges during BFS expansion
    let results = engine
        .recall_with_expansion("temporal test architecture", 10, 2, None)
        .unwrap();

    // The target memory may still appear via direct BM25 scoring (both contain "architecture"),
    // but it should NOT appear with expansion_path referencing the expired edge.
    // The key assertion: if target is found, it must be via "direct" scoring, not expansion.
    for r in &results {
        if r.result.memory.id == "temp-exp-target" {
            assert_eq!(
                r.expansion_path, "direct",
                "expired edge should not be used for expansion; got path: {}",
                r.expansion_path
            );
        }
    }
}

#[test]
fn recall_expansion_includes_active_edge_memories() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();
    let future = now + chrono::Duration::hours(24);

    let m1 = make_memory("temp-act-seed", "active edge seed about modules");
    let m2 = make_memory(
        "temp-act-target",
        "active edge target totally unrelated xyzzy",
    );
    engine.persist_memory(&m1).unwrap();
    engine.persist_memory(&m2).unwrap();

    // Create an active edge (valid_to in the future)
    {
        let mut graph = engine.lock_graph().unwrap();
        let edge = Edge {
            id: "temp-active-edge".to_string(),
            src: "temp-act-seed".to_string(),
            dst: "temp-act-target".to_string(),
            relationship: RelationshipType::LeadsTo,
            weight: 0.8,
            properties: HashMap::new(),
            created_at: now,
            valid_from: Some(now - chrono::Duration::hours(1)),
            valid_to: Some(future),
        };
        let _ = graph.add_edge(edge);
    }

    let results = engine
        .recall_with_expansion("active edge seed modules", 10, 2, None)
        .unwrap();

    let found = results
        .iter()
        .any(|r| r.result.memory.id == "temp-act-target");
    assert!(
        found,
        "memory connected via active (valid_to > now) edge should be reachable via expansion"
    );
}

#[test]
fn recall_expansion_excludes_future_valid_from_edges() {
    let engine = CodememEngine::for_testing();
    let now = chrono::Utc::now();
    let future = now + chrono::Duration::hours(24);

    let m1 = make_memory("temp-fut-seed", "future edge seed about patterns");
    let m2 = make_memory(
        "temp-fut-target",
        "future edge target completely unrelated abcdef",
    );
    engine.persist_memory(&m1).unwrap();
    engine.persist_memory(&m2).unwrap();

    // Create edge with future valid_from (not yet active)
    {
        let mut graph = engine.lock_graph().unwrap();
        let edge = Edge {
            id: "temp-future-edge".to_string(),
            src: "temp-fut-seed".to_string(),
            dst: "temp-fut-target".to_string(),
            relationship: RelationshipType::LeadsTo,
            weight: 0.8,
            properties: HashMap::new(),
            created_at: now,
            valid_from: Some(future), // not yet active
            valid_to: None,
        };
        let _ = graph.add_edge(edge);
    }

    let results = engine
        .recall_with_expansion("future edge seed patterns", 10, 2, None)
        .unwrap();

    // Target should not appear via expansion (edge not yet active)
    for r in &results {
        if r.result.memory.id == "temp-fut-target" {
            assert_eq!(
                r.expansion_path, "direct",
                "future valid_from edge should not be used for expansion; got path: {}",
                r.expansion_path
            );
        }
    }
}

// ── Recall with expansion ───────────────────────────────────────────

#[test]
fn recall_with_expansion_finds_graph_connected_memories() {
    let engine = CodememEngine::for_testing();

    // Create two memories
    let m1 = make_memory("exp1", "primary architecture decision about modules");
    let m2 = make_memory("exp2", "secondary implementation detail about modules");
    engine.persist_memory(&m1).unwrap();
    engine.persist_memory(&m2).unwrap();

    // Link them in the graph
    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        let edge = Edge {
            id: "exp1-exp2".to_string(),
            src: "exp1".to_string(),
            dst: "exp2".to_string(),
            relationship: RelationshipType::LeadsTo,
            weight: 0.8,
            properties: HashMap::new(),
            created_at: now,
            valid_from: None,
            valid_to: None,
        };
        let _ = graph.add_edge(edge);
    }

    let results = engine
        .recall_with_expansion("architecture decision", 10, 2, None)
        .unwrap();
    assert!(!results.is_empty(), "should find memories via expansion");
}

#[test]
fn recall_with_expansion_namespace_filter() {
    let engine = CodememEngine::for_testing();

    let m1 = make_memory_with_opts(
        "ens1",
        "expansion test alpha namespace",
        MemoryType::Context,
        Some("alpha"),
        &[],
        0.7,
        0.9,
    );
    let m2 = make_memory_with_opts(
        "ens2",
        "expansion test beta namespace",
        MemoryType::Context,
        Some("beta"),
        &[],
        0.7,
        0.9,
    );
    engine.persist_memory(&m1).unwrap();
    engine.persist_memory(&m2).unwrap();

    let results = engine
        .recall_with_expansion("expansion test", 10, 1, Some("alpha"))
        .unwrap();
    for r in &results {
        assert_eq!(
            r.result.memory.namespace.as_deref(),
            Some("alpha"),
            "expansion should respect namespace filter"
        );
    }
}

// ── Namespace stats ─────────────────────────────────────────────────

#[test]
fn namespace_stats_empty_namespace() {
    let engine = CodememEngine::for_testing();
    let stats = engine.namespace_stats("nonexistent").unwrap();
    assert_eq!(stats.count, 0);
    assert_eq!(stats.avg_importance, 0.0);
    assert_eq!(stats.avg_confidence, 0.0);
    assert!(stats.type_distribution.is_empty());
    assert!(stats.oldest.is_none());
    assert!(stats.newest.is_none());
}

#[test]
fn namespace_stats_computes_correctly() {
    let engine = CodememEngine::for_testing();

    engine
        .persist_memory(&make_memory_with_opts(
            "ns1",
            "first memory in stats ns",
            MemoryType::Decision,
            Some("stats-ns"),
            &["tag-a"],
            0.6,
            0.8,
        ))
        .unwrap();
    engine
        .persist_memory(&make_memory_with_opts(
            "ns2",
            "second memory in stats ns",
            MemoryType::Context,
            Some("stats-ns"),
            &["tag-a", "tag-b"],
            0.8,
            1.0,
        ))
        .unwrap();

    let stats = engine.namespace_stats("stats-ns").unwrap();
    assert_eq!(stats.count, 2);
    assert!((stats.avg_importance - 0.7).abs() < 0.01);
    assert!((stats.avg_confidence - 0.9).abs() < 0.01);
    assert_eq!(stats.type_distribution.get("decision"), Some(&1));
    assert_eq!(stats.type_distribution.get("context"), Some(&1));
    assert_eq!(stats.tag_frequency.get("tag-a"), Some(&2));
    assert_eq!(stats.tag_frequency.get("tag-b"), Some(&1));
    assert!(stats.oldest.is_some());
    assert!(stats.newest.is_some());
}

// ── Delete namespace ────────────────────────────────────────────────

#[test]
fn delete_namespace_removes_all_memories() {
    let engine = CodememEngine::for_testing();

    for i in 0..3 {
        engine
            .persist_memory(&make_memory_with_opts(
                &format!("del-{i}"),
                &format!("memory {i} to delete"),
                MemoryType::Context,
                Some("to-delete"),
                &[],
                0.7,
                0.9,
            ))
            .unwrap();
    }
    // One in a different namespace
    engine
        .persist_memory(&make_memory_with_opts(
            "keep-1",
            "memory to keep",
            MemoryType::Context,
            Some("keep-ns"),
            &[],
            0.7,
            0.9,
        ))
        .unwrap();

    let deleted = engine.delete_namespace("to-delete").unwrap();
    assert_eq!(deleted, 3, "should delete exactly 3 memories");

    // Verify they're gone from storage
    for i in 0..3 {
        let m = engine.storage.get_memory(&format!("del-{i}")).unwrap();
        assert!(m.is_none(), "deleted memory should be gone");
    }

    // Verify the other namespace is intact
    let kept = engine.storage.get_memory("keep-1").unwrap();
    assert!(kept.is_some(), "memory in other namespace should remain");
}

#[test]
fn delete_namespace_returns_zero_for_empty() {
    let engine = CodememEngine::for_testing();
    let deleted = engine.delete_namespace("nonexistent-ns").unwrap();
    assert_eq!(deleted, 0);
}

// ── is_edge_active ──────────────────────────────────────────────────

#[test]
fn is_edge_active_no_bounds() {
    let now = chrono::Utc::now();
    let edge = Edge {
        id: "e1".to_string(),
        src: "a".to_string(),
        dst: "b".to_string(),
        relationship: RelationshipType::RelatesTo,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: now,
        valid_from: None,
        valid_to: None,
    };
    assert!(
        crate::recall::is_edge_active(&edge, now),
        "edge with no bounds should be active"
    );
}

#[test]
fn is_edge_active_expired() {
    let now = chrono::Utc::now();
    let past = now - chrono::Duration::hours(1);
    let edge = Edge {
        id: "e2".to_string(),
        src: "a".to_string(),
        dst: "b".to_string(),
        relationship: RelationshipType::RelatesTo,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: now,
        valid_from: None,
        valid_to: Some(past),
    };
    assert!(
        !crate::recall::is_edge_active(&edge, now),
        "edge with valid_to in the past should be inactive"
    );
}

#[test]
fn is_edge_active_future_start() {
    let now = chrono::Utc::now();
    let future = now + chrono::Duration::hours(1);
    let edge = Edge {
        id: "e3".to_string(),
        src: "a".to_string(),
        dst: "b".to_string(),
        relationship: RelationshipType::RelatesTo,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: now,
        valid_from: Some(future),
        valid_to: None,
    };
    assert!(
        !crate::recall::is_edge_active(&edge, now),
        "edge with valid_from in the future should be inactive"
    );
}

#[test]
fn is_edge_active_within_window() {
    let now = chrono::Utc::now();
    let past = now - chrono::Duration::hours(1);
    let future = now + chrono::Duration::hours(1);
    let edge = Edge {
        id: "e4".to_string(),
        src: "a".to_string(),
        dst: "b".to_string(),
        relationship: RelationshipType::RelatesTo,
        weight: 1.0,
        properties: HashMap::new(),
        created_at: now,
        valid_from: Some(past),
        valid_to: Some(future),
    };
    assert!(
        crate::recall::is_edge_active(&edge, now),
        "edge within valid window should be active"
    );
}
