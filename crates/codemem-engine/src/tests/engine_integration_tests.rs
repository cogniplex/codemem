use crate::CodememEngine;
use codemem_core::{
    Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind, RelationshipType,
};
use codemem_storage::Storage;
use std::collections::HashMap;

fn make_memory(id: &str, content: &str) -> MemoryNode {
    let now = chrono::Utc::now();
    MemoryNode {
        id: id.to_string(),
        content: content.to_string(),
        memory_type: MemoryType::Context,
        importance: 0.7,
        confidence: 0.9,
        access_count: 0,
        content_hash: Storage::content_hash(content),
        tags: vec!["test".to_string()],
        metadata: HashMap::new(),
        namespace: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

// ── Test #3: Dirty flag lifecycle ───────────────────────────────────

#[test]
fn dirty_flag_lifecycle() {
    let engine = CodememEngine::for_testing();

    // Initially not dirty
    assert!(!engine.is_dirty(), "engine should start clean");

    // persist_memory_no_save sets dirty
    let m1 = make_memory("dirty-1", "dirty flag test memory one");
    engine.persist_memory_no_save(&m1).unwrap();
    assert!(
        engine.is_dirty(),
        "should be dirty after persist_memory_no_save"
    );

    // save_index clears dirty
    engine.save_index();
    assert!(!engine.is_dirty(), "should be clean after save_index");

    // persist_memory_no_save sets dirty again
    let m2 = make_memory("dirty-2", "dirty flag test memory two");
    engine.persist_memory_no_save(&m2).unwrap();
    assert!(
        engine.is_dirty(),
        "should be dirty again after second persist_memory_no_save"
    );

    // persist_memory (save=true) clears dirty
    let m3 = make_memory("dirty-3", "dirty flag test memory three");
    engine.persist_memory(&m3).unwrap();
    assert!(
        !engine.is_dirty(),
        "should be clean after persist_memory with save=true"
    );
}

// ── Test #7: Recall with camelCase queries ──────────────────────────

#[test]
fn recall_camel_case_query() {
    let engine = CodememEngine::for_testing();

    let mem = make_memory("camel-1", "processRequest handles incoming HTTP data");
    engine.persist_memory(&mem).unwrap();

    // camelCase query
    let results = engine
        .recall("processRequest", 5, None, None, &[], None, None)
        .unwrap();
    assert!(
        !results.is_empty(),
        "recall with camelCase query should find the memory"
    );
    assert_eq!(results[0].memory.id, "camel-1");

    // snake_case query should also match (cross-convention)
    let results2 = engine
        .recall("process_request", 5, None, None, &[], None, None)
        .unwrap();
    assert!(
        !results2.is_empty(),
        "recall with snake_case query should also find the memory"
    );
    assert_eq!(results2[0].memory.id, "camel-1");
}

// ── Test #8: Betweenness lazy compute ───────────────────────────────

#[test]
fn betweenness_lazy_compute_on_recall() {
    let engine = CodememEngine::for_testing();

    // Build a non-trivial graph: A -> B -> C
    let now = chrono::Utc::now();
    {
        let mut graph = engine.lock_graph().unwrap();
        for (id, kind) in &[
            ("sym:AlphaFunc", NodeKind::Function),
            ("sym:BetaFunc", NodeKind::Function),
            ("sym:GammaFunc", NodeKind::Function),
        ] {
            let node = GraphNode {
                id: id.to_string(),
                kind: *kind,
                label: id.to_string(),
                payload: HashMap::new(),
                centrality: 0.0,
                memory_id: None,
                namespace: None,
            };
            graph.add_node(node).unwrap();
        }
        for (id, src, dst) in &[
            ("e1", "sym:AlphaFunc", "sym:BetaFunc"),
            ("e2", "sym:BetaFunc", "sym:GammaFunc"),
        ] {
            let edge = Edge {
                id: id.to_string(),
                src: src.to_string(),
                dst: dst.to_string(),
                relationship: RelationshipType::Calls,
                weight: 0.8,
                properties: HashMap::new(),
                created_at: now,
                valid_from: None,
                valid_to: None,
            };
            graph.add_edge(edge).unwrap();
        }

        // Compute only PageRank, NOT betweenness
        graph.recompute_centrality_with_options(false);

        // Verify betweenness is empty: all nodes should return 0.0
        for id in &["sym:AlphaFunc", "sym:BetaFunc", "sym:GammaFunc"] {
            assert_eq!(
                graph.get_betweenness(id),
                0.0,
                "betweenness for {id} should be 0.0 after recompute_centrality_with_options(false)"
            );
        }
    }

    // Persist a memory and recall — recall triggers ensure_betweenness_computed
    let mem = make_memory("between-1", "alpha beta gamma function calls");
    engine.persist_memory(&mem).unwrap();

    let _results = engine
        .recall("alpha function", 5, None, None, &[], None, None)
        .unwrap();

    // After recall, betweenness should be populated for the bridge node
    let graph = engine.lock_graph().unwrap();
    // BetaFunc is on the path A->B->C, so it should have non-zero betweenness
    let beta_betweenness = graph.get_betweenness("sym:BetaFunc");
    assert!(
        beta_betweenness > 0.0,
        "BetaFunc betweenness should be > 0 after recall triggers lazy compute, got {beta_betweenness}"
    );
}

// ── Test #9: Batch persist efficiency ───────────────────────────────

#[test]
fn batch_persist_then_single_save() {
    let engine = CodememEngine::for_testing();

    let m1 = make_memory("batch-1", "first batch memory about rust ownership");
    let m2 = make_memory("batch-2", "second batch memory about borrowing rules");
    let m3 = make_memory("batch-3", "third batch memory about lifetime annotations");

    // Persist without saving
    engine.persist_memory_no_save(&m1).unwrap();
    engine.persist_memory_no_save(&m2).unwrap();
    engine.persist_memory_no_save(&m3).unwrap();
    assert!(
        engine.is_dirty(),
        "should be dirty after batch persist_memory_no_save"
    );

    // Single save clears dirty
    engine.save_index();
    assert!(!engine.is_dirty(), "should be clean after save_index");

    // All 3 memories retrievable from storage
    for id in &["batch-1", "batch-2", "batch-3"] {
        let mem = engine.storage.get_memory(id).unwrap();
        assert!(mem.is_some(), "memory {id} should be in storage");
    }

    // All 3 in BM25 index (score > 0 for matching query)
    let bm25 = engine.lock_bm25().unwrap();
    assert!(
        bm25.score("rust ownership", "batch-1") > 0.0,
        "batch-1 should be in BM25 index"
    );
    assert!(
        bm25.score("borrowing rules", "batch-2") > 0.0,
        "batch-2 should be in BM25 index"
    );
    assert!(
        bm25.score("lifetime annotations", "batch-3") > 0.0,
        "batch-3 should be in BM25 index"
    );
}
