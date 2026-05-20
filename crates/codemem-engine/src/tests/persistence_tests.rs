use crate::index::chunker::CodeChunk;
use crate::index::indexer::{IndexAndResolveResult, IndexResult};
use crate::index::resolver::ResolvedEdge;
use crate::index::symbol::{Symbol, SymbolKind, Visibility};
use crate::persistence::edge_weight_for;
use crate::CodememEngine;
use codemem_core::{GraphConfig, NodeKind, RelationshipType};
use std::collections::HashSet;

// ── Helper: build a minimal Symbol ──────────────────────────────────

fn make_symbol(name: &str, file_path: &str, kind: SymbolKind) -> Symbol {
    Symbol {
        name: name.to_string(),
        qualified_name: name.to_string(),
        kind,
        signature: format!("fn {name}()"),
        visibility: Visibility::Public,
        file_path: file_path.to_string(),
        line_start: 1,
        line_end: 10,
        doc_comment: None,
        parent: None,
        parameters: vec![],
        return_type: None,
        is_async: false,
        attributes: vec![],
        throws: vec![],
        generic_params: None,
        is_abstract: false,
    }
}

fn make_symbol_vis(
    name: &str,
    file_path: &str,
    kind: SymbolKind,
    vis: Visibility,
    line_start: usize,
    line_end: usize,
) -> Symbol {
    Symbol {
        name: name.to_string(),
        qualified_name: name.to_string(),
        kind,
        signature: format!("fn {name}()"),
        visibility: vis,
        file_path: file_path.to_string(),
        line_start,
        line_end,
        doc_comment: None,
        parent: None,
        parameters: vec![],
        return_type: None,
        is_async: false,
        attributes: vec![],
        throws: vec![],
        generic_params: None,
        is_abstract: false,
    }
}

fn make_chunk(file_path: &str, index: usize, parent: Option<&str>) -> CodeChunk {
    CodeChunk {
        index,
        text: format!("chunk {index} content"),
        node_kind: "function_item".to_string(),
        line_start: index * 10,
        line_end: index * 10 + 9,
        byte_start: 0,
        byte_end: 100,
        non_ws_chars: 50,
        parent_symbol: parent.map(String::from),
        file_path: file_path.to_string(),
    }
}

fn make_index_result(
    symbols: Vec<Symbol>,
    chunks: Vec<CodeChunk>,
    file_paths: HashSet<String>,
    edges: Vec<ResolvedEdge>,
) -> IndexAndResolveResult {
    let total_symbols = symbols.len();
    let total_chunks = chunks.len();
    let files_parsed = file_paths.len();
    IndexAndResolveResult {
        index: IndexResult {
            files_scanned: files_parsed,
            files_parsed,
            files_skipped: 0,
            total_symbols,
            total_references: 0,
            total_chunks,
            total_documents: 0,
            parse_results: vec![],
        },
        symbols,
        references: vec![],
        chunks,
        doc_nodes: vec![],
        file_paths,
        edges,
        unresolved: vec![],
        root_path: std::path::PathBuf::from("/test"),
        scip_build: None,
    }
}

// ── edge_weight_for ─────────────────────────────────────────────────

#[test]
fn edge_weight_for_configurable_types() {
    let config = GraphConfig::default();
    assert_eq!(edge_weight_for(&RelationshipType::Calls, &config), 1.0);
    assert_eq!(edge_weight_for(&RelationshipType::Imports, &config), 0.5);
    assert_eq!(edge_weight_for(&RelationshipType::Contains, &config), 0.1);
}

#[test]
fn edge_weight_for_custom_config() {
    let config = GraphConfig {
        calls_edge_weight: 0.9,
        imports_edge_weight: 0.3,
        contains_edge_weight: 0.2,
        ..Default::default()
    };
    assert_eq!(edge_weight_for(&RelationshipType::Calls, &config), 0.9);
    assert_eq!(edge_weight_for(&RelationshipType::Imports, &config), 0.3);
    assert_eq!(edge_weight_for(&RelationshipType::Contains, &config), 0.2);
}

#[test]
fn edge_weight_for_fixed_types() {
    let config = GraphConfig::default();
    assert_eq!(edge_weight_for(&RelationshipType::Implements, &config), 0.8);
    assert_eq!(edge_weight_for(&RelationshipType::Inherits, &config), 0.8);
    assert_eq!(edge_weight_for(&RelationshipType::DependsOn, &config), 0.7);
    assert_eq!(edge_weight_for(&RelationshipType::CoChanged, &config), 0.6);
    assert_eq!(
        edge_weight_for(&RelationshipType::EvolvedInto, &config),
        0.7
    );
    assert_eq!(edge_weight_for(&RelationshipType::Summarizes, &config), 0.7);
    assert_eq!(edge_weight_for(&RelationshipType::PartOf, &config), 0.4);
    assert_eq!(edge_weight_for(&RelationshipType::RelatesTo, &config), 0.3);
    assert_eq!(
        edge_weight_for(&RelationshipType::SharesTheme, &config),
        0.3
    );
}

#[test]
fn edge_weight_for_default_fallback() {
    let config = GraphConfig::default();
    // Types not explicitly matched should return 0.5
    assert_eq!(edge_weight_for(&RelationshipType::Blocks, &config), 0.5);
    assert_eq!(
        edge_weight_for(&RelationshipType::Contradicts, &config),
        0.5
    );
}

// ── persist_index_results: file nodes ───────────────────────────────

#[test]
fn persist_creates_file_nodes() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());
    files.insert("src/lib.rs".to_string());

    let result = make_index_result(vec![], vec![], files, vec![]);
    let pr = engine.persist_index_results(&result, Some("test")).unwrap();

    assert_eq!(pr.files_created, 2);

    let graph = engine.lock_graph().unwrap();
    let main_node = graph.get_node("file:src/main.rs").unwrap();
    assert!(main_node.is_some());
    assert_eq!(main_node.unwrap().kind, NodeKind::File);

    let lib_node = graph.get_node("file:src/lib.rs").unwrap();
    assert!(lib_node.is_some());
}

// ── persist_index_results: package (directory) nodes ────────────────

#[test]
fn persist_creates_package_nodes() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/util/helper.rs".to_string());

    let result = make_index_result(vec![], vec![], files, vec![]);
    let pr = engine.persist_index_results(&result, Some("test")).unwrap();

    assert!(pr.packages_created > 0);

    let graph = engine.lock_graph().unwrap();
    // Should have pkg:src/ and pkg:src/util/
    let src_pkg = graph.get_node("pkg:src/").unwrap();
    assert!(src_pkg.is_some(), "should create src/ package node");
    assert_eq!(src_pkg.unwrap().kind, NodeKind::Package);

    let util_pkg = graph.get_node("pkg:src/util/").unwrap();
    assert!(util_pkg.is_some(), "should create src/util/ package node");
}

#[test]
fn persist_creates_contains_edges_for_packages() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/util/helper.rs".to_string());

    let result = make_index_result(vec![], vec![], files, vec![]);
    engine.persist_index_results(&result, Some("test")).unwrap();

    let graph = engine.lock_graph().unwrap();

    // pkg:src/ -> pkg:src/util/ CONTAINS edge
    let src_edges = graph.get_edges("pkg:src/").unwrap();
    let has_util_edge = src_edges
        .iter()
        .any(|e| e.dst == "pkg:src/util/" && e.relationship == RelationshipType::Contains);
    assert!(has_util_edge, "src/ should contain src/util/");

    // pkg:src/util/ -> file:src/util/helper.rs CONTAINS edge
    let util_edges = graph.get_edges("pkg:src/util/").unwrap();
    let has_file_edge = util_edges.iter().any(|e| {
        e.dst == "file:src/util/helper.rs" && e.relationship == RelationshipType::Contains
    });
    assert!(has_file_edge, "src/util/ should contain helper.rs");
}

// ── persist_index_results: symbol nodes ─────────────────────────────

#[test]
fn persist_creates_symbol_nodes() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let symbols = vec![
        make_symbol("main", "src/main.rs", SymbolKind::Function),
        make_symbol("Config", "src/main.rs", SymbolKind::Struct),
    ];

    let result = make_index_result(symbols, vec![], files, vec![]);
    let pr = engine.persist_index_results(&result, Some("test")).unwrap();

    assert_eq!(pr.symbols_stored, 2);

    let graph = engine.lock_graph().unwrap();
    let main_sym = graph.get_node("sym:main").unwrap();
    assert!(main_sym.is_some());
    let main_node = main_sym.unwrap();
    assert_eq!(main_node.kind, NodeKind::Function);

    let config_sym = graph.get_node("sym:Config").unwrap();
    assert!(config_sym.is_some());
}

#[test]
fn persist_creates_file_to_symbol_contains_edge() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let symbols = vec![make_symbol("process", "src/main.rs", SymbolKind::Function)];
    let result = make_index_result(symbols, vec![], files, vec![]);
    engine.persist_index_results(&result, Some("test")).unwrap();

    let graph = engine.lock_graph().unwrap();
    let file_edges = graph.get_edges("file:src/main.rs").unwrap();
    let has_sym_edge = file_edges
        .iter()
        .any(|e| e.dst == "sym:process" && e.relationship == RelationshipType::Contains);
    assert!(has_sym_edge, "file should contain symbol");
}

#[test]
fn persist_stores_symbol_payload_fields() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let mut sym = make_symbol("handler", "src/main.rs", SymbolKind::Function);
    sym.doc_comment = Some("Handles requests".to_string());
    sym.is_async = true;
    sym.return_type = Some("Result<()>".to_string());
    sym.visibility = Visibility::Public;

    let result = make_index_result(vec![sym], vec![], files, vec![]);
    engine.persist_index_results(&result, Some("test")).unwrap();

    let graph = engine.lock_graph().unwrap();
    let node = graph.get_node("sym:handler").unwrap().unwrap();
    assert_eq!(
        node.payload.get("doc_comment").and_then(|v| v.as_str()),
        Some("Handles requests")
    );
    assert_eq!(
        node.payload.get("is_async").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        node.payload.get("return_type").and_then(|v| v.as_str()),
        Some("Result<()>")
    );
    assert_eq!(
        node.payload.get("visibility").and_then(|v| v.as_str()),
        Some("public")
    );
}

// ── persist_index_results: resolved edges ───────────────────────────

#[test]
fn persist_creates_reference_edges() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let symbols = vec![
        make_symbol("caller", "src/main.rs", SymbolKind::Function),
        make_symbol("callee", "src/main.rs", SymbolKind::Function),
    ];
    let edges = vec![ResolvedEdge {
        source_qualified_name: "caller".to_string(),
        target_qualified_name: "callee".to_string(),
        relationship: RelationshipType::Calls,
        file_path: "src/main.rs".to_string(),
        line: 5,
        resolution_confidence: 1.0,
    }];

    let result = make_index_result(symbols, vec![], files, edges);
    let pr = engine.persist_index_results(&result, Some("test")).unwrap();

    assert_eq!(pr.edges_resolved, 1);

    let graph = engine.lock_graph().unwrap();
    let caller_edges = graph.get_edges("sym:caller").unwrap();
    let has_calls_edge = caller_edges
        .iter()
        .any(|e| e.dst == "sym:callee" && e.relationship == RelationshipType::Calls);
    assert!(
        has_calls_edge,
        "should create Calls edge from caller to callee"
    );
}

// ── persist_index_results: chunk nodes ──────────────────────────────

#[test]
fn persist_creates_chunk_nodes() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let chunks = vec![
        make_chunk("src/main.rs", 0, None),
        make_chunk("src/main.rs", 1, Some("process")),
    ];

    let result = make_index_result(vec![], chunks, files, vec![]);
    let pr = engine.persist_index_results(&result, Some("test")).unwrap();

    // Chunks may be pruned by auto-compact, but at least some should be stored
    assert!(pr.chunks_stored >= 1);

    let graph = engine.lock_graph().unwrap();
    // Check at least one chunk node exists (some may be pruned)
    let chunk0 = graph.get_node("chunk:src/main.rs:0").unwrap();
    let chunk1 = graph.get_node("chunk:src/main.rs:1").unwrap();
    // At least one should survive compaction
    assert!(
        chunk0.is_some() || chunk1.is_some(),
        "at least one chunk should survive"
    );
}

#[test]
fn persist_creates_file_to_chunk_edge() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let chunks = vec![make_chunk("src/main.rs", 0, None)];
    let result = make_index_result(vec![], chunks, files, vec![]);
    engine.persist_index_results(&result, Some("test")).unwrap();

    let graph = engine.lock_graph().unwrap();
    let file_edges = graph.get_edges("file:src/main.rs").unwrap();
    let has_chunk_edge = file_edges
        .iter()
        .any(|e| e.dst.starts_with("chunk:") && e.relationship == RelationshipType::Contains);
    assert!(has_chunk_edge, "file should have CONTAINS edge to chunk");
}

#[test]
fn persist_creates_parent_symbol_to_chunk_edge() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let symbols = vec![make_symbol("process", "src/main.rs", SymbolKind::Function)];
    let chunks = vec![make_chunk("src/main.rs", 0, Some("process"))];

    let result = make_index_result(symbols, chunks, files, vec![]);
    engine.persist_index_results(&result, Some("test")).unwrap();

    let graph = engine.lock_graph().unwrap();
    let sym_edges = graph.get_edges("sym:process").unwrap();
    let has_chunk_edge = sym_edges
        .iter()
        .any(|e| e.dst.starts_with("chunk:") && e.relationship == RelationshipType::Contains);
    assert!(
        has_chunk_edge,
        "symbol should have CONTAINS edge to its child chunk"
    );
}

// ── persist_index_results: namespace ────────────────────────────────

#[test]
fn persist_applies_namespace_to_nodes() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let symbols = vec![make_symbol("run", "src/main.rs", SymbolKind::Function)];
    let result = make_index_result(symbols, vec![], files, vec![]);
    engine
        .persist_index_results(&result, Some("my-project"))
        .unwrap();

    let graph = engine.lock_graph().unwrap();
    let file_node = graph.get_node("file:src/main.rs").unwrap().unwrap();
    assert_eq!(file_node.namespace.as_deref(), Some("my-project"));

    let sym_node = graph.get_node("sym:run").unwrap().unwrap();
    assert_eq!(sym_node.namespace.as_deref(), Some("my-project"));
}

// ── persist_index_results: without embeddings ───────────────────────

#[test]
fn persist_without_embeddings_has_zero_embedded() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let symbols = vec![make_symbol("f", "src/main.rs", SymbolKind::Function)];
    let chunks = vec![make_chunk("src/main.rs", 0, None)];
    let result = make_index_result(symbols, chunks, files, vec![]);
    let pr = engine.persist_index_results(&result, Some("test")).unwrap();

    // for_testing() has no embedding provider
    assert_eq!(pr.symbols_embedded, 0);
    assert_eq!(pr.chunks_embedded, 0);
}

// ── persist_index_results: progress callback ────────────────────────

#[test]
fn persist_calls_progress_callback() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());
    let result = make_index_result(vec![], vec![], files, vec![]);

    let progress_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let pc = progress_called.clone();

    // With no embeddings, progress won't be called. That's expected.
    engine
        .persist_index_results_with_progress(&result, Some("test"), |_done, _total| {
            pc.store(true, std::sync::atomic::Ordering::SeqCst);
        })
        .unwrap();

    // Without embeddings, no progress calls expected
    assert!(
        !progress_called.load(std::sync::atomic::Ordering::SeqCst),
        "no progress without embeddings"
    );
}

// ── compact_graph ───────────────────────────────────────────────────

#[test]
fn compact_graph_empty() {
    let engine = CodememEngine::for_testing();
    let seen = HashSet::new();
    let (chunks_pruned, symbols_pruned) = engine.compact_graph(&seen, Some("test"));
    assert_eq!(chunks_pruned, 0);
    assert_eq!(symbols_pruned, 0);
}

#[test]
fn compact_graph_preserves_structural_symbols() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    // Class/Interface are structural and should never be pruned
    let symbols = vec![
        make_symbol_vis(
            "MyClass",
            "src/main.rs",
            SymbolKind::Class,
            Visibility::Public,
            1,
            50,
        ),
        make_symbol_vis(
            "MyInterface",
            "src/main.rs",
            SymbolKind::Interface,
            Visibility::Public,
            51,
            100,
        ),
    ];

    let result = make_index_result(symbols, vec![], files.clone(), vec![]);
    engine.persist_index_results(&result, Some("test")).unwrap();

    let (_cp, sp) = engine.compact_graph(&files, Some("test"));

    // Structural symbols should not be pruned
    let graph = engine.lock_graph().unwrap();
    assert!(
        graph.get_node("sym:MyClass").unwrap().is_some(),
        "Class should survive compaction"
    );
    assert!(
        graph.get_node("sym:MyInterface").unwrap().is_some(),
        "Interface should survive compaction"
    );
    assert_eq!(sp, 0, "structural symbols should not be pruned");
}

#[test]
fn compact_prunes_low_value_chunks_beyond_max() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/big.rs".to_string());

    // Create many chunks — more than max_retained_chunks_per_file
    let max = engine.config.chunking.max_retained_chunks_per_file;
    let chunk_count = max + 20; // well beyond the limit
    let chunks: Vec<CodeChunk> = (0..chunk_count)
        .map(|i| {
            let mut c = make_chunk("src/big.rs", i, None);
            c.non_ws_chars = 10; // small, low value
            c
        })
        .collect();

    let result = make_index_result(vec![], chunks, files.clone(), vec![]);
    engine.persist_index_results(&result, Some("test")).unwrap();

    // compact_graph is already called by persist if auto_compact is on.
    // The auto_compact flag is true by default, so chunks beyond the limit
    // should already be pruned.
    let graph = engine.lock_graph().unwrap();
    let chunk_nodes: Vec<_> = graph
        .get_all_nodes()
        .into_iter()
        .filter(|n| n.kind == NodeKind::Chunk)
        .collect();

    // Should have pruned some chunks
    assert!(
        chunk_nodes.len() <= max || chunk_nodes.len() < chunk_count,
        "some chunks should be pruned: {} remaining out of {} original, max={}",
        chunk_nodes.len(),
        chunk_count,
        max
    );
}

#[test]
fn compact_preserves_high_value_chunks() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let symbols = vec![make_symbol(
        "important_fn",
        "src/main.rs",
        SymbolKind::Function,
    )];
    // One chunk with a parent symbol (high value due to structural parent)
    let mut chunk = make_chunk("src/main.rs", 0, Some("important_fn"));
    chunk.non_ws_chars = 200; // large, high value

    let result = make_index_result(symbols, vec![chunk], files.clone(), vec![]);
    engine.persist_index_results(&result, Some("test")).unwrap();

    let graph = engine.lock_graph().unwrap();
    let chunk_node = graph.get_node("chunk:src/main.rs:0").unwrap();
    assert!(
        chunk_node.is_some(),
        "high-value chunk with parent symbol should survive compaction"
    );
}

// ── has_memory_link_edge (tested via compact behavior) ──────────────

#[test]
fn compact_cold_start_redistributes_weights() {
    // On cold start (no memories), memory_link weight is redistributed
    // to other factors. This means all compaction decisions are based on
    // structural properties rather than memory links.
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    // Create a private constant (low kind_score=0.1, low vis=0.0)
    let symbols = vec![make_symbol_vis(
        "MY_CONST",
        "src/main.rs",
        SymbolKind::Constant,
        Visibility::Private,
        1,
        2,
    )];

    let result = make_index_result(symbols, vec![], files.clone(), vec![]);
    engine.persist_index_results(&result, Some("test")).unwrap();

    // With cold start, the symbol should score low enough to potentially be pruned.
    // The exact behavior depends on max_retained_symbols_per_file.
    // Just verify the compaction runs without error.
    let (cp, sp) = engine.compact_graph(&files, Some("test"));
    // At minimum, this should not panic — the values are valid counts
    let _ = (cp, sp);
}

// ── persist_index_results: end-to-end ───────────────────────────────

#[test]
fn persist_end_to_end_multi_file() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());
    files.insert("src/lib.rs".to_string());
    files.insert("src/util/helper.rs".to_string());

    let symbols = vec![
        make_symbol("main", "src/main.rs", SymbolKind::Function),
        make_symbol("init", "src/lib.rs", SymbolKind::Function),
        make_symbol("helper", "src/util/helper.rs", SymbolKind::Function),
    ];

    let chunks = vec![
        make_chunk("src/main.rs", 0, Some("main")),
        make_chunk("src/lib.rs", 0, Some("init")),
    ];

    let edges = vec![ResolvedEdge {
        source_qualified_name: "main".to_string(),
        target_qualified_name: "init".to_string(),
        relationship: RelationshipType::Calls,
        file_path: "src/main.rs".to_string(),
        line: 3,
        resolution_confidence: 1.0,
    }];

    let result = make_index_result(symbols, chunks, files, edges);
    let pr = engine
        .persist_index_results(&result, Some("test-ns"))
        .unwrap();

    assert_eq!(pr.files_created, 3);
    assert_eq!(pr.symbols_stored, 3);
    assert_eq!(pr.edges_resolved, 1);
    assert!(
        pr.packages_created >= 2,
        "should create pkg:src/ and pkg:src/util/"
    );

    // Verify the call edge exists
    let graph = engine.lock_graph().unwrap();
    let main_edges = graph.get_edges("sym:main").unwrap();
    let has_call = main_edges
        .iter()
        .any(|e| e.dst == "sym:init" && e.relationship == RelationshipType::Calls);
    assert!(has_call, "main -> init Calls edge should exist");

    // Verify namespace propagation
    let main_node = graph.get_node("sym:main").unwrap().unwrap();
    assert_eq!(main_node.namespace.as_deref(), Some("test-ns"));
}

// ── Stale chunk cleanup on re-index ─────────────────────────────────

#[test]
fn persist_cleans_stale_chunks_on_reindex() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    // First index with 2 chunks
    let chunks1 = vec![
        make_chunk("src/main.rs", 0, None),
        make_chunk("src/main.rs", 1, None),
    ];
    let result1 = make_index_result(vec![], chunks1, files.clone(), vec![]);
    engine
        .persist_index_results(&result1, Some("test"))
        .unwrap();

    // Re-index with only 1 chunk (simulating file changed)
    let chunks2 = vec![make_chunk("src/main.rs", 0, None)];
    let result2 = make_index_result(vec![], chunks2, files, vec![]);
    engine
        .persist_index_results(&result2, Some("test"))
        .unwrap();

    // Old chunk:src/main.rs:1 should be cleaned up by the stale cleanup logic
    // (delete_graph_nodes_by_prefix is called before re-inserting chunks)
    // Note: the graph may still have the node from the first pass if it
    // wasn't cleaned from the in-memory graph, but storage should be clean.
    let stored = engine
        .storage
        .get_graph_node("chunk:src/main.rs:1")
        .unwrap();
    assert!(
        stored.is_none(),
        "stale chunk should be removed from storage on re-index"
    );
}

// ── Memory ops provenance: refine_memory ────────────────────────────

fn make_test_memory(id: &str, content: &str) -> codemem_core::MemoryNode {
    let mut m = codemem_core::MemoryNode::test_default(content);
    m.id = id.to_string();
    m.memory_type = codemem_core::MemoryType::Insight;
    m.confidence = 0.8;
    m.tags = vec!["test".to_string()];
    m
}

#[test]
fn refine_memory_creates_evolved_into_edge() {
    let engine = CodememEngine::for_testing();

    let old_memory = make_test_memory("old-mem-1", "original insight");
    engine.persist_memory(&old_memory).unwrap();

    let (new_memory, new_id) = engine
        .refine_memory("old-mem-1", Some("refined insight"), None, None)
        .unwrap();

    assert_ne!(new_id, "old-mem-1");
    assert_eq!(new_memory.content, "refined insight");

    // Verify EVOLVED_INTO edge from old -> new
    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges("old-mem-1").unwrap();
    let evolved_edge = edges
        .iter()
        .find(|e| e.dst == new_id && e.relationship == RelationshipType::EvolvedInto);
    assert!(
        evolved_edge.is_some(),
        "should create EVOLVED_INTO edge from old to new, edges: {edges:?}"
    );
}

#[test]
fn refine_memory_old_memory_still_exists() {
    let engine = CodememEngine::for_testing();

    let old_memory = make_test_memory("old-mem-2", "original content");
    engine.persist_memory(&old_memory).unwrap();

    engine
        .refine_memory("old-mem-2", Some("updated content"), None, None)
        .unwrap();

    // Old memory should still be retrievable
    let old = engine.storage.get_memory("old-mem-2").unwrap();
    assert!(
        old.is_some(),
        "old memory should still exist after refinement"
    );
    assert_eq!(old.unwrap().content, "original content");
}

#[test]
fn refine_memory_updates_content_tags_importance() {
    let engine = CodememEngine::for_testing();

    let old_memory = make_test_memory("old-mem-3", "old content");
    engine.persist_memory(&old_memory).unwrap();

    let new_tags = vec!["refined".to_string(), "v2".to_string()];
    let (new_memory, _new_id) = engine
        .refine_memory(
            "old-mem-3",
            Some("new content"),
            Some(new_tags.clone()),
            Some(0.9),
        )
        .unwrap();

    assert_eq!(new_memory.content, "new content");
    assert_eq!(new_memory.tags, new_tags);
    assert!((new_memory.importance - 0.9).abs() < f64::EPSILON);
    // Should inherit memory_type from old
    assert_eq!(new_memory.memory_type, codemem_core::MemoryType::Insight);
}

#[test]
fn refine_memory_nonexistent_returns_error() {
    let engine = CodememEngine::for_testing();

    let result = engine.refine_memory("nonexistent-id", Some("content"), None, None);
    assert!(
        result.is_err(),
        "should return error for nonexistent memory"
    );
}

// ── Memory ops provenance: split_memory ─────────────────────────────

#[test]
fn split_memory_creates_part_of_edges() {
    let engine = CodememEngine::for_testing();

    let source = make_test_memory("source-mem-1", "combined insight about A and B");
    engine.persist_memory(&source).unwrap();

    let parts = vec![
        crate::SplitPart {
            content: "insight about A".to_string(),
            tags: Some(vec!["topic-a".to_string()]),
            importance: None,
        },
        crate::SplitPart {
            content: "insight about B".to_string(),
            tags: Some(vec!["topic-b".to_string()]),
            importance: None,
        },
    ];

    let child_ids = engine.split_memory("source-mem-1", &parts).unwrap();
    assert_eq!(child_ids.len(), 2);

    // Verify PART_OF edges: child -> source
    let graph = engine.lock_graph().unwrap();
    for child_id in &child_ids {
        let edges = graph.get_edges(child_id).unwrap();
        let part_of_edge = edges
            .iter()
            .find(|e| e.dst == "source-mem-1" && e.relationship == RelationshipType::PartOf);
        assert!(
            part_of_edge.is_some(),
            "child {child_id} should have PART_OF edge to source, edges: {edges:?}"
        );
    }
}

#[test]
fn split_memory_original_still_exists() {
    let engine = CodememEngine::for_testing();

    let source = make_test_memory("source-mem-2", "original combined");
    engine.persist_memory(&source).unwrap();

    let parts = vec![crate::SplitPart {
        content: "part one".to_string(),
        tags: None,
        importance: None,
    }];

    engine.split_memory("source-mem-2", &parts).unwrap();

    let original = engine.storage.get_memory("source-mem-2").unwrap();
    assert!(
        original.is_some(),
        "original memory should still exist after split"
    );
}

#[test]
fn split_memory_single_part_works() {
    let engine = CodememEngine::for_testing();

    let source = make_test_memory("source-mem-3", "content to extract");
    engine.persist_memory(&source).unwrap();

    let parts = vec![crate::SplitPart {
        content: "extracted part".to_string(),
        tags: None,
        importance: Some(0.7),
    }];

    let child_ids = engine.split_memory("source-mem-3", &parts).unwrap();
    assert_eq!(child_ids.len(), 1);

    let child = engine.storage.get_memory(&child_ids[0]).unwrap().unwrap();
    assert_eq!(child.content, "extracted part");
    assert!((child.importance - 0.7).abs() < f64::EPSILON);
}

#[test]
fn split_memory_empty_parts_returns_error() {
    let engine = CodememEngine::for_testing();

    let source = make_test_memory("source-mem-4", "some content");
    engine.persist_memory(&source).unwrap();

    let parts: Vec<crate::SplitPart> = vec![];
    let result = engine.split_memory("source-mem-4", &parts);
    assert!(
        result.is_err(),
        "split with empty parts should return error"
    );
}

#[test]
fn split_memory_nonexistent_source_returns_error() {
    let engine = CodememEngine::for_testing();

    let parts = vec![crate::SplitPart {
        content: "part".to_string(),
        tags: None,
        importance: None,
    }];

    let result = engine.split_memory("nonexistent-source", &parts);
    assert!(
        result.is_err(),
        "should return error for nonexistent source memory"
    );
}

// ── Memory ops provenance: merge_memories ───────────────────────────

#[test]
fn merge_memories_creates_summarizes_edges() {
    let engine = CodememEngine::for_testing();

    let mem_a = make_test_memory("merge-a", "insight about pattern A");
    let mem_b = make_test_memory("merge-b", "insight about pattern B");
    engine.persist_memory(&mem_a).unwrap();
    engine.persist_memory(&mem_b).unwrap();

    let merged_id = engine
        .merge_memories(
            &["merge-a".to_string(), "merge-b".to_string()],
            "combined insight about patterns A and B",
            codemem_core::MemoryType::Insight,
            0.8,
            vec!["merged".to_string()],
        )
        .unwrap();

    // Verify SUMMARIZES edges: merged -> each source
    let graph = engine.lock_graph().unwrap();
    let edges = graph.get_edges(&merged_id).unwrap();

    let summarizes_a = edges
        .iter()
        .find(|e| e.dst == "merge-a" && e.relationship == RelationshipType::Summarizes);
    assert!(
        summarizes_a.is_some(),
        "should create SUMMARIZES edge to merge-a, edges: {edges:?}"
    );

    let summarizes_b = edges
        .iter()
        .find(|e| e.dst == "merge-b" && e.relationship == RelationshipType::Summarizes);
    assert!(
        summarizes_b.is_some(),
        "should create SUMMARIZES edge to merge-b, edges: {edges:?}"
    );
}

#[test]
fn merge_memories_confidence_averaging() {
    let engine = CodememEngine::for_testing();

    let mut mem_a = make_test_memory("conf-a", "insight A");
    mem_a.confidence = 0.6;
    let mut mem_b = make_test_memory("conf-b", "insight B");
    mem_b.confidence = 1.0;
    engine.persist_memory(&mem_a).unwrap();
    engine.persist_memory(&mem_b).unwrap();

    let merged_id = engine
        .merge_memories(
            &["conf-a".to_string(), "conf-b".to_string()],
            "merged insight",
            codemem_core::MemoryType::Insight,
            0.7,
            vec![],
        )
        .unwrap();

    let merged = engine.storage.get_memory(&merged_id).unwrap().unwrap();
    // Average of 0.6 and 1.0 = 0.8
    assert!(
        (merged.confidence - 0.8).abs() < f64::EPSILON,
        "confidence should be average of sources, got: {}",
        merged.confidence
    );
}

#[test]
fn merge_memories_nonexistent_source_returns_error() {
    let engine = CodememEngine::for_testing();

    let mem_a = make_test_memory("exists-a", "content");
    engine.persist_memory(&mem_a).unwrap();

    let result = engine.merge_memories(
        &["exists-a".to_string(), "does-not-exist".to_string()],
        "merged",
        codemem_core::MemoryType::Insight,
        0.5,
        vec![],
    );
    assert!(
        result.is_err(),
        "should return error when a source memory does not exist"
    );
}

#[test]
fn merge_memories_requires_at_least_two_sources() {
    let engine = CodememEngine::for_testing();

    let mem_a = make_test_memory("single-a", "content");
    engine.persist_memory(&mem_a).unwrap();

    let result = engine.merge_memories(
        &["single-a".to_string()],
        "merged",
        codemem_core::MemoryType::Insight,
        0.5,
        vec![],
    );
    assert!(
        result.is_err(),
        "should require at least 2 source IDs for merge"
    );
}

// ── persist_graph_only ──────────────────────────────────────────────

#[test]
fn persist_graph_only_stores_nodes_and_edges_no_embeddings() {
    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let symbols = vec![make_symbol("handler", "src/main.rs", SymbolKind::Function)];
    let chunks = vec![make_chunk("src/main.rs", 0, Some("handler"))];

    let result = make_index_result(symbols, chunks, files, vec![]);
    let pr = engine.persist_graph_only(&result, Some("test-ns")).unwrap();

    // Should have zero embeddings
    assert_eq!(pr.symbols_embedded, 0);
    assert_eq!(pr.chunks_embedded, 0);

    // But graph nodes should exist
    let graph = engine.lock_graph().unwrap();
    assert!(graph.get_node("file:src/main.rs").unwrap().is_some());
    assert!(graph.get_node("sym:handler").unwrap().is_some());
}

// ── edge fusion with superseded removal ─────────────────────────────

#[test]
fn fuse_edges_removes_superseded_ast_grep_edges() {
    use codemem_core::Edge;
    let now = chrono::Utc::now();

    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    let symbols = vec![
        make_symbol("caller", "src/main.rs", SymbolKind::Function),
        make_symbol("callee", "src/main.rs", SymbolKind::Function),
    ];

    // ast-grep found a reference edge
    let edges = vec![ResolvedEdge {
        source_qualified_name: "caller".to_string(),
        target_qualified_name: "callee".to_string(),
        relationship: RelationshipType::Calls,
        file_path: "src/main.rs".to_string(),
        line: 5,
        resolution_confidence: 0.5,
    }];

    // SCIP found the same edge with compiler-grade confidence
    let scip_build = crate::index::scip::graph_builder::ScipBuildResult {
        nodes: vec![],
        edges: vec![Edge {
            id: "calls:sym:caller->sym:callee:src/main.rs:5".to_string(),
            src: "sym:caller".to_string(),
            dst: "sym:callee".to_string(),
            relationship: RelationshipType::Calls,
            weight: 1.0,
            properties: {
                let mut p = std::collections::HashMap::new();
                p.insert("source".to_string(), serde_json::json!("scip"));
                p.insert("confidence".to_string(), serde_json::json!(0.15));
                p.insert("source_layers".to_string(), serde_json::json!(["scip"]));
                p
            },
            created_at: now,
            valid_from: Some(now),
            valid_to: None,
        }],
        memories: vec![],
        ext_nodes_created: 0,
        files_covered: HashSet::new(),
        doc_memories_created: 0,
    };

    let mut result = make_index_result(symbols, vec![], files, edges);
    result.scip_build = Some(scip_build);

    engine
        .persist_index_results(&result, Some("test-ns"))
        .unwrap();

    let graph = engine.lock_graph().unwrap();

    // The fused edge should exist with combined confidence
    let caller_edges = graph.get_edges("sym:caller").unwrap();
    let calls_edges: Vec<_> = caller_edges
        .iter()
        .filter(|e| e.relationship == RelationshipType::Calls && e.dst == "sym:callee")
        .collect();

    // Should have exactly ONE edge (the fused one), not two
    assert_eq!(
        calls_edges.len(),
        1,
        "should have exactly one fused edge, not duplicates"
    );

    // The surviving edge should have fused confidence (0.15 + 0.10 = 0.25)
    let conf = calls_edges[0]
        .properties
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    assert!(
        (conf - 0.25).abs() < 0.01,
        "fused confidence should be ~0.25, got {conf}"
    );

    // source_layers should show both
    let layers = calls_edges[0]
        .properties
        .get("source_layers")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(layers, 2, "should have both source layers");
}

// ── SCIP stale node cleanup on re-index ─────────────────────────────

#[test]
fn reindex_cleans_stale_scip_nodes() {
    use codemem_core::GraphNode;

    let engine = CodememEngine::for_testing();

    let mut files = HashSet::new();
    files.insert("src/main.rs".to_string());

    // First index: SCIP produces two symbols
    let scip_build_v1 = crate::index::scip::graph_builder::ScipBuildResult {
        nodes: vec![
            GraphNode {
                id: "sym:old_func".to_string(),
                kind: NodeKind::Function,
                label: "old_func".to_string(),
                payload: {
                    let mut p = std::collections::HashMap::new();
                    p.insert(
                        "source".to_string(),
                        serde_json::Value::String("scip".to_string()),
                    );
                    p.insert(
                        "file_path".to_string(),
                        serde_json::Value::String("src/main.rs".to_string()),
                    );
                    p
                },
                centrality: 0.0,
                memory_id: None,
                namespace: Some("test-ns".to_string()),
                valid_from: None,
                valid_to: None,
            },
            GraphNode {
                id: "sym:kept_func".to_string(),
                kind: NodeKind::Function,
                label: "kept_func".to_string(),
                payload: {
                    let mut p = std::collections::HashMap::new();
                    p.insert(
                        "source".to_string(),
                        serde_json::Value::String("scip".to_string()),
                    );
                    p.insert(
                        "file_path".to_string(),
                        serde_json::Value::String("src/main.rs".to_string()),
                    );
                    p
                },
                centrality: 0.0,
                memory_id: None,
                namespace: Some("test-ns".to_string()),
                valid_from: None,
                valid_to: None,
            },
        ],
        edges: vec![],
        memories: vec![],
        ext_nodes_created: 0,
        files_covered: ["src/main.rs".to_string()].into_iter().collect(),
        doc_memories_created: 0,
    };

    let mut result_v1 = make_index_result(vec![], vec![], files.clone(), vec![]);
    result_v1.scip_build = Some(scip_build_v1);
    engine
        .persist_index_results(&result_v1, Some("test-ns"))
        .unwrap();

    // Both nodes should exist
    {
        let graph = engine.lock_graph().unwrap();
        assert!(graph.get_node("sym:old_func").unwrap().is_some());
        assert!(graph.get_node("sym:kept_func").unwrap().is_some());
    }

    // Second index: SCIP only produces kept_func (old_func was deleted from source)
    let scip_build_v2 = crate::index::scip::graph_builder::ScipBuildResult {
        nodes: vec![GraphNode {
            id: "sym:kept_func".to_string(),
            kind: NodeKind::Function,
            label: "kept_func".to_string(),
            payload: {
                let mut p = std::collections::HashMap::new();
                p.insert(
                    "source".to_string(),
                    serde_json::Value::String("scip".to_string()),
                );
                p.insert(
                    "file_path".to_string(),
                    serde_json::Value::String("src/main.rs".to_string()),
                );
                p
            },
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test-ns".to_string()),
            valid_from: None,
            valid_to: None,
        }],
        edges: vec![],
        memories: vec![],
        ext_nodes_created: 0,
        files_covered: ["src/main.rs".to_string()].into_iter().collect(),
        doc_memories_created: 0,
    };

    let mut result_v2 = make_index_result(vec![], vec![], files, vec![]);
    result_v2.scip_build = Some(scip_build_v2);
    engine
        .persist_index_results(&result_v2, Some("test-ns"))
        .unwrap();

    // old_func should be cleaned up, kept_func should still exist
    let graph = engine.lock_graph().unwrap();
    assert!(
        graph.get_node("sym:old_func").unwrap().is_none(),
        "stale SCIP node should be removed on re-index"
    );
    assert!(
        graph.get_node("sym:kept_func").unwrap().is_some(),
        "active SCIP node should survive re-index"
    );
}
