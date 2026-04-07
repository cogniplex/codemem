use crate::api::types::*;
use std::collections::HashMap;

// ── StatsResponse ───────────────────────────────────────────────────────────

#[test]
fn stats_response_serializes_to_json() {
    let stats = StatsResponse {
        memory_count: 42,
        embedding_count: 10,
        node_count: 100,
        edge_count: 200,
        session_count: 3,
        namespace_count: 2,
    };

    let json = serde_json::to_value(&stats).unwrap();
    assert_eq!(json["memory_count"], 42);
    assert_eq!(json["embedding_count"], 10);
    assert_eq!(json["node_count"], 100);
    assert_eq!(json["edge_count"], 200);
    assert_eq!(json["session_count"], 3);
    assert_eq!(json["namespace_count"], 2);
}

// ── HealthResponse ──────────────────────────────────────────────────────────

#[test]
fn health_response_serializes_all_components() {
    let health = HealthResponse {
        storage: ComponentHealth {
            status: "ok".to_string(),
            detail: None,
        },
        vector: ComponentHealth {
            status: "ok".to_string(),
            detail: Some("768 vectors".to_string()),
        },
        graph: ComponentHealth {
            status: "error".to_string(),
            detail: Some("lock poisoned".to_string()),
        },
        embeddings: ComponentHealth {
            status: "unavailable".to_string(),
            detail: Some("No provider configured".to_string()),
        },
    };

    let json = serde_json::to_value(&health).unwrap();
    assert_eq!(json["storage"]["status"], "ok");
    assert!(json["storage"]["detail"].is_null());
    assert_eq!(json["vector"]["detail"], "768 vectors");
    assert_eq!(json["graph"]["status"], "error");
    assert_eq!(json["embeddings"]["status"], "unavailable");
}

// ── MemoryItem ──────────────────────────────────────────────────────────────

#[test]
fn memory_item_serializes_correctly() {
    let item = MemoryItem {
        id: "mem-1".to_string(),
        content: "Test memory content".to_string(),
        memory_type: "insight".to_string(),
        importance: 0.8,
        confidence: 0.95,
        access_count: 5,
        tags: vec!["rust".to_string(), "testing".to_string()],
        namespace: Some("my-project".to_string()),
        created_at: "2025-01-01T00:00:00Z".to_string(),
        updated_at: "2025-01-02T00:00:00Z".to_string(),
    };

    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(json["id"], "mem-1");
    assert_eq!(json["memory_type"], "insight");
    assert_eq!(json["importance"], 0.8);
    assert_eq!(json["confidence"], 0.95);
    assert_eq!(json["access_count"], 5);
    assert_eq!(json["tags"].as_array().unwrap().len(), 2);
    assert_eq!(json["namespace"], "my-project");
}

#[test]
fn memory_item_with_null_namespace() {
    let item = MemoryItem {
        id: "mem-2".to_string(),
        content: "No namespace".to_string(),
        memory_type: "context".to_string(),
        importance: 0.5,
        confidence: 1.0,
        access_count: 0,
        tags: vec![],
        namespace: None,
        created_at: "2025-01-01T00:00:00Z".to_string(),
        updated_at: "2025-01-01T00:00:00Z".to_string(),
    };

    let json = serde_json::to_value(&item).unwrap();
    assert!(json["namespace"].is_null());
    assert!(json["tags"].as_array().unwrap().is_empty());
}

// ── MemoryListResponse ──────────────────────────────────────────────────────

#[test]
fn memory_list_response_serializes_pagination() {
    let resp = MemoryListResponse {
        memories: vec![],
        total: 100,
        offset: 20,
        limit: 10,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["total"], 100);
    assert_eq!(json["offset"], 20);
    assert_eq!(json["limit"], 10);
    assert!(json["memories"].as_array().unwrap().is_empty());
}

// ── SearchResultItem & ScoreBreakdown ───────────────────────────────────────

#[test]
fn search_result_item_serializes_score_breakdown() {
    let item = SearchResultItem {
        id: "sr-1".to_string(),
        content: "Search result".to_string(),
        memory_type: "decision".to_string(),
        score: 0.92,
        score_breakdown: ScoreBreakdownResponse {
            vector_similarity: 0.25,
            graph_strength: 0.20,
            token_overlap: 0.15,
            temporal: 0.10,
            tag_matching: 0.05,
            importance: 0.10,
            confidence: 0.05,
            recency: 0.02,
        },
        tags: vec!["arch".to_string()],
        namespace: None,
    };

    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(json["score"], 0.92);
    assert_eq!(json["score_breakdown"]["vector_similarity"], 0.25);
    assert_eq!(json["score_breakdown"]["graph_strength"], 0.20);
    assert_eq!(json["score_breakdown"]["token_overlap"], 0.15);
    assert_eq!(json["score_breakdown"]["temporal"], 0.10);
    assert_eq!(json["score_breakdown"]["tag_matching"], 0.05);
    assert_eq!(json["score_breakdown"]["importance"], 0.10);
    assert_eq!(json["score_breakdown"]["confidence"], 0.05);
    assert_eq!(json["score_breakdown"]["recency"], 0.02);
}

// ── Graph types ─────────────────────────────────────────────────────────────

#[test]
fn graph_node_response_serialization() {
    let node = GraphNodeResponse {
        id: "node-1".to_string(),
        kind: "File".to_string(),
        label: "src/main.rs".to_string(),
        centrality: 0.85,
        memory_id: Some("mem-123".to_string()),
        namespace: Some("codemem".to_string()),
        payload: std::collections::HashMap::new(),
    };

    let json = serde_json::to_value(&node).unwrap();
    assert_eq!(json["kind"], "File");
    assert_eq!(json["centrality"], 0.85);
    assert_eq!(json["memory_id"], "mem-123");
}

#[test]
fn graph_edge_response_serialization() {
    let edge = GraphEdgeResponse {
        id: "edge-1".to_string(),
        src: "node-1".to_string(),
        dst: "node-2".to_string(),
        relationship: "IMPORTS".to_string(),
        weight: 1.0,
    };

    let json = serde_json::to_value(&edge).unwrap();
    assert_eq!(json["src"], "node-1");
    assert_eq!(json["dst"], "node-2");
    assert_eq!(json["relationship"], "IMPORTS");
    assert_eq!(json["weight"], 1.0);
}

#[test]
fn subgraph_response_empty() {
    let resp = SubgraphResponse {
        nodes: vec![],
        edges: vec![],
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert!(json["nodes"].as_array().unwrap().is_empty());
    assert!(json["edges"].as_array().unwrap().is_empty());
}

#[test]
fn communities_response_serialization() {
    let mut communities = HashMap::new();
    communities.insert("node-a".to_string(), 0);
    communities.insert("node-b".to_string(), 0);
    communities.insert("node-c".to_string(), 1);

    let resp = CommunitiesResponse {
        communities,
        num_communities: 2,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["num_communities"], 2);
    assert_eq!(json["communities"]["node-a"], 0);
    assert_eq!(json["communities"]["node-c"], 1);
}

#[test]
fn pagerank_response_serialization() {
    let resp = PagerankResponse {
        scores: vec![
            PagerankEntry {
                node_id: "n1".to_string(),
                label: "main.rs".to_string(),
                score: 0.15,
            },
            PagerankEntry {
                node_id: "n2".to_string(),
                label: "lib.rs".to_string(),
                score: 0.10,
            },
        ],
    };

    let json = serde_json::to_value(&resp).unwrap();
    let scores = json["scores"].as_array().unwrap();
    assert_eq!(scores.len(), 2);
    assert_eq!(scores[0]["node_id"], "n1");
    assert_eq!(scores[0]["score"], 0.15);
}

// ── Browse types ────────────────────────────────────────────────────────────

#[test]
fn browse_node_item_serialization() {
    let item = BrowseNodeItem {
        id: "b1".to_string(),
        kind: "Function".to_string(),
        label: "process_data".to_string(),
        centrality: 0.5,
        namespace: Some("my-ns".to_string()),
        degree: 12,
    };

    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(json["degree"], 12);
    assert_eq!(json["kind"], "Function");
}

#[test]
fn browse_response_with_kind_counts() {
    let mut kinds = HashMap::new();
    kinds.insert("File".to_string(), 50);
    kinds.insert("Function".to_string(), 200);

    let resp = BrowseResponse {
        nodes: vec![],
        total: 250,
        kinds,
        edge_count: 500,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["total"], 250);
    assert_eq!(json["edge_count"], 500);
    assert_eq!(json["kinds"]["File"], 50);
    assert_eq!(json["kinds"]["Function"], 200);
}

// ── Vector types ────────────────────────────────────────────────────────────

#[test]
fn vector_point_serialization() {
    let point = VectorPoint {
        id: "v1".to_string(),
        x: 1.5,
        y: -2.3,
        z: 0.7,
        memory_type: "insight".to_string(),
        importance: 0.9,
        namespace: None,
        label: "A memory about Rust".to_string(),
    };

    let json = serde_json::to_value(&point).unwrap();
    assert_eq!(json["x"], 1.5);
    assert_eq!(json["y"], -2.3);
    assert_eq!(json["z"], 0.7);
    assert_eq!(json["memory_type"], "insight");
    assert_eq!(json["importance"], 0.9);
}

// ── Namespace types ─────────────────────────────────────────────────────────

#[test]
fn namespace_item_serialization() {
    let item = NamespaceItem {
        name: "test-project".to_string(),
        memory_count: 42,
    };

    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(json["name"], "test-project");
    assert_eq!(json["memory_count"], 42);
}

#[test]
fn namespace_stats_response_serialization() {
    let mut type_dist = HashMap::new();
    type_dist.insert("insight".to_string(), 10);
    type_dist.insert("decision".to_string(), 5);

    let resp = NamespaceStatsResponse {
        namespace: "my-ns".to_string(),
        memory_count: 15,
        avg_importance: 0.7,
        avg_confidence: 0.9,
        type_distribution: type_dist,
        tag_frequency: HashMap::new(),
        oldest: None,
        newest: None,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["namespace"], "my-ns");
    assert_eq!(json["memory_count"], 15);
    assert_eq!(json["type_distribution"]["insight"], 10);
}

// ── Session types ───────────────────────────────────────────────────────────

#[test]
fn session_response_serialization() {
    let resp = SessionResponse {
        id: "sess-1".to_string(),
        namespace: Some("ns".to_string()),
        started_at: "2025-01-01T00:00:00Z".to_string(),
        ended_at: None,
        memory_count: 3,
        summary: None,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["id"], "sess-1");
    assert!(json["ended_at"].is_null());
    assert!(json["summary"].is_null());
    assert_eq!(json["memory_count"], 3);
}

// ── Timeline & Distribution types ───────────────────────────────────────────

#[test]
fn timeline_bucket_serialization() {
    let mut counts = HashMap::new();
    counts.insert("insight".to_string(), 5);
    counts.insert("decision".to_string(), 2);

    let bucket = TimelineBucket {
        date: "2025-03-01".to_string(),
        counts,
        total: 7,
    };

    let json = serde_json::to_value(&bucket).unwrap();
    assert_eq!(json["date"], "2025-03-01");
    assert_eq!(json["total"], 7);
    assert_eq!(json["counts"]["insight"], 5);
}

#[test]
fn distribution_response_serialization() {
    let mut type_counts = HashMap::new();
    type_counts.insert("insight".to_string(), 30);

    let resp = DistributionResponse {
        type_counts,
        importance_histogram: vec![0, 0, 1, 5, 10, 8, 4, 2, 0, 0],
        total: 30,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["total"], 30);
    let hist = json["importance_histogram"].as_array().unwrap();
    assert_eq!(hist.len(), 10);
    assert_eq!(hist[4], 10);
}

// ── Pattern & Consolidation types ───────────────────────────────────────────

#[test]
fn pattern_response_serialization() {
    let resp = PatternResponse {
        pattern_type: "file_hotspot".to_string(),
        description: "src/main.rs accessed frequently".to_string(),
        frequency: 42,
        confidence: 0.85,
        related_memories: vec!["m1".to_string(), "m2".to_string()],
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["pattern_type"], "file_hotspot");
    assert_eq!(json["frequency"], 42);
    assert_eq!(json["confidence"], 0.85);
    assert_eq!(json["related_memories"].as_array().unwrap().len(), 2);
}

#[test]
fn consolidation_status_response_serialization() {
    let resp = ConsolidationStatusResponse {
        cycles: vec![
            ConsolidationCycleStatus {
                cycle: "decay".to_string(),
                last_run: Some("2025-01-01T00:00:00Z".to_string()),
                affected_count: 10,
            },
            ConsolidationCycleStatus {
                cycle: "creative".to_string(),
                last_run: None,
                affected_count: 0,
            },
        ],
    };

    let json = serde_json::to_value(&resp).unwrap();
    let cycles = json["cycles"].as_array().unwrap();
    assert_eq!(cycles.len(), 2);
    assert_eq!(cycles[0]["cycle"], "decay");
    assert_eq!(cycles[0]["affected_count"], 10);
    assert!(cycles[1]["last_run"].is_null());
}

// ── Metrics & Config types ──────────────────────────────────────────────────

#[test]
fn metrics_response_serialization() {
    let mut latency = HashMap::new();
    latency.insert("recall_p50".to_string(), 5.2);
    latency.insert("recall_p95".to_string(), 12.1);

    let resp = MetricsResponse {
        tool_calls_total: 500,
        latency_percentiles: latency,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["tool_calls_total"], 500);
    assert_eq!(json["latency_percentiles"]["recall_p50"], 5.2);
}

// ── Agent / Recipe types ────────────────────────────────────────────────────

#[test]
fn recipe_list_response_serialization() {
    let resp = RecipeListResponse {
        id: "full-analysis".to_string(),
        name: "Full Analysis".to_string(),
        description: "Run everything".to_string(),
        steps: vec![
            RecipeStep {
                tool: "index_codebase".to_string(),
                description: "Index all files".to_string(),
            },
            RecipeStep {
                tool: "detect_patterns".to_string(),
                description: "Find patterns".to_string(),
            },
        ],
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["id"], "full-analysis");
    let steps = json["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["tool"], "index_codebase");
}

// ── Insight types ───────────────────────────────────────────────────────────

#[test]
fn activity_insights_response_serialization() {
    let resp = ActivityInsightsResponse {
        insights: vec![],
        git_summary: GitSummary {
            total_annotated_files: 50,
            top_authors: vec!["Alice".to_string(), "Bob".to_string()],
        },
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["git_summary"]["total_annotated_files"], 50);
    assert_eq!(
        json["git_summary"]["top_authors"].as_array().unwrap().len(),
        2
    );
}

#[test]
fn security_insights_response_serialization() {
    let resp = SecurityInsightsResponse {
        insights: vec![],
        sensitive_file_count: 3,
        endpoint_count: 15,
        security_function_count: 7,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["sensitive_file_count"], 3);
    assert_eq!(json["endpoint_count"], 15);
    assert_eq!(json["security_function_count"], 7);
}

#[test]
fn performance_insights_response_serialization() {
    let resp = PerformanceInsightsResponse {
        insights: vec![],
        high_coupling_nodes: vec![CouplingNode {
            node_id: "n1".to_string(),
            label: "core.rs".to_string(),
            coupling_score: 25,
        }],
        max_depth: 8,
        critical_path: vec![],
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["max_depth"], 8);
    let coupling = json["high_coupling_nodes"].as_array().unwrap();
    assert_eq!(coupling.len(), 1);
    assert_eq!(coupling[0]["coupling_score"], 25);
}

#[test]
fn code_health_insights_response_serialization() {
    let resp = CodeHealthInsightsResponse {
        insights: vec![],
        file_hotspots: vec![],
        decision_chains: vec![],
        pagerank_leaders: vec![],
        community_count: 5,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["community_count"], 5);
    assert!(json["file_hotspots"].as_array().unwrap().is_empty());
}

// ── Generic types ───────────────────────────────────────────────────────────

#[test]
fn message_response_serialization() {
    let resp = MessageResponse {
        message: "Operation completed".to_string(),
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["message"], "Operation completed");
}

#[test]
fn id_response_serialization() {
    let resp = IdResponse {
        id: "abc-123".to_string(),
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["id"], "abc-123");
}

// ── Deserialization tests ───────────────────────────────────────────────────

#[test]
fn memory_list_query_deserializes_with_type_rename() {
    let json = serde_json::json!({
        "namespace": "ns1",
        "type": "insight",
        "offset": 10,
        "limit": 20,
        "sort": "importance"
    });

    let query: MemoryListQuery = serde_json::from_value(json).unwrap();
    assert_eq!(query.namespace.as_deref(), Some("ns1"));
    assert_eq!(query.memory_type.as_deref(), Some("insight"));
    assert_eq!(query.offset, Some(10));
    assert_eq!(query.limit, Some(20));
    assert_eq!(query.sort.as_deref(), Some("importance"));
}

#[test]
fn memory_list_query_deserializes_with_all_optional() {
    let json = serde_json::json!({});
    let query: MemoryListQuery = serde_json::from_value(json).unwrap();
    assert!(query.namespace.is_none());
    assert!(query.memory_type.is_none());
    assert!(query.offset.is_none());
    assert!(query.limit.is_none());
}

#[test]
fn store_memory_request_deserializes() {
    let json = serde_json::json!({
        "content": "Test memory",
        "memory_type": "decision",
        "importance": 0.9,
        "tags": ["rust", "api"],
        "namespace": "test-ns"
    });

    let req: StoreMemoryRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.content, "Test memory");
    assert_eq!(req.memory_type.as_deref(), Some("decision"));
    assert_eq!(req.importance, Some(0.9));
    assert_eq!(req.tags.as_ref().unwrap().len(), 2);
}

#[test]
fn store_memory_request_minimal() {
    let json = serde_json::json!({ "content": "minimal" });
    let req: StoreMemoryRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.content, "minimal");
    assert!(req.memory_type.is_none());
    assert!(req.importance.is_none());
    assert!(req.tags.is_none());
    assert!(req.namespace.is_none());
}

#[test]
fn search_query_deserializes_with_type_rename() {
    let json = serde_json::json!({
        "q": "rust ownership",
        "type": "insight",
        "k": 5
    });

    let query: SearchQuery = serde_json::from_value(json).unwrap();
    assert_eq!(query.q, "rust ownership");
    assert_eq!(query.memory_type.as_deref(), Some("insight"));
    assert_eq!(query.k, Some(5));
}

#[test]
fn scoring_weights_update_deserializes_partial() {
    let json = serde_json::json!({
        "vector_similarity": 0.3,
        "graph_strength": 0.25
    });

    let update: ScoringWeightsUpdate = serde_json::from_value(json).unwrap();
    assert_eq!(update.vector_similarity, Some(0.3));
    assert_eq!(update.graph_strength, Some(0.25));
    assert!(update.token_overlap.is_none());
    assert!(update.temporal.is_none());
    assert!(update.tag_matching.is_none());
    assert!(update.importance.is_none());
    assert!(update.confidence.is_none());
    assert!(update.recency.is_none());
}

#[test]
fn update_memory_request_deserializes() {
    let json = serde_json::json!({
        "content": "Updated content",
        "importance": 0.7
    });

    let req: UpdateMemoryRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.content.as_deref(), Some("Updated content"));
    assert_eq!(req.importance, Some(0.7));
}

#[test]
fn run_recipe_request_deserializes() {
    let json = serde_json::json!({
        "recipe": "full-analysis",
        "repo_id": "repo-1",
        "namespace": "my-ns"
    });

    let req: RunRecipeRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.recipe, "full-analysis");
    assert_eq!(req.repo_id.as_deref(), Some("repo-1"));
    assert_eq!(req.namespace.as_deref(), Some("my-ns"));
}
