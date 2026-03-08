use super::*;
use chrono::Utc;
use std::collections::HashMap;

#[test]
fn memory_type_roundtrip() {
    for mt in [
        MemoryType::Decision,
        MemoryType::Pattern,
        MemoryType::Preference,
        MemoryType::Style,
        MemoryType::Habit,
        MemoryType::Insight,
        MemoryType::Context,
    ] {
        let s = mt.to_string();
        let parsed: MemoryType = s.parse().unwrap();
        assert_eq!(mt, parsed);
    }
}

#[test]
fn relationship_type_roundtrip() {
    for rt in [
        RelationshipType::RelatesTo,
        RelationshipType::LeadsTo,
        RelationshipType::PartOf,
        RelationshipType::Reinforces,
        RelationshipType::Contradicts,
        RelationshipType::EvolvedInto,
        RelationshipType::DerivedFrom,
        RelationshipType::InvalidatedBy,
        RelationshipType::DependsOn,
        RelationshipType::Imports,
        RelationshipType::Extends,
        RelationshipType::Calls,
        RelationshipType::Contains,
        RelationshipType::Supersedes,
        RelationshipType::Blocks,
        RelationshipType::Implements,
        RelationshipType::Inherits,
        RelationshipType::SimilarTo,
        RelationshipType::PrecededBy,
        RelationshipType::Exemplifies,
        RelationshipType::Explains,
        RelationshipType::SharesTheme,
        RelationshipType::Summarizes,
    ] {
        let s = rt.to_string();
        let parsed: RelationshipType = s.parse().unwrap();
        assert_eq!(rt, parsed);
    }
}

#[test]
fn relationship_type_co_changed_roundtrip() {
    let rt = RelationshipType::CoChanged;
    let s = rt.to_string();
    assert_eq!(s, "CO_CHANGED");
    let parsed: RelationshipType = s.parse().unwrap();
    assert_eq!(parsed, RelationshipType::CoChanged);

    // Also verify JSON round-trip
    let json = serde_json::to_string(&rt).unwrap();
    let from_json: RelationshipType = serde_json::from_str(&json).unwrap();
    assert_eq!(from_json, RelationshipType::CoChanged);
}

#[test]
fn node_kind_roundtrip() {
    for nk in [
        NodeKind::File,
        NodeKind::Package,
        NodeKind::Function,
        NodeKind::Class,
        NodeKind::Module,
        NodeKind::Memory,
        NodeKind::Method,
        NodeKind::Interface,
        NodeKind::Type,
        NodeKind::Constant,
        NodeKind::Endpoint,
        NodeKind::Test,
    ] {
        let s = nk.to_string();
        let parsed: NodeKind = s.parse().unwrap();
        assert_eq!(nk, parsed);
    }
}

#[test]
fn node_kind_chunk_roundtrip() {
    let nk = NodeKind::Chunk;
    let s = nk.to_string();
    assert_eq!(s, "chunk");
    let parsed: NodeKind = s.parse().unwrap();
    assert_eq!(parsed, NodeKind::Chunk);

    // Also verify JSON round-trip
    let json = serde_json::to_string(&nk).unwrap();
    let from_json: NodeKind = serde_json::from_str(&json).unwrap();
    assert_eq!(from_json, NodeKind::Chunk);
}

#[test]
fn default_vector_config() {
    let config = VectorConfig::default();
    assert_eq!(config.dimensions, 768);
    assert_eq!(config.m, 16);
    assert_eq!(config.ef_construction, 200);
    assert_eq!(config.ef_search, 100);
}

#[test]
fn scoring_weights_default_sum_to_one() {
    let weights = ScoringWeights::default();
    let sum = weights.vector_similarity
        + weights.graph_strength
        + weights.token_overlap
        + weights.temporal
        + weights.tag_matching
        + weights.importance
        + weights.confidence
        + weights.recency;
    assert!((sum - 1.0).abs() < f64::EPSILON);
}

#[test]
fn total_with_weights_custom() {
    let breakdown = ScoreBreakdown {
        vector_similarity: 1.0,
        graph_strength: 0.0,
        token_overlap: 0.0,
        temporal: 0.0,
        tag_matching: 0.0,
        importance: 0.0,
        confidence: 0.0,
        recency: 0.0,
    };
    // Weight only vector_similarity at 1.0, rest 0.0
    let weights = ScoringWeights {
        vector_similarity: 1.0,
        graph_strength: 0.0,
        token_overlap: 0.0,
        temporal: 0.0,
        tag_matching: 0.0,
        importance: 0.0,
        confidence: 0.0,
        recency: 0.0,
    };
    let total = breakdown.total_with_weights(&weights);
    assert!((total - 1.0).abs() < f64::EPSILON);
}

#[test]
fn score_breakdown_all_zeros() {
    let breakdown = ScoreBreakdown::default();
    let weights = ScoringWeights::default();
    let total = breakdown.total_with_weights(&weights);
    assert!((total - 0.0).abs() < f64::EPSILON);
}

#[test]
fn score_breakdown_equal_weights() {
    let breakdown = ScoreBreakdown {
        vector_similarity: 1.0,
        graph_strength: 1.0,
        token_overlap: 1.0,
        temporal: 1.0,
        tag_matching: 1.0,
        importance: 1.0,
        confidence: 1.0,
        recency: 1.0,
    };
    let weights = ScoringWeights {
        vector_similarity: 0.125,
        graph_strength: 0.125,
        token_overlap: 0.125,
        temporal: 0.125,
        tag_matching: 0.125,
        importance: 0.125,
        confidence: 0.125,
        recency: 0.125,
    };
    let total = breakdown.total_with_weights(&weights);
    assert!((total - 1.0).abs() < f64::EPSILON);
}

#[test]
fn score_breakdown_weights_not_summing_to_one() {
    let breakdown = ScoreBreakdown {
        vector_similarity: 1.0,
        graph_strength: 1.0,
        token_overlap: 0.0,
        temporal: 0.0,
        tag_matching: 0.0,
        importance: 0.0,
        confidence: 0.0,
        recency: 0.0,
    };
    // Weights intentionally sum to 0.5
    let weights = ScoringWeights {
        vector_similarity: 0.3,
        graph_strength: 0.2,
        token_overlap: 0.0,
        temporal: 0.0,
        tag_matching: 0.0,
        importance: 0.0,
        confidence: 0.0,
        recency: 0.0,
    };
    let total = breakdown.total_with_weights(&weights);
    assert!((total - 0.5).abs() < f64::EPSILON);
}

#[test]
fn pattern_type_display() {
    assert_eq!(PatternType::RepeatedSearch.to_string(), "repeated_search");
    assert_eq!(PatternType::FileHotspot.to_string(), "file_hotspot");
    assert_eq!(PatternType::DecisionChain.to_string(), "decision_chain");
    assert_eq!(PatternType::ToolPreference.to_string(), "tool_preference");
}

#[test]
fn detected_pattern_serialization() {
    let pattern = DetectedPattern {
        pattern_type: PatternType::RepeatedSearch,
        description: "Search for 'error handling' appears 5 times".to_string(),
        frequency: 5,
        related_memories: vec!["mem-1".to_string(), "mem-2".to_string()],
        confidence: 0.85,
    };
    let json = serde_json::to_string(&pattern).unwrap();
    let parsed: DetectedPattern = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.pattern_type, PatternType::RepeatedSearch);
    assert_eq!(parsed.frequency, 5);
    assert_eq!(parsed.related_memories.len(), 2);
}

// ── MemoryNode JSON round-trip ──────────────────────────────────────────────

fn make_memory_node() -> MemoryNode {
    let now = Utc::now();
    let mut metadata = HashMap::new();
    metadata.insert(
        "source".to_string(),
        serde_json::Value::String("test".to_string()),
    );
    metadata.insert(
        "count".to_string(),
        serde_json::Value::Number(serde_json::Number::from(42)),
    );

    MemoryNode {
        id: "mem-abc-123".to_string(),
        content: "Use early returns for error handling".to_string(),
        memory_type: MemoryType::Decision,
        importance: 0.8,
        confidence: 0.95,
        access_count: 3,
        content_hash: content_hash("Use early returns for error handling"),
        tags: vec!["rust".to_string(), "error-handling".to_string()],
        metadata,
        namespace: Some("my-project".to_string()),
        session_id: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    }
}

#[test]
fn memory_node_json_roundtrip() {
    let node = make_memory_node();
    let json = serde_json::to_string(&node).unwrap();
    let parsed: MemoryNode = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.id, node.id);
    assert_eq!(parsed.content, node.content);
    assert_eq!(parsed.memory_type, node.memory_type);
    assert!((parsed.importance - node.importance).abs() < f64::EPSILON);
    assert!((parsed.confidence - node.confidence).abs() < f64::EPSILON);
    assert_eq!(parsed.access_count, node.access_count);
    assert_eq!(parsed.content_hash, node.content_hash);
    assert_eq!(parsed.tags, node.tags);
    assert_eq!(parsed.metadata.len(), 2);
    assert_eq!(parsed.namespace, Some("my-project".to_string()));
    assert_eq!(parsed.created_at, node.created_at);
    assert_eq!(parsed.updated_at, node.updated_at);
    assert_eq!(parsed.last_accessed_at, node.last_accessed_at);
}

#[test]
fn memory_node_empty_content() {
    let now = Utc::now();
    let node = MemoryNode {
        id: "mem-empty".to_string(),
        content: String::new(),
        memory_type: MemoryType::Context,
        importance: 0.0,
        confidence: 0.0,
        access_count: 0,
        content_hash: content_hash(""),
        tags: vec![],
        metadata: HashMap::new(),
        namespace: None,
        session_id: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    let json = serde_json::to_string(&node).unwrap();
    let parsed: MemoryNode = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.content, "");
    assert!(parsed.tags.is_empty());
    assert!(parsed.metadata.is_empty());
    assert!(parsed.namespace.is_none());
}

#[test]
fn memory_node_unicode_content() {
    let now = Utc::now();
    let unicode_content =
        "Unicode: \u{1F600} \u{1F680} \u{4E16}\u{754C} caf\u{E9} \u{00FC}ber \u{2603}\u{FE0F}";
    let node = MemoryNode {
        id: "mem-unicode".to_string(),
        content: unicode_content.to_string(),
        memory_type: MemoryType::Insight,
        importance: 0.5,
        confidence: 0.5,
        access_count: 1,
        content_hash: content_hash(unicode_content),
        tags: vec!["\u{1F3F7}\u{FE0F}tag".to_string()],
        metadata: HashMap::new(),
        namespace: None,
        session_id: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    let json = serde_json::to_string(&node).unwrap();
    let parsed: MemoryNode = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.content, unicode_content);
    assert_eq!(parsed.tags[0], "\u{1F3F7}\u{FE0F}tag");
}

#[test]
fn memory_node_empty_tags() {
    let now = Utc::now();
    let node = MemoryNode {
        id: "mem-no-tags".to_string(),
        content: "no tags here".to_string(),
        memory_type: MemoryType::Pattern,
        importance: 0.3,
        confidence: 0.7,
        access_count: 0,
        content_hash: content_hash("no tags here"),
        tags: vec![],
        metadata: HashMap::new(),
        namespace: None,
        session_id: None,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };
    let json = serde_json::to_string(&node).unwrap();
    let parsed: MemoryNode = serde_json::from_str(&json).unwrap();
    assert!(parsed.tags.is_empty());
}

// ── Edge JSON round-trip ────────────────────────────────────────────────────

#[test]
fn edge_json_roundtrip_with_temporal_fields() {
    let now = Utc::now();
    let valid_from = now - chrono::Duration::days(7);
    let valid_to = now + chrono::Duration::days(30);

    let mut props = HashMap::new();
    props.insert(
        "reason".to_string(),
        serde_json::Value::String("refactored".to_string()),
    );

    let edge = Edge {
        id: "edge-1".to_string(),
        src: "node-a".to_string(),
        dst: "node-b".to_string(),
        relationship: RelationshipType::EvolvedInto,
        weight: 0.9,
        properties: props,
        created_at: now,
        valid_from: Some(valid_from),
        valid_to: Some(valid_to),
    };

    let json = serde_json::to_string(&edge).unwrap();
    let parsed: Edge = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.id, "edge-1");
    assert_eq!(parsed.src, "node-a");
    assert_eq!(parsed.dst, "node-b");
    assert_eq!(parsed.relationship, RelationshipType::EvolvedInto);
    assert!((parsed.weight - 0.9).abs() < f64::EPSILON);
    assert_eq!(parsed.valid_from, Some(valid_from));
    assert_eq!(parsed.valid_to, Some(valid_to));
    assert_eq!(parsed.properties.len(), 1);
}

#[test]
fn edge_json_roundtrip_without_temporal_fields() {
    let now = Utc::now();
    let edge = Edge {
        id: "edge-2".to_string(),
        src: "node-x".to_string(),
        dst: "node-y".to_string(),
        relationship: RelationshipType::Contains,
        weight: 0.1,
        properties: HashMap::new(),
        created_at: now,
        valid_from: None,
        valid_to: None,
    };

    let json = serde_json::to_string(&edge).unwrap();
    let parsed: Edge = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.id, "edge-2");
    assert!(parsed.valid_from.is_none());
    assert!(parsed.valid_to.is_none());
}

// ── GraphNode JSON round-trip ───────────────────────────────────────────────

#[test]
fn graph_node_json_roundtrip() {
    let mut payload = HashMap::new();
    payload.insert(
        "file_path".to_string(),
        serde_json::Value::String("src/main.rs".to_string()),
    );
    payload.insert(
        "line_count".to_string(),
        serde_json::Value::Number(serde_json::Number::from(150)),
    );

    let node = GraphNode {
        id: "gn-1".to_string(),
        kind: NodeKind::Function,
        label: "handle_request".to_string(),
        payload,
        centrality: 0.72,
        memory_id: Some("mem-linked".to_string()),
        namespace: Some("api-server".to_string()),
    };

    let json = serde_json::to_string(&node).unwrap();
    let parsed: GraphNode = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.id, "gn-1");
    assert_eq!(parsed.kind, NodeKind::Function);
    assert_eq!(parsed.label, "handle_request");
    assert_eq!(parsed.payload.len(), 2);
    assert!((parsed.centrality - 0.72).abs() < f64::EPSILON);
    assert_eq!(parsed.memory_id, Some("mem-linked".to_string()));
    assert_eq!(parsed.namespace, Some("api-server".to_string()));
}
