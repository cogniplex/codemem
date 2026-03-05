use super::*;

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
