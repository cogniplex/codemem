use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::CodememError;

// ── Memory Types ────────────────────────────────────────────────────────────

/// The 7 memory types inspired by AutoMem research.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Architectural and design decisions made during development.
    Decision,
    /// Recurring code patterns observed across files.
    Pattern,
    /// Team/project preferences (e.g., "prefers explicit error types").
    Preference,
    /// Coding style norms (e.g., "early returns, max 20 lines").
    Style,
    /// Workflow habits (e.g., "tests written before implementation").
    Habit,
    /// Cross-domain insights discovered during consolidation.
    Insight,
    /// File contents and structural context from code exploration.
    Context,
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decision => write!(f, "decision"),
            Self::Pattern => write!(f, "pattern"),
            Self::Preference => write!(f, "preference"),
            Self::Style => write!(f, "style"),
            Self::Habit => write!(f, "habit"),
            Self::Insight => write!(f, "insight"),
            Self::Context => write!(f, "context"),
        }
    }
}

impl std::str::FromStr for MemoryType {
    type Err = CodememError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "decision" => Ok(Self::Decision),
            "pattern" => Ok(Self::Pattern),
            "preference" => Ok(Self::Preference),
            "style" => Ok(Self::Style),
            "habit" => Ok(Self::Habit),
            "insight" => Ok(Self::Insight),
            "context" => Ok(Self::Context),
            _ => Err(CodememError::InvalidMemoryType(s.to_string())),
        }
    }
}

// ── Relationship Types ──────────────────────────────────────────────────────

/// 15 relationship types for the knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RelationshipType {
    // General
    RelatesTo,
    LeadsTo,
    PartOf,
    // Knowledge
    Reinforces,
    Contradicts,
    EvolvedInto,
    DerivedFrom,
    InvalidatedBy,
    // Code-specific
    DependsOn,
    Imports,
    Extends,
    Calls,
    Contains,
    Supersedes,
    Blocks,
    // Structural (auto-created by indexing)
    /// Implements interface/trait.
    Implements,
    /// Class inheritance.
    Inherits,
    // Semantic (auto-created by enrichment)
    /// Semantic similarity > threshold.
    SimilarTo,
    /// Temporal adjacency.
    PrecededBy,
    /// Memory exemplifies a pattern.
    Exemplifies,
    /// Insight explains a pattern.
    Explains,
    /// High similarity across types (consolidation).
    SharesTheme,
    /// Meta-memory summarizes a cluster.
    Summarizes,
}

impl std::fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RelatesTo => write!(f, "RELATES_TO"),
            Self::LeadsTo => write!(f, "LEADS_TO"),
            Self::PartOf => write!(f, "PART_OF"),
            Self::Reinforces => write!(f, "REINFORCES"),
            Self::Contradicts => write!(f, "CONTRADICTS"),
            Self::EvolvedInto => write!(f, "EVOLVED_INTO"),
            Self::DerivedFrom => write!(f, "DERIVED_FROM"),
            Self::InvalidatedBy => write!(f, "INVALIDATED_BY"),
            Self::DependsOn => write!(f, "DEPENDS_ON"),
            Self::Imports => write!(f, "IMPORTS"),
            Self::Extends => write!(f, "EXTENDS"),
            Self::Calls => write!(f, "CALLS"),
            Self::Contains => write!(f, "CONTAINS"),
            Self::Supersedes => write!(f, "SUPERSEDES"),
            Self::Blocks => write!(f, "BLOCKS"),
            Self::Implements => write!(f, "IMPLEMENTS"),
            Self::Inherits => write!(f, "INHERITS"),
            Self::SimilarTo => write!(f, "SIMILAR_TO"),
            Self::PrecededBy => write!(f, "PRECEDED_BY"),
            Self::Exemplifies => write!(f, "EXEMPLIFIES"),
            Self::Explains => write!(f, "EXPLAINS"),
            Self::SharesTheme => write!(f, "SHARES_THEME"),
            Self::Summarizes => write!(f, "SUMMARIZES"),
        }
    }
}

impl std::str::FromStr for RelationshipType {
    type Err = CodememError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "RELATES_TO" => Ok(Self::RelatesTo),
            "LEADS_TO" => Ok(Self::LeadsTo),
            "PART_OF" => Ok(Self::PartOf),
            "REINFORCES" => Ok(Self::Reinforces),
            "CONTRADICTS" => Ok(Self::Contradicts),
            "EVOLVED_INTO" => Ok(Self::EvolvedInto),
            "DERIVED_FROM" => Ok(Self::DerivedFrom),
            "INVALIDATED_BY" => Ok(Self::InvalidatedBy),
            "DEPENDS_ON" => Ok(Self::DependsOn),
            "IMPORTS" => Ok(Self::Imports),
            "EXTENDS" => Ok(Self::Extends),
            "CALLS" => Ok(Self::Calls),
            "CONTAINS" => Ok(Self::Contains),
            "SUPERSEDES" => Ok(Self::Supersedes),
            "BLOCKS" => Ok(Self::Blocks),
            "IMPLEMENTS" => Ok(Self::Implements),
            "INHERITS" => Ok(Self::Inherits),
            "SIMILAR_TO" => Ok(Self::SimilarTo),
            "PRECEDED_BY" => Ok(Self::PrecededBy),
            "EXEMPLIFIES" => Ok(Self::Exemplifies),
            "EXPLAINS" => Ok(Self::Explains),
            "SHARES_THEME" => Ok(Self::SharesTheme),
            "SUMMARIZES" => Ok(Self::Summarizes),
            _ => Err(CodememError::InvalidRelationshipType(s.to_string())),
        }
    }
}

// ── Graph Node Types ────────────────────────────────────────────────────────

/// Node types in the knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    File,
    Package,
    Function,
    Class,
    Module,
    Memory,
    /// Class method (distinct from standalone function).
    Method,
    /// TypeScript interface, Go interface, Rust trait.
    Interface,
    /// Type alias, typedef.
    Type,
    /// Const, static, enum variant.
    Constant,
    /// REST/gRPC endpoint definition.
    Endpoint,
    /// Test function.
    Test,
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File => write!(f, "file"),
            Self::Package => write!(f, "package"),
            Self::Function => write!(f, "function"),
            Self::Class => write!(f, "class"),
            Self::Module => write!(f, "module"),
            Self::Memory => write!(f, "memory"),
            Self::Method => write!(f, "method"),
            Self::Interface => write!(f, "interface"),
            Self::Type => write!(f, "type"),
            Self::Constant => write!(f, "constant"),
            Self::Endpoint => write!(f, "endpoint"),
            Self::Test => write!(f, "test"),
        }
    }
}

impl std::str::FromStr for NodeKind {
    type Err = CodememError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "file" => Ok(Self::File),
            "package" => Ok(Self::Package),
            "function" => Ok(Self::Function),
            "class" => Ok(Self::Class),
            "module" => Ok(Self::Module),
            "memory" => Ok(Self::Memory),
            "method" => Ok(Self::Method),
            "interface" => Ok(Self::Interface),
            "type" => Ok(Self::Type),
            "constant" => Ok(Self::Constant),
            "endpoint" => Ok(Self::Endpoint),
            "test" => Ok(Self::Test),
            _ => Err(CodememError::InvalidNodeKind(s.to_string())),
        }
    }
}

// ── Core Data Structures ────────────────────────────────────────────────────

/// A memory node stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub importance: f64,
    pub confidence: f64,
    pub access_count: u32,
    pub content_hash: String,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub namespace: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
}

/// A graph edge connecting two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub src: String,
    pub dst: String,
    pub relationship: RelationshipType,
    pub weight: f64,
    pub properties: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

/// A graph node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    pub payload: HashMap<String, serde_json::Value>,
    pub centrality: f64,
    pub memory_id: Option<String>,
    pub namespace: Option<String>,
}

/// A search result with hybrid scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub memory: MemoryNode,
    pub score: f64,
    pub score_breakdown: ScoreBreakdown,
}

/// Breakdown of the 9-component hybrid scoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Vector cosine similarity (25%)
    pub vector_similarity: f64,
    /// Graph relationship strength (25%)
    pub graph_strength: f64,
    /// Content token overlap (15%)
    pub token_overlap: f64,
    /// Temporal alignment (10%)
    pub temporal: f64,
    /// Tag matching (10%)
    pub tag_matching: f64,
    /// Importance score (5%)
    pub importance: f64,
    /// Memory confidence (5%)
    pub confidence: f64,
    /// Recency boost (5%)
    pub recency: f64,
}

impl ScoreBreakdown {
    /// Compute the weighted total score using default weights.
    pub fn total(&self) -> f64 {
        self.vector_similarity * 0.25
            + self.graph_strength * 0.25
            + self.token_overlap * 0.15
            + self.temporal * 0.10
            + self.tag_matching * 0.10
            + self.importance * 0.05
            + self.confidence * 0.05
            + self.recency * 0.05
    }

    /// Compute the weighted total score using configurable weights.
    pub fn total_with_weights(&self, weights: &ScoringWeights) -> f64 {
        self.vector_similarity * weights.vector_similarity
            + self.graph_strength * weights.graph_strength
            + self.token_overlap * weights.token_overlap
            + self.temporal * weights.temporal
            + self.tag_matching * weights.tag_matching
            + self.importance * weights.importance
            + self.confidence * weights.confidence
            + self.recency * weights.recency
    }
}

// ── Scoring Weights ──────────────────────────────────────────────────────────

/// Configurable weights for the 9-component hybrid scoring system.
/// All weights should sum to 1.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoringWeights {
    pub vector_similarity: f64,
    pub graph_strength: f64,
    pub token_overlap: f64,
    pub temporal: f64,
    pub tag_matching: f64,
    pub importance: f64,
    pub confidence: f64,
    pub recency: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            vector_similarity: 0.25,
            graph_strength: 0.25,
            token_overlap: 0.15,
            temporal: 0.10,
            tag_matching: 0.10,
            importance: 0.05,
            confidence: 0.05,
            recency: 0.05,
        }
    }
}

impl ScoringWeights {
    /// Normalize weights so they sum to 1.0.
    pub fn normalized(&self) -> Self {
        let sum = self.vector_similarity
            + self.graph_strength
            + self.token_overlap
            + self.temporal
            + self.tag_matching
            + self.importance
            + self.confidence
            + self.recency;
        if sum == 0.0 {
            return Self::default();
        }
        Self {
            vector_similarity: self.vector_similarity / sum,
            graph_strength: self.graph_strength / sum,
            token_overlap: self.token_overlap / sum,
            temporal: self.temporal / sum,
            tag_matching: self.tag_matching / sum,
            importance: self.importance / sum,
            confidence: self.confidence / sum,
            recency: self.recency / sum,
        }
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the HNSW vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VectorConfig {
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            metric: DistanceMetric::Cosine,
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

/// Distance metric for vector search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    Cosine,
    L2,
    InnerProduct,
}

/// Configuration for the graph engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphConfig {
    pub max_expansion_hops: usize,
    pub bridge_similarity_threshold: f64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_expansion_hops: 2,
            bridge_similarity_threshold: 0.8,
        }
    }
}

// ── Cross-Session Pattern Detection ─────────────────────────────────────────

/// A pattern detected across sessions by analyzing memory metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Type of pattern detected.
    pub pattern_type: PatternType,
    /// Human-readable description of the pattern.
    pub description: String,
    /// How many times this pattern was observed.
    pub frequency: usize,
    /// IDs of memories related to this pattern.
    pub related_memories: Vec<String>,
    /// Confidence in the detection (0.0 to 1.0).
    pub confidence: f64,
}

/// Types of cross-session patterns that can be detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternType {
    /// Same search pattern (Grep/Glob) used multiple times across sessions.
    RepeatedSearch,
    /// A file that is read or edited frequently across sessions.
    FileHotspot,
    /// A sequence of file explorations forming a navigation path.
    ExplorationPath,
    /// Multiple edits/writes to the same file over time, forming a decision chain.
    DecisionChain,
    /// Disproportionate usage of certain tools over others.
    ToolPreference,
}

impl std::fmt::Display for PatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RepeatedSearch => write!(f, "repeated_search"),
            Self::FileHotspot => write!(f, "file_hotspot"),
            Self::ExplorationPath => write!(f, "exploration_path"),
            Self::DecisionChain => write!(f, "decision_chain"),
            Self::ToolPreference => write!(f, "tool_preference"),
        }
    }
}

// ── Sessions ────────────────────────────────────────────────────────────

/// A session representing a single interaction period with an AI assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub namespace: Option<String>,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub memory_count: u32,
    pub summary: Option<String>,
}

#[cfg(test)]
mod tests {
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
    fn score_breakdown_weights_sum_to_one() {
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
        let total = breakdown.total();
        assert!((total - 1.0).abs() < f64::EPSILON);
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
    fn scoring_weights_normalized() {
        let weights = ScoringWeights {
            vector_similarity: 2.0,
            graph_strength: 2.0,
            token_overlap: 2.0,
            temporal: 2.0,
            tag_matching: 2.0,
            importance: 2.0,
            confidence: 2.0,
            recency: 2.0,
        };
        let norm = weights.normalized();
        let sum = norm.vector_similarity
            + norm.graph_strength
            + norm.token_overlap
            + norm.temporal
            + norm.tag_matching
            + norm.importance
            + norm.confidence
            + norm.recency;
        assert!((sum - 1.0).abs() < f64::EPSILON);
        // All equal => each should be 0.125
        assert!((norm.vector_similarity - 0.125).abs() < f64::EPSILON);
    }

    #[test]
    fn scoring_weights_normalized_zero_returns_default() {
        let weights = ScoringWeights {
            vector_similarity: 0.0,
            graph_strength: 0.0,
            token_overlap: 0.0,
            temporal: 0.0,
            tag_matching: 0.0,
            importance: 0.0,
            confidence: 0.0,
            recency: 0.0,
        };
        let norm = weights.normalized();
        let default = ScoringWeights::default();
        assert!((norm.vector_similarity - default.vector_similarity).abs() < f64::EPSILON);
        assert!((norm.graph_strength - default.graph_strength).abs() < f64::EPSILON);
    }

    #[test]
    fn total_with_weights_matches_total_for_defaults() {
        let breakdown = ScoreBreakdown {
            vector_similarity: 0.8,
            graph_strength: 0.6,
            token_overlap: 0.5,
            temporal: 0.9,
            tag_matching: 0.3,
            importance: 0.7,
            confidence: 0.95,
            recency: 0.4,
        };
        let default_weights = ScoringWeights::default();
        let total = breakdown.total();
        let total_with = breakdown.total_with_weights(&default_weights);
        assert!((total - total_with).abs() < f64::EPSILON);
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
        assert_eq!(PatternType::ExplorationPath.to_string(), "exploration_path");
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
}
