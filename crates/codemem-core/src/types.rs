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

/// 24 relationship types for the knowledge graph.
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
    // Temporal (auto-created by git enrichment)
    /// Files that frequently change together in commits.
    CoChanged,
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
            Self::CoChanged => write!(f, "CO_CHANGED"),
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
            "CO_CHANGED" => Ok(Self::CoChanged),
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
    /// A CST-aware code chunk.
    Chunk,
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
            Self::Chunk => write!(f, "chunk"),
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
            "chunk" => Ok(Self::Chunk),
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
    /// When this edge becomes valid (None = always valid).
    pub valid_from: Option<DateTime<Utc>>,
    /// When this edge expires (None = never expires).
    pub valid_to: Option<DateTime<Utc>>,
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
            + self.graph_strength * 0.20
            + self.token_overlap * 0.15
            + self.temporal * 0.10
            + self.tag_matching * 0.05
            + self.importance * 0.10
            + self.confidence * 0.10
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
            graph_strength: 0.20,
            token_overlap: 0.15,
            temporal: 0.10,
            tag_matching: 0.05,
            importance: 0.10,
            confidence: 0.10,
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
    /// Default edge weight for CONTAINS relationship (structural, low).
    pub contains_edge_weight: f64,
    /// Default edge weight for CALLS relationship (high signal).
    pub calls_edge_weight: f64,
    /// Default edge weight for IMPORTS relationship.
    pub imports_edge_weight: f64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_expansion_hops: 2,
            bridge_similarity_threshold: 0.8,
            contains_edge_weight: 0.1,
            calls_edge_weight: 1.0,
            imports_edge_weight: 0.5,
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

// ── Repository ──────────────────────────────────────────────────────────

/// A registered repository tracked by the control plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Repository {
    pub id: String,
    pub path: String,
    pub name: Option<String>,
    pub namespace: Option<String>,
    pub created_at: String,
    pub last_indexed_at: Option<String>,
    pub status: String,
}

// ── Session Activity ─────────────────────────────────────────────────

/// Summary of session activity counts for trigger-based auto-insights.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionActivitySummary {
    pub files_read: usize,
    pub files_edited: usize,
    pub searches: usize,
    pub total_actions: usize,
}

#[cfg(test)]
#[path = "tests/types_tests.rs"]
mod tests;
