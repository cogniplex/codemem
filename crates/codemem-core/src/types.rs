use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::CodememError;

/// Compute a SHA-256 content hash for deduplication.
pub fn content_hash(content: &str) -> String {
    let hash = Sha256::digest(content.as_bytes());
    hash.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

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
    /// Not auto-generated; available for manual use via the MCP `store_memory` tool.
    Preference,
    /// Coding style norms (e.g., "early returns, max 20 lines").
    /// Not auto-generated; available for manual use via the MCP `store_memory` tool.
    Style,
    /// Workflow habits (e.g., "tests written before implementation").
    /// Not auto-generated; available for manual use via the MCP `store_memory` tool.
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

/// Relationship types for the knowledge graph.
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
    // SCIP-derived relationships
    /// "variable X has type Y" — from SCIP `Relationship.is_type_definition`.
    TypeDefinition,
    /// Function reads symbol — from SCIP `symbol_roles & ReadAccess`.
    Reads,
    /// Function writes symbol — from SCIP `symbol_roles & WriteAccess`.
    Writes,
    /// Method → parent method (virtual dispatch) — derived from SCIP `is_implementation` on methods.
    Overrides,
    // Cross-service relationships (API surface detection)
    /// Service A's HTTP client call targets Service B's endpoint.
    HttpCalls,
    /// Producer publishes to an event channel/topic.
    PublishesTo,
    /// Consumer subscribes to an event channel/topic.
    SubscribesTo,
    // Temporal (git history layer)
    /// Commit modified a symbol/file.
    ModifiedBy,
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
            Self::TypeDefinition => write!(f, "TYPE_DEFINITION"),
            Self::Reads => write!(f, "READS"),
            Self::Writes => write!(f, "WRITES"),
            Self::Overrides => write!(f, "OVERRIDES"),
            Self::HttpCalls => write!(f, "HTTP_CALLS"),
            Self::PublishesTo => write!(f, "PUBLISHES_TO"),
            Self::SubscribesTo => write!(f, "SUBSCRIBES_TO"),
            Self::ModifiedBy => write!(f, "MODIFIED_BY"),
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
            "TYPE_DEFINITION" => Ok(Self::TypeDefinition),
            "READS" => Ok(Self::Reads),
            "WRITES" => Ok(Self::Writes),
            "OVERRIDES" => Ok(Self::Overrides),
            "HTTP_CALLS" => Ok(Self::HttpCalls),
            "PUBLISHES_TO" => Ok(Self::PublishesTo),
            "SUBSCRIBES_TO" => Ok(Self::SubscribesTo),
            "MODIFIED_BY" => Ok(Self::ModifiedBy),
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
    /// TypeScript interface, Go interface.
    Interface,
    /// Type alias, typedef.
    Type,
    /// Const, static.
    Constant,
    /// REST/gRPC endpoint definition.
    Endpoint,
    /// Test function.
    Test,
    /// A CST-aware code chunk.
    Chunk,
    // SCIP-derived node kinds
    /// Dependency symbols — stubs with hover docs, no source location.
    /// ID format: `ext:{manager}:{package}:{qualified_name}`.
    External,
    /// Rust trait — semantically distinct from `Interface`.
    Trait,
    /// Enum type — distinct from `Constant`.
    Enum,
    /// Enum member/variant.
    EnumVariant,
    /// Struct/class field.
    Field,
    /// Generic type parameter (e.g., `T` in `Vec<T>`).
    TypeParameter,
    /// Rust macros, C preprocessor macros.
    Macro,
    /// JS/TS/Python properties — distinct from struct fields.
    Property,
    // Temporal (git history layer)
    /// A git commit.
    Commit,
    /// A pull request (detected from merge/squash commit patterns).
    PullRequest,
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
            Self::External => write!(f, "external"),
            Self::Trait => write!(f, "trait"),
            Self::Enum => write!(f, "enum"),
            Self::EnumVariant => write!(f, "enum_variant"),
            Self::Field => write!(f, "field"),
            Self::TypeParameter => write!(f, "type_parameter"),
            Self::Macro => write!(f, "macro"),
            Self::Property => write!(f, "property"),
            Self::Commit => write!(f, "commit"),
            Self::PullRequest => write!(f, "pull_request"),
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
            "external" => Ok(Self::External),
            "trait" => Ok(Self::Trait),
            "enum" => Ok(Self::Enum),
            "enum_variant" => Ok(Self::EnumVariant),
            "field" => Ok(Self::Field),
            "type_parameter" => Ok(Self::TypeParameter),
            "macro" => Ok(Self::Macro),
            "property" => Ok(Self::Property),
            "commit" => Ok(Self::Commit),
            "pull_request" => Ok(Self::PullRequest),
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
    /// The session during which this memory was created (auto-populated by the engine).
    #[serde(default)]
    pub session_id: Option<String>,
    /// Repository identifier (auto-populated from scope context).
    #[serde(default)]
    pub repo: Option<String>,
    /// Git ref (branch/tag) this memory belongs to (auto-populated from scope context).
    #[serde(default)]
    pub git_ref: Option<String>,
    /// When this memory expires. `None` means it never expires.
    /// Session memories get a default TTL; enrichment memories expire on reindex.
    #[serde(default)]
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
}

impl MemoryNode {
    /// Create a test/example memory with minimal fields.
    /// Uses `MemoryType::Context` and default values for importance (0.5),
    /// confidence (1.0), empty tags, and empty metadata.
    pub fn test_default(content: &str) -> Self {
        Self::new(content.to_string(), MemoryType::Context)
    }

    /// Create a new MemoryNode with sensible defaults.
    /// Auto-generates id, content_hash, and timestamps.
    pub fn new(content: impl Into<String>, memory_type: MemoryType) -> Self {
        let content = content.into();
        let now = chrono::Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content_hash: content_hash(&content),
            content,
            memory_type,
            importance: 0.5,
            confidence: 1.0,
            access_count: 0,
            tags: Vec::new(),
            metadata: HashMap::new(),
            namespace: None,
            session_id: None,
            repo: None,
            git_ref: None,
            expires_at: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        }
    }
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
    /// When this node becomes valid (None = always valid).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    /// When this node expires (None = never expires).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<DateTime<Utc>>,
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
    /// Graph relationship strength (20%)
    pub graph_strength: f64,
    /// Content token overlap (15%)
    pub token_overlap: f64,
    /// Temporal alignment (10%)
    pub temporal: f64,
    /// Tag matching (5%)
    pub tag_matching: f64,
    /// Importance score (10%)
    pub importance: f64,
    /// Memory confidence (10%)
    pub confidence: f64,
    /// Recency boost (5%)
    pub recency: f64,
}

impl ScoreBreakdown {
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

// ── Raw Graph Metrics ────────────────────────────────────────────────────────

/// Raw graph metrics for a memory node, collected from its neighbors.
///
/// Returned by `GraphEngine::raw_graph_metrics_for_memory()` so that the
/// scoring formula can live in the engine crate.
#[derive(Debug, Clone)]
pub struct RawGraphMetrics {
    /// Highest PageRank score among code-graph neighbors.
    pub max_pagerank: f64,
    /// Highest betweenness centrality among code-graph neighbors.
    pub max_betweenness: f64,
    /// Number of code-graph neighbors (sym:, file:, chunk:, pkg:).
    pub code_neighbor_count: usize,
    /// Sum of edge weights connecting this memory to code-graph neighbors.
    pub total_edge_weight: f64,
    /// Number of memory-to-memory neighbors (UUID-based IDs).
    pub memory_neighbor_count: usize,
    /// Sum of edge weights connecting this memory to other memory nodes.
    pub memory_edge_weight: f64,
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

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VectorConfig {
    /// Backend type: "hnsw" (default), "pgvector", or "qdrant".
    pub backend: String,
    /// Connection URL for remote backends (e.g., "http://localhost:6333" for Qdrant).
    pub url: Option<String>,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            backend: "hnsw".to_string(),
            url: None,
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
    /// Backend type: "petgraph" (default) or "neo4j".
    pub backend: String,
    /// Connection URL for remote backends (e.g., "bolt://localhost:7687" for Neo4j).
    pub url: Option<String>,
    /// Default edge weight for CONTAINS relationship (structural, low).
    pub contains_edge_weight: f64,
    /// Default edge weight for CALLS relationship (high signal).
    pub calls_edge_weight: f64,
    /// Default edge weight for IMPORTS relationship.
    pub imports_edge_weight: f64,
    /// Default edge weight for TYPE_DEFINITION relationship.
    pub type_definition_edge_weight: f64,
    /// Default edge weight for READS relationship.
    pub reads_edge_weight: f64,
    /// Default edge weight for WRITES relationship.
    pub writes_edge_weight: f64,
    /// Default edge weight for OVERRIDES relationship (virtual dispatch).
    pub overrides_edge_weight: f64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            backend: "petgraph".to_string(),
            url: None,
            contains_edge_weight: 0.1,
            calls_edge_weight: 1.0,
            imports_edge_weight: 0.5,
            type_definition_edge_weight: 0.6,
            reads_edge_weight: 0.3,
            writes_edge_weight: 0.4,
            overrides_edge_weight: 0.8,
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
            Self::DecisionChain => write!(f, "decision_chain"),
            Self::ToolPreference => write!(f, "tool_preference"),
        }
    }
}

// ── Scope Context ───────────────────────────────────────────────────────

/// Scoping context threaded through storage and engine operations.
///
/// Replaces the flat `namespace: &str` with a richer context that includes
/// repository identity, git branch, and optional user/session scoping.
/// Backward compatible: `namespace()` derives the same directory-basename
/// value used before, and all new fields are optional or have defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeContext {
    /// Repository identifier (e.g., "github.com/org/repo" or directory basename).
    pub repo: String,
    /// Git ref: branch name, tag, or commit SHA. Defaults to "main".
    pub git_ref: String,
    /// Base ref for overlay resolution (e.g., "main" when on a feature branch).
    /// When set, queries fall back to base_ref for data not in the overlay.
    pub base_ref: Option<String>,
    /// User identifier (for user-scoped memories in team mode).
    pub user: Option<String>,
    /// Session identifier (for session-scoped memories).
    pub session: Option<String>,
}

impl ScopeContext {
    /// Create a scope from a local directory path by detecting the git branch.
    ///
    /// Runs `git rev-parse --abbrev-ref HEAD` to detect the current branch and
    /// `git rev-parse --show-toplevel` + remote URL for repo identity.
    /// Falls back to directory basename if git is not available.
    pub fn from_local(path: &std::path::Path) -> Self {
        let repo = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("unknown")
            .to_string();

        let git_ref = std::process::Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .current_dir(path)
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "main".to_string());

        // If on a feature branch, set base_ref to the repo's default branch.
        let base_ref = if git_ref != "main" && git_ref != "master" && git_ref != "HEAD" {
            Some(detect_default_branch(path))
        } else {
            None
        };

        Self {
            repo,
            git_ref,
            base_ref,
            user: None,
            session: None,
        }
    }

    /// Derive namespace from scope (backward compatible with directory-basename convention).
    pub fn namespace(&self) -> &str {
        &self.repo
    }

    /// Whether this scope is on a feature branch (has a base_ref overlay).
    pub fn is_overlay(&self) -> bool {
        self.base_ref.is_some()
    }
}

/// Detect the default branch for a git repository.
/// Tries `git symbolic-ref refs/remotes/origin/HEAD` first (most reliable),
/// then falls back to checking if "main" exists, then "master", then "main".
fn detect_default_branch(path: &std::path::Path) -> String {
    // Try symbolic-ref (works when origin/HEAD is set)
    if let Some(default) = std::process::Command::new("git")
        .args(["symbolic-ref", "refs/remotes/origin/HEAD"])
        .current_dir(path)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
    {
        // Output is like "refs/remotes/origin/main" — extract the branch name
        if let Some(branch) = default.rsplit('/').next() {
            return branch.to_string();
        }
    }

    // Fallback: check if "main" branch exists locally
    if std::process::Command::new("git")
        .args(["rev-parse", "--verify", "main"])
        .current_dir(path)
        .output()
        .is_ok_and(|o| o.status.success())
    {
        return "main".to_string();
    }

    // Fallback: check if "master" branch exists locally
    if std::process::Command::new("git")
        .args(["rev-parse", "--verify", "master"])
        .current_dir(path)
        .output()
        .is_ok_and(|o| o.status.success())
    {
        return "master".to_string();
    }

    // Last resort
    "main".to_string()
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
///
/// The `created_at` and `last_indexed_at` fields are stored as RFC 3339
/// strings for SQLite compatibility.
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

// ── Node Memory Query ────────────────────────────────────────────────

/// A memory connected to a graph node, with relationship and distance metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMemoryResult {
    pub memory: MemoryNode,
    pub relationship: String,
    pub depth: usize,
}

/// Coverage status for a single graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCoverageEntry {
    pub node_id: String,
    pub memory_count: usize,
    pub has_coverage: bool,
}

// ── Enrichment Analyses ─────────────────────────────────────────────

/// The 14 enrichment analysis types supported by `enrich_codebase`.
///
/// Keep in sync with the MCP tool schema (`mcp/mod.rs` `enrich_codebase` inputSchema)
/// and the dispatch logic in `mcp/tools_enrich.rs`.
pub const ENRICHMENT_ANALYSES: &[&str] = &[
    "git",
    "security",
    "performance",
    "complexity",
    "code_smells",
    "security_scan",
    "architecture",
    "test_mapping",
    "api_surface",
    "doc_coverage",
    "hot_complex",
    "blame",
    "quality",
    "change_impact",
];

// ── Cross-Repo Data ─────────────────────────────────────────────────────────

/// Data for a single unresolved reference, used by batch storage.
#[derive(Debug, Clone)]
pub struct UnresolvedRefData {
    pub source_qualified_name: String,
    pub target_name: String,
    pub namespace: String,
    pub file_path: String,
    pub line: usize,
    pub ref_kind: String,
    pub package_hint: Option<String>,
}

#[cfg(test)]
#[path = "tests/types_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/scope_tests.rs"]
mod scope_tests;
