/// Unified error type for Codemem.
#[derive(Debug, thiserror::Error)]
pub enum CodememError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Vector error: {0}")]
    Vector(String),

    #[error("Graph error: {0}")]
    Graph(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("Hook error: {0}")]
    Hook(String),

    #[error("Invalid memory type: {0}")]
    InvalidMemoryType(String),

    #[error("Invalid relationship type: {0}")]
    InvalidRelationshipType(String),

    #[error("Invalid node kind: {0}")]
    InvalidNodeKind(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Duplicate content (hash: {0})")]
    Duplicate(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
