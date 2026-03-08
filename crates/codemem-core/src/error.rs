/// Unified error type for Codemem.
///
/// Variants like `Storage(String)` intentionally carry a stringified error
/// message rather than the original typed error. This avoids coupling
/// `codemem-core` to storage-specific dependencies (e.g. `rusqlite`),
/// keeping the core crate lightweight and backend-agnostic.
#[derive(Debug, thiserror::Error)]
pub enum CodememError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Vector error: {0}")]
    Vector(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

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

    #[error("Invalid input: {0}")]
    InvalidInput(String),

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

impl From<toml::de::Error> for CodememError {
    fn from(e: toml::de::Error) -> Self {
        CodememError::Config(e.to_string())
    }
}

#[cfg(test)]
#[path = "tests/error_tests.rs"]
mod tests;
