use super::*;

// ── Display impls ───────────────────────────────────────────────────────────

#[test]
fn display_storage_error() {
    let err = CodememError::Storage("disk full".to_string());
    assert_eq!(err.to_string(), "Storage error: disk full");
}

#[test]
fn display_vector_error() {
    let err = CodememError::Vector("index corrupted".to_string());
    assert_eq!(err.to_string(), "Vector error: index corrupted");
}

#[test]
fn display_embedding_error() {
    let err = CodememError::Embedding("model not found".to_string());
    assert_eq!(err.to_string(), "Embedding error: model not found");
}

#[test]
fn display_hook_error() {
    let err = CodememError::Hook("session start failed".to_string());
    assert_eq!(err.to_string(), "Hook error: session start failed");
}

#[test]
fn display_invalid_memory_type() {
    let err = CodememError::InvalidMemoryType("foobar".to_string());
    assert_eq!(err.to_string(), "Invalid memory type: foobar");
}

#[test]
fn display_invalid_relationship_type() {
    let err = CodememError::InvalidRelationshipType("UNKNOWN".to_string());
    assert_eq!(err.to_string(), "Invalid relationship type: UNKNOWN");
}

#[test]
fn display_invalid_node_kind() {
    let err = CodememError::InvalidNodeKind("widget".to_string());
    assert_eq!(err.to_string(), "Invalid node kind: widget");
}

#[test]
fn display_not_found() {
    let err = CodememError::NotFound("mem-123".to_string());
    assert_eq!(err.to_string(), "Not found: mem-123");
}

#[test]
fn display_invalid_input() {
    let err = CodememError::InvalidInput("empty query".to_string());
    assert_eq!(err.to_string(), "Invalid input: empty query");
}

#[test]
fn display_duplicate() {
    let err = CodememError::Duplicate("abc123".to_string());
    assert_eq!(err.to_string(), "Duplicate content (hash: abc123)");
}

#[test]
fn display_internal() {
    let err = CodememError::Internal("unexpected state".to_string());
    assert_eq!(err.to_string(), "Internal error: unexpected state");
}

#[test]
fn display_config() {
    let err = CodememError::Config("invalid weight".to_string());
    assert_eq!(err.to_string(), "Configuration error: invalid weight");
}

#[test]
fn display_lock_poisoned() {
    let err = CodememError::LockPoisoned("scoring weights".to_string());
    assert_eq!(err.to_string(), "Lock poisoned: scoring weights");
}

// ── From conversions ────────────────────────────────────────────────────────

#[test]
fn from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let err: CodememError = io_err.into();
    match &err {
        CodememError::Io(inner) => {
            assert_eq!(inner.kind(), std::io::ErrorKind::NotFound);
        }
        other => panic!("Expected Io variant, got: {other:?}"),
    }
    assert!(err.to_string().contains("file missing"));
}

#[test]
fn from_serde_json_error() {
    let json_err = serde_json::from_str::<serde_json::Value>("not valid json").unwrap_err();
    let err: CodememError = json_err.into();
    match &err {
        CodememError::Json(_) => {}
        other => panic!("Expected Json variant, got: {other:?}"),
    }
    assert!(err.to_string().starts_with("JSON error:"));
}

#[test]
fn from_toml_de_error() {
    let toml_err: toml::de::Error = toml::from_str::<toml::Value>("[broken").unwrap_err();
    let err: CodememError = toml_err.into();
    match &err {
        CodememError::Config(msg) => {
            assert!(!msg.is_empty());
        }
        other => panic!("Expected Config variant, got: {other:?}"),
    }
    assert!(err.to_string().starts_with("Configuration error:"));
}
