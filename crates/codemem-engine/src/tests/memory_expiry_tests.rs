use crate::CodememEngine;
use codemem_core::{MemoryNode, MemoryType};

#[test]
fn session_memory_gets_auto_expires_at() {
    let engine = CodememEngine::for_testing();
    engine.set_active_session(Some("test-session".to_string()));

    let memory = MemoryNode::new("session memory content", MemoryType::Context);
    engine.persist_memory(&memory).unwrap();

    let stored = engine
        .storage()
        .get_memory_no_touch(&memory.id)
        .unwrap()
        .unwrap();
    assert!(
        stored.expires_at.is_some(),
        "Session memory should have auto-set expires_at"
    );
    // Default TTL is 168 hours (7 days)
    let expected_min = chrono::Utc::now() + chrono::Duration::hours(167);
    let expected_max = chrono::Utc::now() + chrono::Duration::hours(169);
    let expires = stored.expires_at.unwrap();
    assert!(
        expires > expected_min && expires < expected_max,
        "expires_at should be ~168 hours from now, got {:?}",
        expires
    );
}

#[test]
fn non_session_memory_has_no_expires_at() {
    let engine = CodememEngine::for_testing();
    // No session started

    let memory = MemoryNode::new("permanent memory content xyz", MemoryType::Decision);
    engine.persist_memory(&memory).unwrap();

    let stored = engine
        .storage()
        .get_memory_no_touch(&memory.id)
        .unwrap()
        .unwrap();
    assert!(
        stored.expires_at.is_none(),
        "Non-session memory should not have expires_at"
    );
}

#[test]
fn explicit_expires_at_not_overwritten() {
    let engine = CodememEngine::for_testing();
    engine.set_active_session(Some("test-session-2".to_string()));

    let mut memory = MemoryNode::new("explicit expiry memory abc", MemoryType::Context);
    let explicit_expiry = chrono::Utc::now() + chrono::Duration::hours(1);
    memory.expires_at = Some(explicit_expiry);

    engine.persist_memory(&memory).unwrap();

    let stored = engine
        .storage()
        .get_memory_no_touch(&memory.id)
        .unwrap()
        .unwrap();
    assert_eq!(
        stored.expires_at.unwrap().timestamp(),
        explicit_expiry.timestamp(),
        "Explicit expires_at should not be overwritten by session TTL"
    );
}

#[test]
fn sweep_expired_memories_cleans_up() {
    let engine = CodememEngine::for_testing();

    // Store a memory that's already expired
    let mut expired = MemoryNode::new("already expired content xyz", MemoryType::Context);
    expired.expires_at = Some(chrono::Utc::now() - chrono::Duration::hours(1));
    engine.persist_memory(&expired).unwrap();

    // Store a live memory
    let live = MemoryNode::new("live memory content abc", MemoryType::Context);
    engine.persist_memory(&live).unwrap();

    // Sweep should delete the expired one
    engine.sweep_expired_memories();

    assert!(
        engine
            .storage()
            .get_memory_no_touch(&expired.id)
            .unwrap()
            .is_none(),
        "Expired memory should be deleted by sweep"
    );
    assert!(
        engine
            .storage()
            .get_memory_no_touch(&live.id)
            .unwrap()
            .is_some(),
        "Live memory should survive sweep"
    );
}

#[test]
fn store_memory_with_ttl_hours_via_mcp() {
    // This tests the engine-level TTL behavior, not MCP parsing.
    // MCP parsing of ttl_hours/expires_at is tested in MCP tool tests.
    let engine = CodememEngine::for_testing();

    let mut memory = MemoryNode::new("ttl memory content for test", MemoryType::Context);
    memory.expires_at = Some(chrono::Utc::now() + chrono::Duration::hours(48));
    engine.persist_memory(&memory).unwrap();

    let stored = engine
        .storage()
        .get_memory_no_touch(&memory.id)
        .unwrap()
        .unwrap();
    assert!(stored.expires_at.is_some());
    let ttl_remaining = stored.expires_at.unwrap() - chrono::Utc::now();
    assert!(
        ttl_remaining.num_hours() >= 47 && ttl_remaining.num_hours() <= 49,
        "TTL should be ~48 hours"
    );
}
