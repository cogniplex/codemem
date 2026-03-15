use crate::CodememEngine;
use codemem_core::ScopeContext;

#[test]
fn set_and_get_scope() {
    let engine = CodememEngine::for_testing();
    assert!(engine.scope().is_none());

    let scope = ScopeContext {
        repo: "my-repo".to_string(),
        git_ref: "feat/auth".to_string(),
        base_ref: Some("main".to_string()),
        user: Some("alice".to_string()),
        session: None,
    };
    engine.set_scope(Some(scope.clone()));

    let retrieved = engine.scope().unwrap();
    assert_eq!(retrieved.repo, "my-repo");
    assert_eq!(retrieved.git_ref, "feat/auth");
    assert_eq!(retrieved.base_ref.as_deref(), Some("main"));
    assert_eq!(retrieved.user.as_deref(), Some("alice"));
}

#[test]
fn scope_namespace_derives_from_repo() {
    let engine = CodememEngine::for_testing();

    engine.set_scope(Some(ScopeContext {
        repo: "codemem".to_string(),
        git_ref: "main".to_string(),
        base_ref: None,
        user: None,
        session: None,
    }));

    assert_eq!(engine.scope_namespace().as_deref(), Some("codemem"));
}

#[test]
fn scope_namespace_none_when_no_scope() {
    let engine = CodememEngine::for_testing();
    assert!(engine.scope_namespace().is_none());
}

#[test]
fn clear_scope() {
    let engine = CodememEngine::for_testing();

    engine.set_scope(Some(ScopeContext {
        repo: "test".to_string(),
        git_ref: "main".to_string(),
        base_ref: None,
        user: None,
        session: None,
    }));
    assert!(engine.scope().is_some());

    engine.set_scope(None);
    assert!(engine.scope().is_none());
}

#[test]
fn persist_memory_auto_populates_repo_git_ref_from_scope() {
    use codemem_core::MemoryNode;

    let engine = CodememEngine::for_testing();
    engine.set_scope(Some(ScopeContext {
        repo: "my-project".to_string(),
        git_ref: "feat/auth".to_string(),
        base_ref: Some("main".to_string()),
        user: None,
        session: None,
    }));

    let memory = MemoryNode::new(
        "scoped memory content xyz",
        codemem_core::MemoryType::Context,
    );
    engine.persist_memory(&memory).unwrap();

    let stored = engine
        .storage()
        .get_memory_no_touch(&memory.id)
        .unwrap()
        .unwrap();
    assert_eq!(stored.repo.as_deref(), Some("my-project"));
    assert_eq!(stored.git_ref.as_deref(), Some("feat/auth"));
}

#[test]
fn explicit_repo_git_ref_not_overwritten_by_scope() {
    use codemem_core::MemoryNode;

    let engine = CodememEngine::for_testing();
    engine.set_scope(Some(ScopeContext {
        repo: "scope-repo".to_string(),
        git_ref: "scope-branch".to_string(),
        base_ref: None,
        user: None,
        session: None,
    }));

    let mut memory = MemoryNode::new(
        "explicit repo memory abc",
        codemem_core::MemoryType::Context,
    );
    memory.repo = Some("explicit-repo".to_string());
    memory.git_ref = Some("explicit-branch".to_string());
    engine.persist_memory(&memory).unwrap();

    let stored = engine
        .storage()
        .get_memory_no_touch(&memory.id)
        .unwrap()
        .unwrap();
    assert_eq!(stored.repo.as_deref(), Some("explicit-repo"));
    assert_eq!(stored.git_ref.as_deref(), Some("explicit-branch"));
}

#[test]
fn recall_filters_by_git_ref() {
    use codemem_core::MemoryNode;

    let engine = CodememEngine::for_testing();

    // Store memory on main
    let mut m_main = MemoryNode::new(
        "main branch memory content",
        codemem_core::MemoryType::Context,
    );
    m_main.git_ref = Some("main".to_string());
    engine.persist_memory(&m_main).unwrap();

    // Store memory on feat/auth
    let mut m_feat = MemoryNode::new(
        "feature branch memory content",
        codemem_core::MemoryType::Context,
    );
    m_feat.git_ref = Some("feat/auth".to_string());
    engine.persist_memory(&m_feat).unwrap();

    // Recall with git_ref filter for main
    let rq = crate::RecallQuery {
        query: "memory content",
        k: 10,
        memory_type_filter: None,
        namespace_filter: None,
        exclude_tags: &[],
        min_importance: None,
        min_confidence: None,
        git_ref_filter: Some("main"),
    };
    let results = engine.recall(&rq).unwrap();
    assert!(
        results
            .iter()
            .all(|r| r.memory.git_ref.as_deref() == Some("main")),
        "All results should be from main branch"
    );
    assert!(
        results.iter().any(|r| r.memory.id == m_main.id),
        "Main memory should be in results"
    );
    assert!(
        results.iter().all(|r| r.memory.id != m_feat.id),
        "Feature memory should be filtered out"
    );
}
