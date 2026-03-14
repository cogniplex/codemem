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
