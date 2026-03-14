use crate::ScopeContext;

#[test]
fn from_local_detects_repo_name() {
    // from_local uses the directory basename as repo
    let scope = ScopeContext::from_local(std::path::Path::new("/tmp/test-repo"));
    assert_eq!(scope.repo, "test-repo");
    assert_eq!(scope.namespace(), "test-repo");
}

#[test]
fn from_local_defaults_to_main_when_no_git() {
    // Non-git directory should default to "main"
    let tmp = std::env::temp_dir().join(format!("codemem-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&tmp).unwrap();
    let scope = ScopeContext::from_local(&tmp);
    std::fs::remove_dir_all(&tmp).ok();
    assert!(
        scope.git_ref == "main" || scope.git_ref == "master",
        "Non-git dir should default to main/master, got: {}",
        scope.git_ref
    );
    assert!(
        scope.base_ref.is_none(),
        "main/master should have no base_ref"
    );
}

#[test]
fn is_overlay_false_for_main() {
    let scope = ScopeContext {
        repo: "test".to_string(),
        git_ref: "main".to_string(),
        base_ref: None,
        user: None,
        session: None,
    };
    assert!(!scope.is_overlay());
}

#[test]
fn is_overlay_true_for_feature_branch() {
    let scope = ScopeContext {
        repo: "test".to_string(),
        git_ref: "feat/auth".to_string(),
        base_ref: Some("main".to_string()),
        user: None,
        session: None,
    };
    assert!(scope.is_overlay());
}

#[test]
fn namespace_returns_repo() {
    let scope = ScopeContext {
        repo: "my-project".to_string(),
        git_ref: "develop".to_string(),
        base_ref: Some("main".to_string()),
        user: Some("alice".to_string()),
        session: Some("sess-123".to_string()),
    };
    assert_eq!(scope.namespace(), "my-project");
}
