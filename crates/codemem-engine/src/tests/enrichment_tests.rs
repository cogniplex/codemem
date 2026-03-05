use crate::enrichment::resolve_path;
use std::path::{Path, PathBuf};

#[test]
fn resolve_path_with_root_joins() {
    let result = resolve_path("src/main.rs", Some(Path::new("/home/user/project")));
    assert_eq!(result, PathBuf::from("/home/user/project/src/main.rs"));
}

#[test]
fn resolve_path_without_root_returns_as_is() {
    let result = resolve_path("src/main.rs", None);
    assert_eq!(result, PathBuf::from("src/main.rs"));
}

#[test]
fn resolve_path_nested_relative() {
    let result = resolve_path("a/b/c.rs", Some(Path::new("/root")));
    assert_eq!(result, PathBuf::from("/root/a/b/c.rs"));
}

#[test]
fn resolve_path_root_only_file() {
    let result = resolve_path("lib.rs", Some(Path::new("/project")));
    assert_eq!(result, PathBuf::from("/project/lib.rs"));
}
