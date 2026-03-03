use super::*;

#[test]
fn short_path_absolute() {
    assert_eq!(short_path("/home/user/project/src/main.rs"), "src/main.rs");
}

#[test]
fn short_path_relative() {
    assert_eq!(short_path("src/main.rs"), "src/main.rs");
}

#[test]
fn short_path_single_component() {
    assert_eq!(short_path("main.rs"), "main.rs");
}

#[test]
fn short_path_empty() {
    assert_eq!(short_path(""), "");
}
