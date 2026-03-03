use super::*;

#[test]
fn diff_simple_edit() {
    let old = "fn main() {\n    println!(\"hello\");\n}\n";
    let new = "fn main() {\n    println!(\"world\");\n}\n";
    let summary = compute_diff("src/main.rs", old, new);
    assert_eq!(summary.change_type, ChangeType::Modified);
    assert_eq!(summary.lines_added, 1);
    assert_eq!(summary.lines_removed, 1);
}

#[test]
fn semantic_summary_function_addition() {
    let old = "// module\n";
    let new = "// module\nfn new_helper() {\n    todo!()\n}\n";
    let summary = compute_diff("src/lib.rs", old, new);
    assert!(summary
        .semantic_summary
        .contains("Added function new_helper"));
}

#[test]
fn semantic_summary_function_removal() {
    let old = "fn helper() {\n    todo!()\n}\nfn main() {}\n";
    let new = "fn main() {}\n";
    let summary = compute_diff("src/lib.rs", old, new);
    assert!(summary.semantic_summary.contains("Removed function helper"));
}

#[test]
fn semantic_summary_import_changes() {
    let old = "use std::io;\nfn main() {}\n";
    let new = "use std::io;\nuse std::fs;\nfn main() {}\n";
    let summary = compute_diff("src/main.rs", old, new);
    assert!(summary.semantic_summary.contains("Updated imports"));
}

#[test]
fn semantic_summary_type_addition() {
    let old = "// types\n";
    let new = "// types\nstruct Config {\n    name: String,\n}\n";
    let summary = compute_diff("src/types.rs", old, new);
    assert!(summary.semantic_summary.contains("Added type Config"));
}

#[test]
fn empty_diff() {
    let content = "fn main() {}\n";
    let summary = compute_diff("src/main.rs", content, content);
    assert_eq!(summary.lines_added, 0);
    assert_eq!(summary.lines_removed, 0);
}

#[test]
fn change_type_added() {
    let summary = compute_diff("new.rs", "", "fn new() {}\n");
    assert_eq!(summary.change_type, ChangeType::Added);
}

#[test]
fn change_type_deleted() {
    let summary = compute_diff("old.rs", "fn old() {}\n", "");
    assert_eq!(summary.change_type, ChangeType::Deleted);
}

#[test]
fn extract_fn_name_works() {
    assert_eq!(extract_fn_name("fn hello("), Some("hello".to_string()));
    assert_eq!(
        extract_fn_name("async fn fetch_data()"),
        Some("fetch_data".to_string())
    );
    assert_eq!(
        extract_fn_name("def process(x):"),
        Some("process".to_string())
    );
    assert_eq!(extract_fn_name("no function here"), None);
}

#[test]
fn extract_type_name_works() {
    assert_eq!(
        extract_type_name("struct MyStruct {"),
        Some("MyStruct".to_string())
    );
    assert_eq!(extract_type_name("enum Color {"), Some("Color".to_string()));
    assert_eq!(
        extract_type_name("trait Display {"),
        Some("Display".to_string())
    );
    assert_eq!(extract_type_name("no type here"), None);
}
