use super::*;
use std::fs;

#[test]
fn index_temp_directory() {
    // Create a temp directory with a Rust file
    let dir = std::env::temp_dir().join("codemem_index_test");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    fs::write(
        dir.join("main.rs"),
        b"pub fn hello() { println!(\"hello\"); }\n",
    )
    .unwrap();

    fs::write(
        dir.join("lib.rs"),
        b"pub struct Config { pub debug: bool }\n",
    )
    .unwrap();

    // Also create a non-Rust file that should be skipped
    fs::write(dir.join("readme.txt"), b"This is not Rust").unwrap();

    let mut indexer = Indexer::new();
    let result = indexer.index_directory(&dir).unwrap();

    assert_eq!(result.files_scanned, 2, "Should scan 2 .rs files");
    assert_eq!(result.files_parsed, 2, "Should parse 2 .rs files");
    assert_eq!(
        result.files_skipped, 0,
        "No files should be skipped on first run"
    );
    assert!(
        result.total_symbols >= 2,
        "Should have at least 2 symbols (hello, Config)"
    );

    // Run again - all files should be skipped (incremental)
    let result2 = indexer.index_directory(&dir).unwrap();
    assert_eq!(result2.files_scanned, 2);
    assert_eq!(
        result2.files_parsed, 0,
        "All files should be skipped on second run"
    );
    assert_eq!(result2.files_skipped, 2);

    // Cleanup
    let _ = fs::remove_dir_all(&dir);
}
