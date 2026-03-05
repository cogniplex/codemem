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

#[test]
fn index_and_resolve_produces_relative_paths() {
    let dir = std::env::temp_dir().join("codemem_index_rel_test");
    let _ = fs::remove_dir_all(&dir);
    let src = dir.join("src");
    fs::create_dir_all(&src).unwrap();

    fs::write(
        src.join("main.rs"),
        b"pub fn greet() { println!(\"hi\"); }\n",
    )
    .unwrap();

    let mut indexer = Indexer::new();
    let result = indexer.index_and_resolve(&dir).unwrap();

    // File paths should be relative to root, not absolute
    assert!(
        result.file_paths.iter().all(|p| !p.starts_with('/')),
        "All file paths should be relative, got: {:?}",
        result.file_paths
    );
    assert!(
        result.file_paths.contains("src/main.rs"),
        "Should contain 'src/main.rs', got: {:?}",
        result.file_paths
    );

    // Symbols should have relative file_path
    for sym in &result.symbols {
        assert!(
            !sym.file_path.starts_with('/'),
            "Symbol file_path should be relative, got: {}",
            sym.file_path
        );
    }

    // root_path should be an absolute canonicalized path
    assert!(
        result.root_path.is_absolute(),
        "root_path should be absolute, got: {:?}",
        result.root_path
    );

    // Cleanup
    let _ = fs::remove_dir_all(&dir);
}
