use super::*;
use ast_grep_core::tree_sitter::LanguageExt;

fn parse_and_chunk(source: &str, ext: &str, config: &ChunkConfig) -> Vec<CodeChunk> {
    let path = format!("test.{ext}");

    let engine = crate::index::engine::AstGrepEngine::new();
    let lang = engine.find_language(ext).expect("extractor for extension");

    let symbols = engine.extract_symbols(lang, source, &path);
    let root = lang.lang.ast_grep(source);
    chunk_file(&root, source, &path, &symbols, config)
}

#[test]
fn test_chunk_config_default() {
    let config = ChunkConfig::default();
    assert_eq!(config.max_chunk_size, 1500);
    assert_eq!(config.min_chunk_size, 50);
}

#[test]
fn small_file_produces_single_chunk() {
    let source = r#"pub fn hello() { println!("hello"); }"#;
    let config = ChunkConfig::default();
    let chunks = parse_and_chunk(source, "rs", &config);
    assert_eq!(chunks.len(), 1, "A small file should produce one chunk");
    assert_eq!(chunks[0].index, 0);
    assert_eq!(chunks[0].file_path, "test.rs");
    assert!(chunks[0].non_ws_chars > 0);
}

#[test]
fn large_file_produces_multiple_chunks() {
    // Generate a large Rust file that exceeds default max_chunk_size
    let mut source = String::new();
    for i in 0..50 {
        source.push_str(&format!(
            "pub fn function_{i}(x: i32) -> i32 {{\n    let result = x * {i} + 1;\n    result\n}}\n\n"
        ));
    }

    let config = ChunkConfig {
        max_chunk_size: 200,
        min_chunk_size: 20,
    };
    let chunks = parse_and_chunk(&source, "rs", &config);
    assert!(
        chunks.len() > 1,
        "A large file with small max_chunk_size should produce multiple chunks, got {}",
        chunks.len()
    );

    // Verify each chunk is within max_chunk_size
    for chunk in &chunks {
        assert!(
            chunk.non_ws_chars <= config.max_chunk_size,
            "Chunk {} has {} non-ws chars, exceeding max {}",
            chunk.index,
            chunk.non_ws_chars,
            config.max_chunk_size,
        );
    }
}

#[test]
fn parent_symbol_resolution() {
    let source = r#"
pub struct Config {
    pub debug: bool,
}

impl Config {
    pub fn new() -> Self {
        Config { debug: false }
    }

    pub fn enable_debug(&mut self) {
        self.debug = true;
    }
}
"#;

    let config = ChunkConfig {
        max_chunk_size: 50,
        min_chunk_size: 10,
    };
    let chunks = parse_and_chunk(source, "rs", &config);

    // At least one chunk should have a parent symbol
    let with_parent: Vec<_> = chunks
        .iter()
        .filter(|c| c.parent_symbol.is_some())
        .collect();
    assert!(
        !with_parent.is_empty(),
        "At least one chunk should have a resolved parent symbol"
    );
}

#[test]
fn deep_nesting_still_produces_chunks() {
    let source = r#"
pub fn outer() {
    if true {
        if true {
            if true {
                if true {
                    println!("deeply nested");
                }
            }
        }
    }
}
"#;

    let config = ChunkConfig {
        max_chunk_size: 30,
        min_chunk_size: 5,
    };
    let chunks = parse_and_chunk(source, "rs", &config);
    assert!(
        !chunks.is_empty(),
        "Should produce at least one chunk from deeply nested code"
    );
}

#[test]
fn custom_chunk_config() {
    let source = "pub fn a() {}\npub fn b() {}\npub fn c() {}\n";
    let config = ChunkConfig {
        max_chunk_size: 5000,
        min_chunk_size: 1,
    };
    let chunks = parse_and_chunk(source, "rs", &config);

    // With a very large max, everything should fit in one chunk
    assert_eq!(
        chunks.len(),
        1,
        "With large max_chunk_size, file should be a single chunk"
    );
}

#[test]
fn empty_file_produces_zero_chunks() {
    let source = "";
    let config = ChunkConfig::default();
    let chunks = parse_and_chunk(source, "rs", &config);
    assert!(chunks.is_empty(), "Empty file should produce zero chunks");
}

#[test]
fn whitespace_only_file_produces_zero_chunks() {
    let source = "   \n\n  \t  \n";
    let config = ChunkConfig::default();
    let chunks = parse_and_chunk(source, "rs", &config);
    assert!(
        chunks.is_empty(),
        "Whitespace-only file should produce zero chunks"
    );
}

#[test]
fn multi_language_rust() {
    let source = r#"
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

pub struct Greeter {
    pub prefix: String,
}

impl Greeter {
    pub fn new(prefix: &str) -> Self {
        Self { prefix: prefix.to_string() }
    }
}
"#;

    let config = ChunkConfig::default();
    let chunks = parse_and_chunk(source, "rs", &config);
    assert!(!chunks.is_empty(), "Rust file should produce chunks");

    for chunk in &chunks {
        assert_eq!(chunk.file_path, "test.rs");
    }
}

#[test]
fn multi_language_typescript() {
    let source = r#"
export function greet(name: string): string {
    return `Hello, ${name}!`;
}

export class Greeter {
    prefix: string;
    constructor(prefix: string) {
        this.prefix = prefix;
    }
}
"#;

    let config = ChunkConfig::default();
    let chunks = parse_and_chunk(source, "ts", &config);
    assert!(!chunks.is_empty(), "TypeScript file should produce chunks");

    for chunk in &chunks {
        assert_eq!(chunk.file_path, "test.ts");
    }
}

#[test]
fn chunks_are_contiguous_indices() {
    let source = r#"
pub fn a() { }
pub fn b() { }
pub fn c() { }
pub fn d() { }
"#;

    let config = ChunkConfig {
        max_chunk_size: 20,
        min_chunk_size: 5,
    };
    let chunks = parse_and_chunk(source, "rs", &config);

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.index, i,
            "Chunk indices should be contiguous starting from 0"
        );
    }
}

#[test]
fn chunk_line_ranges_are_valid() {
    let source = r#"
pub fn first() { }
pub fn second() { }
pub fn third() { }
"#;

    let config = ChunkConfig::default();
    let chunks = parse_and_chunk(source, "rs", &config);

    for chunk in &chunks {
        assert!(
            chunk.line_end >= chunk.line_start,
            "line_end ({}) should be >= line_start ({})",
            chunk.line_end,
            chunk.line_start
        );
        assert!(
            chunk.byte_end >= chunk.byte_start,
            "byte_end ({}) should be >= byte_start ({})",
            chunk.byte_end,
            chunk.byte_start
        );
    }
}
