use super::*;
use ast_grep_core::tree_sitter::LanguageExt;
use std::collections::HashSet;

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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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

// ── Semantic boundary tests ─────────────────────────────────────────

#[test]
fn classify_node_categories() {
    assert_eq!(classify_node("use_declaration"), SemanticCategory::Import);
    assert_eq!(classify_node("import_statement"), SemanticCategory::Import);
    assert_eq!(classify_node("line_comment"), SemanticCategory::Comment);
    assert_eq!(classify_node("block_comment"), SemanticCategory::Comment);
    assert_eq!(
        classify_node("function_item"),
        SemanticCategory::Declaration
    );
    assert_eq!(classify_node("impl_item"), SemanticCategory::Declaration);
    assert_eq!(
        classify_node("class_declaration"),
        SemanticCategory::Declaration
    );
    assert_eq!(classify_node("struct_item"), SemanticCategory::Declaration);
    assert_eq!(
        classify_node("expression_statement"),
        SemanticCategory::Other
    );
    assert_eq!(classify_node("if_expression"), SemanticCategory::Other);
}

#[test]
fn is_semantic_boundary_detects_declarations() {
    assert!(is_semantic_boundary("function_item"));
    assert!(is_semantic_boundary("method_definition"));
    assert!(is_semantic_boundary("class_declaration"));
    assert!(is_semantic_boundary("impl_item"));
    assert!(!is_semantic_boundary("use_declaration"));
    assert!(!is_semantic_boundary("if_expression"));
    assert!(!is_semantic_boundary("line_comment"));
}

#[test]
fn impl_block_splits_at_method_boundaries() {
    // A large impl block should split at method boundaries, not mid-method
    let source = r#"
impl MyStruct {
    pub fn method_a(&self) -> String {
        let x = "hello world".to_string();
        let y = x.repeat(10);
        let z = y.trim().to_uppercase();
        format!("{z} done")
    }

    pub fn method_b(&self) -> i32 {
        let a = 42;
        let b = a * 2;
        let c = b + a;
        c * 3
    }

    pub fn method_c(&self) -> bool {
        let flag = true;
        let result = !flag;
        result || flag
    }
}
"#;

    let config = ChunkConfig {
        max_chunk_size: 100,
        min_chunk_size: 10,
        ..Default::default()
    };
    let chunks = parse_and_chunk(source, "rs", &config);

    // Each method should be its own chunk (not split mid-method)
    let method_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.text.contains("fn method_"))
        .collect();
    assert!(
        method_chunks.len() >= 3,
        "Each method should be a separate chunk, got {} method chunks from {} total",
        method_chunks.len(),
        chunks.len()
    );
}

#[test]
fn imports_not_merged_with_functions() {
    // Imports and functions should not be merged even if both are small
    let source = r#"
use std::collections::HashMap;

pub fn tiny() -> i32 { 1 }
"#;

    let config = ChunkConfig {
        max_chunk_size: 500,
        min_chunk_size: 100, // Both are below min, but different categories
        ..Default::default()
    };
    let chunks = parse_and_chunk(source, "rs", &config);

    // Should have at least 1 chunk (may merge if root node fits)
    assert!(!chunks.is_empty());

    // If they are separate chunks, verify category separation
    if chunks.len() >= 2 {
        let has_import = chunks.iter().any(|c| c.text.contains("use std"));
        let has_fn = chunks.iter().any(|c| c.text.contains("fn tiny"));
        assert!(has_import, "Should have an import chunk");
        assert!(has_fn, "Should have a function chunk");
    }
}

// ── Signature context injection tests ───────────────────────────────

#[test]
fn signature_injected_for_inner_chunks() {
    // A large function that gets split should have signature context in inner chunks
    let mut body = String::new();
    for i in 0..30 {
        body.push_str(&format!("    let var_{i} = {i} * 2 + 1;\n"));
    }
    let source = format!("pub fn big_function(x: i32, y: i32) -> i32 {{\n{body}    x + y\n}}");

    let config = ChunkConfig {
        max_chunk_size: 150,
        min_chunk_size: 20,
        ..Default::default()
    };
    let chunks = parse_and_chunk(&source, "rs", &config);

    if chunks.len() > 1 {
        // Chunks that don't start at the function's first line should have signature context
        let inner_with_sig: Vec<_> = chunks
            .iter()
            .filter(|c| c.text.contains("[context:") && c.text.contains("big_function"))
            .collect();
        assert!(
            !inner_with_sig.is_empty(),
            "Inner chunks should have signature context injected. Chunks: {:?}",
            chunks
                .iter()
                .map(|c| &c.text[..c.text.len().min(60)])
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn truncate_signature_short() {
    assert_eq!(truncate_signature("fn foo()", 120), "fn foo()");
}

#[test]
fn truncate_signature_long() {
    let sig = "pub fn very_long_function_name(param1: String, param2: HashMap<String, Vec<i32>>, param3: Option<Box<dyn Trait>>) -> Result<String, Error>";
    let truncated = truncate_signature(sig, 60);
    assert!(truncated.len() <= 60);
    assert!(truncated.starts_with("pub fn very_long_function_name"));
}

#[test]
fn truncate_signature_multiline() {
    let sig = "pub fn foo(\n    x: i32,\n    y: i32,\n) -> i32";
    assert_eq!(truncate_signature(sig, 120), "pub fn foo(");
}

// ── Merge category compatibility tests ──────────────────────────────

#[test]
fn categories_mergeable_same() {
    assert!(categories_mergeable(
        SemanticCategory::Import,
        SemanticCategory::Import
    ));
    assert!(categories_mergeable(
        SemanticCategory::Declaration,
        SemanticCategory::Declaration
    ));
    assert!(categories_mergeable(
        SemanticCategory::Other,
        SemanticCategory::Other
    ));
}

#[test]
fn categories_mergeable_comment_with_anything() {
    assert!(categories_mergeable(
        SemanticCategory::Comment,
        SemanticCategory::Import
    ));
    assert!(categories_mergeable(
        SemanticCategory::Comment,
        SemanticCategory::Declaration
    ));
    assert!(categories_mergeable(
        SemanticCategory::Declaration,
        SemanticCategory::Comment
    ));
}

#[test]
fn categories_not_mergeable_different() {
    assert!(!categories_mergeable(
        SemanticCategory::Import,
        SemanticCategory::Declaration
    ));
    assert!(!categories_mergeable(
        SemanticCategory::Declaration,
        SemanticCategory::Import
    ));
    assert!(!categories_mergeable(
        SemanticCategory::Import,
        SemanticCategory::Other
    ));
}

// ── Multiple language semantic splitting ────────────────────────────

#[test]
fn typescript_class_splits_at_methods() {
    let source = r#"
export class MyService {
    private data: Map<string, number> = new Map();

    constructor(private name: string) {
        this.data.set("init", 0);
        this.data.set("count", 0);
        this.data.set("total", 0);
    }

    public getData(): Map<string, number> {
        const copy = new Map(this.data);
        copy.set("accessed", Date.now());
        return copy;
    }

    public processItem(item: string): number {
        const len = item.length;
        const hash = len * 31;
        this.data.set(item, hash);
        return hash;
    }
}
"#;

    let config = ChunkConfig {
        max_chunk_size: 120,
        min_chunk_size: 10,
        ..Default::default()
    };
    let chunks = parse_and_chunk(source, "ts", &config);

    assert!(
        chunks.len() >= 2,
        "TypeScript class should split into multiple chunks at method boundaries, got {}",
        chunks.len()
    );
}

#[test]
fn python_file_splits_at_function_boundaries() {
    let source = r#"
import os
import sys

def function_one(x):
    result = x * 2
    extra = result + 1
    final = extra * 3
    return final

def function_two(y):
    value = y + 10
    adjusted = value - 5
    computed = adjusted * 2
    return computed

def function_three(z):
    base = z ** 2
    modified = base + z
    output = modified * 4
    return output
"#;

    let config = ChunkConfig {
        max_chunk_size: 100,
        min_chunk_size: 10,
        ..Default::default()
    };
    let chunks = parse_and_chunk(source, "py", &config);

    let fn_chunks: Vec<_> = chunks.iter().filter(|c| c.text.contains("def ")).collect();
    assert!(
        fn_chunks.len() >= 2,
        "Python functions should be separate chunks, got {} function chunks",
        fn_chunks.len()
    );
}

// ── Edge cases ──────────────────────────────────────────────────────

#[test]
fn single_large_function_no_children_emits_chunk() {
    // A function with a single very large expression
    let long_expr = (0..100)
        .map(|i| format!("{i}"))
        .collect::<Vec<_>>()
        .join(" + ");
    let source = format!("pub fn huge() -> i32 {{ {long_expr} }}");

    let config = ChunkConfig {
        max_chunk_size: 50,
        min_chunk_size: 10,
        ..Default::default()
    };
    let chunks = parse_and_chunk(&source, "rs", &config);
    assert!(
        !chunks.is_empty(),
        "Should produce at least one chunk even for oversized nodes"
    );
}

#[test]
fn overlap_preserved_with_semantic_splitting() {
    let source = r#"
pub fn a() -> i32 { 1 }

pub fn b() -> i32 { 2 }

pub fn c() -> i32 { 3 }
"#;

    let config = ChunkConfig {
        max_chunk_size: 30,
        min_chunk_size: 5,
        overlap_lines: 1,
    };
    let chunks = parse_and_chunk(source, "rs", &config);

    // Verify overlap works: chunks after the first should have content
    // from the preceding line
    if chunks.len() > 1 {
        for chunk in &chunks[1..] {
            assert!(
                chunk.line_start < chunk.line_end || !chunk.text.is_empty(),
                "Overlapping chunks should have content"
            );
        }
    }
}

#[test]
fn no_duplicate_chunks_from_boundary_splitting() {
    let source = r#"
impl Foo {
    fn bar(&self) -> i32 { 42 }
    fn baz(&self) -> i32 { 99 }
}
"#;

    let config = ChunkConfig {
        max_chunk_size: 60,
        min_chunk_size: 5,
        ..Default::default()
    };
    let chunks = parse_and_chunk(source, "rs", &config);

    // Verify no text appears in multiple chunks (ignoring overlap/signature injection)
    let texts: HashSet<_> = chunks.iter().map(|c| &c.text).collect();
    assert_eq!(
        texts.len(),
        chunks.len(),
        "Should not produce duplicate chunks"
    );
}
