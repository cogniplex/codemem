use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_rust(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang = tree_sitter_rust::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("failed to set Rust language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_rust_function() {
    let source = r#"
/// Adds two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    assert_eq!(symbols.len(), 1);
    let sym = &symbols[0];
    assert_eq!(sym.name, "add");
    assert_eq!(sym.qualified_name, "add");
    assert_eq!(sym.kind, SymbolKind::Function);
    assert_eq!(sym.visibility, Visibility::Public);
    assert!(sym.signature.contains("pub fn add(a: i32, b: i32) -> i32"));
    assert_eq!(sym.doc_comment.as_deref(), Some("Adds two numbers."));
    assert!(sym.parent.is_none());
}

#[test]
fn extract_rust_struct_and_impl() {
    let source = r#"
pub struct Foo {
    x: i32,
}

impl Foo {
    pub fn new(x: i32) -> Self {
        Self { x }
    }

    fn private_method(&self) -> i32 {
        self.x
    }
}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    // Should have: Struct(Foo), Method(new), Method(private_method)
    assert_eq!(symbols.len(), 3, "Expected 3 symbols, got: {:#?}", symbols);

    let foo = symbols.iter().find(|s| s.name == "Foo").unwrap();
    assert_eq!(foo.kind, SymbolKind::Struct);
    assert_eq!(foo.visibility, Visibility::Public);

    let new_method = symbols.iter().find(|s| s.name == "new").unwrap();
    assert_eq!(new_method.kind, SymbolKind::Method);
    assert_eq!(new_method.qualified_name, "Foo::new");
    assert_eq!(new_method.visibility, Visibility::Public);
    assert_eq!(new_method.parent.as_deref(), Some("Foo"));

    let private = symbols.iter().find(|s| s.name == "private_method").unwrap();
    assert_eq!(private.kind, SymbolKind::Method);
    assert_eq!(private.visibility, Visibility::Private);
    assert_eq!(private.parent.as_deref(), Some("Foo"));
}

#[test]
fn extract_rust_imports() {
    let source = r#"
use std::collections::HashMap;
use crate::parser::CodeParser;
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.rs");

    assert_eq!(
        references.len(),
        2,
        "Expected 2 import refs, got: {:#?}",
        references
    );
    assert!(references.iter().all(|r| r.kind == ReferenceKind::Import));
    assert!(references
        .iter()
        .any(|r| r.target_name == "std::collections::HashMap"));
    assert!(references
        .iter()
        .any(|r| r.target_name == "crate::parser::CodeParser"));
}

#[test]
fn extract_rust_test_function() {
    let source = r#"
#[test]
fn it_works() {
    assert_eq!(2 + 2, 4);
}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    assert_eq!(symbols.len(), 1);
    let sym = &symbols[0];
    assert_eq!(sym.name, "it_works");
    assert_eq!(sym.kind, SymbolKind::Test);
}

#[test]
fn extract_rust_trait_and_impl() {
    let source = r#"
pub trait Greeter {
    fn greet(&self) -> String;
}

pub struct Bot;

impl Greeter for Bot {
    fn greet(&self) -> String {
        "Hello".to_string()
    }
}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();

    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");
    assert!(symbols
        .iter()
        .any(|s| s.name == "Greeter" && s.kind == SymbolKind::Interface));
    assert!(symbols
        .iter()
        .any(|s| s.name == "Bot" && s.kind == SymbolKind::Struct));

    let references = extractor.extract_references(&tree, source.as_bytes(), "test.rs");
    assert!(references
        .iter()
        .any(|r| r.kind == ReferenceKind::Implements && r.target_name == "Greeter"));
}

#[test]
fn extract_rust_enum_and_const() {
    let source = r#"
pub enum Color {
    Red,
    Green,
    Blue,
}

pub const MAX_SIZE: usize = 1024;
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    assert!(symbols
        .iter()
        .any(|s| s.name == "Color" && s.kind == SymbolKind::Enum));
    assert!(symbols
        .iter()
        .any(|s| s.name == "MAX_SIZE" && s.kind == SymbolKind::Constant));
}

#[test]
fn extract_rust_pub_crate_visibility() {
    let source = r#"
pub(crate) fn internal_fn() {}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    assert_eq!(symbols.len(), 1);
    assert_eq!(symbols[0].visibility, Visibility::Crate);
}

#[test]
fn extract_rust_nested_module() {
    let source = r#"
pub mod outer {
    pub fn outer_fn() {}

    pub mod inner {
        pub fn inner_fn() {}
    }
}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    assert!(
        symbols
            .iter()
            .any(|s| s.name == "outer" && s.kind == SymbolKind::Module),
        "Expected outer module, got: {:#?}",
        symbols
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.qualified_name == "outer::outer_fn" && s.kind == SymbolKind::Function),
        "Expected outer::outer_fn function, got: {:#?}",
        symbols
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.qualified_name == "outer::inner" && s.kind == SymbolKind::Module),
        "Expected outer::inner module, got: {:#?}",
        symbols
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.qualified_name == "outer::inner::inner_fn"
                && s.kind == SymbolKind::Function),
        "Expected outer::inner::inner_fn function, got: {:#?}",
        symbols
    );
}

#[test]
fn extract_rust_call_references() {
    let source = r#"
fn caller() {
    let x = foo();
    bar(x);
}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.rs");

    let calls: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Call)
        .collect();
    assert!(calls.iter().any(|r| r.target_name == "foo"));
    assert!(calls.iter().any(|r| r.target_name == "bar"));
}

#[test]
fn extract_rust_type_alias() {
    let source = r#"
pub type Result<T> = std::result::Result<T, MyError>;
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    assert_eq!(symbols.len(), 1);
    assert_eq!(symbols[0].name, "Result");
    assert_eq!(symbols[0].kind, SymbolKind::Type);
}

#[test]
fn extract_rust_static_item() {
    let source = r#"
pub static GLOBAL: &str = "hello";
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    assert_eq!(symbols.len(), 1);
    assert_eq!(symbols[0].name, "GLOBAL");
    assert_eq!(symbols[0].kind, SymbolKind::Constant);
}

#[test]
fn extract_rust_macro_invocation() {
    let source = r#"
fn main() {
    println!("hello");
    vec![1, 2, 3];
}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.rs");

    let macro_calls: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Call && r.target_name.ends_with('!'))
        .collect();
    assert!(
        macro_calls.iter().any(|r| r.target_name == "println!"),
        "Expected println! macro call, got: {:?}",
        macro_calls
    );
    assert!(
        macro_calls.iter().any(|r| r.target_name == "vec!"),
        "Expected vec! macro call, got: {:?}",
        macro_calls
    );
}

#[test]
fn extract_rust_multi_line_doc_comment() {
    let source = r#"
/// First line.
/// Second line.
/// Third line.
pub fn documented() {}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    assert_eq!(symbols.len(), 1);
    assert_eq!(
        symbols[0].doc_comment.as_deref(),
        Some("First line.\nSecond line.\nThird line.")
    );
}

#[test]
fn extract_function_in_module_is_function_not_method() {
    let source = r#"
mod mymod {
    pub fn not_a_method() {}
}
"#;
    let tree = parse_rust(source);
    let extractor = RustExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.rs");

    let func = symbols.iter().find(|s| s.name == "not_a_method").unwrap();
    assert_eq!(
        func.kind,
        SymbolKind::Function,
        "Function inside a module should be Function, not Method"
    );
    assert_eq!(func.qualified_name, "mymod::not_a_method");
}
