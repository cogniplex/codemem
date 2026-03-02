use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_python(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang = tree_sitter_python::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("failed to set Python language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_python_function() {
    let source = r#"
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.py");

    assert_eq!(symbols.len(), 1);
    let sym = &symbols[0];
    assert_eq!(sym.name, "add");
    assert_eq!(sym.kind, SymbolKind::Function);
    assert_eq!(sym.visibility, Visibility::Public);
    assert!(sym.signature.contains("def add(a"));
    assert_eq!(sym.doc_comment.as_deref(), Some("Adds two numbers."));
}

#[test]
fn extract_python_class_with_methods() {
    let source = r#"
class Dog:
    """A dog class."""

    def __init__(self, name: str):
        self.name = name

    def bark(self) -> str:
        return "Woof!"
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.py");

    assert!(symbols
        .iter()
        .any(|s| s.name == "Dog" && s.kind == SymbolKind::Class));
    assert!(symbols
        .iter()
        .any(|s| s.name == "__init__" && s.kind == SymbolKind::Method));
    assert!(symbols
        .iter()
        .any(|s| s.name == "bark" && s.kind == SymbolKind::Method));

    let bark = symbols.iter().find(|s| s.name == "bark").unwrap();
    assert_eq!(bark.qualified_name, "Dog.bark");
    assert_eq!(bark.parent.as_deref(), Some("Dog"));
}

#[test]
fn extract_python_imports() {
    let source = r#"
import os
import sys
from pathlib import Path
from collections import defaultdict
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.py");

    let imports: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Import)
        .collect();
    assert!(
        imports.iter().any(|r| r.target_name == "os"),
        "refs: {:#?}",
        imports
    );
    assert!(
        imports.iter().any(|r| r.target_name == "sys"),
        "refs: {:#?}",
        imports
    );
    assert!(
        imports.iter().any(|r| r.target_name == "pathlib"),
        "refs: {:#?}",
        imports
    );
}

#[test]
fn extract_python_class_inheritance() {
    let source = r#"
class Animal:
    pass

class Dog(Animal):
    pass
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.py");

    let inherits: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Inherits)
        .collect();
    assert!(
        inherits
            .iter()
            .any(|r| r.target_name == "Animal" && r.source_qualified_name == "Dog"),
        "Expected Dog inherits Animal, got: {:#?}",
        inherits
    );
}

#[test]
fn extract_python_test_function() {
    let source = r#"
def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 2 - 1 == 1
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.py");

    assert!(
        symbols.iter().all(|s| s.kind == SymbolKind::Test),
        "Expected all test functions, got: {:#?}",
        symbols
    );
}

#[test]
fn extract_python_constants() {
    let source = r#"
MAX_SIZE = 1024
DEFAULT_NAME = "world"
_private = True
regular_var = 42
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.py");

    assert!(
        symbols
            .iter()
            .any(|s| s.name == "MAX_SIZE" && s.kind == SymbolKind::Constant),
        "Expected MAX_SIZE constant, got: {:#?}",
        symbols
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name == "DEFAULT_NAME" && s.kind == SymbolKind::Constant),
        "Expected DEFAULT_NAME constant, got: {:#?}",
        symbols
    );
    // _private and regular_var should not be extracted as constants
    assert!(
        !symbols.iter().any(|s| s.name == "_private"),
        "Should not extract _private"
    );
    assert!(
        !symbols.iter().any(|s| s.name == "regular_var"),
        "Should not extract regular_var"
    );
}

#[test]
fn extract_python_private_function() {
    let source = r#"
def _helper():
    pass

def __really_private():
    pass

def public_fn():
    pass
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.py");

    let helper = symbols.iter().find(|s| s.name == "_helper").unwrap();
    assert_eq!(helper.visibility, Visibility::Private);

    let private = symbols
        .iter()
        .find(|s| s.name == "__really_private")
        .unwrap();
    assert_eq!(private.visibility, Visibility::Private);

    let public = symbols.iter().find(|s| s.name == "public_fn").unwrap();
    assert_eq!(public.visibility, Visibility::Public);
}

#[test]
fn extract_python_nested_class() {
    let source = r#"
class Outer:
    class Inner:
        def inner_method(self):
            pass
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.py");

    assert!(
        symbols.iter().any(|s| s.qualified_name == "Outer.Inner"),
        "Expected Outer.Inner, got: {:#?}",
        symbols
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.qualified_name == "Outer.Inner.inner_method"),
        "Expected Outer.Inner.inner_method, got: {:#?}",
        symbols
    );
}

#[test]
fn extract_python_call_references() {
    let source = r#"
def caller():
    x = foo()
    bar(x)
    baz.method(x)
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.py");

    let calls: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Call)
        .collect();
    assert!(
        calls.iter().any(|r| r.target_name == "foo"),
        "calls: {:#?}",
        calls
    );
    assert!(
        calls.iter().any(|r| r.target_name == "bar"),
        "calls: {:#?}",
        calls
    );
}

#[test]
fn extract_python_multiline_docstring() {
    let source = r#"
def documented():
    """
    First line.
    Second line.
    Third line.
    """
    pass
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.py");

    assert_eq!(symbols.len(), 1);
    let doc = symbols[0].doc_comment.as_deref().unwrap();
    assert!(doc.contains("First line."), "doc: {doc}");
    assert!(doc.contains("Third line."), "doc: {doc}");
}

#[test]
fn extract_python_multiple_inheritance() {
    let source = r#"
class MyClass(Base1, Base2, Base3):
    pass
"#;
    let tree = parse_python(source);
    let extractor = PythonExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.py");

    let inherits: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Inherits)
        .collect();
    assert_eq!(
        inherits.len(),
        3,
        "Expected 3 inheritance refs, got: {:#?}",
        inherits
    );
}
