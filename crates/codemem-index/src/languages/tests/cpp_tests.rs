use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_cpp(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang = tree_sitter_cpp::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("failed to set C++ language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_cpp_function() {
    let source = r#"
// Adds two integers.
int add(int a, int b) {
    return a + b;
}
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.cpp");

    let add = symbols.iter().find(|s| s.name == "add").unwrap();
    assert_eq!(add.kind, SymbolKind::Function);
    assert_eq!(add.visibility, Visibility::Public);
    assert!(
        add.signature.contains("int add(int a, int b)"),
        "signature was: {}",
        add.signature
    );
    assert_eq!(add.doc_comment.as_deref(), Some("Adds two integers."));
}

#[test]
fn extract_cpp_class() {
    let source = r#"
class MyClass {
public:
    void doSomething() {}
private:
    int x;
};
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "my_class.cpp");

    let cls = symbols.iter().find(|s| s.name == "MyClass").unwrap();
    assert_eq!(cls.kind, SymbolKind::Class);
    assert_eq!(cls.visibility, Visibility::Public);

    // Method inside the class should be extracted.
    let method = symbols.iter().find(|s| s.name == "doSomething").unwrap();
    assert_eq!(method.kind, SymbolKind::Method);
    assert_eq!(method.qualified_name, "MyClass::doSomething");
    assert_eq!(method.parent.as_deref(), Some("MyClass"));
}

#[test]
fn extract_cpp_struct() {
    let source = r#"
struct Point {
    double x;
    double y;
};
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "point.hpp");

    let point = symbols.iter().find(|s| s.name == "Point").unwrap();
    assert_eq!(point.kind, SymbolKind::Struct);
    assert_eq!(point.visibility, Visibility::Public);
}

#[test]
fn extract_cpp_method() {
    let source = r#"
class Calculator {
public:
    int multiply(int a, int b) {
        return a * b;
    }
};
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "calc.cpp");

    let method = symbols.iter().find(|s| s.name == "multiply").unwrap();
    assert_eq!(method.kind, SymbolKind::Method);
    assert_eq!(method.qualified_name, "Calculator::multiply");
    assert_eq!(method.visibility, Visibility::Public);
    assert_eq!(method.parent.as_deref(), Some("Calculator"));
}

#[test]
fn extract_cpp_namespace() {
    let source = r#"
namespace mylib {
    int helper() {
        return 42;
    }
}
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "lib.cpp");

    let ns = symbols.iter().find(|s| s.name == "mylib").unwrap();
    assert_eq!(ns.kind, SymbolKind::Module);
    assert_eq!(ns.signature, "namespace mylib");

    let helper = symbols.iter().find(|s| s.name == "helper").unwrap();
    assert_eq!(helper.kind, SymbolKind::Function);
    assert_eq!(helper.qualified_name, "mylib::helper");
}

#[test]
fn extract_cpp_enum() {
    let source = r#"
enum Color {
    Red,
    Green,
    Blue
};
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "color.h");

    let color = symbols.iter().find(|s| s.name == "Color").unwrap();
    assert_eq!(color.kind, SymbolKind::Enum);
}

#[test]
fn extract_cpp_includes() {
    let source = r#"
#include <stdio.h>
#include "myheader.h"
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "main.c");

    let imports: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Import)
        .collect();
    assert!(
        imports.iter().any(|r| r.target_name == "stdio.h"),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports.iter().any(|r| r.target_name == "myheader.h"),
        "imports: {:#?}",
        imports
    );
}

#[test]
fn extract_cpp_inheritance() {
    let source = r#"
class Base {};

class Derived : public Base {
public:
    void foo() {}
};
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "inherit.cpp");

    let inherits: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Inherits)
        .collect();
    assert!(
        inherits.iter().any(|r| r.target_name == "Base"),
        "inherits: {:#?}",
        inherits
    );
    // The source should be the Derived class.
    let base_ref = inherits.iter().find(|r| r.target_name == "Base").unwrap();
    assert!(
        base_ref.source_qualified_name.contains("Derived"),
        "source_qualified_name: {}",
        base_ref.source_qualified_name
    );
}

#[test]
fn extract_cpp_test_function() {
    let source = r#"
void test_addition() {
    assert(add(1, 2) == 3);
}

void TEST_Subtraction() {
    assert(sub(3, 1) == 2);
}
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test_math.cpp");

    let test1 = symbols.iter().find(|s| s.name == "test_addition").unwrap();
    assert_eq!(test1.kind, SymbolKind::Test);

    let test2 = symbols
        .iter()
        .find(|s| s.name == "TEST_Subtraction")
        .unwrap();
    assert_eq!(test2.kind, SymbolKind::Test);
}

#[test]
fn extract_c_function() {
    let source = r#"
/* Computes the factorial. */
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

static int internal_helper(int x) {
    return x + 1;
}
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "math.c");

    let fact = symbols.iter().find(|s| s.name == "factorial").unwrap();
    assert_eq!(fact.kind, SymbolKind::Function);
    assert_eq!(fact.visibility, Visibility::Public);
    assert!(
        fact.doc_comment.is_some(),
        "Expected doc comment for factorial"
    );

    let helper = symbols
        .iter()
        .find(|s| s.name == "internal_helper")
        .unwrap();
    assert_eq!(helper.kind, SymbolKind::Function);
    assert_eq!(helper.visibility, Visibility::Private);
}

#[test]
fn extract_cpp_visibility() {
    let source = r#"
class Widget {
public:
    void show() {}
    void hide() {}
protected:
    void resize() {}
private:
    void init() {}
};
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "widget.cpp");

    let show = symbols.iter().find(|s| s.name == "show").unwrap();
    assert_eq!(show.visibility, Visibility::Public);

    let hide = symbols.iter().find(|s| s.name == "hide").unwrap();
    assert_eq!(hide.visibility, Visibility::Public);

    let resize = symbols.iter().find(|s| s.name == "resize").unwrap();
    assert_eq!(resize.visibility, Visibility::Protected);

    let init = symbols.iter().find(|s| s.name == "init").unwrap();
    assert_eq!(init.visibility, Visibility::Private);
}

#[test]
fn extract_cpp_preproc_def() {
    let source = r#"
#define MAX_SIZE 1024
#define VERSION "2.0"
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "config.h");

    let max_size = symbols.iter().find(|s| s.name == "MAX_SIZE").unwrap();
    assert_eq!(max_size.kind, SymbolKind::Constant);

    let version = symbols.iter().find(|s| s.name == "VERSION").unwrap();
    assert_eq!(version.kind, SymbolKind::Constant);
}

#[test]
fn extract_cpp_call_references() {
    let source = r#"
void caller() {
    int x = foo();
    bar(x);
}
"#;
    let tree = parse_cpp(source);
    let extractor = CppExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "main.cpp");

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
