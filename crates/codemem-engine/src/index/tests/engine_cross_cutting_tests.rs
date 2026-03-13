use crate::index::engine::AstGrepEngine;
use crate::index::symbol::SymbolKind;

fn extract(engine: &AstGrepEngine, ext: &str, source: &str) -> Vec<crate::index::symbol::Symbol> {
    let lang = engine.find_language(ext).expect("unsupported ext");
    engine.extract_symbols(lang, source, &format!("test.{ext}"))
}

// ── Go test detection ───────────────────────────────────────────────

#[test]
fn go_test_detection() {
    let engine = AstGrepEngine::new();
    let source = r#"
package main

func TestAdd(t *testing.T) {
    result := add(1, 2)
}

func BenchmarkAdd(b *testing.B) {
    add(1, 2)
}
"#;
    let lang = engine.find_language("go").unwrap();
    let syms = engine.extract_symbols(lang, source, "math_test.go");
    let test_fn = syms.iter().find(|s| s.name == "TestAdd");
    assert!(test_fn.is_some(), "missing TestAdd");
    assert_eq!(test_fn.unwrap().kind, SymbolKind::Test);
}

// ── Rust test detection ─────────────────────────────────────────────

#[test]
fn rust_test_detection() {
    let engine = AstGrepEngine::new();
    let source = r#"
#[test]
fn test_addition() {
    assert_eq!(1 + 1, 2);
}

fn regular_function() {}
"#;
    let syms = extract(&engine, "rs", source);
    let test_fn = syms.iter().find(|s| s.name == "test_addition");
    assert!(test_fn.is_some(), "missing test_addition");
    assert_eq!(test_fn.unwrap().kind, SymbolKind::Test);

    let regular = syms.iter().find(|s| s.name == "regular_function");
    assert!(regular.is_some());
    assert_eq!(regular.unwrap().kind, SymbolKind::Function);
}

// ── Python test detection ───────────────────────────────────────────

#[test]
fn python_test_detection() {
    let engine = AstGrepEngine::new();
    let source = r#"
def test_something():
    assert True

def helper():
    pass
"#;
    let syms = extract(&engine, "py", source);
    let test_fn = syms.iter().find(|s| s.name == "test_something");
    assert!(test_fn.is_some());
    assert_eq!(test_fn.unwrap().kind, SymbolKind::Test);

    let helper = syms.iter().find(|s| s.name == "helper");
    assert!(helper.is_some());
    assert_eq!(helper.unwrap().kind, SymbolKind::Function);
}

// ── Visibility detection ────────────────────────────────────────────

#[test]
fn rust_visibility_detection() {
    use crate::index::symbol::Visibility;

    let engine = AstGrepEngine::new();
    let source = r#"
pub fn public_fn() {}
pub(crate) fn crate_fn() {}
fn private_fn() {}
"#;
    let syms = extract(&engine, "rs", source);
    let public = syms.iter().find(|s| s.name == "public_fn").unwrap();
    assert_eq!(public.visibility, Visibility::Public);

    let crate_vis = syms.iter().find(|s| s.name == "crate_fn").unwrap();
    assert_eq!(crate_vis.visibility, Visibility::Crate);

    let private = syms.iter().find(|s| s.name == "private_fn").unwrap();
    assert_eq!(private.visibility, Visibility::Private);
}

#[test]
fn python_visibility_detection() {
    use crate::index::symbol::Visibility;

    let engine = AstGrepEngine::new();
    let source = r#"
def public_function():
    pass

def _private_function():
    pass

def __mangled_function():
    pass
"#;
    let syms = extract(&engine, "py", source);
    let public = syms.iter().find(|s| s.name == "public_function").unwrap();
    assert_eq!(public.visibility, Visibility::Public);

    let private = syms.iter().find(|s| s.name == "_private_function").unwrap();
    assert_eq!(private.visibility, Visibility::Private);
}

#[test]
fn go_visibility_detection() {
    use crate::index::symbol::Visibility;

    let engine = AstGrepEngine::new();
    let source = r#"
package main

func PublicFunc() {}
func privateFunc() {}
"#;
    let syms = extract(&engine, "go", source);
    let public = syms.iter().find(|s| s.name == "PublicFunc").unwrap();
    assert_eq!(public.visibility, Visibility::Public);

    let private = syms.iter().find(|s| s.name == "privateFunc").unwrap();
    assert_eq!(private.visibility, Visibility::Private);
}

// ── Doc comment extraction ──────────────────────────────────────────

#[test]
fn rust_doc_comment_extraction() {
    let engine = AstGrepEngine::new();
    let source = r#"
/// This is a documented function.
/// It does something important.
pub fn documented() {}

fn undocumented() {}
"#;
    let syms = extract(&engine, "rs", source);
    let documented = syms.iter().find(|s| s.name == "documented").unwrap();
    assert!(documented.doc_comment.is_some());
    let doc = documented.doc_comment.as_ref().unwrap();
    assert!(doc.contains("documented function"));

    let undocumented = syms.iter().find(|s| s.name == "undocumented").unwrap();
    assert!(undocumented.doc_comment.is_none());
}

#[test]
fn python_docstring_extraction() {
    let engine = AstGrepEngine::new();
    let source = r#"
def my_function():
    """This function does something."""
    pass

def no_doc():
    pass
"#;
    let syms = extract(&engine, "py", source);
    let my_fn = syms.iter().find(|s| s.name == "my_function").unwrap();
    assert!(my_fn.doc_comment.is_some());
    assert!(my_fn
        .doc_comment
        .as_ref()
        .unwrap()
        .contains("does something"));

    let no_doc = syms.iter().find(|s| s.name == "no_doc").unwrap();
    assert!(no_doc.doc_comment.is_none());
}

// ── Signature extraction ────────────────────────────────────────────

#[test]
fn rust_signature_extraction() {
    let engine = AstGrepEngine::new();
    let source = r#"
pub fn process(input: &str, count: usize) -> Result<String, Error> {
    Ok(input.to_string())
}
"#;
    let syms = extract(&engine, "rs", source);
    let process = syms.iter().find(|s| s.name == "process").unwrap();
    assert!(process.signature.contains("fn process"));
    assert!(process.signature.contains("input: &str"));
}

#[test]
fn python_signature_extraction() {
    let engine = AstGrepEngine::new();
    let source = r#"
def calculate(x: int, y: int) -> float:
    return x / y
"#;
    let syms = extract(&engine, "py", source);
    let calc = syms.iter().find(|s| s.name == "calculate").unwrap();
    // Python signature is text up to the `:` delimiter
    assert!(
        calc.signature.contains("calculate"),
        "signature should contain function name, got: {}",
        calc.signature
    );
}

// ── Scope / qualified name ──────────────────────────────────────────

#[test]
fn rust_nested_scope() {
    let engine = AstGrepEngine::new();
    let source = r#"
mod outer {
    pub fn top_level() {}

    mod inner {
        pub fn nested_fn() {}
    }
}
"#;
    let syms = extract(&engine, "rs", source);
    let nested = syms.iter().find(|s| s.name == "nested_fn").unwrap();
    assert_eq!(
        nested.qualified_name, "outer::inner::nested_fn",
        "expected outer::inner::nested_fn, got: {}",
        nested.qualified_name
    );
}

#[test]
fn ts_class_method_scope() {
    let engine = AstGrepEngine::new();
    let source = r#"
class MyService {
    getData() {
        return [];
    }
}
"#;
    let syms = extract(&engine, "ts", source);
    let get_data = syms.iter().find(|s| s.name == "getData").unwrap();
    assert_eq!(get_data.qualified_name, "MyService.getData");
    assert_eq!(get_data.parent.as_deref(), Some("MyService"));
}

// ── From<SymbolKind> for NodeKind ────────────────────────────────────

#[test]
fn symbol_kind_to_node_kind() {
    use codemem_core::NodeKind;

    assert_eq!(NodeKind::from(SymbolKind::Function), NodeKind::Function);
    assert_eq!(NodeKind::from(SymbolKind::Method), NodeKind::Method);
    assert_eq!(NodeKind::from(SymbolKind::Class), NodeKind::Class);
    assert_eq!(NodeKind::from(SymbolKind::Struct), NodeKind::Class);
    assert_eq!(NodeKind::from(SymbolKind::Enum), NodeKind::Enum);
    assert_eq!(NodeKind::from(SymbolKind::Interface), NodeKind::Interface);
    assert_eq!(NodeKind::from(SymbolKind::Type), NodeKind::Type);
    assert_eq!(NodeKind::from(SymbolKind::Constant), NodeKind::Constant);
    assert_eq!(NodeKind::from(SymbolKind::Module), NodeKind::Module);
    assert_eq!(NodeKind::from(SymbolKind::Test), NodeKind::Test);
    assert_eq!(NodeKind::from(SymbolKind::Field), NodeKind::Field);
    assert_eq!(NodeKind::from(SymbolKind::Constructor), NodeKind::Method);
}
