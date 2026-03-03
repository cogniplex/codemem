use super::*;

#[test]
fn parse_rust_file() {
    let parser = CodeParser::new();
    let source = b"pub fn hello() { println!(\"hello\"); }";
    let result = parser.parse_file("src/main.rs", source);
    assert!(result.is_some());
    let result = result.unwrap();
    assert_eq!(result.language, "rust");
    assert!(!result.symbols.is_empty());
}

#[test]
fn unsupported_extension_returns_none() {
    let parser = CodeParser::new();
    let result = parser.parse_file("file.xyz", b"some content");
    assert!(result.is_none());
}

#[test]
fn supported_extensions_includes_rs() {
    let parser = CodeParser::new();
    assert!(parser.supports_extension("rs"));
    assert!(parser.supports_extension("py"));
    assert!(parser.supports_extension("go"));
    assert!(parser.supports_extension("java"));
    assert!(parser.supports_extension("scala"));
    assert!(parser.supports_extension("rb"));
    assert!(parser.supports_extension("cs"));
    assert!(parser.supports_extension("kt"));
    assert!(parser.supports_extension("swift"));
    assert!(parser.supports_extension("php"));
    assert!(parser.supports_extension("tf"));
    assert!(!parser.supports_extension("xyz"));
}
