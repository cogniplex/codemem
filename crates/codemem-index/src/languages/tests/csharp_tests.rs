use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_csharp(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang = tree_sitter_c_sharp::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("failed to set C# language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_csharp_class_with_methods() {
    let source = r#"
public class MyService {
    public void Run() {}
    private int Calculate(int x) { return x * 2; }
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "MyService.cs");

    let class = symbols.iter().find(|s| s.name == "MyService").unwrap();
    assert_eq!(class.kind, SymbolKind::Class);
    assert_eq!(class.visibility, Visibility::Public);
    assert!(
        class.signature.contains("class MyService"),
        "signature: {}",
        class.signature
    );

    let run = symbols.iter().find(|s| s.name == "Run").unwrap();
    assert_eq!(run.kind, SymbolKind::Method);
    assert_eq!(run.qualified_name, "MyService.Run");
    assert_eq!(run.parent.as_deref(), Some("MyService"));
    assert_eq!(run.visibility, Visibility::Public);

    let calc = symbols.iter().find(|s| s.name == "Calculate").unwrap();
    assert_eq!(calc.kind, SymbolKind::Method);
    assert_eq!(calc.qualified_name, "MyService.Calculate");
    assert_eq!(calc.visibility, Visibility::Private);
}

#[test]
fn extract_csharp_interface() {
    let source = r#"
public interface IRepository {
    void Save(object entity);
    object FindById(int id);
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "IRepository.cs");

    let iface = symbols.iter().find(|s| s.name == "IRepository").unwrap();
    assert_eq!(iface.kind, SymbolKind::Interface);
    assert_eq!(iface.visibility, Visibility::Public);
}

#[test]
fn extract_csharp_namespace() {
    let source = r#"
namespace MyApp.Services {
    public class UserService {
        public void Create() {}
    }
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "UserService.cs");

    let ns = symbols.iter().find(|s| s.name == "MyApp.Services").unwrap();
    assert_eq!(ns.kind, SymbolKind::Module);

    let class = symbols.iter().find(|s| s.name == "UserService").unwrap();
    assert_eq!(class.kind, SymbolKind::Class);
    assert_eq!(class.qualified_name, "MyApp.Services.UserService");
    assert_eq!(class.parent.as_deref(), Some("MyApp.Services"));

    let method = symbols.iter().find(|s| s.name == "Create").unwrap();
    assert_eq!(method.kind, SymbolKind::Method);
    assert_eq!(method.qualified_name, "MyApp.Services.UserService.Create");
}

#[test]
fn extract_csharp_using_directives() {
    let source = r#"
using System;
using System.Collections.Generic;
using System.Linq;

public class App {}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "App.cs");

    let imports: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Import)
        .collect();
    assert!(
        imports.iter().any(|r| r.target_name == "System"),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports
            .iter()
            .any(|r| r.target_name == "System.Collections.Generic"),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports.iter().any(|r| r.target_name == "System.Linq"),
        "imports: {:#?}",
        imports
    );
}

#[test]
fn extract_csharp_inheritance() {
    let source = r#"
public class MyService : BaseService, IRunnable, IDisposable {
    public void Run() {}
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "MyService.cs");

    let inherits: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Inherits)
        .collect();
    assert!(
        inherits
            .iter()
            .any(|r| r.target_name.contains("BaseService")),
        "inherits: {:#?}",
        inherits
    );

    let implements: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Implements)
        .collect();
    assert!(
        implements.iter().any(|r| r.target_name == "IRunnable"),
        "implements: {:#?}",
        implements
    );
    assert!(
        implements.iter().any(|r| r.target_name == "IDisposable"),
        "implements: {:#?}",
        implements
    );
}

#[test]
fn extract_csharp_enum() {
    let source = r#"
public enum Color {
    Red,
    Green,
    Blue
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Color.cs");

    let en = symbols.iter().find(|s| s.name == "Color").unwrap();
    assert_eq!(en.kind, SymbolKind::Enum);
    assert_eq!(en.visibility, Visibility::Public);
}

#[test]
fn extract_csharp_struct() {
    let source = r#"
public struct Point {
    public int X;
    public int Y;
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Point.cs");

    let st = symbols.iter().find(|s| s.name == "Point").unwrap();
    assert_eq!(st.kind, SymbolKind::Struct);
    assert_eq!(st.visibility, Visibility::Public);
}

#[test]
fn extract_csharp_constructor() {
    let source = r#"
public class Server {
    private int port;

    public Server(int port) {
        this.port = port;
    }
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Server.cs");

    let ctor = symbols
        .iter()
        .find(|s| s.name == "Server" && s.kind == SymbolKind::Method)
        .unwrap();
    assert_eq!(ctor.qualified_name, "Server.Server");
    assert!(
        ctor.signature.contains("Server(int port)"),
        "signature: {}",
        ctor.signature
    );
    assert_eq!(ctor.parent.as_deref(), Some("Server"));
}

#[test]
fn extract_csharp_xml_doc_comment() {
    let source = r#"
/// <summary>
/// A utility class for string operations.
/// Provides helper methods for formatting.
/// </summary>
public class StringUtils {
    /// <summary>
    /// Formats the given input string.
    /// </summary>
    public string Format(string input) {
        return input.Trim();
    }
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "StringUtils.cs");

    let class = symbols.iter().find(|s| s.name == "StringUtils").unwrap();
    let doc = class
        .doc_comment
        .as_ref()
        .expect("expected XML doc on class");
    assert!(
        doc.contains("utility class for string operations"),
        "doc: {}",
        doc
    );

    let method = symbols.iter().find(|s| s.name == "Format").unwrap();
    let method_doc = method
        .doc_comment
        .as_ref()
        .expect("expected XML doc on method");
    assert!(
        method_doc.contains("Formats the given input string"),
        "method_doc: {}",
        method_doc
    );
}

#[test]
fn extract_csharp_visibility() {
    let source = r#"
public class Example {
    public void PublicMethod() {}
    private void PrivateMethod() {}
    protected void ProtectedMethod() {}
    internal void InternalMethod() {}
    void DefaultMethod() {}
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Example.cs");

    let public_m = symbols.iter().find(|s| s.name == "PublicMethod").unwrap();
    assert_eq!(public_m.visibility, Visibility::Public);

    let private_m = symbols.iter().find(|s| s.name == "PrivateMethod").unwrap();
    assert_eq!(private_m.visibility, Visibility::Private);

    let protected_m = symbols
        .iter()
        .find(|s| s.name == "ProtectedMethod")
        .unwrap();
    assert_eq!(protected_m.visibility, Visibility::Protected);

    let internal_m = symbols.iter().find(|s| s.name == "InternalMethod").unwrap();
    assert_eq!(internal_m.visibility, Visibility::Public); // internal -> Public

    let default_m = symbols.iter().find(|s| s.name == "DefaultMethod").unwrap();
    assert_eq!(default_m.visibility, Visibility::Private); // default -> Private
}

#[test]
fn extract_csharp_invocation_references() {
    let source = r#"
public class App {
    public void Run() {
        Console.WriteLine("hello");
        DoWork();
    }

    private void DoWork() {}
}
"#;
    let tree = parse_csharp(source);
    let extractor = CSharpExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "App.cs");

    let calls: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Call)
        .collect();
    assert!(
        calls.iter().any(|r| r.target_name.contains("WriteLine")),
        "calls: {:#?}",
        calls
    );
    assert!(
        calls.iter().any(|r| r.target_name == "DoWork"),
        "calls: {:#?}",
        calls
    );
}
