use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_java(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang = tree_sitter_java::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("failed to set Java language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_java_class() {
    let source = r#"
public class MyService {
    public void run() {}
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "MyService.java");

    let class = symbols.iter().find(|s| s.name == "MyService").unwrap();
    assert_eq!(class.kind, SymbolKind::Class);
    assert_eq!(class.visibility, Visibility::Public);
    assert!(
        class.signature.contains("class MyService"),
        "signature: {}",
        class.signature
    );
}

#[test]
fn extract_java_interface() {
    let source = r#"
public interface Repository {
    void save(Object entity);
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Repository.java");

    let iface = symbols.iter().find(|s| s.name == "Repository").unwrap();
    assert_eq!(iface.kind, SymbolKind::Interface);
    assert_eq!(iface.visibility, Visibility::Public);
}

#[test]
fn extract_java_enum() {
    let source = r#"
public enum Color {
    RED, GREEN, BLUE
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Color.java");

    let en = symbols.iter().find(|s| s.name == "Color").unwrap();
    assert_eq!(en.kind, SymbolKind::Enum);
    assert_eq!(en.visibility, Visibility::Public);
}

#[test]
fn extract_java_method() {
    let source = r#"
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Calculator.java");

    let method = symbols.iter().find(|s| s.name == "add").unwrap();
    assert_eq!(method.kind, SymbolKind::Method);
    assert_eq!(method.qualified_name, "Calculator.add");
    assert_eq!(method.parent.as_deref(), Some("Calculator"));
    assert!(
        method.signature.contains("int add(int a, int b)"),
        "signature: {}",
        method.signature
    );
}

#[test]
fn extract_java_constructor() {
    let source = r#"
public class Server {
    private int port;

    public Server(int port) {
        this.port = port;
    }
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Server.java");

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
fn extract_java_static_constant() {
    let source = r#"
public class Config {
    public static final int MAX_SIZE = 1024;
    public static final String VERSION = "1.0";
    private int mutableField = 0;
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Config.java");

    let constants: Vec<_> = symbols
        .iter()
        .filter(|s| s.kind == SymbolKind::Constant)
        .collect();
    assert!(
        constants.iter().any(|s| s.name == "MAX_SIZE"),
        "Expected MAX_SIZE, got: {:#?}",
        constants
    );
    assert!(
        constants.iter().any(|s| s.name == "VERSION"),
        "Expected VERSION, got: {:#?}",
        constants
    );
    // mutableField should NOT appear as a constant
    assert!(
        !constants.iter().any(|s| s.name == "mutableField"),
        "mutableField should not be a constant"
    );
}

#[test]
fn extract_java_imports() {
    let source = r#"
import java.util.List;
import java.util.Map;
import java.io.IOException;

public class App {}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "App.java");

    let imports: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Import)
        .collect();
    assert!(
        imports.iter().any(|r| r.target_name == "java.util.List"),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports.iter().any(|r| r.target_name == "java.util.Map"),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports
            .iter()
            .any(|r| r.target_name == "java.io.IOException"),
        "imports: {:#?}",
        imports
    );
}

#[test]
fn extract_java_inheritance() {
    let source = r#"
public class MyService extends BaseService implements Runnable, Serializable {
    public void run() {}
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "MyService.java");

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
        implements.iter().any(|r| r.target_name == "Runnable"),
        "implements: {:#?}",
        implements
    );
    assert!(
        implements.iter().any(|r| r.target_name == "Serializable"),
        "implements: {:#?}",
        implements
    );
}

#[test]
fn extract_java_test_method() {
    let source = r#"
import org.junit.Test;

public class CalculatorTest {
    @Test
    public void testAdd() {
        assert(1 + 1 == 2);
    }

    public void helperMethod() {}
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "CalculatorTest.java");

    let test_method = symbols.iter().find(|s| s.name == "testAdd").unwrap();
    assert_eq!(test_method.kind, SymbolKind::Test);

    let helper = symbols.iter().find(|s| s.name == "helperMethod").unwrap();
    assert_eq!(helper.kind, SymbolKind::Method);
}

#[test]
fn extract_java_visibility() {
    let source = r#"
public class Example {
    public void publicMethod() {}
    private void privateMethod() {}
    protected void protectedMethod() {}
    void packagePrivateMethod() {}
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Example.java");

    let public_m = symbols.iter().find(|s| s.name == "publicMethod").unwrap();
    assert_eq!(public_m.visibility, Visibility::Public);

    let private_m = symbols.iter().find(|s| s.name == "privateMethod").unwrap();
    assert_eq!(private_m.visibility, Visibility::Private);

    let protected_m = symbols
        .iter()
        .find(|s| s.name == "protectedMethod")
        .unwrap();
    assert_eq!(protected_m.visibility, Visibility::Public); // protected → Public

    let pkg_m = symbols
        .iter()
        .find(|s| s.name == "packagePrivateMethod")
        .unwrap();
    assert_eq!(pkg_m.visibility, Visibility::Private); // package-private → Private
}

#[test]
fn extract_java_javadoc() {
    let source = r#"
/**
 * A utility class for string operations.
 * Provides helper methods for formatting.
 */
public class StringUtils {
    /**
     * Formats the given input string.
     * @param input the raw string
     * @return the formatted string
     */
    public String format(String input) {
        return input.trim();
    }
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "StringUtils.java");

    let class = symbols.iter().find(|s| s.name == "StringUtils").unwrap();
    let doc = class
        .doc_comment
        .as_ref()
        .expect("expected Javadoc on class");
    assert!(
        doc.contains("utility class for string operations"),
        "doc: {}",
        doc
    );

    let method = symbols.iter().find(|s| s.name == "format").unwrap();
    let method_doc = method
        .doc_comment
        .as_ref()
        .expect("expected Javadoc on method");
    assert!(
        method_doc.contains("Formats the given input string"),
        "method_doc: {}",
        method_doc
    );
}

#[test]
fn extract_java_method_invocation_references() {
    let source = r#"
public class App {
    public void run() {
        System.out.println("hello");
        doWork();
    }

    private void doWork() {}
}
"#;
    let tree = parse_java(source);
    let extractor = JavaExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "App.java");

    let calls: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Call)
        .collect();
    assert!(
        calls.iter().any(|r| r.target_name.contains("println")),
        "calls: {:#?}",
        calls
    );
    assert!(
        calls.iter().any(|r| r.target_name == "doWork"),
        "calls: {:#?}",
        calls
    );
}
