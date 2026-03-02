use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_php(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang = tree_sitter_php::LANGUAGE_PHP;
    parser
        .set_language(&lang.into())
        .expect("failed to set PHP language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_php_class_with_methods() {
    let source = r#"<?php
class UserService {
    public function findById(int $id): ?User {
        return null;
    }

    private function validate(array $data): bool {
        return true;
    }
}
"#;
    let tree = parse_php(source);
    let extractor = PhpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "UserService.php");

    let class = symbols.iter().find(|s| s.name == "UserService").unwrap();
    assert_eq!(class.kind, SymbolKind::Class);
    assert_eq!(class.visibility, Visibility::Public);
    assert!(
        class.signature.contains("class UserService"),
        "signature: {}",
        class.signature
    );

    let find_method = symbols.iter().find(|s| s.name == "findById").unwrap();
    assert_eq!(find_method.kind, SymbolKind::Method);
    assert_eq!(find_method.qualified_name, "UserService::findById");
    assert_eq!(find_method.visibility, Visibility::Public);
    assert_eq!(find_method.parent.as_deref(), Some("UserService"));

    let validate_method = symbols.iter().find(|s| s.name == "validate").unwrap();
    assert_eq!(validate_method.kind, SymbolKind::Method);
    assert_eq!(validate_method.qualified_name, "UserService::validate");
    assert_eq!(validate_method.visibility, Visibility::Private);
    assert_eq!(validate_method.parent.as_deref(), Some("UserService"));
}

#[test]
fn extract_php_function() {
    let source = r#"<?php
function greet(string $name): string {
    return "Hello, " . $name;
}
"#;
    let tree = parse_php(source);
    let extractor = PhpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "helpers.php");

    let func = symbols.iter().find(|s| s.name == "greet").unwrap();
    assert_eq!(func.kind, SymbolKind::Function);
    assert_eq!(func.visibility, Visibility::Public);
    assert!(
        func.signature.contains("function greet"),
        "signature: {}",
        func.signature
    );
}

#[test]
fn extract_php_namespace() {
    let source = r#"<?php
namespace App\Models {
    class User {
        public function getName(): string {
            return "";
        }
    }
}
"#;
    let tree = parse_php(source);
    let extractor = PhpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "User.php");

    let ns = symbols.iter().find(|s| s.name == "App\\Models").unwrap();
    assert_eq!(ns.kind, SymbolKind::Module);

    let class = symbols.iter().find(|s| s.name == "User").unwrap();
    assert_eq!(class.kind, SymbolKind::Class);
    assert_eq!(class.qualified_name, "App\\Models\\User");
    assert_eq!(class.parent.as_deref(), Some("App\\Models"));

    let method = symbols.iter().find(|s| s.name == "getName").unwrap();
    assert_eq!(method.kind, SymbolKind::Method);
    assert_eq!(method.qualified_name, "App\\Models\\User::getName");
}

#[test]
fn extract_php_use_declarations() {
    let source = r#"<?php
use App\Models\User;
use App\Services\AuthService;
use Illuminate\Http\Request;

class Controller {}
"#;
    let tree = parse_php(source);
    let extractor = PhpExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "Controller.php");

    let imports: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Import)
        .collect();
    assert!(
        imports
            .iter()
            .any(|r| r.target_name.contains("App\\Models\\User")),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports
            .iter()
            .any(|r| r.target_name.contains("App\\Services\\AuthService")),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports
            .iter()
            .any(|r| r.target_name.contains("Illuminate\\Http\\Request")),
        "imports: {:#?}",
        imports
    );
}

#[test]
fn extract_php_inheritance() {
    let source = r#"<?php
class UserController extends Controller implements JsonSerializable, Countable {
    public function index(): void {}
}
"#;
    let tree = parse_php(source);
    let extractor = PhpExtractor::new();
    let references =
        extractor.extract_references(&tree, source.as_bytes(), "UserController.php");

    let inherits: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Inherits)
        .collect();
    assert!(
        inherits
            .iter()
            .any(|r| r.target_name.contains("Controller")),
        "inherits: {:#?}",
        inherits
    );

    let implements: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Implements)
        .collect();
    assert!(
        implements
            .iter()
            .any(|r| r.target_name == "JsonSerializable"),
        "implements: {:#?}",
        implements
    );
    assert!(
        implements.iter().any(|r| r.target_name == "Countable"),
        "implements: {:#?}",
        implements
    );
}

#[test]
fn extract_php_interface_and_trait() {
    let source = r#"<?php
interface Cacheable {
    public function getCacheKey(): string;
}

trait HasTimestamps {
    public function getCreatedAt(): string {
        return "";
    }
}
"#;
    let tree = parse_php(source);
    let extractor = PhpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "traits.php");

    let iface = symbols.iter().find(|s| s.name == "Cacheable").unwrap();
    assert_eq!(iface.kind, SymbolKind::Interface);

    let tr = symbols.iter().find(|s| s.name == "HasTimestamps").unwrap();
    assert_eq!(tr.kind, SymbolKind::Interface);

    let method = symbols.iter().find(|s| s.name == "getCreatedAt").unwrap();
    assert_eq!(method.kind, SymbolKind::Method);
    assert_eq!(method.qualified_name, "HasTimestamps::getCreatedAt");
}

#[test]
fn extract_php_phpdoc() {
    let source = r#"<?php
/**
 * A repository for managing users.
 * Provides CRUD operations.
 */
class UserRepository {
    /**
     * Find a user by their ID.
     * @param int $id
     * @return User|null
     */
    public function find(int $id): ?User {
        return null;
    }
}
"#;
    let tree = parse_php(source);
    let extractor = PhpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "UserRepository.php");

    let class = symbols.iter().find(|s| s.name == "UserRepository").unwrap();
    let doc = class
        .doc_comment
        .as_ref()
        .expect("expected PHPDoc on class");
    assert!(
        doc.contains("repository for managing users"),
        "doc: {}",
        doc
    );

    let method = symbols.iter().find(|s| s.name == "find").unwrap();
    let method_doc = method
        .doc_comment
        .as_ref()
        .expect("expected PHPDoc on method");
    assert!(
        method_doc.contains("Find a user by their ID"),
        "method_doc: {}",
        method_doc
    );
}

#[test]
fn extract_php_function_calls() {
    let source = r#"<?php
class App {
    public function run(): void {
        doWork();
        process($data);
    }
}
"#;
    let tree = parse_php(source);
    let extractor = PhpExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "App.php");

    let calls: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Call)
        .collect();
    assert!(
        calls.iter().any(|r| r.target_name.contains("doWork")),
        "calls: {:#?}",
        calls
    );
    assert!(
        calls.iter().any(|r| r.target_name.contains("process")),
        "calls: {:#?}",
        calls
    );
}

#[test]
fn extract_php_visibility_modifiers() {
    let source = r#"<?php
class Example {
    public function publicMethod(): void {}
    private function privateMethod(): void {}
    protected function protectedMethod(): void {}
    function defaultMethod(): void {}
}
"#;
    let tree = parse_php(source);
    let extractor = PhpExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Example.php");

    let public_m = symbols.iter().find(|s| s.name == "publicMethod").unwrap();
    assert_eq!(public_m.visibility, Visibility::Public);

    let private_m = symbols.iter().find(|s| s.name == "privateMethod").unwrap();
    assert_eq!(private_m.visibility, Visibility::Private);

    let protected_m = symbols
        .iter()
        .find(|s| s.name == "protectedMethod")
        .unwrap();
    assert_eq!(protected_m.visibility, Visibility::Protected);

    let default_m = symbols.iter().find(|s| s.name == "defaultMethod").unwrap();
    assert_eq!(default_m.visibility, Visibility::Public); // Default is public
}
