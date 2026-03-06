use crate::index::engine::AstGrepEngine;
use crate::index::symbol::SymbolKind;

fn extract(engine: &AstGrepEngine, ext: &str, source: &str) -> Vec<crate::index::symbol::Symbol> {
    let lang = engine.find_language(ext).expect("unsupported ext");
    engine.extract_symbols(lang, source, &format!("test.{ext}"))
}

// ── Go ───────────────────────────────────────────────────────────────

#[test]
fn go_struct_and_interface() {
    let engine = AstGrepEngine::new();
    let source = r#"
package main

type Server struct {
    host string
    port int
}

type Handler interface {
    Handle(req Request) Response
}
"#;
    let syms = extract(&engine, "go", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"Server"), "missing Server, got: {names:?}");
    assert!(
        names.contains(&"Handler"),
        "missing Handler, got: {names:?}"
    );

    let server = syms.iter().find(|s| s.name == "Server").unwrap();
    assert_eq!(server.kind, SymbolKind::Struct);

    let handler = syms.iter().find(|s| s.name == "Handler").unwrap();
    assert_eq!(handler.kind, SymbolKind::Interface);
}

#[test]
fn go_type_alias() {
    let engine = AstGrepEngine::new();
    let source = r#"
package main

type MyString string
"#;
    let syms = extract(&engine, "go", source);
    let my_string = syms.iter().find(|s| s.name == "MyString");
    assert!(my_string.is_some(), "missing MyString type alias");
    assert_eq!(my_string.unwrap().kind, SymbolKind::Type);
}

#[test]
fn go_grouped_type_declaration() {
    let engine = AstGrepEngine::new();
    let source = r#"
package main

type (
    Foo struct{}
    Bar interface{}
)
"#;
    let syms = extract(&engine, "go", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"Foo"), "missing Foo, got: {names:?}");
    assert!(names.contains(&"Bar"), "missing Bar, got: {names:?}");
}

#[test]
fn go_const_declaration() {
    let engine = AstGrepEngine::new();
    let source = r#"
package main

const MaxSize = 100

const (
    StatusOK    = 200
    StatusError = 500
)
"#;
    let syms = extract(&engine, "go", source);
    let consts: Vec<&str> = syms
        .iter()
        .filter(|s| s.kind == SymbolKind::Constant)
        .map(|s| s.name.as_str())
        .collect();
    assert!(
        consts.contains(&"MaxSize"),
        "missing MaxSize, got: {consts:?}"
    );
    assert!(
        consts.contains(&"StatusOK"),
        "missing StatusOK, got: {consts:?}"
    );
    assert!(
        consts.contains(&"StatusError"),
        "missing StatusError, got: {consts:?}"
    );
}

#[test]
fn go_var_declaration() {
    let engine = AstGrepEngine::new();
    let source = r#"
package main

var DefaultTimeout = 30

var (
    ErrNotFound = errors.New("not found")
    ErrTimeout  = errors.New("timeout")
)
"#;
    let syms = extract(&engine, "go", source);
    let vars: Vec<&str> = syms
        .iter()
        .filter(|s| s.kind == SymbolKind::Constant)
        .map(|s| s.name.as_str())
        .collect();
    assert!(
        vars.contains(&"DefaultTimeout"),
        "missing DefaultTimeout, got: {vars:?}"
    );
    assert!(
        vars.contains(&"ErrNotFound"),
        "missing ErrNotFound, got: {vars:?}"
    );
    assert!(
        vars.contains(&"ErrTimeout"),
        "missing ErrTimeout, got: {vars:?}"
    );
}

#[test]
fn go_function_and_method() {
    let engine = AstGrepEngine::new();
    let source = r#"
package main

func main() {}

func (s *Server) Start() error {
    return nil
}
"#;
    let syms = extract(&engine, "go", source);
    let main_fn = syms.iter().find(|s| s.name == "main");
    assert!(main_fn.is_some());
    assert_eq!(main_fn.unwrap().kind, SymbolKind::Function);

    let start = syms.iter().find(|s| s.name == "Start");
    assert!(start.is_some());
    assert_eq!(start.unwrap().kind, SymbolKind::Method);
    assert_eq!(start.unwrap().parent.as_deref(), Some("Server"));
}

// ── Swift ────────────────────────────────────────────────────────────

#[test]
fn swift_class_struct_enum() {
    let engine = AstGrepEngine::new();
    let source = r#"
class MyClass {
    func doSomething() {}
}

struct Point {
    var x: Int
    var y: Int
}

enum Direction {
    case north
    case south
}

protocol Drawable {
    func draw()
}
"#;
    let syms = extract(&engine, "swift", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();

    assert!(
        names.contains(&"MyClass"),
        "missing MyClass, got: {names:?}"
    );

    let protocol = syms.iter().find(|s| s.name == "Drawable");
    assert!(protocol.is_some(), "missing Drawable, got: {names:?}");
    assert_eq!(protocol.unwrap().kind, SymbolKind::Interface);
}

// ── Kotlin ───────────────────────────────────────────────────────────

#[test]
fn kotlin_interface() {
    let engine = AstGrepEngine::new();
    let source = r#"
interface Repository {
    fun findAll(): List<Entity>
}

class UserService {
    fun getUser(id: Int): User? = null
}
"#;
    let syms = extract(&engine, "kt", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();

    let repo = syms.iter().find(|s| s.name == "Repository");
    assert!(repo.is_some(), "missing Repository, got: {names:?}");
    assert_eq!(repo.unwrap().kind, SymbolKind::Interface);

    let service = syms.iter().find(|s| s.name == "UserService");
    assert!(service.is_some(), "missing UserService, got: {names:?}");
    assert_eq!(service.unwrap().kind, SymbolKind::Class);
}

// ── C# ───────────────────────────────────────────────────────────────

#[test]
fn csharp_property_and_field() {
    let engine = AstGrepEngine::new();
    let source = r#"
namespace MyApp {
    class User {
        public string Name { get; set; }
        private int age;
    }
}
"#;
    let syms = extract(&engine, "cs", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"User"), "missing User, got: {names:?}");
}

// ── TypeScript ───────────────────────────────────────────────────────

#[test]
fn ts_class_fields() {
    let engine = AstGrepEngine::new();
    let source = r#"
class MyComponent {
    name: string = "default";
    count = 0;
    onClick = () => {};
}
"#;
    let syms = extract(&engine, "ts", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.contains(&"MyComponent"),
        "missing MyComponent, got: {names:?}"
    );
    // onClick should be extracted as a method (arrow field)
    let onclick = syms.iter().find(|s| s.name == "onClick");
    assert!(onclick.is_some(), "missing onClick, got: {names:?}");
}

// ── Ruby ─────────────────────────────────────────────────────────────

#[test]
fn ruby_constants() {
    let engine = AstGrepEngine::new();
    let source = r#"
MAX_SIZE = 100
DEFAULT_NAME = "world"
"#;
    let syms = extract(&engine, "rb", source);
    let consts: Vec<&str> = syms
        .iter()
        .filter(|s| s.kind == SymbolKind::Constant)
        .map(|s| s.name.as_str())
        .collect();
    assert!(
        consts.contains(&"MAX_SIZE"),
        "missing MAX_SIZE, got: {consts:?}"
    );
    assert!(
        consts.contains(&"DEFAULT_NAME"),
        "missing DEFAULT_NAME, got: {consts:?}"
    );
}

// ── Java ─────────────────────────────────────────────────────────────

#[test]
fn java_instance_fields() {
    let engine = AstGrepEngine::new();
    let source = r#"
public class User {
    private String name;
    public static final int MAX_AGE = 150;
}
"#;
    let syms = extract(&engine, "java", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"User"), "missing User, got: {names:?}");
    assert!(
        names.contains(&"name"),
        "missing name field, got: {names:?}"
    );
    assert!(
        names.contains(&"MAX_AGE"),
        "missing MAX_AGE, got: {names:?}"
    );
}

// ── C++ Symbol Tests ────────────────────────────────────────────────

#[test]
fn cpp_class_and_function() {
    let engine = AstGrepEngine::new();
    let source = r#"
class Shape {
public:
    virtual void draw() = 0;
};

void render() {
    // ...
}
"#;
    let syms = extract(&engine, "cpp", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"Shape"), "missing Shape, got: {names:?}");
    // render may include parens in some AST grammars
    assert!(
        names.iter().any(|n| n.contains("render")),
        "missing render, got: {names:?}"
    );
}

#[test]
fn cpp_type_alias() {
    let engine = AstGrepEngine::new();
    let source = r#"
using StringVec = std::vector<std::string>;
"#;
    let syms = extract(&engine, "cpp", source);
    let alias = syms.iter().find(|s| s.name == "StringVec");
    assert!(
        alias.is_some(),
        "missing StringVec, got: {:?}",
        syms.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    assert_eq!(alias.unwrap().kind, SymbolKind::Type);
}

// ── Scala Symbol Tests ──────────────────────────────────────────────

#[test]
fn scala_class_and_object() {
    let engine = AstGrepEngine::new();
    let source = r#"
class MyService {
    def process(): Unit = {}
}

object MyService {
    def apply(): MyService = new MyService()
}

trait Serializable {
    def serialize(): String
}
"#;
    let syms = extract(&engine, "scala", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.contains(&"MyService"),
        "missing MyService, got: {names:?}"
    );
    assert!(
        names.contains(&"Serializable"),
        "missing Serializable, got: {names:?}"
    );

    let trait_sym = syms.iter().find(|s| s.name == "Serializable");
    assert_eq!(trait_sym.unwrap().kind, SymbolKind::Interface);
}

// ── HCL Symbol Tests ────────────────────────────────────────────────

#[test]
fn hcl_resource_and_variable() {
    let engine = AstGrepEngine::new();
    let source = r#"
resource "aws_instance" "web" {
    ami           = "ami-12345"
    instance_type = "t2.micro"
}

variable "region" {
    default = "us-east-1"
}
"#;
    let syms = extract(&engine, "tf", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.iter().any(|n| n.contains("aws_instance")),
        "missing aws_instance resource, got: {names:?}"
    );
}

// ── PHP Symbol Tests ────────────────────────────────────────────────

#[test]
fn php_class_and_method() {
    let engine = AstGrepEngine::new();
    let source = r#"<?php
class UserController {
    public function index() {
        return [];
    }

    private $name;
}
"#;
    let syms = extract(&engine, "php", source);
    let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.contains(&"UserController"),
        "missing UserController, got: {names:?}"
    );
    assert!(
        names.contains(&"index"),
        "missing index method, got: {names:?}"
    );
}
