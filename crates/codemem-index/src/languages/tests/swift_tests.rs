use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_swift(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang = tree_sitter_swift::LANGUAGE.into();
    parser
        .set_language(&lang)
        .expect("failed to set Swift language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_swift_class_with_methods() {
    let source = r#"
public class UserService {
    public func findUser(id: Int) -> User? {
        return nil
    }

    private func validate(name: String) -> Bool {
        return !name.isEmpty
    }
}
"#;
    let tree = parse_swift(source);
    let extractor = SwiftExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "UserService.swift");

    let class = symbols.iter().find(|s| s.name == "UserService");
    assert!(
        class.is_some(),
        "Expected class UserService, got symbols: {:#?}",
        symbols
    );
    let class = class.unwrap();
    assert_eq!(class.kind, SymbolKind::Class);
    assert!(
        class.signature.contains("class UserService"),
        "signature: {}",
        class.signature
    );

    let find_user = symbols.iter().find(|s| s.name == "findUser");
    assert!(
        find_user.is_some(),
        "Expected method findUser, got symbols: {:#?}",
        symbols
    );
    let find_user = find_user.unwrap();
    assert_eq!(find_user.kind, SymbolKind::Method);
    assert_eq!(find_user.qualified_name, "UserService.findUser");
    assert_eq!(find_user.parent.as_deref(), Some("UserService"));

    let validate = symbols.iter().find(|s| s.name == "validate");
    assert!(
        validate.is_some(),
        "Expected method validate, got symbols: {:#?}",
        symbols
    );
    let validate = validate.unwrap();
    assert_eq!(validate.kind, SymbolKind::Method);
    assert_eq!(validate.qualified_name, "UserService.validate");
}

#[test]
fn extract_swift_struct() {
    let source = r#"
public struct Point {
    var x: Double
    var y: Double

    func distance(to other: Point) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        return (dx * dx + dy * dy).squareRoot()
    }
}
"#;
    let tree = parse_swift(source);
    let extractor = SwiftExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Point.swift");

    let s = symbols.iter().find(|s| s.name == "Point");
    assert!(
        s.is_some(),
        "Expected struct Point, got symbols: {:#?}",
        symbols
    );
    let s = s.unwrap();
    assert_eq!(s.kind, SymbolKind::Struct);
    assert!(
        s.signature.contains("struct Point"),
        "signature: {}",
        s.signature
    );

    let distance = symbols.iter().find(|sym| sym.name == "distance");
    assert!(
        distance.is_some(),
        "Expected method distance, got symbols: {:#?}",
        symbols
    );
    let distance = distance.unwrap();
    assert_eq!(distance.kind, SymbolKind::Method);
    assert_eq!(distance.qualified_name, "Point.distance");
    assert_eq!(distance.parent.as_deref(), Some("Point"));
}

#[test]
fn extract_swift_protocol() {
    let source = r#"
public protocol Drawable {
    func draw()
    func resize(width: Int, height: Int)
}
"#;
    let tree = parse_swift(source);
    let extractor = SwiftExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Drawable.swift");

    let proto = symbols.iter().find(|s| s.name == "Drawable");
    assert!(
        proto.is_some(),
        "Expected protocol Drawable, got symbols: {:#?}",
        symbols
    );
    let proto = proto.unwrap();
    assert_eq!(proto.kind, SymbolKind::Interface);
    assert!(
        proto.signature.contains("protocol Drawable"),
        "signature: {}",
        proto.signature
    );
}

#[test]
fn extract_swift_enum() {
    let source = r#"
public enum Direction {
    case north
    case south
    case east
    case west
}
"#;
    let tree = parse_swift(source);
    let extractor = SwiftExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Direction.swift");

    let en = symbols.iter().find(|s| s.name == "Direction");
    assert!(
        en.is_some(),
        "Expected enum Direction, got symbols: {:#?}",
        symbols
    );
    let en = en.unwrap();
    assert_eq!(en.kind, SymbolKind::Enum);
    assert!(
        en.signature.contains("enum Direction"),
        "signature: {}",
        en.signature
    );
}

#[test]
fn extract_swift_imports() {
    let source = r#"
import Foundation
import UIKit
import SwiftUI

class App {}
"#;
    let tree = parse_swift(source);
    let extractor = SwiftExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "App.swift");

    let imports: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Import)
        .collect();
    assert!(
        imports.iter().any(|r| r.target_name == "Foundation"),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports.iter().any(|r| r.target_name == "UIKit"),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports.iter().any(|r| r.target_name == "SwiftUI"),
        "imports: {:#?}",
        imports
    );
}

#[test]
fn extract_swift_free_function() {
    let source = r#"
func greet(name: String) -> String {
    return "Hello, \(name)!"
}
"#;
    let tree = parse_swift(source);
    let extractor = SwiftExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Greet.swift");

    let func = symbols.iter().find(|s| s.name == "greet");
    assert!(
        func.is_some(),
        "Expected function greet, got symbols: {:#?}",
        symbols
    );
    let func = func.unwrap();
    assert_eq!(func.kind, SymbolKind::Function);
    assert!(func.parent.is_none());
}
