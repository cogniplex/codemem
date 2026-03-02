use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_ts(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang = tree_sitter_typescript::LANGUAGE_TSX;
    parser
        .set_language(&lang.into())
        .expect("failed to set TypeScript/TSX language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_ts_function() {
    let source = r#"
/** Adds two numbers. */
export function add(a: number, b: number): number {
    return a + b;
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

    assert_eq!(symbols.len(), 1, "Expected 1 symbol, got: {:#?}", symbols);
    let sym = &symbols[0];
    assert_eq!(sym.name, "add");
    assert_eq!(sym.qualified_name, "add");
    assert_eq!(sym.kind, SymbolKind::Function);
    assert_eq!(sym.visibility, Visibility::Public);
    assert!(
        sym.signature.contains("function add"),
        "Signature should contain 'function add', got: {}",
        sym.signature
    );
    assert_eq!(sym.doc_comment.as_deref(), Some("Adds two numbers."));
    assert!(sym.parent.is_none());
}

#[test]
fn extract_ts_class_and_methods() {
    let source = r#"
export class Greeter {
    private name: string;

    constructor(name: string) {
        this.name = name;
    }

    public greet(): string {
        return `Hello, ${this.name}`;
    }

    private internal(): void {}
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

    // Should have: Class(Greeter), Method(constructor), Method(greet), Method(internal)
    let greeter = symbols.iter().find(|s| s.name == "Greeter");
    assert!(
        greeter.is_some(),
        "Expected Greeter class, got: {:#?}",
        symbols
    );
    let greeter = greeter.unwrap();
    assert_eq!(greeter.kind, SymbolKind::Class);
    assert_eq!(greeter.visibility, Visibility::Public);

    let greet = symbols.iter().find(|s| s.name == "greet");
    assert!(
        greet.is_some(),
        "Expected greet method, got: {:#?}",
        symbols
    );
    let greet = greet.unwrap();
    assert_eq!(greet.kind, SymbolKind::Method);
    assert_eq!(greet.visibility, Visibility::Public);
    assert_eq!(greet.qualified_name, "Greeter.greet");
    assert_eq!(greet.parent.as_deref(), Some("Greeter"));

    let internal = symbols.iter().find(|s| s.name == "internal");
    assert!(
        internal.is_some(),
        "Expected internal method, got: {:#?}",
        symbols
    );
    assert_eq!(internal.unwrap().visibility, Visibility::Private);
}

#[test]
fn extract_ts_interface() {
    let source = r#"
/** Represents a shape. */
export interface Shape {
    area(): number;
    perimeter(): number;
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

    let shape = symbols.iter().find(|s| s.name == "Shape");
    assert!(
        shape.is_some(),
        "Expected Shape interface, got: {:#?}",
        symbols
    );
    let shape = shape.unwrap();
    assert_eq!(shape.kind, SymbolKind::Interface);
    assert_eq!(shape.visibility, Visibility::Public);
    assert_eq!(shape.doc_comment.as_deref(), Some("Represents a shape."));
}

#[test]
fn extract_ts_type_alias_and_enum() {
    let source = r#"
export type Result<T> = { ok: true; value: T } | { ok: false; error: string };

export enum Color {
    Red,
    Green,
    Blue,
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

    assert!(
        symbols
            .iter()
            .any(|s| s.name == "Result" && s.kind == SymbolKind::Type),
        "Expected Result type alias, got: {:#?}",
        symbols
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name == "Color" && s.kind == SymbolKind::Enum),
        "Expected Color enum, got: {:#?}",
        symbols
    );
}

#[test]
fn extract_ts_imports() {
    let source = r#"
import { useState, useEffect } from 'react';
import * as path from 'path';
import express from 'express';
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.ts");

    let imports: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Import)
        .collect();
    assert_eq!(
        imports.len(),
        3,
        "Expected 3 import refs, got: {:#?}",
        imports
    );
    assert!(imports.iter().any(|r| r.target_name == "react"));
    assert!(imports.iter().any(|r| r.target_name == "path"));
    assert!(imports.iter().any(|r| r.target_name == "express"));
}

#[test]
fn extract_ts_class_heritage() {
    let source = r#"
class Animal {
    name: string;
}

interface Swimmer {
    swim(): void;
}

class Dog extends Animal implements Swimmer {
    swim(): void {}
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.ts");

    let inherits: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Inherits)
        .collect();
    assert!(
        inherits
            .iter()
            .any(|r| r.target_name == "Animal" && r.source_qualified_name == "Dog"),
        "Expected Dog extends Animal, got: {:#?}",
        inherits
    );

    let implements: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Implements)
        .collect();
    assert!(
        implements
            .iter()
            .any(|r| r.target_name == "Swimmer" && r.source_qualified_name == "Dog"),
        "Expected Dog implements Swimmer, got: {:#?}",
        implements
    );
}

#[test]
fn extract_ts_call_references() {
    let source = r#"
function caller() {
    const x = foo();
    bar(x);
    console.log("hello");
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "test.ts");

    let calls: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Call)
        .collect();
    assert!(
        calls.iter().any(|r| r.target_name == "foo"),
        "Expected call to foo, got: {:#?}",
        calls
    );
    assert!(
        calls.iter().any(|r| r.target_name == "bar"),
        "Expected call to bar, got: {:#?}",
        calls
    );
    assert!(
        calls.iter().any(|r| r.target_name == "console.log"),
        "Expected call to console.log, got: {:#?}",
        calls
    );
}

#[test]
fn extract_ts_arrow_function() {
    let source = r#"
export const multiply = (a: number, b: number): number => a * b;
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

    let multiply = symbols.iter().find(|s| s.name == "multiply");
    assert!(
        multiply.is_some(),
        "Expected multiply arrow function, got: {:#?}",
        symbols
    );
    let multiply = multiply.unwrap();
    assert_eq!(multiply.kind, SymbolKind::Function);
    assert_eq!(multiply.visibility, Visibility::Public);
}

#[test]
fn extract_ts_non_exported_function() {
    let source = r#"
function privateHelper(): void {
    // ...
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

    assert_eq!(symbols.len(), 1, "Expected 1 symbol, got: {:#?}", symbols);
    assert_eq!(symbols[0].name, "privateHelper");
    assert_eq!(symbols[0].visibility, Visibility::Private);
}

#[test]
fn extract_ts_jsdoc_multiline() {
    let source = r#"
/**
 * Processes the given data.
 * @param data - The input data
 * @returns The processed result
 */
export function process(data: string): string {
    return data;
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

    assert_eq!(symbols.len(), 1);
    let doc = symbols[0].doc_comment.as_deref().unwrap();
    assert!(
        doc.contains("Processes the given data."),
        "Expected JSDoc content, got: {}",
        doc
    );
    assert!(doc.contains("@param data"));
    assert!(doc.contains("@returns"));
}

#[test]
fn extract_ts_namespace() {
    let source = r#"
export namespace Validation {
    export function validate(s: string): boolean {
        return s.length > 0;
    }
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

    let ns = symbols
        .iter()
        .find(|s| s.name == "Validation" && s.kind == SymbolKind::Module);
    assert!(
        ns.is_some(),
        "Expected Validation namespace, got: {:#?}",
        symbols
    );
    assert_eq!(ns.unwrap().visibility, Visibility::Public);

    let validate = symbols.iter().find(|s| s.name == "validate");
    assert!(
        validate.is_some(),
        "Expected validate function, got: {:#?}",
        symbols
    );
    let validate = validate.unwrap();
    assert_eq!(validate.qualified_name, "Validation.validate");
    assert_eq!(validate.kind, SymbolKind::Function);
    assert_eq!(validate.parent.as_deref(), Some("Validation"));
}

#[test]
fn extract_ts_protected_method() {
    let source = r#"
class Base {
    protected helper(): void {}
}
"#;
    let tree = parse_ts(source);
    let extractor = TypeScriptExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

    let helper = symbols.iter().find(|s| s.name == "helper");
    assert!(
        helper.is_some(),
        "Expected helper method, got: {:#?}",
        symbols
    );
    assert_eq!(helper.unwrap().visibility, Visibility::Protected);
}
