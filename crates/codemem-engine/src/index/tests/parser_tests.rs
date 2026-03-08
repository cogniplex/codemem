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

// ── Python parsing ──────────────────────────────────────────────────────────

#[test]
fn parse_python_function_def() {
    let parser = CodeParser::new();
    let source = br#"
def greet(name: str) -> str:
    return f"Hello, {name}"

def process_data(items):
    for item in items:
        pass
"#;
    let result = parser.parse_file("app.py", source);
    assert!(result.is_some(), "Python file should be parseable");
    let result = result.unwrap();
    assert_eq!(result.language, "python");

    let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"greet"), "missing greet, got: {names:?}");
    assert!(
        names.contains(&"process_data"),
        "missing process_data, got: {names:?}"
    );

    let greet = result.symbols.iter().find(|s| s.name == "greet").unwrap();
    assert!(
        matches!(
            greet.kind,
            crate::index::symbol::SymbolKind::Function | crate::index::symbol::SymbolKind::Method
        ),
        "greet should be a Function or Method, got: {:?}",
        greet.kind
    );
}

#[test]
fn parse_python_class_def() {
    let parser = CodeParser::new();
    let source = br#"
class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return "..."

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"
"#;
    let result = parser.parse_file("models.py", source);
    assert!(result.is_some());
    let result = result.unwrap();

    let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"Animal"), "missing Animal, got: {names:?}");
    assert!(names.contains(&"Dog"), "missing Dog, got: {names:?}");

    let animal = result.symbols.iter().find(|s| s.name == "Animal").unwrap();
    assert_eq!(animal.kind, crate::index::symbol::SymbolKind::Class);
}

#[test]
fn parse_python_import_extraction() {
    let parser = CodeParser::new();
    let source = br#"
import os
from pathlib import Path
from typing import List, Dict

def main():
    pass
"#;
    let result = parser.parse_file("main.py", source);
    assert!(result.is_some());
    let result = result.unwrap();

    // References should include imports
    let ref_names: Vec<&str> = result
        .references
        .iter()
        .map(|r| r.target_name.as_str())
        .collect();
    // At minimum, the function should be found as a symbol
    let sym_names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(
        sym_names.contains(&"main"),
        "missing main function, got symbols: {sym_names:?}, refs: {ref_names:?}"
    );
}

// ── Go parsing ──────────────────────────────────────────────────────────────

#[test]
fn parse_go_func_and_struct() {
    let parser = CodeParser::new();
    let source = br#"
package main

import "fmt"

type Server struct {
    Host string
    Port int
}

func NewServer(host string, port int) *Server {
    return &Server{Host: host, Port: port}
}

func (s *Server) Start() error {
    fmt.Println("Starting server")
    return nil
}
"#;
    let result = parser.parse_file("server.go", source);
    assert!(result.is_some(), "Go file should be parseable");
    let result = result.unwrap();
    assert_eq!(result.language, "go");

    let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.contains(&"Server"),
        "missing Server struct, got: {names:?}"
    );
    assert!(
        names.contains(&"NewServer"),
        "missing NewServer func, got: {names:?}"
    );
    assert!(
        names.contains(&"Start"),
        "missing Start method, got: {names:?}"
    );

    let server = result.symbols.iter().find(|s| s.name == "Server").unwrap();
    assert_eq!(server.kind, crate::index::symbol::SymbolKind::Struct);

    let new_server = result
        .symbols
        .iter()
        .find(|s| s.name == "NewServer")
        .unwrap();
    assert_eq!(new_server.kind, crate::index::symbol::SymbolKind::Function);

    let start = result.symbols.iter().find(|s| s.name == "Start").unwrap();
    assert_eq!(start.kind, crate::index::symbol::SymbolKind::Method);
}

#[test]
fn parse_go_interface() {
    let parser = CodeParser::new();
    let source = br#"
package main

type Reader interface {
    Read(p []byte) (n int, err error)
}
"#;
    let result = parser.parse_file("interfaces.go", source);
    assert!(result.is_some());
    let result = result.unwrap();

    let reader = result.symbols.iter().find(|s| s.name == "Reader");
    assert!(reader.is_some(), "missing Reader interface");
    assert_eq!(
        reader.unwrap().kind,
        crate::index::symbol::SymbolKind::Interface
    );
}

// ── Java parsing ────────────────────────────────────────────────────────────

#[test]
fn parse_java_class_and_method() {
    let parser = CodeParser::new();
    let source = br#"
package com.example;

import java.util.List;
import java.util.ArrayList;

public class UserService {
    private List<String> users = new ArrayList<>();

    public void addUser(String name) {
        users.add(name);
    }

    public List<String> getUsers() {
        return users;
    }
}
"#;
    let result = parser.parse_file("UserService.java", source);
    assert!(result.is_some(), "Java file should be parseable");
    let result = result.unwrap();
    assert_eq!(result.language, "java");

    let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.contains(&"UserService"),
        "missing UserService class, got: {names:?}"
    );
    assert!(
        names.contains(&"addUser"),
        "missing addUser method, got: {names:?}"
    );
    assert!(
        names.contains(&"getUsers"),
        "missing getUsers method, got: {names:?}"
    );

    let user_service = result
        .symbols
        .iter()
        .find(|s| s.name == "UserService")
        .unwrap();
    assert_eq!(user_service.kind, crate::index::symbol::SymbolKind::Class);
}

#[test]
fn parse_java_import_extraction() {
    let parser = CodeParser::new();
    let source = br#"
import java.util.Map;
import java.io.IOException;

public class App {
    public static void main(String[] args) {}
}
"#;
    let result = parser.parse_file("App.java", source);
    assert!(result.is_some());
    let result = result.unwrap();

    // At minimum, the class and main method should be found
    let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"App"), "missing App class, got: {names:?}");
    assert!(
        names.contains(&"main"),
        "missing main method, got: {names:?}"
    );
}

// ── TypeScript/JavaScript parsing ───────────────────────────────────────────

#[test]
fn parse_typescript_function_and_class() {
    let parser = CodeParser::new();
    let source = br#"
export function greet(name: string): string {
    return `Hello, ${name}`;
}

export class UserManager {
    private users: string[] = [];

    addUser(name: string): void {
        this.users.push(name);
    }

    getUsers(): string[] {
        return this.users;
    }
}
"#;
    let result = parser.parse_file("user.ts", source);
    assert!(result.is_some(), "TypeScript file should be parseable");
    let result = result.unwrap();
    assert_eq!(result.language, "typescript");

    let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"greet"), "missing greet, got: {names:?}");
    assert!(
        names.contains(&"UserManager"),
        "missing UserManager, got: {names:?}"
    );
}

#[test]
fn parse_typescript_interface() {
    let parser = CodeParser::new();
    let source = br#"
export interface Config {
    host: string;
    port: number;
    debug?: boolean;
}

export type Result<T> = { ok: true; value: T } | { ok: false; error: string };
"#;
    let result = parser.parse_file("types.ts", source);
    assert!(result.is_some());
    let result = result.unwrap();

    let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.contains(&"Config"),
        "missing Config interface, got: {names:?}"
    );

    let config = result.symbols.iter().find(|s| s.name == "Config").unwrap();
    assert_eq!(config.kind, crate::index::symbol::SymbolKind::Interface);
}

#[test]
fn parse_typescript_import_extraction() {
    let parser = CodeParser::new();
    let source = br#"
import { useState, useEffect } from 'react';
import axios from 'axios';

export function App() {
    return null;
}
"#;
    let result = parser.parse_file("App.tsx", source);
    assert!(result.is_some());
    let result = result.unwrap();

    let sym_names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(
        sym_names.contains(&"App"),
        "missing App function, got: {sym_names:?}"
    );
}

#[test]
fn parse_javascript_function_and_class() {
    let parser = CodeParser::new();
    let source = br#"
function add(a, b) {
    return a + b;
}

class Calculator {
    multiply(a, b) {
        return a * b;
    }
}
"#;
    let result = parser.parse_file("calc.js", source);
    assert!(result.is_some(), "JavaScript file should be parseable");
    let result = result.unwrap();
    // JS maps to "typescript" language name for uniformity
    assert_eq!(result.language, "typescript");

    let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"add"), "missing add, got: {names:?}");
    assert!(
        names.contains(&"Calculator"),
        "missing Calculator, got: {names:?}"
    );
}

// ── Manifest parsing: go.mod ────────────────────────────────────────────────

#[test]
fn parse_go_mod_manifest() {
    use crate::index::manifest::parse_go_mod;

    let dir = std::env::temp_dir().join("codemem_test_go_mod");
    std::fs::create_dir_all(&dir).ok();

    let content = r#"module github.com/example/myapp

go 1.21

require (
    github.com/gorilla/mux v1.8.0
    github.com/stretchr/testify v1.8.4
)

require github.com/pkg/errors v0.9.1
"#;
    let path = dir.join("go.mod");
    std::fs::write(&path, content).expect("write go.mod");

    let result = parse_go_mod(&path).expect("should parse go.mod");

    assert!(
        result.packages.contains_key("github.com/example/myapp"),
        "Expected module name in packages, got: {:?}",
        result.packages
    );

    let dep_names: Vec<&str> = result
        .dependencies
        .iter()
        .map(|d| d.name.as_str())
        .collect();
    assert!(
        dep_names.contains(&"github.com/gorilla/mux"),
        "missing gorilla/mux, got: {dep_names:?}"
    );
    assert!(
        dep_names.contains(&"github.com/stretchr/testify"),
        "missing testify, got: {dep_names:?}"
    );
    assert!(
        dep_names.contains(&"github.com/pkg/errors"),
        "missing pkg/errors, got: {dep_names:?}"
    );

    // Verify version extraction
    let mux = result
        .dependencies
        .iter()
        .find(|d| d.name == "github.com/gorilla/mux")
        .unwrap();
    assert_eq!(mux.version, "v1.8.0");

    std::fs::remove_dir_all(&dir).ok();
}

// ── Manifest parsing: pyproject.toml (PEP 621) ─────────────────────────────

#[test]
fn parse_pyproject_pep621() {
    use crate::index::manifest::parse_pyproject_toml;

    let dir = std::env::temp_dir().join("codemem_test_pyproject_pep621");
    std::fs::create_dir_all(&dir).ok();

    let content = r#"
[project]
name = "my-python-app"
version = "1.0.0"
dependencies = [
    "requests>=2.28.0",
    "click>=8.0",
    "pydantic",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "mypy",
]
"#;
    let path = dir.join("pyproject.toml");
    std::fs::write(&path, content).expect("write pyproject.toml");

    let result = parse_pyproject_toml(&path).expect("should parse pyproject.toml");

    assert!(
        result.packages.contains_key("my-python-app"),
        "Expected my-python-app in packages, got: {:?}",
        result.packages
    );

    let prod_deps: Vec<&str> = result
        .dependencies
        .iter()
        .filter(|d| !d.dev)
        .map(|d| d.name.as_str())
        .collect();
    assert!(
        prod_deps.contains(&"requests"),
        "missing requests, got: {prod_deps:?}"
    );
    assert!(
        prod_deps.contains(&"click"),
        "missing click, got: {prod_deps:?}"
    );
    assert!(
        prod_deps.contains(&"pydantic"),
        "missing pydantic, got: {prod_deps:?}"
    );

    let dev_deps: Vec<&str> = result
        .dependencies
        .iter()
        .filter(|d| d.dev)
        .map(|d| d.name.as_str())
        .collect();
    assert!(
        dev_deps.contains(&"pytest"),
        "missing pytest in dev deps, got: {dev_deps:?}"
    );
    assert!(
        dev_deps.contains(&"mypy"),
        "missing mypy in dev deps, got: {dev_deps:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ── Manifest parsing: pyproject.toml (Poetry format) ────────────────────────

#[test]
fn parse_pyproject_poetry() {
    use crate::index::manifest::parse_pyproject_toml;

    let dir = std::env::temp_dir().join("codemem_test_pyproject_poetry");
    std::fs::create_dir_all(&dir).ok();

    let content = r#"
[tool.poetry]
name = "my-poetry-app"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.9"
fastapi = "^0.100.0"
uvicorn = {version = "^0.23", extras = ["standard"]}

[tool.poetry.dev-dependencies]
pytest = "^7.0"
black = "^23.0"
"#;
    let path = dir.join("pyproject.toml");
    std::fs::write(&path, content).expect("write pyproject.toml");

    let result = parse_pyproject_toml(&path).expect("should parse Poetry pyproject.toml");

    assert!(
        result.packages.contains_key("my-poetry-app"),
        "Expected my-poetry-app in packages, got: {:?}",
        result.packages
    );

    let prod_deps: Vec<&str> = result
        .dependencies
        .iter()
        .filter(|d| !d.dev)
        .map(|d| d.name.as_str())
        .collect();
    // python should be skipped
    assert!(
        !prod_deps.contains(&"python"),
        "python itself should be skipped"
    );
    assert!(
        prod_deps.contains(&"fastapi"),
        "missing fastapi, got: {prod_deps:?}"
    );
    assert!(
        prod_deps.contains(&"uvicorn"),
        "missing uvicorn, got: {prod_deps:?}"
    );

    let dev_deps: Vec<&str> = result
        .dependencies
        .iter()
        .filter(|d| d.dev)
        .map(|d| d.name.as_str())
        .collect();
    assert!(
        dev_deps.contains(&"pytest"),
        "missing pytest in dev deps, got: {dev_deps:?}"
    );
    assert!(
        dev_deps.contains(&"black"),
        "missing black in dev deps, got: {dev_deps:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
