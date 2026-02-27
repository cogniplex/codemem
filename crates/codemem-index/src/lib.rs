//! codemem-index: Tree-sitter based code indexing for the Codemem memory engine.
//!
//! Provides a code parsing pipeline that extracts symbols (functions, structs,
//! traits, etc.) and references (calls, imports, implementations) from source
//! files using tree-sitter grammars. Supports incremental indexing with SHA-256
//! change detection and reference resolution into knowledge graph edges.
//!
//! # Architecture
//!
//! - **parser** — Tree-sitter parsing coordinator that dispatches to language extractors
//! - **extractor** — Trait for per-language symbol/reference extraction
//! - **languages** — Language-specific extractors (currently: Rust, TypeScript)
//! - **resolver** — Resolves unresolved references to target symbols, producing graph edges
//! - **indexer** — Main pipeline: directory walking, change detection, parsing
//! - **incremental** — SHA-256 based file change detection
//! - **manifest** — Manifest file parsing for cross-repo dependency detection

pub mod extractor;
pub mod incremental;
pub mod indexer;
pub mod languages;
pub mod manifest;
pub mod parser;
pub mod resolver;
pub mod symbol;

pub use extractor::LanguageExtractor;
pub use indexer::{IndexResult, Indexer};
pub use manifest::{Dependency, ManifestResult, Workspace};
pub use parser::{CodeParser, ParseResult};
pub use resolver::{ReferenceResolver, ResolvedEdge};
pub use symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
