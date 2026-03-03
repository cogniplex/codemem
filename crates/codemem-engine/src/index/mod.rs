//! Code indexing engine: ast-grep based code parsing, symbol extraction,
//! reference resolution, and incremental indexing.

pub mod chunker;
pub mod engine;
pub mod incremental;
pub mod indexer;
pub mod manifest;
pub mod parser;
pub mod resolver;
pub mod rule_loader;
pub mod symbol;

pub use chunker::{ChunkConfig, CodeChunk};
pub use indexer::{IndexAndResolveResult, IndexProgress, IndexResult, Indexer};
pub use manifest::{Dependency, ManifestResult, Workspace};
pub use parser::{CodeParser, ParseResult};
pub use resolver::{ReferenceResolver, ResolvedEdge};
pub use symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
