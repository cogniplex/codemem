//! Code indexing engine: ast-grep based code parsing, symbol extraction,
//! reference resolution, cross-repo linking, and API surface detection.

pub mod api_surface;
pub mod blocklist;
pub mod chunker;
pub mod engine;
pub mod incremental;
pub mod indexer;
pub mod linker;
pub mod manifest;
pub mod parser;
pub mod resolver;
pub mod rule_loader;
pub mod scip;
pub mod spec_parser;
pub mod symbol;

pub use api_surface::{
    detect_client_calls, detect_endpoints, detect_endpoints_from_references, detect_event_calls,
    match_endpoint, match_event_channels, normalize_path_pattern, ApiSurfaceResult,
    DetectedClientCall, DetectedEndpoint, DetectedEventCall,
};
pub use chunker::{ChunkConfig, CodeChunk};
pub use indexer::{IndexAndResolveResult, IndexProgress, IndexResult, Indexer};
pub use linker::{
    backward_link, extract_packages, forward_link, match_symbol, CrossRepoEdge, LinkResult,
    PendingRef, RegisteredPackage, SymbolMatch,
};
pub use manifest::{Dependency, ManifestResult, Workspace};
pub use parser::{CodeParser, ParseResult};
pub use resolver::{ReferenceResolver, ResolveResult, ResolvedEdge, UnresolvedRef};
pub use scip::orchestrate::{OrchestrationResult, ScipLanguage, ScipOrchestrator};
pub use spec_parser::{
    parse_asyncapi, parse_openapi, scan_api_specs, AsyncApiParseResult, SpecChannel, SpecEndpoint,
    SpecFileResult, SpecParseResult,
};
pub use symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
