# SCIP Integration: Compiler-Grade Code Intelligence

## Problem

Codemem's code graph is built from tree-sitter/ast-grep parsing, which provides structural extraction (functions, classes, imports) but lacks semantic precision:

- **References are heuristic** — `CALLS` and `IMPORTS` edges rely on name matching. `handle_special_reference()` in `references.rs` drops unresolved refs entirely.
- **No type information** — `return_type`, `generic_params` fields exist on `SymbolInfo` but are sparsely populated.
- **No cross-package resolution** — calls into `node_modules/` or `site-packages/` are dropped or linked to file nodes without symbol context.
- **No overload disambiguation** — multiple functions with the same name produce ambiguous edges.
- **Limited graph vocabulary** — the graph has 13 node kinds and 24 edge types, but SCIP provides 87 symbol kinds, 37 syntax kinds, and richer relationship types that the graph doesn't capture.
- **Confidence is uniform** — all ast-grep edges get the same weight regardless of resolution certainty.

The merged LSP enrichment (Pyright for Python, tsc for TypeScript) addresses some of this but requires running server processes per language. SCIP provides the same compiler-grade data in a simpler, offline, batch-friendly format.

## Architecture: SCIP Primary, ast-grep Fallback

### Pipeline

```
Current:
  ast-grep/OXC (always) → basic graph → LSP enrichment (optional) → upgraded graph

New:
  SCIP indexers (primary) → compiler-grade graph
  ast-grep/OXC (fallback) → basic graph for files where no SCIP indexer ran
  LSP enrichment          → removed (SCIP replaces it)
```

For each source file, exactly one path runs — SCIP if an indexer covered it, ast-grep otherwise. No duplicate nodes, no conflicting edges.

### What gets removed

| Component | Why |
|-----------|-----|
| **LSP enrichment** (Pyright, tsc) | Replaced by scip-python, scip-typescript (same compiler engines, offline) |
| **Ruff** (planned Python parser) | Never implemented, now unnecessary |

### What stays unchanged

| Component | Role |
|-----------|------|
| **ast-grep** (tree-sitter + YAML rules) | Fallback for files not covered by SCIP |
| **OXC** (JS/TS parser) | Fallback for JS/TS when scip-typescript not available |
| **Cross-repo linker** | SCIP feeds package info directly into it |
| **Enrichment pipeline** (git, security, complexity, etc.) | Operates on the graph, independent of parser |
| **Code chunks for embedding** | tree-sitter CST-aware chunking runs for all files regardless of SCIP |

### Supported SCIP Indexers

| Language | Indexer | Install | Command |
|----------|---------|---------|---------|
| **Rust** | rust-analyzer | Ships with rust-analyzer | `rust-analyzer scip .` |
| **TypeScript/JavaScript** | scip-typescript | `npm i -g @sourcegraph/scip-typescript` | `scip-typescript index` |
| **Python** | scip-python | `npm i -g @sourcegraph/scip-python` | `scip-python index . --project-name=X` |
| **Java/Scala/Kotlin** | scip-java | Docker or Coursier | `scip-java index` |
| **C/C++** | scip-clang | Binary from GitHub releases | `scip-clang --compdb-path=compile_commands.json` |
| **C#/Visual Basic** | scip-dotnet | `dotnet tool install -g scip-dotnet` | `scip-dotnet index` |
| **Ruby** | scip-ruby | `gem install scip-ruby` | `bundle exec scip-ruby` |
| **Go** | scip-go | `go install .../scip-go@latest` | `scip-go` |
| **PHP** | scip-php | GitHub binary | `scip-php index` |
| **Dart** | scip-dart | GitHub binary | `scip-dart` |

Languages without SCIP indexers (HCL, Swift, etc.) fall back to ast-grep automatically.

### How it works

1. **Auto-detect project languages** from manifests (`Cargo.toml`, `package.json`, `pyproject.toml`, `go.mod`, etc.)
2. **Auto-detect available indexers** on PATH
3. **Run matching indexers** as part of `codemem index`, producing `.scip` files
4. **Build graph from SCIP** for covered files (compiler-grade nodes + edges)
5. **Fall back to ast-grep/OXC** for remaining files (structural extraction)
6. **Merge into single graph** — both paths produce the same node/edge types

## What is SCIP?

SCIP (Source Code Intelligence Protocol) is a language-agnostic protobuf format for code intelligence, created by Sourcegraph. A `.scip` file is a serialized protobuf `Index` containing:

- **Every symbol occurrence** in the codebase with exact source ranges
- **Definition/reference classification** via a role bitmask per occurrence
- **Cross-file and cross-package resolution** via globally unique symbol strings
- **Type/hover documentation** attached to symbol definitions
- **Structural relationships** (implements, type-definition, reference) between symbols
- **Diagnostic information** (compiler errors/warnings) per occurrence

### Why add SCIP

| Dimension | ast-grep + LSP (current) | SCIP primary + ast-grep fallback (new) |
|-----------|--------------------------|---------------------------------------|
| **Primary accuracy** | Structural (heuristic refs) | Compiler-grade where indexer available |
| **Fallback** | LSP upgrade (server lifecycle) | ast-grep (same as today, no change) |
| **Cross-package** | Only with LSP, inconsistent | Built-in via SCIP symbol string format |
| **Languages at compiler-grade** | 2 (Python, TypeScript via LSP) | 10 (all SCIP indexers) |
| **Integration complexity** | Server lifecycle, async protocol | Read protobuf file, iterate |
| **Code chunks** | ast-grep (unchanged) | ast-grep (unchanged) |

## SCIP Schema Overview

```
Index
  metadata: Metadata { project_root, tool_info, version, text_encoding }
  documents[]: Document
    relative_path: string          # "src/auth/jwt.rs"
    language: string               # "rust"
    occurrences[]: Occurrence
      range: int32[]               # [line, startCol, endCol] or [startLine, startCol, endLine, endCol]
      symbol: string               # "rust-analyzer cargo my_crate 1.0.0 auth/jwt/validate()."
      symbol_roles: int32          # bitmask: Definition=0x1, Import=0x2, WriteAccess=0x4, ReadAccess=0x8
      syntax_kind: SyntaxKind      # IdentifierFunction, IdentifierType, etc.
      diagnostics[]: Diagnostic    # Compiler errors/warnings at this location
      enclosing_range: int32[]     # Scope of enclosing definition
    symbols[]: SymbolInformation
      symbol: string               # Same symbol string
      documentation: string[]      # Hover docs (first element = type signature)
      relationships[]: Relationship
        symbol: string             # Target symbol
        is_implementation: bool    # "X implements Y"
        is_type_definition: bool   # "X is the type definition of Y"
        is_reference: bool         # "X references Y"
        is_definition: bool        # "X defines Y"
      kind: Kind                   # Function, Method, Class, Struct, Trait, Enum, Field, ... (87 values)
      display_name: string
      enclosing_symbol: string     # Parent symbol for locals
  external_symbols[]: SymbolInformation  # Hover docs for dependency symbols
```

### Symbol String Format

The symbol string is the key to SCIP's cross-repo linking. It's a deterministic, globally unique identifier:

```
<scheme> ' ' <manager> ' ' <package-name> ' ' <version> ' ' <descriptor-path>

Examples:
  rust-analyzer cargo serde 1.0.0 Serialize#serialize().
  scip-typescript npm @types/node 18.0.0 fs/readFileSync().
  scip-python pip django 4.2.0 django/http/HttpRequest#GET.
  local 42                                                      # File-scoped, not globally unique
```

Descriptor suffixes encode symbol kind:
- `/` = namespace, `#` = type, `.` = term, `(<disambiguator>).` = method
- `[name]` = type parameter, `(name)` = parameter, `!` = macro

**Two repos referencing the same dependency produce identical symbol strings** — this is what enables cross-repo go-to-definition and find-references.

### Occurrence Role Bitmask

| Bit | Name | Meaning |
|-----|------|---------|
| 0x1 | Definition | Symbol is defined here |
| 0x2 | Import | Symbol is imported here |
| 0x4 | WriteAccess | Symbol is written here |
| 0x8 | ReadAccess | Symbol is read here |
| 0x10 | Generated | In generated code |
| 0x20 | Test | In test code |
| 0x40 | ForwardDefinition | Forward declaration |

Roles are OR'd — a single occurrence can be `Definition | WriteAccess` (0x5).

### Index File Characteristics

| Project size | Approx .scip size | Indexing time |
|-------------|-------------------|---------------|
| Small (10K LOC) | 1-5 MB | Seconds |
| Medium (100K LOC) | 10-50 MB | 30s - 2 min |
| Large monorepo (1M+ LOC) | 100-500 MB | 5-30 min |

SCIP is 5-10x smaller than equivalent LSIF indexes (protobuf binary vs JSON). Indexing is **not incremental** — each run produces a complete index.

### SCIP CLI Tool

```bash
# Install
go install github.com/sourcegraph/scip/cmd/scip@latest

# Inspect an index
scip print index.scip              # Human-readable dump
scip print --json index.scip       # JSON format
scip stats index.scip              # Summary statistics
scip snapshot --from=index.scip    # Visual caret-annotated snapshots
scip lint index.scip               # Validate index quality
```

## Rust Integration

### Crate

```toml
[dependencies]
scip = "0.6.1"
protobuf = "=3.7.2"   # Needed for Message::parse_from_bytes
```

The `scip` crate uses **rust-protobuf** (not prost). Types live in `scip::types::*`.

### Reading a .scip File

```rust
use std::fs;
use protobuf::Message;
use scip::types::Index;

let bytes = fs::read("index.scip")?;
let index = Index::parse_from_bytes(&bytes)?;

for doc in &index.documents {
    println!("File: {}", doc.relative_path);

    for occ in &doc.occurrences {
        let is_def = (occ.symbol_roles & 0x1) != 0;
        let (start_line, start_col, end_line, end_col) = match occ.range.len() {
            3 => (occ.range[0], occ.range[1], occ.range[0], occ.range[2]),
            4 => (occ.range[0], occ.range[1], occ.range[2], occ.range[3]),
            _ => continue,
        };
        // Process occurrence...
    }

    for sym in &doc.symbols {
        // sym.symbol — globally unique symbol string
        // sym.documentation — hover docs
        // sym.relationships — implements, type-def, reference edges
        // sym.kind — Function, Method, Class, Trait, etc.
    }
}

// Symbols from external dependencies (hover docs without full index)
for ext in &index.external_symbols {
    // ext.symbol, ext.documentation, ext.kind
}
```

### Symbol Parsing

```rust
use scip::symbol;

// Parse symbol string into structured parts
let sym = symbol::parse_symbol(
    "rust-analyzer cargo my_crate 1.0.0 auth/jwt/validate()."
)?;
// sym.scheme = "rust-analyzer"
// sym.package.manager = "cargo"
// sym.package.name = "my_crate"
// sym.package.version = "1.0.0"
// sym.descriptors = [Namespace("auth"), Namespace("jwt"), Method("validate")]

// Check locality
assert!(symbol::is_local_symbol("local 42"));
assert!(symbol::is_global_symbol("rust-analyzer cargo foo 1.0 Bar#"));
```

## Integration with Codemem

### Expanding the Graph Model

SCIP provides significantly richer data than what the current graph captures. Rather than squeezing SCIP into the existing 13 node kinds, we should expand the graph to represent what SCIP actually gives us.

#### New Node Kinds

SCIP's `SymbolInformation.Kind` has 87 values. We don't need all of them, but the current 13 node kinds miss important categories:

| New NodeKind | SCIP Kind(s) | ID format | Why needed |
|-------------|-------------|-----------|-----------|
| `External` | Any kind from `external_symbols` | `ext:{manager}:{package}:{qualified_name}` | Dependency symbols — stubs with hover docs, no source location |
| `Trait` | Trait (53) | `sym:{name}` | Currently lumped into `Interface` — Rust traits are semantically distinct |
| `Enum` | Enum (11) | `sym:{name}` | Currently lumped into `Constant` — enums are types, not values |
| `EnumVariant` | EnumMember (12) | `sym:{parent}::{name}` | SCIP distinguishes enum members from constants |
| `Field` | Field (15) | `sym:{parent}::{name}` | Currently extracted but no dedicated node kind |
| `TypeParameter` | TypeParameter (58) | `sym:{parent}::[{name}]` | Generics — `T` in `Vec<T>`, enables type-level graph queries |
| `Macro` | Macro (25) | `sym:{name}` | Rust macros, C preprocessor — important for understanding code flow |
| `Property` | Property (41) | `sym:{parent}::{name}` | JS/TS/Python properties distinct from struct fields |

#### New Edge Types

SCIP relationships and occurrence roles map to edges the graph doesn't have:

| New Edge | Source | SCIP data | Why needed |
|----------|--------|-----------|-----------|
| `TYPE_DEFINITION` | symbol → its type | `Relationship.is_type_definition` | "variable X has type Y" — type-level graph |
| `READS` | function → symbol | `symbol_roles & ReadAccess` | Distinguish read vs write access |
| `WRITES` | function → symbol | `symbol_roles & WriteAccess` | Mutation tracking |
| `FORWARD_DECLARES` | declaration → definition | `symbol_roles & ForwardDefinition` | C/C++ headers, Rust trait declarations |
| `OVERRIDES` | method → parent method | Derived from `is_implementation` on methods | Virtual dispatch chains |

#### Expanded Node Payloads

SCIP provides data that should be stored on existing nodes, not just new ones:

| Existing NodeKind | New payload fields from SCIP | Source |
|-------------------|----------------------------|--------|
| Function/Method | `resolved_return_type`, `resolved_param_types`, `generic_params` | `SymbolInformation.documentation[0]` (type signature) |
| Class/Struct | `field_types`, `implements_list` (resolved) | `Relationship.is_implementation` |
| All symbols | `scip_symbol` (canonical ID), `reference_count`, `is_test`, `is_generated` | Occurrence flags + counting |
| All symbols | `diagnostics` (compiler warnings) | `Occurrence.diagnostics` |
| File | `language` (compiler-verified) | `Document.language` |

#### Documentation as Memories

SCIP provides rich hover documentation for every symbol (`SymbolInformation.documentation`). Rather than bloating node payloads, store documentation as **memories attached to the symbol node** via `RELATES_TO` edges:

```
sym:auth::jwt::JwtValidator::validate
    ├── RELATES_TO → memory: "Validates a JWT token against the configured key set.
    │                         Returns Claims on success, JwtError on failure.
    │                         Checks exp, nbf, iss claims."
    │                (type: context, tags: [scip-doc, auto-generated], importance: 0.4)
    └── RELATES_TO → memory: "fn validate(&self, token: &str) -> Result<Claims, JwtError>"
                     (type: context, tags: [scip-signature, auto-generated], importance: 0.3)
```

This approach:
- Keeps node payloads lean (no multi-KB doc strings in JSON)
- Makes docs searchable via recall (embeddings + BM25)
- Allows docs to participate in distillation (low-importance auto-generated → candidates for consolidation)
- Plays well with the memory catalog (area summaries can reference doc memories)

For external symbols specifically, the hover doc memory is often the only context we have — making it even more valuable as a memory rather than a payload field.

### Node and Edge Mapping

**SCIP symbol string → codemem node ID:**

```
SCIP: rust-analyzer cargo my_crate 1.0.0 auth/jwt/JwtValidator#validate().
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          descriptors → qualified_name

Codemem: sym:auth::jwt::JwtValidator::validate
```

The conversion strips scheme/package/version and joins descriptors with the language-appropriate separator (`::` for Rust, `.` for Python/Java/TS).

**Edge mapping:**

| SCIP data | Codemem edge | How |
|-----------|-------------|-----|
| Occurrence with `Definition` in file A, reference in file B | `CALLS` or `IMPORTS` edge B→A | Reference in B points to definition in A |
| Relationship `is_implementation` | `IMPLEMENTS` edge | Direct mapping |
| Relationship `is_type_definition` | `TYPE_DEFINITION` edge (new) | Type-level graph |
| Occurrence with `Import` role | `IMPORTS` edge | Direct mapping |
| Occurrence with `ReadAccess` | `READS` edge (new) | Data flow tracking |
| Occurrence with `WriteAccess` | `WRITES` edge (new) | Mutation tracking |
| Occurrence with `Test` role | Node annotation `is_test = true` | Not an edge |
| Occurrence with `Generated` role | Node annotation `is_generated = true` | Not an edge |

**Confidence:** SCIP edges get `confidence: 1.0` and `properties.source: "scip"`. When matching existing ast-grep edges, upgrade rather than duplicate.

### External Symbol Nodes

For symbols from dependencies (`index.external_symbols` + cross-package references):

```
NodeKind::External
ID: ext:{manager}:{package}:{qualified_name}
Example: ext:cargo:serde:Serialize::serialize
Payload: {
    "package_manager": "cargo",
    "package_name": "serde",
    "package_version": "1.0.0",
    "kind": "Method",
    "scip_symbol": "rust-analyzer cargo serde 1.0.0 Serialize#serialize()."
}
+ RELATES_TO → memory with hover documentation
```

External nodes are lightweight stubs. They feed directly into the cross-repo linker: `package_manager` + `package_name` populate `package_registry` and `unresolved_refs.package_hint`.

### Indexing Pipeline

```
codemem index <directory>
  │
  ├─ Phase 1: Detect languages + available SCIP indexers
  │   - Read manifests (Cargo.toml, package.json, pyproject.toml, go.mod, ...)
  │   - Check PATH for indexers (rust-analyzer, scip-typescript, scip-python, ...)
  │   - Track which file extensions are covered by SCIP vs need ast-grep fallback
  │
  ├─ Phase 2: Run SCIP indexers (parallel where possible)
  │   - rust-analyzer scip .                         → index-rust.scip
  │   - scip-typescript index                        → index-ts.scip
  │   - scip-python index . --project-name={ns}      → index-py.scip
  │   - etc.
  │
  ├─ Phase 3: Build graph from merged SCIP indexes
  │   - Create sym: nodes from definitions (with expanded kinds, type info)
  │   - Create ext: nodes from external_symbols
  │   - Create edges: CALLS, IMPORTS, IMPLEMENTS, TYPE_DEFINITION, READS, WRITES
  │   - Attach hover docs as memories → RELATES_TO edges to sym: nodes
  │   - All edges at confidence 1.0
  │   - Track which files were covered (by relative_path in SCIP Documents)
  │
  ├─ Phase 4: ast-grep/OXC fallback (files NOT in any SCIP index)
  │   - Standard ast-grep pipeline for uncovered files (unchanged from today)
  │   - Edges at default confidence (~0.7)
  │
  ├─ Phase 5: Code chunks for embedding (all files)
  │   - tree-sitter CST-aware chunking (unchanged from today)
  │
  └─ Phase 6: Cross-repo linking, package registration
      - Extract package info from SCIP symbol strings → feed into cross-repo linker
      - ast-grep unresolved refs also feed into linker (unchanged)
```

**Why indexers can't be a generic library:** The `scip` Rust crate reads/writes the protobuf format, but **producing** an index requires the actual compiler for each language — Rust needs cargo + the type system, TypeScript needs Node.js + tsc, Python needs Pyright + virtualenv, etc. Codemem orchestrates them transparently.

### SCIP Symbol → Codemem Node ID

SCIP runs before ast-grep. For SCIP-covered files, nodes are created from SCIP data. The qualified name is extracted from SCIP descriptors:

```rust
fn scip_symbol_to_node_id(scip_symbol: &str, lang_separator: &str) -> Option<String> {
    let sym = symbol::parse_symbol(scip_symbol).ok()?;
    let parts: Vec<&str> = sym.descriptors.iter()
        .map(|d| d.name.as_str())
        .collect();
    if parts.is_empty() { return None; }
    Some(format!("sym:{}", parts.join(lang_separator)))
}

// Rust: sym:auth::jwt::JwtValidator::validate
// Python: sym:auth.jwt.JwtValidator.validate
// TypeScript: sym:auth.jwt.JwtValidator.validate
```

The node ID format is identical to what ast-grep produces — same `sym:` prefix, same qualified name convention. This means if both paths somehow run on the same file, SCIP upgrades the existing node rather than duplicating it.

### Configuration

```toml
[scip]
enabled = true                  # Master switch for SCIP integration
auto_detect_indexers = true     # Check PATH for available indexers
cache_index = true              # Cache .scip files between runs
cache_ttl_hours = 24            # Re-index if cache older than this
create_external_nodes = true    # Create ext: nodes for dependency symbols
max_references_per_symbol = 100 # Skip utility symbols with excessive fan-out
store_docs_as_memories = true   # Attach hover docs as memories to nodes

[scip.indexers]
# Override auto-detected commands (empty = auto-detect from PATH)
rust = ""                       # e.g., "rust-analyzer scip ."
typescript = ""                 # e.g., "scip-typescript index"
python = ""                     # e.g., "scip-python index . --project-name={namespace}"
java = ""                       # e.g., "scip-java index"
go = ""                         # e.g., "scip-go"
```

### CLI and MCP Interface

```bash
# Index with SCIP (auto-detects and runs indexers)
codemem index .

# Index without SCIP (fast, ast-grep only)
codemem index . --skip-scip

# Show SCIP index stats without modifying graph
codemem scip-stats [path/to/index.scip]

# Dry run: show what SCIP would add to the graph
codemem index . --scip-dry-run
```

MCP tool:
```json
{
  "name": "index_codebase",
  "params": {
    "path": ".",
    "skip_scip": false
  }
}
```

## LSP Removal

SCIP fully replaces the LSP enrichment pipeline. The LSP code (`crates/codemem-engine/src/index/lsp/`) should be removed:

| LSP feature | SCIP equivalent |
|------------|----------------|
| Pyright batch mode (Python) | scip-python (same Pyright engine, offline) |
| tsc --noEmit (TypeScript) | scip-typescript (same tsc engine, offline) |
| Edge confidence upgrade | SCIP edges are confidence 1.0 by default |
| External node creation | SCIP `external_symbols` + cross-package references |
| Type annotation enrichment | SCIP `SymbolInformation.documentation` (type signatures) |

**What to delete:**
- `crates/codemem-engine/src/index/lsp/mod.rs` — orchestration
- `crates/codemem-engine/src/index/lsp/pyright.rs` — Python LSP
- `crates/codemem-engine/src/index/lsp/tsserver.rs` — TypeScript LSP
- `crates/codemem-engine/src/persistence/lsp.rs` — LSP result application
- Related CLI flags, config options, and tests

The cross-repo linker and package registry (Phase 2) stay — SCIP feeds directly into them via package info extracted from symbol strings. API surface linking (Phase 4) is independent and stays.

### What SCIP Adds Beyond Current ast-grep

| Capability | Current (ast-grep) | With SCIP |
|-----------|-------------------|-----------|
| Symbol extraction | Structural (CST pattern matching) | Semantic (compiler analysis) |
| Reference resolution | Heuristic name matching, drops unresolved | Compiler-grade, cross-file, cross-package |
| Type information | Signature text only | Parsed types, generics, return types via hover docs |
| Overload disambiguation | Ambiguous | Resolved via method disambiguator in symbol string |
| Cross-package refs | Not tracked | Full resolution with package manager + name + version |
| Implements/inherits | Pattern-based (fragile) | Explicit `Relationship` with `is_implementation` flag |
| Import resolution | File-level only | Symbol-level (knows exactly which symbol is imported) |
| Test detection | Name-based (`test_*`, `*_test`) | Occurrence flag (`symbol_roles & 0x20`) |
| Edge confidence | Uniform (no differentiation) | 1.0 for SCIP-resolved, upgrades existing edges |
| Hover documentation | Not available | Full docs from `SymbolInformation.documentation` |
| Diagnostic info | Not available | Compiler errors/warnings per occurrence |

## Implementation Plan

| Phase | What | Files | Depends on |
|-------|------|-------|------------|
| **1** | Add `scip` + `protobuf` dependencies | `crates/codemem-engine/Cargo.toml` | -- |
| **2** | Expand graph model: new NodeKinds (External, Trait, Enum, EnumVariant, Field, TypeParameter, Macro, Property) + new edge types (TYPE_DEFINITION, READS, WRITES, FORWARD_DECLARES, OVERRIDES) | `crates/codemem-core/src/types.rs` | -- |
| **3** | SCIP reader module: parse `.scip` protobuf, extract definitions/references/externals/relationships into intermediate structs | `crates/codemem-engine/src/index/scip.rs` (new) | Phase 1 |
| **4** | SCIP → graph builder: create nodes + edges from SCIP data, attach docs as memories | `crates/codemem-engine/src/index/scip.rs` | Phases 2, 3 |
| **5** | Indexer orchestration: auto-detect languages + available indexers, run them, merge `.scip` outputs | `crates/codemem-engine/src/index/scip.rs` | Phase 3 |
| **6** | Wire into `codemem index`: SCIP runs first, track covered files, ast-grep runs on remainder | `crates/codemem-engine/src/index/indexer.rs` | Phases 4, 5 |
| **7** | Remove LSP enrichment code | `crates/codemem-engine/src/index/lsp/`, `crates/codemem-engine/src/persistence/lsp.rs` | Phase 6 |
| **8** | Configuration: `[scip]` section in config.toml | `crates/codemem-core/src/config.rs` | -- |
| **9** | Package extraction: feed SCIP package info into cross-repo linker | `crates/codemem-engine/src/index/linker.rs` | Phase 4 |
| **10** | Diagnostic insights: store compiler errors/warnings as enrichment findings | `crates/codemem-engine/src/enrichment/scip_diagnostics.rs` (new) | Phase 4 |
| **11** | CLI + MCP: `codemem scip-stats`, dry-run flag, `enrich_scip` MCP tool | `crates/codemem/src/cli/`, `crates/codemem/src/mcp/` | Phase 6 |

Phases 1-2 and 8 can be done in parallel. Critical path is 1 → 3 → 4 → 6.

## Expected Impact

| Metric | Before (ast-grep only) | With SCIP |
|--------|----------------------|-----------|
| Edge confidence | Uniform ~0.7 | 1.0 for SCIP-resolved, ~0.7 for remainder |
| Cross-file reference coverage | ~60-70% (heuristic) | ~95%+ (compiler-grade) |
| Cross-package references | 0 | All resolved with package info |
| External dependency nodes | 0 | Full stubs with hover docs |
| Type information | Signature text | Parsed types + generics |
| Implements/inherits accuracy | Pattern-based (~80%) | Compiler-verified (100%) |
| Import resolution granularity | File-level | Symbol-level |

## Resolved Design Decisions

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Run indexers inside codemem or external? | **Inside** — `codemem index` auto-detects and runs indexers | Transparent UX. SCIP format is lang-agnostic; indexers are lang-specific but codemem orchestrates them. |
| 2 | Multi-language monorepo: merge or separate? | **Merge** into single enrichment pass | Symbol strings are globally unique across languages. One pass avoids duplicate node lookups. |
| 3 | ast-grep vs SCIP conflicts? | **SCIP wins** — ast-grep skips symbol/ref extraction for SCIP-covered files | Per-file routing: if file appears in a SCIP Document, ast-grep only runs code chunking for it. |
| 4 | Hover doc storage? | **Memories attached to nodes** via RELATES_TO | Keeps payloads lean, makes docs searchable, participates in distillation. |
| 5 | Indexer availability detection? | **Auto-detect from PATH** with config override | `which rust-analyzer`, `which scip-typescript`, etc. Config overrides for custom paths. |
| 6 | LSP integration? | **Remove** | SCIP replaces it entirely — same compiler engines, simpler integration, no server lifecycle. |

## Open Questions

1. **Index freshness:** SCIP indexes become stale when code changes. `codemem index` re-runs indexers, but this adds 30s-30min. Should there be a `--skip-scip` flag for quick re-indexes? Or cache the `.scip` file and only re-index when source files change?

2. **Index storage:** Should codemem cache the `.scip` file for faster re-enrichment, or discard after processing? Caching adds 10-500MB but avoids re-running slow indexers. Recommendation: cache with a configurable TTL.

3. **Incremental processing:** SCIP indexes are full-repo, but graph building could be incremental — only process documents whose `relative_path` changed since last index. This avoids re-processing unchanged files.

4. **Local symbols:** SCIP marks file-scoped symbols as `local <id>`. Skip them? They add volume without cross-file value. Recommendation: skip.

5. **Reference fan-out threshold:** Some symbols (e.g., `String`, `println!`, `console.log`) have thousands of references. Creating an edge for each would bloat the graph. Should we cap at N references per symbol, or skip "utility" symbols entirely? Recommendation: skip symbols with >100 references in a single file (likely stdlib/utility).

6. **OXC future:** With SCIP handling TS/JS when scip-typescript is available, OXC only runs as fallback. Keep it as-is for now — it's the better TS/JS fallback than ast-grep, and removing it is a separate decision.
