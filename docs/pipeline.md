# Index & Enrich Pipeline

This document traces the journey of data through Codemem's index and enrichment pipeline -- from source files on disk to a fully annotated knowledge graph with insights.

## Overview

```
Source Files → Index (tree-sitter + SCIP) → Persist (nodes, edges, embeddings, compaction)
                                                  ↓
                                             Enrich (14 analyses) → Insight memories
                                                  ↓
                                             Temporal (git commits → graph nodes)
                                                  ↓
                                             PageRank + Louvain → Annotated graph
```

The pipeline runs via two entry points:
- **CLI**: `codemem analyze` (full pipeline) or `codemem index` (indexing only)
- **CLI**: `codemem review` (diff-aware blast radius analysis, reads diff from stdin)

---

## Step 1: Indexing

**Entry**: `Indexer::index_and_resolve(root)` in `codemem-engine/src/index/`

### 1a. File Discovery

The indexer walks the project directory, skipping:
- Files matched by `.gitignore` rules
- Binary files and known non-code extensions
- Files larger than the configured size limit

### 1b. AST Parsing (tree-sitter via ast-grep)

Each file is parsed using language-specific tree-sitter grammars. Codemem supports 14 languages: Rust, TypeScript, JavaScript, JSX/TSX, Python, Go, C, C++, Java, Ruby, C#, Kotlin, Swift, PHP, Scala, HCL/Terraform.

**Extracted per file:**
- **Symbols**: Functions, methods, classes, structs, enums, interfaces, types, constants, modules, tests, fields, properties, constructors, enum variants, macros, decorators
- **References**: Import statements, function calls, type references (with Rust grouped import decomposition: `std::{HashMap, HashSet}`)
- **Metadata**: Parameters, return types, visibility, doc comments, async/abstract flags, generic params, attributes

### 1c. Chunking

Files are split into overlapping code chunks for embedding. Each chunk:
- Covers a contiguous range of lines
- Is assigned to a parent symbol via O(log n) interval lookup (`SymbolIntervalIndex`)
- Carries `node_kind` labels (comma-separated when merged)

### 1d. Manifest Parsing

Package manifests are parsed to create Package nodes and DEPENDS_ON edges:
- `Cargo.toml` (Rust)
- `package.json` (Node.js)
- `go.mod` (Go)
- `pyproject.toml` (Python, PEP 621 + Poetry)

### 1e. SCIP Enrichment (optional)

When SCIP indexers are installed and enabled (`[scip] enabled = true` in config), codemem runs them to produce compiler-grade cross-references. The process:

1. **Language detection**: Scan manifest files (Cargo.toml, package.json, go.mod, pyproject.toml) to identify languages
2. **Indexer discovery**: Check PATH for the corresponding SCIP indexer (rust-analyzer, scip-typescript, etc.)
3. **Index generation**: Run the indexer to produce a `.scip` protobuf file
4. **Edge fusion**: SCIP edges (confidence 0.15) are fused with ast-grep pattern edges (confidence 0.10). When both sources identify the same edge, their confidence scores add (0.25 total). SCIP-only edges supersede ast-grep duplicates
5. **Containment hierarchy**: File → module → class → method containment tree built from SCIP's nested symbol structure
6. **External nodes**: Dependency symbols (from `ext:` references) create External nodes with package manager metadata
7. **Caching**: `.scip` files cached at `~/.codemem/scip-cache/{namespace}/` with configurable TTL (default 24h)

Intra-class edges are collapsed into parent metadata when `collapse_intra_class_edges = true` (default). Fan-out limits prevent hub symbols from creating excessive edges.

### 1f. Reference Resolution

References are resolved against the symbol table to create typed edges:
- `CALLS` (function/method invocations)
- `IMPORTS` (import statements)
- `EXTENDS` / `IMPLEMENTS` / `INHERITS` (type relationships)
- `CONTAINS` (structural parent-child)
- `DEPENDS_ON` (package dependencies)

References are deduplicated by (source, target, kind) before edge creation.

---

## Step 2: Persistence

**Entry**: `CodememEngine::persist_index_results()` in `codemem-engine/src/persistence/`

### 2a. Graph Nodes

Each symbol and file becomes a graph node with kind-specific attributes:
- `file:src/main.rs` (kind: File)
- `sym:MyStruct::my_method` (kind: Method)
- `pkg:src/` (kind: Package)
- `chunk:src/main.rs:10-25` (kind: Chunk)

Nodes are persisted via multi-row INSERT batching (respecting SQLite's 999-parameter limit).

### 2b. Graph Edges

Resolved references become weighted edges. Edge weights are computed by `edge_weight_for()` based on relationship type:
- CALLS: 1.0
- IMPORTS: 0.8
- CONTAINS: 0.6
- DEPENDS_ON: 0.5
- etc.

Edges are also batched via multi-row INSERT.

### 2c. Embeddings

Symbols and chunks are embedded using the configured provider (Candle/Ollama/OpenAI). Text is contextually enriched before embedding:
- Symbol: qualified name + kind + file path + doc comment excerpt
- Chunk: file path + parent symbol + content

Embeddings are batched (chunks of 64) and inserted into both the HNSW vector index (via batch insert) and SQLite storage (via multi-row INSERT). The embedding mutex is acquired per-batch rather than for the full pipeline to reduce lock contention.

### 2d. Compaction

After persistence, a two-pass compaction prunes low-value nodes:

**Chunk scoring** (keep top N per file):
- Centrality (PageRank/betweenness)
- Structural parent linkage
- Memory link (has associated memories)
- Content density

**Symbol scoring** (prune disconnected/trivial symbols):
- Call connectivity
- Visibility (pub > pub(crate) > private)
- Kind importance (class > function > field)
- Memory link
- Code size

Cold-start-aware: when no memories exist, the memory_link weight is redistributed to other scoring factors.

### 2e. Change Detection

An incremental `ChangeDetector` tracks file hashes. On subsequent runs, only modified files are re-indexed. The detector state is persisted to storage.

---

## Step 3: Enrichment

**Entry**: `CodememEngine::run_enrichments()` in `codemem-engine/src/enrichment/mod.rs`

The enrichment pipeline runs 14 analyses over the indexed graph. Each analysis:
1. Reads existing graph nodes and edges
2. Performs its analysis (git history, static checks, graph algorithms, etc.)
3. Stores findings as Insight-type memories tagged `static-analysis` + `track:<analysis>`
4. Links insights to relevant graph nodes via RELATES_TO edges

### Dispatch

```rust
// Simplified from enrichment/mod.rs
run_analysis!("git",           enrich_git_history(path, days, namespace));
run_analysis!("security",      enrich_security(namespace));
run_analysis!("performance",   enrich_performance(10, namespace));
run_analysis!("complexity",    enrich_complexity(namespace, project_root));
run_analysis!("code_smells",   enrich_code_smells(namespace, project_root));
run_analysis!("security_scan", enrich_security_scan(namespace, project_root));
run_analysis!("architecture",  enrich_architecture(namespace));
run_analysis!("test_mapping",  enrich_test_mapping(namespace));
run_analysis!("api_surface",   enrich_api_surface(namespace));
run_analysis!("doc_coverage",  enrich_doc_coverage(namespace));
run_analysis!("hot_complex",   enrich_hot_complex(namespace));
run_analysis!("blame",         enrich_blame(path, namespace));
run_analysis!("quality",       enrich_quality_stratification(namespace));
// change_impact requires explicit file_path, excluded from run-all
```

### The 14 Analyses

| Analysis | Source | What it produces |
|----------|--------|-----------------|
| **git** | Git log | Commit counts, churn rates, CO_CHANGED edges, activity insights |
| **security** | Graph nodes | Auth patterns, validation checks, trust boundary identification |
| **performance** | Graph centrality | Performance hotspots ranked by connectivity |
| **complexity** | Source files | Cyclomatic and cognitive complexity scores per function |
| **code_smells** | Source files | Long methods, deep nesting, large classes, parameter lists |
| **security_scan** | Source files | Hardcoded secrets, unsafe patterns, injection risks |
| **architecture** | Graph structure | Module boundaries, layering violations, dependency patterns |
| **test_mapping** | Graph nodes | Test-to-code mapping, coverage gaps |
| **api_surface** | Graph nodes | Public API endpoints, handler documentation |
| **doc_coverage** | Graph nodes | Documentation coverage percentage per module |
| **hot_complex** | Git + complexity | Files that are both frequently changed AND complex |
| **blame** | Git blame | File ownership, contributor distribution |
| **quality** | All metrics | Quality stratification: critical/important/standard/low-priority |
| **change_impact** | Git + graph | Blast radius of changes to a specific file |

### Insight Dedup

Before storing, each insight is checked for semantic near-duplicates (cosine > 0.90 against existing insights with the same track tag). Duplicates are silently skipped to prevent bloat from repeated analysis runs.

---

## Step 3.5: Temporal Graph

**Entry**: `temporal::ingest_temporal_layer()` in `codemem-engine/src/enrichment/temporal.rs`

Integrated into `enrich_git_history()`, the temporal layer records git history as first-class graph nodes:

1. **Commit parsing**: `git log` output is parsed with parent hashes and subjects
2. **Commit nodes**: Each commit becomes a `Commit` node (kind: Commit) with author, date, and subject metadata
3. **ModifiedBy edges**: File-level edges for all commits; symbol-level edges for recent commits (30-day cutoff)
4. **PR detection**: Squash/merge commit patterns are detected and annotated
5. **Bot compaction**: Dependabot, Renovate, and lock-file-only commits are flagged
6. **Deleted symbol expiry**: Files/symbols removed in recent commits get `valid_to` set, marking them as expired
7. **Incremental ingestion**: A sentinel node tracks the last-processed commit, avoiding duplicate work

All graph nodes carry `valid_from` and `valid_to` timestamps. Expired nodes (where `valid_to < now`) are automatically skipped in recall scoring, graph linking, and code search. The `graph_traverse` tool accepts an `at_time` parameter for point-in-time filtered traversal.

---

## Step 4: Graph Analysis

After enrichment, the full pipeline computes:

### PageRank

Standard PageRank (damping=0.85, 100 iterations, tolerance=1e-6) over the entire graph. Scores are cached and used by:
- `find_important_nodes` tool (top-K query)
- Hybrid scoring's graph_strength component
- Compaction scoring

### Louvain Community Detection

Louvain algorithm (resolution=1.0) partitions the graph into clusters of tightly connected nodes. Used by:
- `find_related_groups` tool
- Architecture analysis enrichment
- Cluster-based consolidation

---

## Full Pipeline Summary

```
codemem analyze /path/to/project
  │
  ├─ Step 1: Index
  │   ├─ Walk files (skip .gitignore, binaries)
  │   ├─ Parse ASTs (14 languages via tree-sitter)
  │   ├─ Extract symbols, references, chunks
  │   ├─ Parse manifests (Cargo.toml, package.json, etc.)
  │   └─ SCIP enrichment (if indexers installed)
  │       ├─ Auto-detect languages + indexers
  │       ├─ Run indexers → .scip protobuf
  │       └─ Fuse edges (ast-grep 0.10 + SCIP 0.15 = 0.25)
  │
  ├─ Step 2: Persist
  │   ├─ Upsert graph nodes (file, sym, pkg, chunk, commit)
  │   ├─ Resolve references → typed edges
  │   ├─ Embed symbols + chunks (batched, contextual)
  │   └─ Compact: prune low-value chunks + symbols
  │
  ├─ Step 3: Enrich (14 analyses)
  │   ├─ Git history, blame, hot+complex
  │   ├─ Security, security_scan
  │   ├─ Performance, complexity, code_smells
  │   ├─ Architecture, test_mapping, API surface
  │   ├─ Doc coverage, quality stratification
  │   └─ Each → Insight memories tagged static-analysis
  │
  ├─ Step 3.5: Temporal Graph
  │   ├─ Parse git log → Commit nodes
  │   ├─ Create ModifiedBy edges (file + symbol level)
  │   ├─ Detect squash/merge PRs, compact bot commits
  │   └─ Set valid_to on deleted files/symbols
  │
  └─ Step 4: Graph Analysis
      ├─ PageRank (top-10 displayed)
      └─ Louvain communities (count displayed)
```

Run `codemem stats` afterwards to see updated totals.
