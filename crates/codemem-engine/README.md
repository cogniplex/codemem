# codemem-engine

ast-grep based code indexing for 14 languages with incremental change detection.

## Overview

Parses source files to extract symbols (functions, structs, classes, methods, interfaces, constants) and their references (calls, imports, implements, inherits). Uses a unified `AstGrepEngine` driven by compile-time embedded YAML rules. Produces structured data that feeds into the knowledge graph.

## Supported Languages

Rust (.rs), TypeScript/JavaScript (.ts/.tsx/.js/.jsx), Python (.py), Go (.go), C/C++ (.c/.h/.cpp/.hpp), Java (.java), Ruby (.rb), C# (.cs), Kotlin (.kt/.kts), Swift (.swift), PHP (.php), Scala (.scala/.sc), HCL/Terraform (.tf/.hcl)

## Key Types

- `Indexer` — Main pipeline: walks directories, parses files, detects changes
- `CodeParser` — ast-grep parsing coordinator
- `AstGrepEngine` — Unified extraction engine with YAML-driven rules for all languages
- `ReferenceResolver` — Maps references to qualified names, produces graph edges
- `Symbol` — Extracted code entity (name, kind, signature, visibility, file, line range, doc comment)
- `Reference` — Cross-symbol reference (source, target, kind: Call/Import/Inherits/Implements/TypeUsage)
- `ChangeDetector` — SHA-256 file hashing for incremental re-indexing

## Data Flow

```
Directory → Walk → Parse (ast-grep) → Symbols + References
                                            ↓
                                  ReferenceResolver
                                            ↓
                                  ResolvedEdges (CALLS, IMPORTS, INHERITS, ...)
```
