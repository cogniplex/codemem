# codemem-engine

Domain logic engine for the Codemem memory system.

## Overview

`CodememEngine` is the central struct that holds all backends (storage, embeddings, graph) and orchestrates all domain operations. It handles code indexing, memory persistence, recall, enrichment, consolidation, and session management.

## Modules

| Module | Purpose |
|--------|---------|
| `index/` | ast-grep code indexing for 14 languages, YAML-driven rules, manifest parsing, reference resolution |
| `hooks/` | Lifecycle hook handlers for tool types (Read/Glob/Grep/Edit/Write/Bash/WebFetch/WebSearch/Agent/ListDir), trigger-based auto-insights |
| `watch/` | Real-time file watcher with <50ms debounce and .gitignore support |
| `enrichment/` | 14 enrichment analyses (git history, security, performance, complexity, etc.) |
| `consolidation/` | 5 neuroscience-inspired cycles: decay, creative, cluster, forget, summarize |
| `persistence/` | Index persistence pipeline with batched graph/embedding inserts and compaction |
| `analysis.rs` | Decision chains, session checkpoints, impact analysis |
| `search.rs` | Semantic, text, and hybrid code search |
| `recall.rs` | Unified recall with temporal edge filtering and hybrid scoring |
| `memory_ops.rs` | Memory CRUD with transaction wrapping and session_id auto-population |
| `bm25.rs` | Okapi BM25 scoring with code-aware tokenizer and serialization |
| `scoring.rs` | 8-component hybrid scoring helpers |
| `patterns.rs` | Cross-session pattern detection |
| `compress.rs` | Optional LLM observation compression |
| `metrics.rs` | Operational metrics (latency percentiles, call counters) |

## Supported Languages

Rust, TypeScript/JavaScript/JSX/TSX, Python, Go, C/C++, Java, Ruby, C#, Kotlin, Swift, PHP, Scala, HCL/Terraform

## Data Flow

```
Source Files → Index (ast-grep) → Persist (nodes, edges, embeddings) → Compact
                                                                         ↓
Hooks (tool observations) → Extract → Embed → Store ←── Enrich (14 analyses)
                                                         ↓
                                              Recall (hybrid scoring) → Results
```
