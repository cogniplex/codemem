# codemem-core

Shared types, traits, and errors for the Codemem memory engine.

## Overview

This is the foundation crate with zero internal dependencies. All other Codemem crates depend on it.

## Key Exports

- **Types** (`types.rs`): `MemoryNode`, `Edge`, `Session`, `DetectedPattern`, `ScoringWeights`, `VectorConfig`, `GraphConfig`
- **Enums**: `MemoryType` (7 variants), `RelationshipType` (23 variants), `NodeKind` (12 variants), `PatternType` (5 variants)
- **Traits** (`traits.rs`): `VectorBackend`, `GraphBackend`, `StorageBackend`
- **Config** (`config.rs`): `CodememConfig`, `EmbeddingConfig`, `StorageConfig` — TOML-backed persistent configuration
- **Errors** (`error.rs`): `CodememError` with variants for storage, embedding, graph, config, lock poisoning, and more
