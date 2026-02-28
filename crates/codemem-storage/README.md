# codemem-storage

SQLite persistence layer with WAL mode, CRUD operations, graph storage, and versioned schema migrations.

## Overview

Wraps rusqlite (bundled) with WAL mode, 64MB cache, and foreign key enforcement. Implements the `StorageBackend` trait from `codemem-core`.

## Modules

- `memory.rs` — Memory CRUD (insert, get, update, delete, list)
- `graph_persistence.rs` — Graph node/edge storage, embedding persistence
- `queries.rs` — Stats, session management, pattern queries
- `backend.rs` — `StorageBackend` trait implementation
- `migrations.rs` — Versioned, idempotent SQL migrations tracked in `schema_version` table

## Schema

Migrations live in `src/migrations/` as `.sql` files:
- `001_initial.sql` — Full schema (memories, sessions, graph nodes/edges, embeddings, file hashes)
- `002_compound_indexes.sql` — Compound indexes for namespace/type, importance/access, src/relationship
