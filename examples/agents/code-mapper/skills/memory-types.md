# Memory Types Reference

Use the right memory type for each finding — don't default everything to "insight". At least 50% of stored memories should be Decision or Pattern type.

## Types

### decision
**When to use**: Architectural choices, trade-offs, why something was designed a certain way.

Examples:
- "The auth module uses middleware-based validation rather than per-route checks because it ensures consistent enforcement and reduces boilerplate."
- "SQLite with WAL mode was chosen over PostgreSQL to keep codemem as a single binary with zero external dependencies."
- "The engine uses a 9-component hybrid scoring formula because pure vector similarity misses structurally important memories."

### pattern
**When to use**: Recurring code structures, naming conventions, repeated approaches.

Examples:
- "All API handlers follow the pattern: validate → authorize → execute → respond"
- "Error handling uses typed errors per crate with From impls for cross-crate conversion"
- "Graph algorithms follow a compute-then-cache pattern: results stored in RwLock and invalidated on graph mutation"

### preference
**When to use**: Team/project conventions, preferred libraries, style choices.

Examples:
- "Project prefers explicit error types over anyhow; each crate has its own Error enum"
- "Bun is the preferred package manager for the UI (not npm/yarn)"
- "Tests use tempfile for fixtures rather than committed test data"

### style
**When to use**: Coding style norms, formatting, naming patterns.

Examples:
- "Functions use snake_case, types use PascalCase, constants are SCREAMING_SNAKE_CASE. Max function length ~40 lines."
- "Modules use the pattern: pub mod + re-export in lib.rs for flat public API"
- "SQL queries use multi-line string literals with consistent indentation"

### insight
**When to use**: Cross-cutting architectural observations, system-level findings.

Examples:
- "The auth module is the most interconnected subsystem with 12 inbound dependencies"
- "codemem-engine is the largest crate by far (8 modules), acting as the domain logic layer between storage and transport"
- "The BM25 index and vector index provide complementary recall: BM25 for exact terms, vectors for semantic similarity"

### context
**When to use**: File contents, structural context from exploration.

Examples:
- "src/lib.rs exports the public API surface: McpServer, StdioTransport, types"
- "Package codemem-storage/ contains 7 files implementing the persistence layer: memory CRUD, graph persistence, queries, vector index, migrations"
- "File src/engine/scoring.rs implements the 9-component hybrid scoring formula used by recall"

### habit
**When to use**: Workflow patterns, testing approaches, development practices.

Examples:
- "Tests are co-located in the same file with #[cfg(test)] mod tests"
- "CI runs cargo clippy with -D warnings — all warnings are errors"
- "Benchmarks use Criterion with a 20% regression threshold"

## Guidelines

- **Avoid storing what the graph already knows**: Don't store "file X has 48 symbols" — that's graph data, not a memory.
- **Link to symbols**: Every memory about a function or type should include `links: ["sym:<qualified_name>"]`.
- **Use EXPLAINS relationships**: After storing a decision or insight, use `associate_memories` with `EXPLAINS`.
- **Tag appropriately**: Use `human-verified` for user-clarified findings, `agent-verified` for agent-reviewed static analysis, `needs-review` for uncertain findings.
