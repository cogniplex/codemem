# Codemem Roadmap

A living roadmap for Codemem's evolution from a developer-local single binary to a hosted team-friendly code memory platform. Last updated March 2026.

## Why Now?

The AI coding assistant market is exploding — Cursor, Windsurf, Claude Code, Copilot Workspace. Every one of them loses context between sessions. Memory is the obvious missing piece, and the timing window is narrow: once a dominant memory layer emerges, switching costs lock developers in.

Codemem's advantage is not just features — it's architectural. A single Rust binary with zero runtime dependencies can ship everywhere these assistants run: local machines, CI pipelines, air-gapped environments, containers. No other tool in the space can make that claim.

## Strategic Arc

```
Viral demo --> credibility + community --> daily-driver UX --> hosted team mode --> expand
```

## Competitive Context

The AI memory space is crowded and well-funded (Mem0 $24M, Cognee $7.5M, Supermemory $3M). See [comparison.md](comparison.md) for detailed analysis. Codemem's differentiators:

- **Single Rust binary, zero runtime deps** — every competitor needs Python/Node.js + external databases
- **Offline-first** — fully functional with local Candle BERT
- **Code-aware structural intelligence** — 14 language extractors with CST-aware chunking
- **Deep graph algorithms** — 25 algorithms (PageRank, Louvain, betweenness, SCC) built-in
- **Self-editing memory** — refine/split/merge with provenance tracking
- **9-component hybrid scoring** — vector + graph strength + BM25 + temporal + tags + importance + confidence + recency

Gaps to close: no multi-modal support, no team/hosted mode, local-only scaling limits on large monorepos (redundant indexing across developer machines), no community infrastructure.

## Adoption Funnel

Every phase should move developers through this funnel:

```
Discover (HN post, demo video, MCP registry)
  --> Try (brew install, 5-min getting started)
    --> Use daily (incremental indexing, fast recall)
      --> Contribute (good first issues, Discord)
        --> Advocate (share, blog, talk)
```

---

## Phase 0: Viral Proof Point (~4-6 weeks)

**Goal**: Create one demo that sells the project in 30 seconds.

Before benchmarks, before docs, before anything else — build something shareable that makes developers say "I need this."

### Demo Candidates (pick one or two)

**Repo onboarding demo**: Point Codemem at a well-known OSS repo (e.g., ripgrep, axum, or tokio), run `codemem index` + `codemem enrich`, then ask architectural questions through MCP recall. Show it answering questions that would take a human hours to figure out. Screen recording for Twitter/YouTube/HN.

**Session replay demo**: Record a 10-minute coding session where the developer explores a codebase. Close the session. Open a new one. Show Codemem recalling the full context — what files were read, what patterns were noticed, what decisions were made — without re-exploration.

**Interactive playground**: A hosted web demo where visitors can explore a pre-indexed repo's knowledge graph in the React UI. Zero install, instant value. Pre-index 3-5 popular repos for visitors to browse.

### Launch Sequence

1. Record demo video (< 2 minutes)
2. Write HN "Show HN" post with the video
3. Post to relevant subreddits (r/rust, r/programming, r/ChatGPTCoding)
4. Submit to MCP tool registries

---

## Phase 1: Credibility and Community (v0.9 - v1.0, ~3 months)

**Goal**: Establish trust and build the first community of daily users.

### Benchmarks

- Run LoCoMo and LongMemEval benchmark suites
- Publish results in README and comparison doc
- Even mid-pack results are compelling when paired with "zero deps, offline, <100ms startup"
- Establish a repeatable benchmark CI job so results stay current

### Community Infrastructure

| Item | Why |
|------|-----|
| CONTRIBUTING.md with "good first issues" | Lower the barrier to first PR |
| Discord server | Real-time community, support, feedback |
| Weekly changelog / dev blog | Show momentum, attract attention |
| GitHub Discussions enabled | Async Q&A, feature requests |
| "Awesome codemem" examples | Show real-world usage patterns |
| Sponsor button / Open Collective | Signal longevity, fund hosting costs |
| Architecture walkthrough | Converts curious developers into PR authors — explain the crate structure, data flow, and where to start hacking |

### Distribution

- MCP registry listing (official MCP tool directory)
- Claude Code / Cursor / Windsurf plugin marketplace presence
- Polish Homebrew tap (`brew install cogniplex/tap/codemem`)

### Documentation

- Getting-started guide (first 5 minutes to value)
- MCP tool API reference (all 32 tools with examples)
- Architecture walkthrough for contributors

### Stability

- ~~Freeze MCP tool surface at 32 tools~~ ✅ Done
- ~~Session continuity polish (metadata tracking, checkpoint reliability)~~ ✅ Done — session_id on memories, memory_count via correlated subquery, checkpoint persistence with rich metadata
- ~~Persistence pipeline performance (batch inserts for graph nodes/edges/embeddings)~~ ✅ Done — multi-row INSERT batching, transaction wrapping, vector batch insert, embedding mutex scope reduction

### Testing

Focus on tests that catch real problems, not coverage numbers:

- Integration tests for the critical path: index -> recall -> consolidate
- Regression tests for bugs users actually hit
- Cross-platform CI: add Windows to existing ubuntu + macos matrix

---

## Phase 2: Daily-Driver UX (v1.x, ~3 months)

**Goal**: Make Codemem reliable enough that developers use it every day without thinking about it.

### Incremental Indexing

~~The #1 UX pain point for daily use. Currently, re-indexing means re-scanning the entire codebase.~~ ✅ Done

- ~~File watching + diff-based re-index — only process changed files~~ ✅ Done — `codemem watch` + `ChangeDetector` with file hash tracking
- ~~Track file content hashes to skip unchanged files on startup~~ ✅ Done — namespace-scoped SHA-256 hashes, `--force` flag to override
- Dependents-aware: when `foo.rs` changes, re-resolve references in files that import from it
- Target: re-index a single file change in < 1 second

### Config UX

- Sensible defaults that work out of the box for common project types
- Project-local `.codemem.toml` config (currently only global `~/.codemem/config.toml`)
- Auto-detect project type and apply appropriate settings (Rust project? Set namespace to crate name, configure Rust-specific enrichments)

### Better ast-grep Rules

Rather than replacing ast-grep now, improve it incrementally:

- Add languages users request (based on GitHub issues / Discord feedback)
- Improve cross-file heuristics (better import resolution within ast-grep's pattern model)
- Community-contributed rule sets for popular frameworks

### Recall Quality

Concrete feedback signal: when a memory is recalled and the user's next action references its content (e.g., the memory mentions `AuthService` and the next Edit/Read targets `auth_service.rs`), count that as a "used" signal. When the user re-asks the same question within a session, count the prior recall as a miss.

- Instrument PostToolUse hook to detect content overlap between recalled memories and subsequent tool targets
- Track used/missed ratio per memory and per scoring component
- Tune scoring weights based on real usage data (increase weight of components that correlate with "used" signals)
- Surface confidence scores so users know when to trust vs. verify

---

## Phase 3: Storage Abstraction and Team Mode (v2.x, ~6 months)

**Goal**: One team member (or CI) indexes, everyone benefits. Solve the monorepo redundancy problem.

### Storage Backend: Postgres + pgvector

For team/hosted mode, use PostgreSQL + pgvector — battle-tested, every team already runs Postgres, every cloud provider has managed instances, and pgvector is the production standard for vector search.

**Why Postgres, not SurrealDB:**

| Concern | Postgres | SurrealDB |
|---------|----------|-----------|
| Production maturity | Decades | Still maturing |
| Community size | Massive (contributors know it) | Growing but small |
| Cloud availability | Every provider has managed Postgres | Limited managed offerings |
| Vector search | pgvector is the industry standard | Native HNSW but less battle-tested |
| Graph queries | Recursive CTEs or Apache AGE extension | Native record links |
| Ops knowledge | Your users' ops teams already know it | Learning curve |
| Migration tooling | SQLite -> Postgres is well-understood | Novel migration path |

SurrealDB is architecturally elegant but practically risky for a core dependency. If it matures significantly, it can be added as a third backend later behind the existing trait abstraction.

**Architecture:**

- **Local mode (default)**: SQLite + usearch + petgraph — unchanged, keep what works
- **Team mode**: Postgres (documents + metadata) + pgvector (embeddings + HNSW) + petgraph materialization (graph algorithms)
- Feature flag: `--features postgres` adds the team backend. Default build stays zero-dep

**Implementation plan:**

1. ~~Audit and extend `StorageBackend`, `VectorBackend`, `GraphBackend` traits in codemem-core to cover all raw backend calls~~ ✅ Done — trait-object backends with 15+ new GraphBackend methods, 3+ VectorBackend methods, and factory pattern
2. New `codemem-postgres` crate implementing all three traits
3. Migration tooling: SQLite -> Postgres data migration
4. Graph algorithms: keep petgraph as compute layer, materialize subgraphs from Postgres on demand (Postgres doesn't have PageRank built-in)

### Shared Indexing via CI

The primary use case for team mode:

1. CI pipeline runs `codemem index` + `codemem enrich` on push to main
2. Developers connect to the shared Postgres instance for recall
3. No local indexing needed for large monorepos
4. Individual developers can still store personal memories (scoped to their user within the namespace)

```bash
# Local mode (default, current behavior)
codemem serve

# Team mode (connect to shared Postgres)
codemem serve --postgres postgres://codemem:pass@team-db:5432/codemem
```

### Auth and Permissions

Map to Postgres roles:

- **Reader** — recall, search, graph traversal (most developers)
- **Writer** — store memories, refine, split, merge
- **Admin** — index, enrich, consolidate, configure

### Memory Conflict Resolution

When multiple developers store memories for the same namespace:

- **Code structure** (facts from indexing) — CI is authoritative, last-write-wins
- **Insights** (observations, patterns) — append, all perspectives kept
- **Decisions** — append with authorship, build decision chains

### Sustainability

Team mode implies infrastructure costs (Postgres hosting, CI compute). Funding options, not mutually exclusive:

- **GitHub Sponsors / Open Collective** — individual and corporate sponsorship (set up in Phase 1)
- **Paid hosted tier** — managed Codemem instance with Postgres, CI indexing, and SSO. Free for open-source repos, paid for private repos
- **Support / consulting** — paid onboarding and integration support for teams adopting Codemem at scale

The core project stays Apache 2.0 and fully functional in local mode. Hosted features are open-source too — the paid tier is the managed infrastructure, not the software.

### Deployment

```bash
# Docker compose
docker compose up  # Postgres + pgvector + codemem serve

# Or connect to existing Postgres
codemem serve --postgres postgres://user:pass@host:5432/codemem
```

---

## Phase 4: Expand (v2.x+)

**Goal**: Broaden beyond code-only memory.

### SCIP-Based Symbol Resolution ✅ Done

~~Replace ast-grep with Language Server Protocol~~ Replaced with SCIP (Source Code Intelligence Protocol) — compiler-grade cross-references with zero false positives. See [ADR-008: SCIP over LSP](adr/008-scip-over-lsp.md).

SCIP was chosen over LSP because: (1) offline batch processing fits the index-then-query model better than LSP's interactive protocol, (2) SCIP indexers produce deterministic output suitable for caching, (3) no need to manage language server lifecycle/memory.

### LSP Considerations (deferred)

LSP remains an option for future interactive features (e.g., IDE extensions) but is not needed for the batch indexing pipeline now that SCIP provides compiler-grade edges.

**What LSP gives over ast-grep:**

- **Cross-file symbol resolution** — follow imports to definitions, resolve trait implementations to concrete types. Graph edges (IMPORTS, CALLS, IMPLEMENTS) become precise instead of heuristic
- **Type-aware context** — embed return types and parameter types in contextual embeddings
- **Workspace-wide references** — find all callers across the entire monorepo
- **Language coverage** — rust-analyzer, typescript-language-server, pyright, gopls, clangd all speak the same protocol

**Architecture: two layers, not two systems:**

- **Tree-sitter** for structure — parse syntax trees to identify function/class/method boundaries for chunking. Lightweight, works on any file with zero project setup
- **LSP** for semantics — resolve references, types, cross-file edges. Requires the project to be buildable

Tree-sitter handles "where are the chunks?" and LSP handles "how do the chunks relate?" This is a parse-then-resolve pipeline, not two parallel extraction systems.

**Tradeoff:** LSP servers are heavyweight (rust-analyzer uses 2GB+ RAM on large repos). This reinforces the team/hosted mode value — run LSP indexing once on CI, not on every developer machine.

**Transition plan:**

1. Implement LSP integration using `textDocument/documentSymbol`, `textDocument/references`, `textDocument/definition`
2. Validate per-language that LSP matches or exceeds ast-grep quality
3. Remove ast-grep rules language by language
4. Keep tree-sitter as fallback for when LSP can't start

### Document Ingestion

- PDF, markdown, and Notion import — chunk and embed alongside code
- Useful for design docs, RFCs, ADRs, runbooks that reference code
- Link document memories to code graph nodes

### GitHub / GitLab Integration

Webhook-driven memory capture:

- PR opened: index the diff, store change summary
- PR merged: store decision memory with context
- Issue closed: capture resolution as insight
- Review comment: extract patterns and preferences

### IDE Extensions

- VS Code and JetBrains extensions
- Inline "what does Codemem know about this file/function?" in the editor
- Graph neighbors, related memories, recent changes in a sidebar

### Smarter Recall

- Cross-encoder re-ranking for top-k results (lightweight model, runs after initial retrieval)
- Query expansion using graph neighbors and synonyms from the knowledge graph
- Adaptive scoring weights that learn from the Phase 2 feedback signals
- Context-aware recall thresholds (tighter for focused queries, broader for exploration)

### Cross-Repo Knowledge

- With team mode, enable `get_cross_repo` to traverse memories across repositories
- Shared patterns and decisions across the organization's codebase

---

## Milestones

| Milestone | Target | Metric |
|-----------|--------|--------|
| Demo video + HN launch | Month 1 | 100+ GitHub stars |
| v1.0 release | Month 4 | 500+ stars, 10+ contributors |
| MCP registry + marketplace | Month 5 | 1000+ installs |
| v1.x incremental indexing | Month 7 | Daily active users metric established |
| v2.0 team mode beta | Month 12 | 3+ teams in beta |
| LSP integration (first language) | Month 15 | One language with LSP parity vs ast-grep |
| v2.x document ingestion + GitHub hooks | Month 18 | PDF/markdown ingest working, 1+ webhook integration live |
| IDE extension (VS Code) | Month 20 | Published on VS Code marketplace |

---

## Principles

- **Local-first always works** — hosted mode is additive, never required
- **Ship small, ship often** — weekly releases beat quarterly launches
- **Community over features** — 10 engaged users matter more than 100 unused features
- **Demo-driven development** — every feature should be showable in 30 seconds
- **Prove with benchmarks** — claims backed by reproducible numbers
- **Replace, don't layer** — one symbol extraction system at a time, not parallel stacks
- **Open source** — Apache 2.0, community-driven development
- **De-risk with proven tech** — Postgres over novel databases, SQLite stays the local default
- **Minimize dependencies** — the single-binary, zero-dep story is a core differentiator
