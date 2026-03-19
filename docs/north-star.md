# Codemem North Star: From Local Tool to Central Code Intelligence Platform

A vision document covering the architectural evolution needed to make codemem a central, team-shared, review-aware code intelligence platform. March 2026.

## The Core Problem

Every time a developer opens Cursor, Claude Code, or Windsurf on a large codebase, their AI agent starts from zero. It reads 50-100 files to understand the architecture. It discovers the same patterns, conventions, and relationships that 5 other developers' agents already discovered this week. All that context is lost when the session ends. Next developer starts over.

Multiply by team size x sessions per day. It's massive waste — tokens, time, and developer patience.

**Central codemem eliminates this.** Index once, everyone's agent queries instead of explores. The agent goes from spending 10 minutes reading files to getting instant answers from a pre-built knowledge graph with memories, patterns, and precise symbol relationships.

## The End State

A single codemem deployment serves as the **live knowledge base** for every developer's AI agent and every automated bot. It's not a CI artifact — it's a hot, always-available service that agents query in real-time for their day-to-day coding work.

```
              ┌──────────────────────────────────────┐
              │         Central Codemem Server        │
              │  ┌────────────────────────────────┐  │
              │  │ Scope Layer                    │  │  ← repo + branch + user + session
              │  ├────────────────────────────────┤  │
              │  │ Engine (read: <100ms)          │  │  ← recall, graph traverse, search
              │  ├────────────────────────────────┤  │
              │  │ Storage Layer (pluggable)      │  │  ← SQLite / Postgres+pgvector / Qdrant+Neo4j
              │  └────────────────────────────────┘  │
              └──────────┬───────────────────────────┘
                         │
       ┌─────────────────┼──────────────────┐
       │                 │                  │
  ┌────┴─────┐    ┌──────┴──────┐    ┌─────┴──────┐
  │ Dev agents│    │ CI/CD       │    │ Bots       │
  │ (daily)  │    │ (on push)   │    │ (on PR)    │
  ├──────────┤    ├─────────────┤    ├────────────┤
  │ recall   │    │ index       │    │ review     │
  │ search   │    │ enrich      │    │ blast      │
  │ traverse │    │ update graph│    │ radius     │
  │ store    │    │             │    │            │
  └──────────┘    └─────────────┘    └────────────┘
   Alice on        main branch        PR #42
   feat/auth       updated            review bot
```

### Developer Daily Workflow (the primary use case)

This is the #1 reason to deploy central codemem — not CI, not review bots, but **making every developer's agent instantly productive**:

```
Developer opens Claude Code on the monorepo:

  Agent: "What does the auth module do?"
  └── recall → instant answer from existing memories + graph
      (zero file exploration needed)

  Agent: "I need to change validate_token()"
  └── get_symbol_graph → all callers, dependencies, tests
      (zero grep/find needed)

  Agent: "What patterns does this codebase use for error handling?"
  └── detect_patterns → pre-computed patterns from prior enrichment
      (zero scanning 200 files)

  Agent works on the module, learns something new
  └── store_memory → available to ALL developers immediately
```

### Knowledge Accumulation

When dev A's agent enriches a module (stores a pattern, documents a decision, notes a gotcha), that knowledge is instantly available to dev B's agent. The team's collective understanding of the codebase grows monotonically. No one re-discovers the same thing twice.

```
Week 1: Agent discovers "this module uses the repository pattern"
         → stored as pattern memory, linked to module node
Week 2: New developer's agent asks about the module
         → gets the pattern + all related memories immediately
Week 3: Agent finds a bug pattern "lock ordering issue in auth"
         → stored as decision memory
Week 4: Review bot sees a PR touching auth locks
         → recalls the decision memory, flags it in review
```

---

## 1. Graph Quality: The Foundation Everything Else Depends On

**Status**: SCIP integration in progress but the graph is a flat soup of nodes and edges, not a navigable tree with precise "go to definition" links. Agents enrich based on graph structure and embedded vectors — garbage in, garbage out.

### Problem 1: No Hierarchical Containment

The current graph has `file → CONTAINS → symbol` and that's it — one level deep. But code has nested structure:

```
file:src/auth/mod.rs
  └── sym:auth (Module)
        ├── sym:auth::Authenticator (Trait)
        │     └── sym:auth::Authenticator::validate (Method)
        └── sym:auth::middleware (Module)
              └── sym:auth::middleware::validate_token (Function)
```

This containment tree is what makes the graph **navigable** — drill down from file → module → class → method. Without it, every symbol is a flat peer of every other symbol. You can't answer "what's in this module?" or "what methods does this class have?" by traversing the graph.

SCIP symbol strings already encode this hierarchy in their descriptor chain (`package/module/class#method().`). The qualified name `auth::middleware::validate_token` tells us the containment path. But the graph builder flattens it — it doesn't create module nodes or parent→child CONTAINS edges between symbols.

**Fix**: Parse the SCIP descriptor chain and create **nested CONTAINS edges**:
- `file:src/auth/mod.rs → CONTAINS → sym:auth` (module)
- `sym:auth → CONTAINS → sym:auth::Authenticator` (trait in module)
- `sym:auth::Authenticator → CONTAINS → sym:auth::Authenticator::validate` (method in trait)

This gives us the tree. Every graph traversal that starts at a file can drill down to any depth.

### Problem 2: Imprecise Semantic Edges

All references become edges with equal treatment — a `use` import, a function call, a field read, and a type annotation all look similar in the graph. The graph should distinguish:

**What we need** (precise, cross-file, semantically typed):
```
sym:auth::middleware::validate_token
  ──CALLS──→ sym:auth::jwt::decode           (line 47, cross-file)
  ──CALLS──→ sym:db::sessions::lookup        (line 52, cross-file)
  ──READS──→ sym:config::AUTH_TIMEOUT         (line 38, cross-file)
  ──IMPLEMENTS──→ sym:auth::Authenticator::validate
```

These are the edges that make blast radius work — follow CALLS transitively to find impact. The SCIP role bitmask (IMPORT, READ_ACCESS, WRITE_ACCESS) partially distinguishes these, but:

- scip-go sets everything to READ_ACCESS (no differentiation)
- Import edges from re-exports create false dependency chains
- Type-annotation references ("return type mentions X") are conflated with actual call-site references

**Fix**: Prioritize **call-site and definition-use edges** over type-mention edges. When SCIP role bits are ambiguous, use heuristics:
- Reference inside a function body at a call-expression position → CALLS
- Reference at an import statement → IMPORTS
- Reference in a type annotation / return type → USES_TYPE (lower weight, don't follow for blast radius)
- Add `--mode lsp` for languages where SCIP is too imprecise (LSP gives exact go-to-definition)

### Problem 3: Node Noise

SCIP indexers produce every symbol occurrence: type parameters, enum variants, fields, re-exports, generated code stubs. A typical TypeScript project produces thousands of nodes that are never useful for architectural understanding.

**Tiered node importance** — not all symbols deserve graph nodes:

| Tier | Kinds | Treatment |
|------|-------|-----------|
| **1 — Always nodes** | `Function`, `Method`, `Class`, `Trait`, `Interface`, `Module`, `File`, `Package`, `Enum` | Full graph nodes with edges |
| **2 — Nodes when cross-file referenced** | `Constant`, `Type`, `Macro`, `Endpoint` | Node only if referenced from another file |
| **3 — Fold into parent** | `Field`, `Property`, `EnumVariant`, `TypeParameter` | Metadata on parent node: `{"fields": ["a", "b"]}` |

Tier 3 preserves searchability (BM25 indexes payload text) without graph pollution.

### Problem 4: Edge Noise

- **Same-file CALLS/READS/WRITES between methods of the same class**: collapse into a single intra-class edge with a count, or drop entirely (CONTAINS captures co-location)
- **Import chains**: if A imports B which re-exports C, create A→C, drop intermediate
- **Fan-out thresholds per kind**: modules are expected to be widely imported; functions are not

### Solution: Multi-Layer Fusion (ast-grep + SCIP + LSP)

Instead of picking one analysis layer or using a fallback chain, run all available layers and **fuse their results** using confidence merging. Each layer contributes partial confidence; agreement between layers increases overall confidence.

| Layer | Strengths | Weaknesses | Cost | Base Confidence |
|-------|-----------|------------|------|-----------------|
| **ast-grep/tree-sitter** | Fast, works on any file, no project setup | No cross-file resolution, heuristic imports | ~ms/file | 0.10 |
| **SCIP** | Batch, cross-file, types, CI-friendly | Indexer quality varies (scip-go imprecise) | ~sec/repo | 0.15 |
| **LSP** | Precise go-to-definition, exact references | Slow, needs buildable project, heavy RAM | ~100ms/query | 0.20 |

They fail in **different ways** — that's exactly when multi-source fusion works well.

**Fusion example — agreement:**
```
ast-grep:  func A "probably calls" func B (heuristic import + name match)  → 0.10
SCIP:      sym:A has reference to sym:B with READ_ACCESS role              → 0.15
LSP:       definition at call site in A resolves to B                      → 0.20
Merged:    A ──CALLS──→ B, confidence: 0.45 (three layers agree)
```

**Fusion example — disagreement (informative):**
```
ast-grep:  A imports B (pattern match on import statement)                 → 0.10
SCIP:      no reference from A to B (missed — re-export?)                  → 0.00
LSP:       A does reference B, through a re-export chain                   → 0.20
Merged:    A ──IMPORTS──→ B, confidence: 0.30
           + metadata: { scip_missed: true, likely_reexport: true }
```

**Running strategy — not all layers every time:**

```
Fast path (CI, every push):
  ast-grep (always) + SCIP (if indexer available)
  → confidence ceiling: 0.25

Thorough path (nightly, or on-demand):
  ast-grep + SCIP + LSP
  → confidence ceiling: 0.45

Incremental (file changed):
  Re-run ast-grep + SCIP for changed files
  LSP only for cross-file edges touching changed symbols

Targeted (agent request):
  Agent asks about a specific symbol → trigger LSP query for that symbol
  to upgrade edge confidence from 0.25 → 0.45
```

**Merge algorithm:**
```rust
struct CandidateEdge {
    src: String,
    dst: String,
    relationship: RelationshipType,
    source_layer: AnalysisLayer,  // AstGrep, Scip, Lsp
    layer_confidence: f64,
    evidence: EdgeEvidence,       // line number, role bits, etc.
}

fn merge_candidates(candidates: Vec<CandidateEdge>) -> Vec<Edge> {
    // Group by (src, dst, relationship)
    // Sum layer confidences (3 layers agree = 0.45)
    // Take most precise relationship type (LSP CALLS > SCIP READ_ACCESS > ast-grep "references")
    // Merge evidence (keep all line numbers, role bits)
    // Flag disagreements in edge properties
}
```

Agents already respect confidence in scoring — higher-confidence edges naturally weight more in recall and blast radius. An edge at 0.10 (single heuristic layer) gets treated very differently from one at 0.45 (three layers confirmed).

### Cross-Repo Linking

With precise go-to-definition, cross-repo linking becomes natural:
- SCIP external symbols already carry `(package_manager, package_name, package_version)`
- When both repos are indexed in central codemem, resolve `pkg:cargo:shared-lib` → the actual symbols in the shared-lib repo's graph
- A change to `shared-lib::auth::validate()` can trace CALLS edges into every consumer repo

This is what makes blast radius work across a monorepo or multi-repo setup.

### Config

```toml
[analysis]
# Which layers to run (order doesn't matter — all run, results fused)
layers = ["ast-grep", "scip"]  # add "lsp" for thorough mode

# Per-layer base confidence (before fusion)
[analysis.confidence]
ast_grep = 0.10
scip = 0.15
lsp = 0.20

# Run LSP selectively: only for symbols where ast-grep and SCIP disagree
lsp_on_disagreement = true

# For monorepos: per-language layer overrides
[analysis.languages.rust]
layers = ["ast-grep", "scip"]  # rust-analyzer SCIP is high quality

[analysis.languages.python]
layers = ["ast-grep", "scip", "lsp"]  # pyright SCIP misses dynamic imports

[scip]
# Which node kinds to include in the graph (default: tier 1 + tier 2)
# Tier 3 kinds are always folded into parent metadata
node_tiers = [1, 2]

# Build nested containment tree from SCIP descriptor chains
hierarchical_containment = true

# Collapse intra-class edges into parent metadata
collapse_intra_class_edges = true

# Distinguish type-annotation refs from call-site refs
# Type refs get USES_TYPE edge (weight 0.1), call refs get CALLS (weight 1.0)
separate_type_refs = true

# Per-kind fan-out thresholds (0 = use global max_references_per_symbol)
[scip.fan_out_limits]
module = 200
function = 30
method = 30
class = 50
```

---

## 2. Memory Expiration

**Status**: Memories are append-only with no TTL. Stale memories accumulate and degrade recall quality.

### Design

Add `expires_at: Option<DateTime<Utc>>` to `MemoryNode`. Expiration is optional — most code-structure memories don't expire, but:

- **Session memories** (conversation context): expire after configurable TTL (default: 7 days)
- **Agent enrichments** tagged `static-analysis`: expire when the underlying code changes (tied to file content hash)
- **User observations**: no expiry by default, but users can set one

### Implementation

1. Add `expires_at` column to `memories` table (nullable timestamp)
2. `recall()` and `search()` filter out expired memories by default (`WHERE expires_at IS NULL OR expires_at > NOW()`)
3. Background cleanup: `codemem gc` command + periodic sweep in server mode that deletes expired memories and their embeddings
4. Hook into file re-indexing: when a file's content hash changes, mark all `static-analysis` memories linked to symbols in that file as expired

### Config

```toml
[memory]
default_session_ttl_hours = 168  # 7 days
gc_interval_minutes = 60
expire_enrichments_on_reindex = true
```

---

## 3. Storage Abstraction

**Status**: Traits exist (`StorageBackend`, `VectorBackend`, `GraphBackend` in `codemem-core/src/traits.rs`). `StorageBackend` is already used as `Box<dyn StorageBackend>` in engine. Vector and graph are concrete types (performance optimization).

### Step 1: Make Vector + Graph Trait-Based in Engine

```rust
// Current:
pub struct CodememEngine {
    pub(crate) storage: Box<dyn StorageBackend>,
    pub(crate) vector: OnceLock<Mutex<HnswIndex>>,      // concrete
    pub(crate) graph: Mutex<GraphEngine>,                 // concrete
}

// Target:
pub struct CodememEngine {
    pub(crate) storage: Box<dyn StorageBackend>,
    pub(crate) vector: OnceLock<Mutex<Box<dyn VectorBackend>>>,
    pub(crate) graph: Mutex<Box<dyn GraphBackend>>,
}
```

Performance cost is vtable indirection on hot paths (vector search, graph traversal). Acceptable for a central server where network latency dominates. For local mode, we could use a generic parameter instead of trait object if profiling shows impact.

### Step 2: Backend Implementations

**Priority order** (based on deployment model):

| Backend | Crate | Trait | Why First |
|---------|-------|-------|-----------|
| Postgres + pgvector | `codemem-postgres` | `StorageBackend` + `VectorBackend` | Single DB for team mode, everyone has Postgres |
| Qdrant | `codemem-qdrant` | `VectorBackend` | Best-in-class vector search for large scale |
| Neo4j | `codemem-neo4j` | `GraphBackend` | Native graph traversal, Cypher queries |

Each is a feature-flagged crate. Default build stays zero-dep (SQLite + usearch + petgraph).

### Step 3: Factory from Config

```toml
[storage]
backend = "sqlite"  # or "postgres"
# postgres_url = "postgres://..."

[vector]
backend = "hnsw"  # or "pgvector", "qdrant"
# qdrant_url = "http://localhost:6333"

[graph]
backend = "petgraph"  # or "neo4j"
# neo4j_url = "bolt://localhost:7687"
```

Engine construction reads config and instantiates the right combination. Mix-and-match is valid (e.g., SQLite metadata + Qdrant vectors + petgraph in-memory graph).

---

## 4. Scope Layer (ScopeContext)

**Status**: Scoping is `namespace: &str` derived from directory basename. No branch or user awareness.

### Design

```rust
/// Scoping context threaded through all storage and engine operations.
pub struct ScopeContext {
    /// Repository identifier (e.g., "github.com/org/repo" or local path hash)
    pub repo: String,
    /// Git ref: branch name, tag, or commit SHA
    pub git_ref: String,
    /// Base ref for overlay resolution (e.g., "main" when on a feature branch)
    pub base_ref: Option<String>,
    /// User identifier (for user-scoped memories)
    pub user: Option<String>,
    /// Session identifier (for session-scoped memories)
    pub session: Option<String>,
}

impl ScopeContext {
    /// For local embedded mode: derive from cwd + git
    pub fn from_local(path: &Path) -> Self { /* git rev-parse for branch, basename for repo */ }

    /// For central server mode: from request headers/params
    pub fn from_request(req: &Request) -> Self { /* extract from auth + query params */ }
}
```

### Branch Overlay Model

Most content is shared between branches. A feature branch differs from main by a handful of files. Don't duplicate the full index.

```
main (base layer)
  ├── 5000 symbol embeddings
  ├── full graph
  │
feat/auth (overlay on main)
  ├── 12 changed symbol embeddings (override)
  ├── graph edge patches (added/removed)
  ├── user memories (alice's observations)
```

**Resolution logic**: when querying on `feat/auth`:
1. Check overlay for the symbol/embedding/node
2. If not found (unchanged), fall back to `base_ref` (main)
3. For graph traversal: merge base graph + overlay patches

**Storage representation**:
- Each embedding/node/edge tagged with `(repo, git_ref)`
- Overlay = rows where `git_ref = "feat/auth"`
- Base = rows where `git_ref = "main"`
- Query: `WHERE git_ref = ? OR (git_ref = ? AND id NOT IN (SELECT id FROM ... WHERE git_ref = ?))`

This is similar to git's object model — objects are content-addressed and shared, refs just point differently.

### Migration Path

1. Add `repo` and `git_ref` columns to relevant tables (memories, graph_nodes, graph_edges, embeddings)
2. Existing data gets `repo = namespace, git_ref = "main"` as default
3. `namespace` becomes derived from `ScopeContext` (backward compatible)
4. New MCP/API calls accept scope parameters, old calls infer from local git

---

## 5. Diff-Aware Review Pipeline

**Status**: No review capability. All the graph infrastructure needed is in place.

### New: `diff_to_symbols()`

The bridge between git and the graph. Given a unified diff, map changed lines to affected symbols:

```rust
pub struct DiffSymbolMapping {
    pub changed_symbols: Vec<String>,    // sym:IDs directly modified
    pub containing_symbols: Vec<String>, // sym:IDs whose body contains changes
    pub changed_files: Vec<String>,      // file:IDs
}

/// Parse a unified diff and resolve changed line ranges to symbol IDs
/// using SCIP line→symbol mapping stored in graph node payloads.
pub fn diff_to_symbols(
    diff: &str,
    engine: &CodememEngine,
    scope: &ScopeContext,
) -> Result<DiffSymbolMapping, CodememError>
```

This uses the `line_start`/`line_end` stored in SCIP node payloads to map diff hunks to symbols.

### New: `blast_radius()`

Multi-hop graph traversal from changed symbols:

```rust
pub struct BlastRadiusReport {
    pub changed_symbols: Vec<SymbolInfo>,
    pub direct_dependents: Vec<SymbolInfo>,     // 1-hop
    pub transitive_dependents: Vec<SymbolInfo>,  // 2+ hops (with decay)
    pub affected_files: Vec<String>,
    pub affected_modules: Vec<String>,
    pub risk_score: f64,                         // based on centrality of changed nodes
    pub missing_changes: Vec<MissingChange>,     // same-pattern symbols not in diff
    pub relevant_memories: Vec<MemoryNode>,       // past observations about these symbols
}

pub struct MissingChange {
    pub symbol: String,
    pub reason: String,  // "same caller pattern as 6 updated endpoints"
}
```

Risk score formula:
```
risk = Σ(changed_symbol.pagerank × change_magnitude) × log(transitive_dependent_count + 1)
```

High PageRank symbol changed + many transitive dependents = high risk.

### New: `codemem review` CLI + MCP Tool

```bash
# CLI
git diff main..HEAD | codemem review --base main --head feat/auth

# MCP tool: review_diff
{
  "diff": "...",         # or "pr_url": "github.com/org/repo/pull/42"
  "base_ref": "main",
  "depth": 3             # traversal depth for transitive impact
}
```

Output: structured `BlastRadiusReport` that an LLM can synthesize into a review comment.

### GitHub Integration

```yaml
# .github/workflows/codemem-review.yml
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          git diff ${{ github.event.pull_request.base.sha }}..HEAD \
            | codemem review --server $CODEMEM_URL --format github \
            | gh pr comment ${{ github.event.number }} --body-file -
```

---

## 6. Dependency Order

These workstreams have real dependencies:

```
                    ┌──────────────────────────┐
                    │ 1. Graph Quality         │  ← MUST be first
                    │    (hierarchical tree,   │     everything downstream depends
                    │     precise edges,       │     on a clean graph
                    │     node tiers,          │
                    │     multi-layer fusion)  │
                    └────────┬─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌──────────────┐ ┌────────────┐ ┌──────────────┐
    │ 2. Memory    │ │ 3. Storage │ │ 4. Scope     │
    │    Expiry    │ │ Abstraction│ │    Context   │
    │              │ │ (traits)   │ │ (repo+branch │
    │              │ │            │ │  +user)      │
    └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
           │               │               │
           │       ┌───────┴───────┐       │
           │       │               │       │
           │       ▼               ▼       │
           │  ┌─────────┐   ┌──────────┐   │
           │  │ Postgres │   │ Qdrant / │   │
           │  │ backend  │   │ Neo4j    │   │
           │  └─────────┘   └──────────┘   │
           │               │               │
           └───────────────┼───────────────┘
                           │
              ┌────────────┼────────────┐
              │                         │
              ▼                         ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ 5. Central Server│    │ 6. Review        │
    │    (daily dev    │    │    Pipeline      │
    │     use, always  │    │    (diff→symbols,│
    │     hot, <100ms) │    │     blast radius)│
    └──────────────────┘    └──────────────────┘
```

**Workstreams 2, 3, 4 can run in parallel** after graph quality is addressed. Central server mode and the review pipeline both need scope + storage, but serve different primary users:

- **Central server** = daily developer use (the primary use case — agents query instead of explore)
- **Review pipeline** = automated bots on PR events (depends on central server being available)

---

## 7. What Not To Build

- **19 vector backends like mem0**: Start with pgvector + Qdrant. Add more when there's demand.
- **Per-request custom extraction prompts**: The graph is structural, not conversational. Custom prompts make sense for mem0's chat-based extraction; our extraction is deterministic (SCIP/LSP).
- **Conversation memory types (session/organizational)**: We're code-first. User memories and session context are secondary to structural graph quality.
- **Multi-modal ingestion (PDF, images)**: Stay focused on code. Document ingestion can come later if there's demand.
- **Custom memory categories like mem0**: Our memory types (`Context`, `Decision`, `Pattern`, `Purpose`) are code-domain-specific. Don't generalize prematurely.

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| **Graph quality** | | |
| Graph nodes per 1K LOC | ~50-200 (noisy) | ~15-40 (meaningful) |
| Containment tree depth | 1 (flat) | 3-5 (file → module → class → method) |
| Edge confidence (avg) | 1.0 (binary) | 0.25-0.45 (multi-layer fused) |
| Cross-file edge precision | Unknown | >90% of CALLS edges are real call sites |
| **Daily developer use** | | |
| Agent file reads before first useful action | ~50-100 | ~0-5 (query codemem instead) |
| Time to first useful answer | Minutes | <1 second |
| Knowledge re-discovery rate | ~100% (every session) | <10% (shared memories) |
| Central query latency (p95) | N/A | <100ms recall, <200ms graph traverse |
| **Review & blast radius** | | |
| Blast radius accuracy | N/A | >90% of affected files identified |
| Missing change detection | N/A | Flags >70% of "you probably forgot this" cases |
| **Infrastructure** | | |
| Branch overlay overhead | N/A | <5% storage increase per feature branch |
| Memory staleness | Unbounded | Zero expired memories after gc cycle |
| Concurrent developer queries | 1 (local) | 50+ (central server) |
