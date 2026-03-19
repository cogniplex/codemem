# Memory Distillation: LLM-Powered Memory Optimization

## Problem

Codemem stores too many low-value memories. Analysis of a real project namespace (263 memories) shows:

- **97% never accessed** (255/263 have `access_count = 0`, total 8 accesses)
- **Prompts (32%):** Verbatim user input including task-notifications, short confirmations, off-topic questions
- **Static analysis (41%):** One memory per finding — per-file untested functions, per-function code smells
- **PostToolUse observations (10%):** Individual file reads/edits, including test fixtures and near-identical repeated edits
- **File watch (8%):** Single-file change events (14/21 are single-file creates at importance 0.35)
- **Session summaries (4%):** Bloated with raw XML, edit diffs; duplicate summaries per session
- **Pending-analysis (3%):** Multiple overlapping entries per session, never consumed
- **Tool failures (2%):** Transient errors (SSH, EISDIR) stored permanently

Every memory costs: 1 embedding (768-dim), 1 graph node, 1 BM25 entry, 1 vector index entry. Noise drowns signal during recall.

## Solution: Two-Tier Storage with Batch Distillation

Instead of storing raw observations immediately through the full pipeline (embed + graph + BM25), introduce a staging buffer with LLM-powered batch processing.

```
Hook fires (prompt, tool use, file watch, error)
    │
    ▼
┌─────────────────────────┐
│  Tier 1: Staging Buffer │  Lightweight INSERT, no embedding/graph/BM25
│  (staged_observations)  │
└────────────┬────────────┘
             │  Trigger: session end, threshold (N items), or manual
             ▼
┌─────────────────────────┐
│  Tier 2: LLM Distill    │  Single batched LLM call classifies, merges,
│  (distill.rs)           │  summarizes raw observations into refined memories
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Full Pipeline           │  persist → embed → graph → BM25 → vector
│  (5-10 refined memories) │
└─────────────────────────┘
```

### Why Batching

A single LLM call processing 20 raw observations is cheaper, faster, and produces better output than 20 individual classification calls. The LLM sees the full session context and can deduplicate, merge related items, and judge importance holistically.

### Latency Budgets

The design has three LLM call sites with different latency profiles:

| Call site | When | User waiting? | Budget |
|-----------|------|---------------|--------|
| Batch distillation | Session end (Stop hook) | Yes — blocks summary generation | <5s target, timeout at 15s |
| Enrichment aggregation | After enrichment run | No — background process | Relaxed, can take 30s+ |
| Catalog summarization | After distillation or on rebuild | No — cached artifact | Relaxed, can take 30s+ |

Distillation is on the critical path: the user's session-end experience depends on it completing quickly. Consider streaming the response or returning a preliminary summary from heuristics while the LLM call completes asynchronously. Catalog and enrichment summarization can be fully async since they produce cached artifacts.

## Components

### 1. Staging Buffer

New table in SQLite:

```sql
CREATE TABLE staged_observations (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    session_id TEXT,
    source TEXT NOT NULL,      -- 'prompt', 'tool_read', 'tool_edit', 'tool_bash',
                               -- 'tool_grep', 'tool_glob', 'tool_write',
                               -- 'file_watch', 'tool_error'
    content TEXT NOT NULL,
    metadata TEXT,             -- JSON: tool_name, file_path, exit_code, etc.
    created_at TEXT NOT NULL,
    batch_id TEXT,             -- Set to UUID when claimed/distilled
    distill_status TEXT NOT NULL DEFAULT 'pending'
                               -- 'pending' → 'claimed' → 'distilled' | 'failed'
                               -- Two-phase: claimed prevents re-processing on crash,
                               -- failed enables retry without data loss
);
CREATE INDEX idx_staged_ns_session ON staged_observations(namespace, session_id);
CREATE INDEX idx_staged_pending ON staged_observations(distill_status) WHERE distill_status = 'pending';
```

**What goes to staging (instead of direct persist):**

| Source | Currently | New behavior |
|--------|-----------|--------------|
| User prompts | Stored verbatim as Context memories | Staged |
| PostToolUse (Read, Edit, Write, Bash, Grep, Glob) | Each stored as Context memory with embedding | Staged |
| Tool errors | Stored as Context memories | Staged |
| File watch events | Stored as Context memories | Staged |
| File change tracking | Stored as pending-analysis | Staged |

**What still goes directly to the full pipeline (unchanged):**

| Source | Why |
|--------|-----|
| Agent results | Already curated by the agent |
| Auto-insight triggers | Already deduplicated per-session |
| Enrichment insights | Handled separately (see Component 3) |
| Manual `store_memory` calls | Explicit user/agent intent |
| Session checkpoints | Structural, not observational |

Staging is cheap: a single INSERT with no embedding, graph, or BM25 work.

### 2. Batch Distillation

New module: `crates/codemem-engine/src/distill.rs`

**Trigger points:**

| Trigger | When | Use case |
|---------|------|----------|
| Session end | Stop hook fires | Primary — processes all staged items for that session |
| Threshold | Staged buffer hits N items (default 30) | Long sessions that haven't ended |
| Manual | `codemem distill` CLI / `distill` MCP tool | On-demand cleanup |

**Prompt complexity note:** The distillation prompt asks the LLM to classify (KEEP/MERGE/DROP), merge related items, summarize, assign types, score importance, and track source IDs — all in one call. This is a lot of structured output to get right reliably, especially with smaller/cheaper models. Consider whether a simpler approach — having the LLM return just refined memory texts + types, and computing importance from source metadata (e.g., number of merged items, presence of edits vs. reads) — would be more robust. Alternatively, a two-pass approach (classify → then summarize kept/merged) trades one LLM call for higher reliability.

**Distillation prompt (single LLM call per batch):**

```
You are a memory distiller for a code exploration engine. Given a batch of raw
observations from a coding session, produce a structured JSON response.

For each observation, classify it as one of:
- KEEP: Worth remembering long-term (architectural decisions, key findings,
  important patterns, non-trivial changes)
- MERGE: Should be combined with related observations into one memory
- DROP: Noise (trivial confirmations, transient errors, system messages,
  temp file operations, task-notifications)

Then produce the final refined memories. Each memory should have:
- content: A concise, information-dense summary (1-3 sentences)
- memory_type: one of "decision", "pattern", "insight", "context"
- importance: 0.0-1.0
- tags: relevant classification tags
- merge_sources: list of source observation IDs that were merged

Rules:
- Merge consecutive edits to the same file into one memory
- Merge related prompts and their tool results into one "intent + outcome" memory
- Drop task-notifications, empty outputs, trivial confirmations ("yes", "ok", "continue")
- Drop transient errors (SSH failures, path-not-found, branch-exists) unless they
  reveal a recurring pattern
- Preserve specific names: functions, files, types, error messages
- A session with 30 raw observations should produce 5-10 refined memories
- Focus on what was learned, decided, or changed — not mechanical steps

Session context:
  Namespace: {namespace}
  Session: {session_id}
  Working directory: {cwd}

Raw observations (JSON array):
{staged_items_json}
```

**Expected response format:**

```json
{
  "memories": [
    {
      "content": "Investigated auth module: read JWT middleware (src/auth/jwt.rs), edited token validation to use RS256 algorithm, updated 3 route handlers to pass token claims through context",
      "memory_type": "decision",
      "importance": 0.7,
      "tags": ["auth", "jwt", "refactoring"],
      "merge_sources": ["obs-1", "obs-3", "obs-5", "obs-8"]
    },
    {
      "content": "Project uses Axum with tower-http CORS layer; routes defined in src/api/mod.rs with nested Router::merge pattern",
      "memory_type": "pattern",
      "importance": 0.5,
      "tags": ["api", "architecture"],
      "merge_sources": ["obs-2"]
    }
  ],
  "dropped": ["obs-4", "obs-6", "obs-7"],
  "drop_reasons": {
    "obs-4": "task-notification system message",
    "obs-6": "trivial confirmation",
    "obs-7": "transient SSH error"
  }
}
```

**After distillation:**

1. Each refined memory goes through the full pipeline (persist → embed → graph → BM25)
2. Auto-linking runs on refined content (detects code references, links to graph nodes)
3. Staged items transition `claimed → distilled` with `batch_id` set (retained for audit, not deleted)
4. Session summary is generated from refined memories (high-quality signal)

**Failure handling:**

If distillation fails (LLM timeout, malformed response, crash mid-batch):
- Items remain in `claimed` status with their `batch_id` — they are not lost
- On next trigger (session end, threshold, or manual), retry claimed items older than 5 minutes
- After 3 failed attempts, mark as `failed` and fall back to heuristic processing for that batch
- Without this, raw observations silently vanish after `staging_ttl_hours` without ever being distilled

### 3. Enrichment Aggregation

Change how the enrichment pipeline stores findings.

**Current flow (per-finding):**

```
code_smells.rs    → store_insight("Code smell: Long function X in file.rs") ×17
test_mapping.rs   → store_insight("Untested: fn foo in bar.rs")            ×49
complexity.rs     → store_insight("High complexity: fn baz")                ×12
...
Total: ~108 individual memories per enrichment run
```

**New flow (collect → group → summarize):**

```
code_smells.rs    → return Vec<Finding>
test_mapping.rs   → return Vec<Finding>
complexity.rs     → return Vec<Finding>
                        │
                        ▼
                  Group by track + directory
                        │
                        ▼
              LLM summarize each group
                        │
                        ▼
              ~15-20 aggregated memories
```

**Aggregation prompt:**

```
Summarize these code analysis findings for directory "{directory}" into one
concise insight. Include specific file/function names for the most important
items. Prioritize actionable findings over informational ones. Max 200 words.

Track: {track_name}
Findings:
{findings_json}
```

**Example output:**

```
Testing gaps in codemem-engine/src/enrichment/: 6 public functions lack tests,
including enrich_architecture (architecture.rs), detect_code_smells (code_smells.rs),
and enrich_security_scan (security_scan.rs). The enrichment pipeline entry point
run_enrichments() is also untested. Priority: security_scan and architecture
analysis affect the full enrichment pipeline output.
```

This collapses 49 `track:testing` memories into ~5-8 directory-level summaries.

### 4. Session Summary via Distilled Memories

Change `cmd_summarize()` to build summaries from distilled memories instead of raw observations.

**Current flow:**

```
Stop hook → collect raw memories since session start → concatenate prompts +
            edits + diffs → template-based summary (avg 857 bytes, includes XML noise)
```

**New flow:**

```
Stop hook → trigger distillation of staged buffer
         → from distilled memories:
            → LLM generates session summary
            → one summary memory + one pending-analysis memory (merged file list)
```

**Summary prompt:**

```
Given these distilled session memories, write a 2-3 sentence session summary
capturing: what was explored, what was changed, and key decisions made.
Be specific — use file names, function names, and technical terms.

Memories:
{distilled_memories_json}

Files modified: {file_list}
```

**Pending-analysis dedup:** Before storing a new pending-analysis entry, check for an existing one with the same `session_id`. If found, merge file lists (set union) via `refine_memory` instead of creating a new entry.

### 5. Recall Improvements

**`include_tags` parameter:** Add to the recall MCP tool alongside the existing `exclude_tags`. Filter during the SQL query phase (before embedding search) for efficiency. Enables targeted queries like "give me all pending-analysis" without embedding overhead.

**Access-weighted decay:** During the decay consolidation cycle, factor in `access_count`:
- `access_count = 0` after 7 days → halve importance
- `access_count = 0` after 30 days → candidate for deletion
- `access_count > 0` → slower decay rate (current behavior)

**Bootstrapping consideration:** New memories start with `access_count = 0` by definition. If an important memory is created but the user doesn't work in that area for 7+ days, it gets halved before anyone ever had a chance to access it. Distilled memories (which are higher quality by design) should get a grace period — either a longer initial decay window (e.g., 14 days for distilled, 7 for raw) or a minimum importance floor that prevents decay below a threshold (e.g., 0.3) for the first 30 days.

### 6. Configuration

Extend `~/.codemem/config.toml`:

```toml
[distillation]
enabled = true              # Master switch for staging + distillation
provider = "auto"           # "auto" = inherit from CODEMEM_COMPRESS_PROVIDER
                            # or explicit: "ollama", "openai", "anthropic"
model = ""                  # Empty = use provider default
batch_threshold = 30        # Trigger mid-session distillation at N staged items
max_batch_size = 50         # Cap items per LLM call (splits into multiple if needed)
staging_ttl_hours = 168     # Clean up unprocessed staging after 7 days

[distillation.enrichment]
aggregation = true          # Aggregate enrichment findings before storing
group_by = "directory"      # "file" (no change), "directory", "crate"/"package"

[distillation.fallback]
# Heuristic rules used when no LLM is configured
drop_task_notifications = true
drop_short_prompts_words = 5
merge_same_file_edits = true
tool_error_ttl_hours = 24

[catalog]
enabled = true              # Generate memory catalog for session context
max_context_tokens = 1000   # Token budget for session-start injection
cwd_detail_depth = 3        # Detail level for current working area
sibling_depth = 1           # Detail level for sibling areas
rebuild_on_distill = true   # Auto-rebuild catalog after distillation
```

`provider = "auto"` reuses `CompressProvider::from_env()` — zero additional configuration for users who already have compression set up.

## Graceful Degradation

The system must work without an LLM. When no provider is configured, a minimal heuristic fallback applies:

| Feature | With LLM | Without LLM (fallback) |
|---------|----------|----------------------|
| Prompt filtering | LLM classifies KEEP/DROP | Drop if < 5 words or matches `<task-notification>` |
| Observation merging | LLM merges related items | Merge consecutive edits to same file |
| Tool error handling | LLM judges transient vs. meaningful | TTL-based expiry (24h default) |
| Enrichment aggregation | LLM summarizes per-directory | Template concatenation, grouped by directory |
| Session summary | LLM generates from distilled memories | Current template-based approach from raw observations |

The fallback is intentionally minimal — a safety net with 5-6 simple rules, not an extensive heuristic system. The staging buffer still provides value without an LLM by enabling deferred processing and same-file edit merging.

## Component 7: Memory Catalog (Agent Brief)

### Problem

At session start, the agent needs to know what memories exist so it can decide when to `recall` for details. Currently, `cmd_context()` injects recent decisions, file hotspots, and detected patterns — but this is a flat list that doesn't scale, and the agent has no sense of what *areas* of knowledge are covered.

For a small repo this works. For a large monorepo (e.g., 50 services, 3 languages, 2000+ memories), injecting a flat list would blow up context or be so truncated it's useless.

### Solution: Hierarchical Memory Catalog

A pre-computed, compact "table of contents" for stored memories, organized by the code graph's natural structure. The agent sees a brief overview and drills into specific areas via `recall` with scoping.

**What the agent sees at session start (example — small repo):**

```
### Memory Catalog (codemem, 45 memories)

Areas with stored knowledge:
- **auth** (12 memories): JWT migration decision, session middleware patterns, RBAC role hierarchy
- **api/routes** (8 memories): REST endpoint conventions, error response format, pagination pattern
- **storage** (6 memories): SQLite WAL mode choice, migration strategy, query batching patterns
- **ui** (4 memories): Zustand store patterns, React Query caching decisions
- **testing** (3 memories): Integration test fixture setup, mock patterns
- **cross-cutting** (12 memories): Error handling flow, logging conventions, config management

Use `recall` with area tags (e.g. recall query="..." include_tags=["auth"]) for details.
```

**What the agent sees at session start (example — monorepo):**

```
### Memory Catalog (mega-corp, 1847 memories)

Top-level areas:
- **services/auth-service** [python] (142 memories): OAuth2 flows, token rotation, rate limiting
- **services/billing** [go] (89 memories): Stripe integration, invoice generation, webhook handling
- **services/notifications** [typescript] (67 memories): Email templates, push notification routing
- **libs/shared-models** [python] (34 memories): Pydantic schemas, serialization conventions
- **infra/terraform** (28 memories): AWS ECS config, RDS setup, IAM policies
- **frontend/web-app** [typescript] (203 memories): Next.js patterns, auth context, API client
- 14 more areas (1284 memories total) — use `memory_catalog` tool for full list

Current directory context (services/auth-service):
- JWT RS256 migration: completed, see decision chain (3 linked memories)
- Rate limiter uses sliding window, Redis-backed (pattern)
- 2 pending items from last session: token refresh endpoint, CORS policy update

Use `recall` with area tags for details. Use `memory_catalog` to browse other areas.
```

### Real-World Monorepo Example

The `example-monorepo` namespace demonstrates the scaling challenge: a single indexing pass by the agent team produced 868 memories.

| Category | Count | % | What it contains |
|----------|-------|---|------------------|
| baseline (file/package summaries) | 435 | 50% | One summary per source file + one per package |
| static-analysis | 192 | 22% | Per-file findings: untested functions, code smells, complexity, API surface |
| other (architecture, patterns, decisions) | 154 | 18% | Agent-discovered architectural decisions, cross-cutting patterns |
| agent-curated | 32 | 4% | Verified/refined findings from symbol-analyst and reviewers |
| prompt | 24 | 3% | User prompts from the indexing session |
| cross-file-pattern | 21 | 2% | Patterns spanning multiple files (naming conventions, shared structures) |
| session-summary | 10 | 1% | Session end summaries |

This is a multi-language monorepo (~60 sub-packages, Python + TypeScript) with areas spanning auth, payments, frontend SPA, partner portal, proxy layer, state management, feature flags, and more.

When a developer opens this repo, all 868 memories compete for recall relevance. The **435 baseline summaries** are individually useful to agents during analysis but become noise for daily development — a developer working on the payment module doesn't need file summaries for all 60 packages. Similarly, **192 static-analysis findings** are granular per-file entries that should be compressed into area-level overviews.

A catalog for this repo would look like:

```
### Memory Catalog (example-monorepo, 868 memories → 45 distilled)

Areas with stored knowledge:
- **auth** (17 memories): Keycloak OAuth2 flow, session migration sync→async,
  Tollbooth token validation, CSRF protection patterns
- **payment** (14 memories): Braintree/Recurly/Adyen triple-provider checkout,
  saga pattern for distributed transactions, idempotency handling
- **homepage/src/mainapp** (12 memories): React SPA core, Redux store config,
  signed-in routing, SEO meta patterns, React Query migration
- **proxy** (10 memories): BFF proxy hierarchy (3 layers), explicit upstream
  routing, header stripping at trust boundaries
- **state-management** (8 memories): Redux + reselect selectors, saga side effects,
  hybrid state (Redux + React Query coexistence)
- **testing** (6 memories): 3 test stacks (pytest/Jest/Playwright), mock-patch
  decorators, fixture composition system
- **security** (5 memories): Input sanitisation, SSRF protection, SQL injection
  guards, trust boundary decisions
- 8 more areas (28 memories) — use `memory_catalog` for full list

Key decisions:
- BFF pattern: React SPA never calls backend services directly
- Sync+async duality: every middleware has both implementations
- Feature flags via Optimizely with workspace-level decisions
```

This compresses 868 memories into ~300 tokens of context. The agent knows what areas have knowledge, what key decisions were made, and can `recall` with area tags for details.

### How It Works

#### Catalog Generation (pre-computed)

The catalog is built from two data sources:

1. **Code graph structure** — packages/modules already form a natural hierarchy via CONTAINS edges. The `summary_tree` tool already traverses this.

2. **Memory-to-code links** — memories are linked to graph nodes via RELATES_TO edges, auto-linking, and file tags. These give each memory a "location" in the code hierarchy.

**Build process:**

```
1. Get all packages/modules from graph (top-level CONTAINS tree)
2. For each area, count linked memories and collect their types
3. LLM-summarize each area's memories into a 1-sentence description
4. Store the catalog as a single special memory (type: catalog, tag: memory-catalog)
5. CWD-aware: at session start, highlight the area matching the working directory
```

#### Monorepo Scaling Strategy

The core challenge: a monorepo with 50 services can't list all areas in the session context. Three mechanisms handle this:

**1. CWD-relative focusing**

The SessionStart hook already receives `cwd`. For a monorepo, this tells us which service/package the user is working in.

```
cwd = /home/user/mega-corp/services/auth-service
      ^^^^^^^^^^^^^^^^^^^^^^^^ repo root
                               ^^^^^^^^^^^^^^^^^^^^ relative path = area focus
```

The catalog shows:
- **Full detail** for the CWD area (recent decisions, pending items, key patterns)
- **1-line summaries** for sibling areas (other services at the same level)
- **Counts only** for distant areas (infra, frontend, etc.)

This gives depth where the user is working and breadth awareness elsewhere.

**2. Area-level summarization instead of memory-level**

Instead of listing individual memories, each area gets one LLM-generated sentence:

```
Input (15 memories for services/billing):
- Decision: Chose Stripe over Braintree for payment processing
- Pattern: All webhook handlers follow verify→parse→dispatch pattern
- Insight: Invoice PDF generation uses wkhtmltopdf, takes ~2s per invoice
- Decision: Idempotency keys stored in Redis with 24h TTL
- ... (11 more)

Output: "Stripe-based payment system with webhook verify→parse→dispatch pattern,
         Redis-backed idempotency (24h TTL), wkhtmltopdf invoice generation"
```

15 memories → 1 sentence (~20 tokens). A 50-service monorepo with 2000 memories becomes ~50 sentences + detail for CWD area ≈ 500-800 tokens in the session context.

**3. Tiered depth control**

| Distance from CWD | Detail level | Token budget |
|---|---|---|
| **Same area** | Full: recent decisions, pending items, key patterns, file hotspots | ~200-400 tokens |
| **Sibling areas** (same parent directory) | 1-line summary + memory count | ~20 tokens each |
| **Distant areas** | Name + count only, grouped if >10 areas | ~5 tokens each |
| **Cross-cutting** (memories without area association) | 1-line summary | ~30 tokens |

Total budget: configurable, default 1000 tokens. The catalog generator fits content within this budget.

#### Catalog Refresh Triggers

The catalog is a cached artifact, not computed on every session start:

| Trigger | Action |
|---|---|
| After distillation (session end) | Regenerate catalog for affected areas |
| After enrichment run | Regenerate for enriched areas |
| After `store_memory` / `refine_memory` | Mark area as stale, regenerate on next session start |
| Manual: `codemem catalog --rebuild` | Full rebuild |

Staleness check: compare memory count + latest `updated_at` per area against cached catalog metadata. Only re-summarize areas that changed.

#### New MCP Tool: `memory_catalog`

For agents that want to browse beyond the session brief:

```json
{
  "name": "memory_catalog",
  "params": {
    "area": "services/billing",    // optional: drill into specific area
    "depth": 2,                     // 1=areas only, 2=with summaries, 3=with top memories
    "limit": 20                     // max areas to return
  }
}
```

Returns the full catalog or a subtree, with more detail than what fits in the session context.

#### Fallback (no LLM)

Without an LLM, area summaries are replaced with top-3 memory titles per area:

```
- **auth** (12 memories): "JWT RS256 migration decision", "Session middleware redesign", "RBAC role hierarchy"
```

Still useful — the agent knows what topics are covered and can `recall` for details. Just less polished than an LLM-generated summary.

### Implementation

| Step | What | Files |
|------|------|-------|
| 1 | Area extraction: map memories to code graph areas via links/tags | `codemem-engine/src/catalog.rs` (new) |
| 2 | Area summarization: LLM or fallback title-list | `codemem-engine/src/catalog.rs` |
| 3 | CWD-relative tiering: focus/sibling/distant depth control | `codemem-engine/src/catalog.rs` |
| 4 | Catalog storage: special memory with `memory-catalog` tag | `codemem-engine/src/catalog.rs` |
| 5 | SessionStart integration: inject catalog into `cmd_context()` | `codemem/src/cli/commands_lifecycle.rs` |
| 6 | MCP tool: `memory_catalog` for on-demand browsing | `codemem/src/mcp/tools_memory.rs` |
| 7 | Refresh triggers: stale detection + incremental rebuild | `codemem-engine/src/catalog.rs` |

## Implementation Plan

| Phase | What | Files | Depends on |
|-------|------|-------|------------|
| **1** | Staging table schema + migration | `codemem-storage/src/migrations.rs` | — |
| **2** | Staging buffer writes: redirect prompts, PostToolUse, tool errors, file watch to staging | `codemem/src/cli/commands_lifecycle.rs`, `codemem-engine/src/hooks/extractors.rs` | Phase 1 |
| **3** | Distillation module with LLM batch call + response parsing | `codemem-engine/src/distill.rs` (new) | Phase 1 |
| **4** | Wire Stop hook to trigger distillation before summary generation | `codemem/src/cli/commands_lifecycle.rs` | Phases 2, 3 |
| **5** | Enrichment aggregation: collect → group → summarize (NOTE: requires refactoring enrichment trait — all analysis modules change from `store_insight()` to `return Vec<Finding>`) | `codemem-engine/src/enrichment/mod.rs` + all analysis modules | Phase 3 |
| **6** | Heuristic fallback for no-LLM mode | `codemem-engine/src/distill.rs` | Phase 3 |
| **7** | Configuration: `[distillation]` + `[catalog]` sections in config.toml | `codemem-core/src/config.rs` | — |
| **8** | Recall `include_tags` + access-weighted decay | `codemem-engine/src/recall.rs`, consolidation | — |
| **9** | CLI command `codemem distill` + MCP tool (include `--dry-run` flag that shows what would be kept/merged/dropped without committing — useful for prompt tuning and building trust) | `codemem/src/cli/`, `codemem/src/mcp/` | Phase 3 |
| **10** | Memory catalog: area extraction + summarization + CWD tiering | `codemem-engine/src/catalog.rs` (new) | Phase 3 |
| **11** | SessionStart catalog injection into `cmd_context()` | `codemem/src/cli/commands_lifecycle.rs` | Phase 10 |
| **12** | `memory_catalog` MCP tool for on-demand browsing | `codemem/src/mcp/tools_memory.rs` | Phase 10 |

Phases 1-4 are the critical path. Phases 7-8 and 10-12 can be done in parallel.

**Note on phasing:** Component 7 (Memory Catalog, Phases 10-12) is nearly independent — it works with existing memories and benefits from distillation but doesn't require it. It could ship first to provide immediate value, then get better automatically when distillation lands.

## Expected Impact

| Metric | Before | With LLM | Without LLM (fallback) |
|--------|--------|----------|----------------------|
| Memories per session | ~25-30 | ~5-10 | ~12-15 |
| Embedding calls per session | ~25-30 | ~5-10 | ~12-15 |
| LLM calls per session | 0 | 1-2 (batched) | 0 |
| Enrichment memories per run | ~108 | ~15-20 | ~20-25 |
| Memory access rate | ~3% | Target >50% | Target >30% |
| Session summary size | ~857 bytes avg | ~200-300 bytes | ~400-500 bytes |

## Open Questions

1. **Staging retention policy:** Should processed staged observations be deleted after distillation, or kept indefinitely for audit? Current proposal: keep for `staging_ttl_hours` (7 days), then delete.

2. **Mid-session distillation:** When the threshold triggers mid-session, should the LLM see previously distilled memories from the same session for context? This improves quality but adds input tokens.

3. **Enrichment re-run staleness:** Should we track file hashes in enrichment memories and skip re-analysis for unchanged files? Adds complexity but prevents stale findings from accumulating.

4. **Batch size tuning:** The 30-item threshold and 50-item max are estimates. Need to benchmark actual token counts across different project types to find the sweet spot for cost vs. quality.

5. **Baseline memory lifecycle:** The agent team (baseline-scanner, symbol-analyst) produces hundreds of file/package summaries (435 in example-monorepo). These are useful during the analysis phase but become noise afterward. Should they be auto-distilled into area summaries after the agent team completes? Or kept as a separate "reference" tier that doesn't pollute recall? **Recommendation:** Auto-distill into area summaries after the agent team completes. The baseline summaries served their purpose during analysis — the catalog is the right home for their compressed form. A "reference" tier adds complexity for marginal benefit since the original source code is always available for re-analysis.

6. **Multi-language area detection:** Monorepos mix languages (Python backend + TypeScript frontend). Area detection should use the code graph package structure, not file extensions. The graph already has this via CONTAINS edges from packages, but need to verify it works across language boundaries (e.g., a `frontend/` package containing `.tsx` files and a `backend/` package containing `.py` files should be separate areas even if in the same directory tree).

7. **Catalog invalidation granularity:** Rebuilding the full catalog on every `store_memory` is expensive for large repos. Need a granular invalidation strategy — track which areas are dirty and only re-summarize those. The area-level checksum (memory count + max updated_at) should be sufficient.

8. **Cross-area memories:** Some memories span multiple areas (e.g., "BFF pattern: React SPA never calls backend services directly" touches both frontend and backend). Should these live in a "cross-cutting" section of the catalog, or be duplicated into both areas?
