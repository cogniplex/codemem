# LSP Enrichment + Cross-Repo Linking

Design plan for adding compiler-grade reference resolution and systematic cross-repository graph connectivity to codemem.

## Goals

1. **Enrich, don't duplicate** — LSP results augment existing ast-grep nodes, never create parallel ones
2. **Systematic cross-repo linking at index time** — when Repo A imports from Repo B, create real edges in the graph, regardless of which repo is indexed first
3. **Service-to-service visibility** — detect HTTP/gRPC endpoint definitions and client calls, link them across repos
4. **Zero required deps** — ast-grep remains the always-available baseline; LSP and API detection are optional enrichment passes

## Target Languages

Python, TypeScript/JS, Go, Java (in priority order).

## Current Limitations

| Gap | Impact |
|-----|--------|
| Unresolved references are discarded | Cross-module and cross-repo calls invisible |
| No package registry | Can't match imports to indexed namespaces |
| Reference resolution is heuristic-only | Ambiguous overloads, type-inferred calls, path aliases all fail |
| `get_cross_repo` is read-only | Scans manifests but creates no edges |
| No API surface detection | Microservice call graphs impossible |

---

## New Modules & Schema

### File Layout

```
codemem-engine/src/
  index/
    resolver.rs          ← modified: preserve unresolved refs
    linker.rs            ← NEW: cross-repo linker
    lsp/
      mod.rs             ← LspEnricher trait + orchestration
      pyright.rs         ← Python enrichment
      tsserver.rs        ← TS/JS enrichment
```

### Schema Migration (v14)

```sql
CREATE TABLE IF NOT EXISTS package_registry (
    package_name TEXT NOT NULL,
    namespace    TEXT NOT NULL,
    version      TEXT DEFAULT '',
    manifest     TEXT DEFAULT '',
    PRIMARY KEY (package_name, namespace)
);

CREATE TABLE IF NOT EXISTS unresolved_refs (
    id           TEXT PRIMARY KEY,
    namespace    TEXT NOT NULL,
    source_node  TEXT NOT NULL,
    target_name  TEXT NOT NULL,
    package_hint TEXT,
    ref_kind     TEXT NOT NULL,
    file_path    TEXT,
    line         INTEGER,
    created_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS api_endpoints (
    id        TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    method    TEXT,
    path      TEXT NOT NULL,
    handler   TEXT,
    schema    TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_unresolved_refs_pkg ON unresolved_refs(package_hint);
CREATE INDEX IF NOT EXISTS idx_unresolved_refs_ns ON unresolved_refs(namespace);
CREATE INDEX IF NOT EXISTS idx_package_registry_ns ON package_registry(namespace);
CREATE INDEX IF NOT EXISTS idx_api_endpoints_path ON api_endpoints(path);
```

---

## The Pipeline (4 Phases)

```
index_codebase("/path/to/repo")
  │
  Phase 1: AST-GREP INDEX (existing, fast, always runs)
  │  Parse all files → sym:, file:, chunk:, pkg: nodes
  │  Resolve references → edges with confidence scores
  │  Output: resolved edges + UNRESOLVED refs (currently discarded)
  │
  Phase 2: REGISTER + LINK (new, fast, always runs)
  │  ├─ Parse manifests → register in package_registry
  │  ├─ Extract package_hint from each unresolved ref's import context
  │  ├─ Store unresolved refs in unresolved_refs table
  │  ├─ Forward link: match THIS repo's unresolved refs against
  │  │   already-indexed namespaces in package_registry
  │  │   → create cross-namespace edges for matches
  │  └─ Backward link: check OTHER repos' unresolved refs
  │      that reference THIS repo's package name
  │      → resolve against THIS repo's sym: nodes
  │      → create cross-namespace edges, delete resolved entries
  │
  Phase 3: LSP ENRICHMENT (new, optional, async)
  │  Only runs if language tooling is available on PATH
  │  ├─ Collect low-confidence edges (<0.8) + remaining unresolved refs
  │  ├─ Batch query: pyright --outputjson / tsc --noEmit --declaration
  │  ├─ For each LSP result:
  │  │   ├─ Match to existing sym: node by (file, line_range, name)
  │  │   ├─ Found in THIS repo → upgrade edge confidence to 1.0, source="lsp"
  │  │   ├─ Found in deps (node_modules/site-packages)?
  │  │   │   → create ext: node + edge, add package_hint
  │  │   │   → check package_registry → if indexed, create cross-ns edge
  │  │   └─ Enrich sym: node payload with type info, return type, generics
  │  └─ Re-run cross-repo linker for newly created ext: nodes
  │
  Phase 4: API SURFACE LINKING (new, optional, after enrichment)
     ├─ Detect endpoint definitions (decorators/annotations):
     │   @app.route, @GetMapping, http.HandleFunc, express.get
     │   → register in api_endpoints table
     ├─ Detect HTTP client calls:
     │   fetch(), requests.post(), http.Get(), RestTemplate
     │   → extract URL patterns
     └─ Match URL patterns against api_endpoints across namespaces
         → create edges: sym:caller → CALLS → ep:service:METHOD:/path
```

---

## Phase 1: Preserve Unresolved Refs

Currently in `resolver.rs`, unresolved refs return `None` and are dropped.

### Change to Resolver Output

```rust
pub struct ResolveResult {
    pub edges: Vec<ResolvedEdge>,
    pub unresolved: Vec<UnresolvedRef>,  // NEW
}

pub struct UnresolvedRef {
    pub source_node: String,          // sym: qualified_name of caller
    pub target_name: String,          // what they tried to call/import
    pub package_hint: Option<String>, // extracted from import context
    pub ref_kind: String,             // "call", "import", "inherits"
    pub file_path: String,
    pub line: usize,
}
```

### Package Hint Extraction Per Language

| Language | Import Statement | `package_hint` |
|----------|-----------------|----------------|
| Python | `from requests.api import get` | `"requests"` — first segment |
| Python | `from myapp.utils import helper` | `"myapp"` — first segment (may be local) |
| TS/JS | `import { x } from '@acme/shared'` | `"@acme/shared"` — full specifier if scoped |
| TS/JS | `import { x } from 'lodash'` | `"lodash"` — first segment |
| Go | `import "github.com/acme/utils"` | `"github.com/acme/utils"` — full module path |
| Java | `import com.acme.shared.Validator` | Lookup against pom.xml groupId:artifactId |

The extraction happens during reference extraction in `engine/references.rs`. Each import-kind reference already has the full import path — the package hint is derived from that path using language-specific rules.

---

## Phase 2: Cross-Repo Linker

### Package Registry

When `index_codebase` runs, manifest parsing already happens via `scan_manifests()`. The new step: write each discovered package name into `package_registry`, keyed by `(package_name, namespace)`.

### Bidirectional Linking

The linker runs after every `index_codebase` and works in both directions:

**Forward link** — resolve THIS repo's dangling refs against already-indexed repos:
```
SELECT ur.* FROM unresolved_refs ur
JOIN package_registry pr ON ur.package_hint = pr.package_name
WHERE ur.namespace = ? AND pr.namespace != ur.namespace
```
For each match, query `sym:` nodes in the target namespace, fuzzy-match `target_name`, and create a cross-namespace edge.

**Backward link** — resolve OTHER repos' dangling refs that point to THIS repo:
```
SELECT ur.* FROM unresolved_refs ur
JOIN package_registry pr ON ur.package_hint = pr.package_name
WHERE pr.namespace = ? AND ur.namespace != ?
```
For each match, resolve `target_name` against THIS repo's symbols, create edge, delete from `unresolved_refs`.

### Cross-Namespace Edge Format

```rust
Edge {
    id: "xref:{src_ns}/{src_sym}→{dst_ns}/{dst_sym}",
    src: "sym:handler.process",           // in namespace "backend"
    dst: "sym:validate",                  // in namespace "shared-lib"
    relationship: "Calls",
    properties: json!({
        "cross_namespace": true,
        "source_namespace": "backend",
        "target_namespace": "shared-lib",
        "confidence": 0.9,
        "link_source": "manifest"         // or "lsp" when LSP confirms
    })
}
```

### Ordering Independence

Whether you index A first or B first, the linker catches up:

```
Scenario 1: A indexed first
  t1: index Repo A → unresolved ref to "shared-lib.validate" → stored
  t2: index Repo B (shared-lib) → registers in package_registry
      → backward linker finds A's unresolved ref → resolves → creates edge

Scenario 2: B indexed first
  t1: index Repo B → registers, nothing to link yet
  t2: index Repo A → unresolved ref → forward linker finds B in registry
      → resolves → creates edge
```

### Symbol Matching Strategy

When matching an unresolved `target_name` against symbols in another namespace:

1. **Exact qualified name match** → confidence 1.0
2. **Suffix match** (e.g., `validate` matches `utils.validate`) → confidence 0.85
3. **Exported symbols only** (public visibility) → preference boost +0.2
4. **Multiple candidates** → pick highest visibility + shortest qualified name (most likely the public API)

---

## Phase 3: LSP Enrichment

### Design Principles

- LSP enriches existing `sym:` nodes — never creates competing symbol nodes
- Node identity matching: **(file_path, line_range, symbol_name)**, not qualified_name
- External dependencies get `ext:` prefix nodes (lightweight stubs, no source code)
- Runs as optional async pass — doesn't block the fast index

### Enricher Trait

```rust
pub trait LspEnricher: Send + Sync {
    /// Check if this enricher's tooling is available on PATH
    fn is_available(&self) -> bool;

    /// File extensions this enricher handles
    fn extensions(&self) -> &[&str];

    /// Batch resolve: given files with unresolved/low-confidence refs,
    /// return enrichment results
    fn enrich(&self, project_root: &Path, targets: &[EnrichmentTarget]) -> Vec<EnrichmentResult>;
}

pub struct EnrichmentTarget {
    pub file_path: String,
    pub refs: Vec<RefToResolve>,
}

pub struct RefToResolve {
    pub source_node: String,
    pub target_name: String,
    pub line: usize,
    pub current_confidence: Option<f64>,  // None = unresolved, Some = low confidence
}

pub struct EnrichmentResult {
    pub resolved_refs: Vec<LspResolvedRef>,
    pub type_annotations: Vec<TypeAnnotation>,
}

pub struct LspResolvedRef {
    pub source_file: String,
    pub source_line: usize,
    pub target_file: String,         // absolute — may be in node_modules etc.
    pub target_line: usize,
    pub target_symbol: String,
    pub is_external: bool,           // true if in deps, not in project
    pub package_name: Option<String>,
}

pub struct TypeAnnotation {
    pub file_path: String,
    pub line: usize,
    pub symbol_name: String,
    pub resolved_type: String,       // e.g., "List[User]", "Promise<Response>"
    pub return_type: Option<String>,
    pub generic_params: Vec<String>,
}
```

### Pyright (Python — highest priority)

```bash
pyright --outputjson --level basic /path/to/project
```

Pyright outputs JSON with diagnostics and resolved symbol locations. Parse the output, match each resolved reference back to existing `sym:` nodes by `(file, line)`.

Why Pyright:
- Single binary or `npm install pyright`
- No project build required — works on raw source
- Handles duck typing, `*args/**kwargs`, dataclasses, type stubs
- Batch mode — one invocation for the whole project

### tsserver (TypeScript/JS)

```bash
tsc --noEmit --declaration --declarationDir /tmp/out
```

Or use the TypeScript compiler API programmatically. Key wins:
- Resolves path aliases (`@/components/Button` → real file)
- Resolves barrel exports (`export * from './utils'`)
- Type narrowing and generic inference

### Edge Upgrades

When LSP confirms an existing ast-grep edge:
```
Before: sym:caller → sym:target (confidence: 0.5, source: "ast-grep")
After:  sym:caller → sym:target (confidence: 1.0, source: "lsp")
```

Add `source` field to edge properties. Use `INSERT OR REPLACE` keyed on edge ID.

### External Node Creation

When LSP resolves to a path outside the project (in `node_modules`, `site-packages`, etc.):

```
LSP: handler.py:15 calls → /venv/lib/python3.11/site-packages/requests/api.py:get()
  ↓
Create: ext:requests.api.get (kind: Function, payload: {package: "requests", version: "2.31.0"})
Edge:   sym:handler.process → CALLS → ext:requests.api.get (confidence: 1.0, source: "lsp")
```

External node ID format: `ext:{package}.{qualified_name}`

### LSP → Linker Feedback Loop

After LSP enrichment, re-run the cross-repo linker because:
- LSP resolved refs now carry accurate `package_name` from dep paths
- New `ext:` nodes may match `sym:` nodes in already-indexed namespaces

```
LSP resolves: handler.py:15 → /venv/lib/shared_lib/utils.py:validate()
  → is_external=true, package="shared-lib"
  → check package_registry("shared-lib") → found, namespace "shared-lib"
  → match validate() against sym: nodes in "shared-lib"
  → create cross-namespace edge: backend/sym:handler → CALLS → shared-lib/sym:validate
```

---

## Phase 4: API Surface Linking

For microservice architectures where repos communicate via HTTP/gRPC, not imports.

### Endpoint Detection (ast-grep rules)

**Python (Flask/FastAPI/Django):**
```yaml
- kind: decorated_definition
  special: python_route_decorator
  # @app.route("/users"), @router.get("/users/{id}"), @api_view(["GET"])
```

**TypeScript (Express/NestJS/Fastify):**
```yaml
- kind: call_expression
  special: ts_route_handler
  # app.get("/users", handler), @Get("/users"), fastify.route({...})
```

**Java (Spring):**
```yaml
- kind: annotation
  special: java_request_mapping
  # @GetMapping("/users"), @RequestMapping(value="/users", method=GET)
```

**Go (net/http, gin, echo):**
```yaml
- kind: call_expression
  special: go_route_handler
  # http.HandleFunc("/users", handler), r.GET("/users", handler)
```

Detected endpoints are registered in the `api_endpoints` table with:
- HTTP method (GET, POST, etc.)
- Path pattern (with parameter placeholders normalized: `/users/{id}`)
- Handler symbol (the `sym:` node that handles the request)

### HTTP Client Call Detection

```yaml
# Python: requests.get("url"), httpx.post("url"), aiohttp.ClientSession
# TS/JS: fetch("url"), axios.get("url"), got("url")
# Java: RestTemplate.exchange("url"), WebClient, HttpClient
# Go: http.Get("url"), http.Post("url"), resty.R().Get("url")
```

Extract URL string literals from client call arguments. Normalize path patterns.

### Cross-Service Edge Creation

Match client call URL patterns against registered `api_endpoints` across all namespaces:

```
sym:order_service.create_order calls requests.post("http://user-service/api/users")
  → extract path: /api/users, method: POST
  → match against api_endpoints: ep:user-service:POST:/api/users
  → handler: sym:user_service.create_user

Edge chain:
  sym:create_order → CALLS → ep:user-service:POST:/api/users → HANDLES → sym:create_user
```

URL matching strategy:
1. **Exact path match** → confidence 1.0
2. **Path with parameter normalization** (`/users/123` matches `/users/{id}`) → confidence 0.9
3. **Path prefix match** → confidence 0.7
4. **Hostname/service name matching** when URL contains service identifier

---

## Edge Source Tracking

All edges carry a `source` field in properties to track provenance:

| Source | Meaning |
|--------|---------|
| `"ast-grep"` | Resolved by heuristic name matching |
| `"manifest"` | Cross-repo link via package registry |
| `"lsp"` | Confirmed by language server / type checker |
| `"lsp-confirmed"` | ast-grep edge upgraded by LSP confirmation |
| `"api-surface"` | HTTP endpoint → client call match |

---

## Query Changes

### `graph_edges_for_namespace` — include cross-namespace edges

Currently filters to edges where both src and dst are in the same namespace. Add an option to include cross-namespace edges:

```sql
-- Existing: intra-namespace only
SELECT ... FROM graph_edges e
JOIN graph_nodes gs ON e.src = gs.id
JOIN graph_nodes gd ON e.dst = gd.id
WHERE gs.namespace = ?1 AND gd.namespace = ?1

-- New: include cross-namespace edges touching this namespace
SELECT ... FROM graph_edges e
JOIN graph_nodes gs ON e.src = gs.id
JOIN graph_nodes gd ON e.dst = gd.id
WHERE gs.namespace = ?1 OR gd.namespace = ?1
```

### `get_cross_repo` — return actual edges, not just manifests

Upgrade from "dump manifest JSON" to returning the live cross-namespace edges:

```json
{
  "packages": [...],
  "dependencies": [...],
  "cross_repo_edges": [
    {
      "source": "backend/sym:handler.process",
      "target": "shared-lib/sym:utils.validate",
      "relationship": "Calls",
      "confidence": 1.0,
      "link_source": "lsp"
    }
  ],
  "unresolved_refs": 12,
  "pending_packages": ["redis", "sqlalchemy"]
}
```

---

## Implementation Order

| Step | What | Depends On | Effort | External Deps |
|------|------|-----------|--------|---------------|
| **1** | Schema migration (3 tables + indexes) | — | S | None |
| **2** | Preserve unresolved refs from resolver | — | S | None |
| **3** | Package hint extraction per language | — | M | None |
| **4** | Package registry population from manifests | Step 1 | S | None |
| **5** | `CrossRepoLinker` (forward + backward) | Steps 1–4 | M | None |
| **6** | Wire linker into `index_codebase` | Step 5 | S | None |
| **7** | `LspEnricher` trait + orchestration | — | S | None |
| **8** | Pyright implementation | Step 7 | M | pyright on PATH |
| **9** | tsserver implementation | Step 7 | M | tsc on PATH |
| **10** | LSP → linker feedback loop | Steps 5, 8/9 | S | None |
| **11** | API endpoint detection rules (4 langs) | — | M | None |
| **12** | HTTP client call detection rules | — | M | None |
| **13** | API surface linker | Steps 1, 11, 12 | M | None |

**Steps 1–6**: Systematic cross-repo linking with zero external dependencies.
**Steps 7–10**: Compiler-grade accuracy for Python and TypeScript.
**Steps 11–13**: Service-to-service call graphs for microservice architectures.

---

## Multi-Language Monorepo Support

### How It Works

Each `LspEnricher` is scoped by file extension, not by repository. In a multi-language monorepo, the orchestration layer partitions files by extension and runs matching enrichers in parallel:

```
monorepo/
  backend/         ← .py files  → Pyright enricher
  frontend/        ← .ts files  → tsserver enricher
  services/auth/   ← .go files  → (future) gopls enricher
  shared/proto/    ← .proto     → skipped (no enricher)
```

### Orchestration Flow

```
1. Collect all files with low-confidence/unresolved refs
2. Partition by extension → {".py": [...], ".ts": [...], ".go": [...]}
3. For each partition:
   a. Find enricher where extensions() matches AND is_available() == true
   b. Detect project root within monorepo (tsconfig.json, pyproject.toml, go.mod)
   c. Run enricher scoped to that project root
4. Run all enrichers in parallel (independent processes)
5. Merge all results into the same graph
6. Re-run cross-repo linker once with combined results
```

### Project Root Detection

In a monorepo, multiple project configs may exist at different levels. The enricher must detect and respect these boundaries:

```
monorepo/
  backend/
    pyproject.toml        ← Pyright runs scoped here
    src/
      api.py
  frontend/
    tsconfig.json         ← tsserver runs scoped here
    src/
      app.ts
  services/
    auth/
      go.mod              ← gopls would run scoped here
      main.go
```

Detection strategy per enricher:

| Enricher | Config File | Behavior |
|----------|------------|----------|
| Pyright | `pyproject.toml`, `pyrightconfig.json` | Walk up from each `.py` file to find nearest config |
| tsserver | `tsconfig.json` | Walk up from each `.ts`/`.tsx` file to find nearest config |
| gopls | `go.mod` | Walk up from each `.go` file to find nearest module root |

If multiple project roots are found, run the enricher once per root with the relevant file subset.

### Cross-Language Linking Within a Monorepo

LSP enrichers cannot cross language boundaries — Pyright knows nothing about TypeScript files and vice versa. However, cross-language connections happen naturally through two mechanisms:

**1. Phase 4 API Surface Linking** catches HTTP calls across languages:
```
backend/api.py:     @app.route("/api/users")  → registers endpoint
frontend/app.ts:    fetch("/api/users")        → detected as client call
                    → edge: sym:App.loadUsers → CALLS → ep:backend:GET:/api/users
```

**2. Shared package imports** when both languages consume a common dependency:
```
backend/api.py:     from shared_models import User    → unresolved, package_hint="shared-models"
frontend/types.ts:  import { User } from '@acme/shared-models'  → unresolved, package_hint="@acme/shared-models"
```
If `shared-models` is indexed as its own namespace, both resolve via the cross-repo linker. If both package names map to the same namespace in the registry (via an alias or naming convention), they connect to the same target symbols.

### Extension Overlap Handling

Multiple enrichers may claim the same extension (e.g., `.js` could be handled by tsserver or a future Node-specific enricher). Resolution order:

1. **Specificity** — enricher that requires a project config matching this file wins (tsserver with `tsconfig.json` beats a generic JS enricher)
2. **Priority config** — user-configurable enricher priority in `~/.codemem/config.toml`:
   ```toml
   [lsp]
   enricher_priority = ["tsserver", "pyright", "gopls"]
   ```
3. **First match** — if no config and no specificity difference, first available enricher wins
4. **No double-enrichment** — once a file is claimed by an enricher, no other enricher processes it

### Monorepo-Specific Config

```toml
# ~/.codemem/config.toml

[lsp]
enabled = true
enricher_priority = ["pyright", "tsserver"]
parallel = true                    # run enrichers concurrently

# Override project root detection for specific paths
[lsp.project_roots]
"backend" = { enricher = "pyright", config = "backend/pyproject.toml" }
"frontend" = { enricher = "tsserver", config = "frontend/tsconfig.json" }
```

This allows explicit mapping when auto-detection isn't sufficient (e.g., workspaces with non-standard layouts).

### Intra-Namespace API Linking

API surface linking (Phase 4) must work **within** a single namespace too, not just across namespaces. In a multi-language monorepo like EMS, the Python and TypeScript halves don't import each other — they communicate over HTTP within the same repo. The endpoint linker should match client calls to endpoint definitions regardless of whether source and target are in the same or different namespaces.

---

## Case Study: epidemic-music-server

A real-world multi-language monorepo indexed in codemem that validates this plan.

### The Architecture

- **3 apps in one repo**: Django BFF (Python) + React SPA "Homepage" (TypeScript) + Partner Portal (TypeScript/legacy)
- ~3000 source files, 200+ URL route patterns, 65+ React feature packages
- Django BFF proxies to 10+ backend microservices (trantor, Braavos, Heimdall, etc.)
- React SPA **never calls backend services directly** — everything routes through the BFF

### What's Currently Invisible

**1. Python↔TypeScript boundary (the biggest gap)**

Django defines proxy routes in `frontend/src/frontend/track/views.py`. React makes `fetch()` calls to those routes from `homepage/src/mainapp/api.ts`. These are completely disconnected in the current graph — two languages, no edges between them.

With Phase 4:
```
Django:  TrackView.get() at /api/track/{id}  →  ep:ems:GET:/api/track/{id}
React:   fetch("/api/track/123") in api.ts   →  matched to ep:ems:GET:/api/track/{id}
Edge:    ts/sym:api.getTrack → CALLS → ep:ems:GET:/api/track/{id} → HANDLES → py/sym:TrackView.get
```

**2. BFF → microservice calls**

The Django BFF has HTTP clients in `frontend/src/frontend/clients/` (glue.py, music.py, playlists.py, heimdall.py, workspace.py, BraavosBackendClient). These make `requests.post()`/`requests.get()` calls to external services. If those services are indexed in separate namespaces:

```
py/sym:BraavosBackendClient.proxy() → requests.post("http://braavos/api/subscriptions")
    ↓ Phase 4 matches against api_endpoints
ep:braavos:POST:/api/subscriptions
    ↓ if braavos namespace is indexed
py/sym:braavos.SubscriptionView.post
```

**3. Type-inferred relationships**

EMS uses 3 state management layers (Redux + React Query + Apollo) with heavy generics:
- Zod schema validation with inferred types
- React Query typed hooks (`useQuery<TrackResponse>`)
- Python decorators (`@service_connection`, middleware chains)
- Dynamic dispatch in `BraavosBackendClient` extending `BaseBraavosBackendClient`

All invisible to ast-grep, all resolvable by Pyright/tsserver in Phase 3.

**4. Migration tracking across languages**

EMS has 4 active migrations (catch-all→explicit proxy, untyped→Zod API, int→UUID users, Recurly→Braavos payments). Cross-language linking would show which React components still call unvalidated Django endpoints vs Zod-validated ones — tracking migration progress across the Python↔TypeScript boundary.

### Lessons for the Plan

| Observation | Implication |
|-------------|------------|
| Python and TS communicate only via HTTP, never imports | Phase 4 (API surface) is essential, not optional — it's the only way to connect multi-language monorepos |
| BFF proxies to 10+ external services via HTTP clients | Phase 2 cross-repo linker + Phase 4 endpoint matching must compose: BFF→microservice→handler |
| Single namespace contains both languages | API surface linker must work intra-namespace, not just cross-namespace |
| 200+ URL patterns, many dynamic | URL pattern normalization and parameter placeholder matching (`/track/{id}`) are critical for accuracy |
| 3 different HTTP client patterns in Django alone | Phase 4 ast-grep rules need to detect `service_connection` facade calls, `ProxyView` class hierarchy, and direct `requests.*` calls |
| Feature flags gate endpoints on both sides | Edge properties should capture feature-flag context when detectable, enabling "this call only happens when flag X is enabled" |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Pyright/tsc not installed | Graceful skip — `is_available()` checks PATH, logs info-level message |
| LSP output format changes | Version-pin expected output schema, test against multiple versions |
| `unresolved_refs` table grows unbounded | TTL-based cleanup — delete refs older than N days with no matching registry entry |
| Cross-namespace edges break namespace isolation | `cross_namespace: true` property allows queries to filter them in or out |
| Package name collisions (e.g., two repos both name themselves "utils") | Registry key is `(package_name, namespace)` — collisions produce multiple candidates, resolved by manifest version/path proximity |
| URL pattern matching false positives | Require hostname/service-name match in addition to path, or flag low-confidence API links for review |
