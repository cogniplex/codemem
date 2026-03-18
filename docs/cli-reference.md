# Codemem CLI Reference

All commands are invoked as `codemem <command> [options]`.

## Global Environment Variables

These environment variables affect all commands that use embeddings (`search`, `serve`, `ingest`, `index`, `watch`, `import`).

### Embedding Provider

| Variable | Values | Default |
|----------|--------|---------|
| `CODEMEM_EMBED_PROVIDER` | `candle`, `ollama`, `openai`, `gemini` | `candle` |
| `CODEMEM_EMBED_MODEL` | model name | provider default |
| `CODEMEM_EMBED_URL` | base URL override | provider default |
| `CODEMEM_EMBED_API_KEY` | API key | also reads `OPENAI_API_KEY` |
| `CODEMEM_EMBED_DIMENSIONS` | integer | `768` |

The `openai` provider works with any OpenAI-compatible API (Voyage AI, Together, Azure, etc.) via `CODEMEM_EMBED_URL`. The `gemini` provider also accepts `GEMINI_API_KEY` or `GOOGLE_API_KEY` as fallbacks.

### Additional Embedding Config

| Variable | Values | Default |
|----------|--------|---------|
| `CODEMEM_EMBED_BATCH_SIZE` | integer | `16` |
| `CODEMEM_EMBED_DTYPE` | `f32`, `f16`, `bf16` | `f32` |
| `CODEMEM_EMBED_CACHE_SIZE` | integer | `10000` |

### Observation Compression

| Variable | Values | Default |
|----------|--------|---------|
| `CODEMEM_COMPRESS_PROVIDER` | `ollama`, `openai`, `anthropic` | disabled |
| `CODEMEM_COMPRESS_MODEL` | model name | provider default |
| `CODEMEM_COMPRESS_URL` | base URL override | provider default |
| `CODEMEM_API_KEY` | API key | also reads `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |

---

## `codemem init`

Initialize Codemem in the current directory. Downloads the embedding model (~440MB BAAI/bge-base-en-v1.5 safetensors), registers 9 lifecycle hooks (SessionStart, UserPromptSubmit, PostToolUse, PostToolUseFailure, Stop, SubagentStart, SubagentStop, SessionEnd, PreCompact), and creates the MCP server configuration.

**Syntax**

```
codemem init [--path <dir>] [--skip-model]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--path <dir>` | Directory to initialize (defaults to current directory) |
| `--skip-model` | Skip embedding model download (useful for CI/testing) |

Automatically detects installed AI assistants (Claude Code, Cursor, Windsurf) and configures hooks and MCP server entries.

**Example**

```bash
codemem init
codemem init --path ~/projects/my-monorepo
```

---

## `codemem search`

Search stored memories using 9-component hybrid scoring (vector similarity, graph strength, token overlap, temporal, tags, importance, confidence, recency).

**Syntax**

```
codemem search <query> [--k <num>] [--namespace <ns>]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--k <num>` | Number of results to return (default: 5) |
| `--namespace <ns>` | Restrict search to a specific namespace (e.g., project path) |

**Example**

```bash
codemem search "error handling pattern"
codemem search "database migration" --k 10
codemem search "auth flow" --namespace /home/user/projects/backend
```

---

## `codemem stats`

Display database statistics including total memory count, graph node and edge counts, and vector index size.

**Syntax**

```
codemem stats
```

**Example**

```bash
codemem stats
```

---

## `codemem serve`

Start the MCP server using JSON-RPC over stdio (default). Composable with `--api` for REST API + embedded frontend, and `--http` for HTTP-based MCP transport.

**Syntax**

```
codemem serve [--api] [--http] [--port <port>]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--api` | Enable REST API + embedded frontend on HTTP |
| `--http` | Use HTTP transport for MCP (instead of stdio) |
| `--port <port>` | HTTP server port (default: 4242, used when `--api` or `--http` is set) |

**Example**

```bash
codemem serve              # JSON-RPC stdio (default)
codemem serve --api        # REST API + UI on port 4242
codemem serve --http       # MCP over HTTP on port 4242
```

---

## `codemem ui`

Open the control plane UI. Alias for `codemem serve --api` with auto-browser-open.

**Syntax**

```
codemem ui [--port <port>] [--no-open]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--port <port>` | HTTP server port (default: 4242) |
| `--no-open` | Don't open browser automatically |

**Example**

```bash
codemem ui
codemem ui --port 8080 --no-open
```

---

## `codemem ingest`

Process a PostToolUse hook payload from stdin. This command is called automatically by the configured hooks whenever an AI assistant uses a tool (Read, Glob, Grep, Edit, Write, Bash, WebFetch, WebSearch, Agent, SendMessage, ListDir). It extracts relevant information, optionally compresses via LLM, and stores it as a memory with embeddings and graph nodes.

Uses the embedding provider configured via `CODEMEM_EMBED_*` env vars (see [Global Environment Variables](#global-environment-variables)). Compression is configured via `CODEMEM_COMPRESS_*` env vars.

**Syntax**

```
codemem ingest
```

**Example**

```bash
echo '{"tool":"Read","path":"/src/main.rs","content":"..."}' | codemem ingest
```

---

## `codemem consolidate`

Run memory consolidation cycles inspired by neuroscience research. Supports four cycle types: decay (daily, reduces confidence of stale memories), creative/REM (weekly, discovers novel associations), cluster (monthly, merges related memories), and forget (optional, removes low-value memories).

**Syntax**

```
codemem consolidate [--cycle <name>] [--status]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--cycle <name>` | Consolidation cycle to run: `decay`, `creative`, `cluster`, or `forget` |
| `--status` | Show consolidation status without running a cycle |

**Example**

```bash
codemem consolidate --cycle decay
codemem consolidate --cycle creative
codemem consolidate --status
```

---

---

## `codemem index`

Index a codebase for structural analysis using ast-grep. Parses source files across 14 supported languages (Rust, TypeScript/JS/JSX, Python, Go, C/C++, Java, Ruby, C#, Kotlin, Swift, PHP, Scala, HCL/Terraform) and extracts functions, structs, classes, imports, and call relationships.

**Syntax**

```
codemem index [--path <dir>] [--verbose]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--path <dir>` | Directory to index (defaults to current directory) |
| `--verbose` | Print detailed progress and per-file output |

**Example**

```bash
codemem index
codemem index --path ~/projects/my-app --verbose
```

---

## `codemem analyze`

Run the full analysis pipeline on a codebase: index (tree-sitter) → SCIP enrichment (if indexers installed) → enrich (git history, security, performance, etc.) → temporal graph (commits, ModifiedBy edges) → PageRank → Louvain clustering.

**Syntax**

```
codemem analyze [--path <dir>] [--namespace <ns>] [--days <num>] [--skip-scip] [--skip-embed] [--skip-enrich] [--force]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--path <dir>` | Directory to analyze (defaults to current directory) |
| `--namespace <ns>` | Namespace for storing results (defaults to directory basename) |
| `--days <num>` | Days of git history to analyze (default: 90) |
| `--skip-scip` | Skip SCIP indexing (fast, ast-grep only) |
| `--skip-embed` | Skip embedding phase (store graph without vectorizing) |
| `--skip-enrich` | Skip enrichment phase (no git/complexity/security analysis) |
| `--force` | Force re-index even when file SHAs haven't changed |

**Example**

```bash
codemem analyze
codemem analyze --path ~/projects/my-app --namespace my-app --days 180
codemem analyze --skip-scip --skip-enrich   # Fast: ast-grep index + PageRank only
```

**Output**

Prints a summary including:
- Index stats (files parsed, symbols extracted, edges resolved, chunks stored)
- Enrichment results (git history insights, security findings, performance hotspots)
- Top 10 important nodes by PageRank score
- Cluster count from Louvain community detection

---

## `codemem review`

Review a diff: map changed lines to graph symbols, compute multi-hop blast radius with risk scoring, surface relevant memories and potentially missing changes. Reads unified diff from stdin.

**Syntax**

```
codemem review [--base <ref>] [--depth <num>] [--format <fmt>]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--base <ref>` | Base ref for scope context (default: `main`) |
| `--depth <num>` | Traversal depth for transitive impact analysis (default: 2) |
| `--format <fmt>` | Output format: `json` (default) or `text` (human-readable) |

**Example**

```bash
git diff main..HEAD | codemem review --format text
git diff --cached | codemem review --depth 3
```

**Output** (text format):

```
Risk Score: 7.2 (high)
Changed Symbols: 3
Direct Dependents: 12
Transitive Dependents: 47
Potentially Missing Changes: 1
```

---

## `codemem export`

Export memories in multiple formats for backup, migration, or external analysis.

**Syntax**

```
codemem export [--namespace <ns>] [--memory-type <type>] [--output <file>] [--format <fmt>]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--namespace <ns>` | Export only memories from a specific namespace |
| `--memory-type <type>` | Export only memories of a given type (Decision, Pattern, Preference, Style, Habit, Insight, Context) |
| `--output <file>` | Output file path (defaults to stdout) |
| `--format <fmt>` | Output format: `jsonl` (default), `json`, `csv`, `markdown`/`md` |

**Example**

```bash
codemem export > all-memories.jsonl
codemem export --format csv --output memories.csv
codemem export --format markdown --output memories.md
codemem export --namespace /home/user/projects/api --output api-memories.jsonl
codemem export --memory-type Pattern --format json --output patterns.json
```

---

## `codemem import`

Import memories from a JSONL file previously created with `codemem export`.

**Syntax**

```
codemem import --input <file> [--skip-duplicates]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--input <file>` | Path to the JSONL file to import |
| `--skip-duplicates` | Skip memories that already exist (matched by SHA-256 content hash) |

**Example**

```bash
codemem import --input memories.jsonl
codemem import --input backup.jsonl --skip-duplicates
```

---

## `codemem watch`

Watch a directory for file changes and automatically re-index modified files. Uses `notify` with 50ms debouncing. Respects `.gitignore` patterns and common ignore directories (`node_modules`, `target`, `.git`, etc.).

**Syntax**

```
codemem watch [--path <dir>]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--path <dir>` | Directory to watch (defaults to current directory) |

**Example**

```bash
codemem watch
codemem watch --path ~/projects/my-app
```

---

## `codemem sessions`

Manage memory sessions for cross-session continuity. Sessions track interaction periods with AI assistants and can be used to scope memories and detect patterns.

### `codemem sessions list`

List all sessions, optionally filtered by namespace.

```bash
codemem sessions list
codemem sessions list --namespace /Users/dev/myproject
```

### `codemem sessions start`

Start a new session. Returns a session ID.

```bash
codemem sessions start
codemem sessions start --namespace /Users/dev/myproject
```

### `codemem sessions end`

End an active session with an optional summary.

```bash
codemem sessions end <session-id>
codemem sessions end <session-id> --summary "Refactored auth module to use JWT"
```

---

## Lifecycle Hook Commands

These commands are called automatically by registered hooks. They are not intended for manual use but are documented here for reference.

---

## `codemem context`

**Hook:** SessionStart

Queries the memory database for relevant prior context and injects it into the new session via `hookSpecificOutput.additionalContext`. Handles the `source` field in the payload:

- `startup` (default): Full context injection with all 5 data sources
- `resume`: Minimal response — session is continuing with context intact
- `compact`: Saves a checkpoint memory, then full context injection
- `clear`: Treated like startup

The injected context includes:

- Recent sessions with summaries (up to 5)
- Key Decision, Insight, and Pattern memories (up to 15)
- File hotspots (most frequently touched files)
- Detected cross-session patterns (repeated searches, etc.)
- Database statistics

Reads JSON `{session_id, cwd, source}` from stdin. Outputs JSON with `hookSpecificOutput.additionalContext` containing compact markdown wrapped in `<codemem-context>` tags.

**Syntax**

```
codemem context
```

---

## `codemem prompt`

**Hook:** UserPromptSubmit

Records the user's prompt as a Context memory (importance 0.3) for session tracking. This enables session summaries to include what the user asked for.

Reads JSON `{session_id, cwd, prompt}` from stdin. Outputs JSON `{continue: true}`.

Prompts shorter than 5 characters are ignored. Trivial prompts (fewer than 30 characters or fewer than 5 words) are also skipped.

**Syntax**

```
codemem prompt
```

---

## `codemem summarize`

**Hook:** Stop

Builds a structured session summary from all memories created during the session. Categorizes memories into files read, files edited, searches performed, decisions made, and user prompts. Stores the summary as an Insight memory and ends the session. Appends a truncated excerpt of `last_assistant_message` for richer context.

Reads JSON `{session_id, cwd, stop_hook_active, last_assistant_message}` from stdin. Outputs JSON `{continue: true}`. Skips execution if `stop_hook_active` is true (prevents recursive hook loops).

**Summary format:** `"Requests: ...; Investigated N file(s): ...; Modified N file(s): ...; Decisions: ...; Searched: ..."`

**Syntax**

```
codemem summarize
```

---

## `codemem agent-result`

**Hook:** SubagentStop

Captures subagent findings from `last_assistant_message` as Insight memories (importance 0.5) with `agent-result` and `agent:<type>` tags. Skips messages shorter than 20 characters. Guards against `stop_hook_active` loops.

Reads JSON `{session_id, cwd, agent_type, agent_id, last_assistant_message, stop_hook_active}` from stdin. Outputs JSON `{}`.

**Syntax**

```
codemem agent-result
```

---

## `codemem agent-start`

**Hook:** SubagentStart

Logs agent spawn events via tracing. Does not store a memory — agent spawns are transient operational events.

Reads JSON `{session_id, cwd, agent_type, agent_id}` from stdin. Outputs JSON `{}`.

**Syntax**

```
codemem agent-start
```

---

## `codemem tool-error`

**Hook:** PostToolUseFailure

Captures tool error patterns as Context memories (importance 0.4) with `error`, `tool-failure`, and `tool:<name>` tags. Includes input context (file_path, command, or pattern) for actionable error messages. Skips user interrupts (`is_interrupt: true`).

Reads JSON `{session_id, cwd, tool_name, tool_input, error, is_interrupt}` from stdin. Outputs JSON `{}`.

**Syntax**

```
codemem tool-error
```

---

## `codemem session-close`

**Hook:** SessionEnd

Cleanly closes the session in the database with the termination reason (e.g., `user_request`, `api_request`, `tool_use`).

Reads JSON `{session_id, cwd, reason}` from stdin. Outputs JSON `{}`.

**Syntax**

```
codemem session-close
```

---

## `codemem checkpoint`

**Hook:** PreCompact

Saves a checkpoint memory before context compaction, summarizing up to 5 recent Decision/Insight/Pattern memories from the namespace. Tagged `pre-compact` + `checkpoint` with importance 0.5. Also triggered internally by SessionStart when `source: compact`.

Reads JSON `{session_id, cwd}` from stdin. Outputs JSON `{}`.

**Syntax**

```
codemem checkpoint
```

---

## `codemem doctor`

Run diagnostic health checks on the Codemem installation. Validates database integrity, schema version, memory count, vector index, embedding provider, and configuration.

**Syntax**

```
codemem doctor
```

**Output:** A checklist of OK/FAIL results for each health check.

---

## `codemem config get <key>`

Read a configuration value by dot-separated key path (e.g., `scoring.vector_similarity`).

**Syntax**

```
codemem config get <key>
```

**Example:**

```
codemem config get scoring.vector_similarity
```

---

## `codemem config set <key> <value>`

Set a configuration value by dot-separated key path. The value is parsed as JSON (number, string, boolean, etc.).

**Syntax**

```
codemem config set <key> <value>
```

**Example:**

```
codemem config set scoring.vector_similarity 0.3
```

---

## `codemem migrate`

Show and apply pending schema migrations. Reports the current schema version and whether any migrations were applied.

**Syntax**

```
codemem migrate
```
