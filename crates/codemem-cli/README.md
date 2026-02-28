# codemem-cli

CLI entry point with 15 commands for the Codemem memory engine.

## Overview

The `codemem` binary. Dispatches to all subsystems: initialization, search, data management, lifecycle hooks, consolidation, visualization, and the MCP server.

## Commands

| Command | Description |
|---------|-------------|
| `init` | Download embedding model, register hooks, configure MCP server |
| `search` | Search memories with 9-component hybrid scoring |
| `stats` | Database and index statistics |
| `serve` | Start MCP server (JSON-RPC over stdio) |
| `ingest` | Process PostToolUse hook payloads |
| `consolidate` | Run memory consolidation cycles |
| `viz` | Interactive PCA visualization dashboard |
| `index` | Index codebase with tree-sitter |
| `export` / `import` | Backup and restore memories (JSONL) |
| `watch` | Real-time file watcher with auto-indexing |
| `sessions` | Session management (list, start, end) |
| `context` | SessionStart hook — inject prior context |
| `prompt` | UserPromptSubmit hook — record user prompts |
| `summarize` | Stop hook — generate session summary |

## Modules

- `commands_init.rs` — Model download, hook registration, MCP config
- `commands_search.rs` — Memory search with scoring
- `commands_data.rs` — Stats, watch
- `commands_lifecycle.rs` — Context, prompt, summarize, sessions
- `commands_consolidation.rs` — Consolidation cycles
- `commands_export.rs` — Export, import, index

See [CLI Reference](../../docs/cli-reference.md) for full usage.
