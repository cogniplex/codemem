# ADR-004: Claude Code Hooks Integration

**Date:** 2026-03-02
**Status:** Accepted

## Context

Codemem needs to observe what the AI assistant is doing — which files it reads, which files it edits, when sessions start and stop — to build an accurate knowledge graph without the agent explicitly calling `store_memory` for every action.

Claude Code provides a lifecycle hooks system: shell commands that execute in response to events (PreToolUse, PostToolUse, SessionStart, Stop, etc.). Each hook receives a JSON payload on stdin and can write JSON to stdout.

The alternative was to require agents to explicitly call MCP tools for every observation, but this adds latency to every tool call and requires agent prompt engineering to ensure compliance.

## Decision

Integrate via Claude Code's hook system. `codemem init` registers hooks in `.claude/settings.json` that invoke the codemem binary with subcommands (`codemem hook post-tool-use`, etc.).

Key design choices:

1. **Hooks use `open_without_migrations()` for speed.** Hook handlers open SQLite without running migrations to avoid blocking the assistant. Migrations run on `codemem mcp serve` or `codemem analyze` instead.

2. **PostToolUse hooks are selective.** Only Edit/Write/MultiEdit tool calls trigger file re-indexing. Read/Glob/Grep calls are observed for focus detection (directory affinity, repeated searches) but don't trigger re-indexing.

3. **Hooks read a single-line JSON payload from stdin.** Shared helpers `read_hook_payload()` and `extract_hook_context()` handle common extraction (cwd, session_id, tool name).

4. **`tool_response` is `serde_json::Value`, not a string.** The `HookPayload::tool_response_text()` method extracts meaningful content based on tool type: `file.content` for Read, `stdout` for Bash, etc.

## Consequences

- The knowledge graph builds up automatically as the developer works — no explicit memory calls needed for code structure.
- Hook latency is critical. Slow hooks block the assistant. Lazy init (ADR-007) was partly motivated by this: from_db_path() was taking ~2.7s loading embeddings when hooks only need SQLite + graph.
- Tight coupling to Claude Code's hook spec. Changes to the spec require updates to our hook handlers.
- Auto-insights (directory focus detection, edit-after-read tracking) provide useful context without agent involvement.
