# codemem-hooks

PostToolUse hook handler for passive capture of AI assistant observations.

## Overview

Parses incoming hook payloads from AI coding assistants (Claude Code, Cursor, Windsurf), extracts relevant information per tool type, and produces structured memories with graph nodes and edges.

## Supported Tools

| Tool | Memory Type | Extraction |
|------|-------------|------------|
| Read | Context | File content, path, extension tags |
| Glob | Pattern | Glob pattern, discovery tags |
| Grep | Pattern | Search regex, match results |
| Edit / MultiEdit | Decision | Semantic diff summary (functions added/removed, import changes) |
| Write | Decision | New file content, path, extension tags |

## Key Features

- SHA-256 content hashing for deduplication
- Diff-aware memory via `similar` crate (semantic summaries of code changes)
- Edge materialization: `EVOLVED_INTO` edges when a file is edited after being read
- Per-tool tag extraction (file extension, directory, glob pattern, search regex)
