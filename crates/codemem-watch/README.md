# codemem-watch

Real-time file watcher with debouncing and .gitignore support.

## Overview

Watches a directory for file changes using `notify` with 50ms debouncing. Filters events through `.gitignore` patterns (via the `ignore` crate) and a hardcoded list of common ignore directories. Emits typed events over crossbeam channels.

## Events

- `FileChanged` — Existing file modified
- `FileCreated` — New file created
- `FileDeleted` — File removed

## Features

- 50ms debounce via `notify-debouncer-mini`
- `.gitignore` parsing via `GitignoreBuilder` with fallback to hardcoded patterns
- 17 watchable file extensions (`.rs`, `.ts`, `.py`, `.go`, `.c`, `.java`, etc.)
- Deduplication within each debounce window

## Usage

```bash
codemem watch                  # Watch current directory
codemem watch --path ~/project # Watch specific directory
```
