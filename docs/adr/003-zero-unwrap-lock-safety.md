# ADR-003: Zero Unwrap and Lock Safety Policy

**Date:** 2026-02-28
**Status:** Accepted

## Context

During v0.3.0 development, there were 699 `.unwrap()` calls in production code. Several were on mutex/rwlock acquisitions, meaning a poisoned lock would panic and crash the MCP server mid-session — losing the user's context with no recovery path.

The MCP server runs as a long-lived process handling concurrent hook invocations (PostToolUse, SessionStart) alongside normal tool calls. A panic in one code path could poison locks that other code paths depend on.

## Decision

1. **Zero `.unwrap()` on mutex/rwlock acquisitions.** All lock acquisitions use typed helper methods that return `CodememError::LockPoisoned` instead of panicking. Example: `lock_vector()`, `lock_bm25()`, `lock_embeddings()`.

2. **Zero `.unwrap()` in production code overall.** All 699 instances replaced with proper error propagation (`?`), `.unwrap_or()`, or `.ok_or()` with domain-specific error variants (`CodememError::Internal`, `CodememError::Config`).

3. **`UnsafeCell<ScoringWeights>` replaced with `RwLock`.** The original implementation used `UnsafeCell` with manual `unsafe impl Send + Sync` to avoid lock overhead on scoring weights. This was replaced with a proper `RwLock` — the performance difference is negligible for a config read, and the safety guarantee is worth it.

## Consequences

- The MCP server never panics on lock contention or poisoning. Errors propagate cleanly to the JSON-RPC response.
- New code must follow this convention — PR review catches any `.unwrap()` on locks.
- Slightly more verbose error handling code, but `?` operator keeps it manageable.
- CI enforces `-D warnings` via RUSTFLAGS, catching most obvious issues at compile time.
