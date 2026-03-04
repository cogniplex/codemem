# Human Input & Ambiguous Findings

Not everything can be determined from code alone. **Ask the user before storing a finding when intent is ambiguous.** A wrong memory is worse than a missing one — it will mislead future sessions.

## When to ask

Pause and ask for human input when you encounter any of these:

| Signal | Why it's ambiguous | Example |
|--------|-------------------|---------|
| **Unreferenced public API** | Might be dead code, or might be a library/SDK surface consumed externally | A `pub fn process_webhook()` with zero internal callers — could be the main entry point for consumers |
| **Multiple transport/protocol layers** | Might look redundant, but could serve different consumers | JSON-RPC stdio + REST HTTP in the same binary — one for IDE plugins, one for a web UI |
| **Unusual dependency direction** | Might be a layering violation, or an intentional inversion | A "core" crate depending on a "storage" crate — could be a conscious trade-off |
| **Empty or stub implementations** | Might be dead code, or a planned extension point | A trait impl that returns `unimplemented!()` — could be WIP or intentional no-op |
| **Config/feature flags with no obvious use** | Might be obsolete, or used in deployment environments you can't see | `ENABLE_LEGACY_AUTH=true` with no code path referencing it in this repo |
| **Unconventional patterns** | Might be a mistake, or a deliberate choice for reasons not in code | `unsafe` block wrapping seemingly safe code — might be for FFI or performance |
| **Contradictory signals** | Code says one thing, comments/docs say another | A comment says "deprecated" but the function has recent commits and callers |

## How to ask

Present your observation, what you think it means, and the alternatives:

```
I found that `codemem-mcp` exposes both JSON-RPC (stdio) and REST HTTP endpoints.
This could be:
  (a) Intentional — different consumers need different protocols (IDE vs web UI)
  (b) Legacy — one transport is being phased out
  (c) Something else

Which is it? This will help me store an accurate architectural decision memory.
```

## How to store after clarification

After the user responds, store the finding with the user's context included:

```
store_memory {
  "content": "codemem-mcp deliberately supports dual transports: JSON-RPC stdio for IDE integration (Claude Code, Cursor) and REST HTTP for the web control plane UI. Both are active and maintained — not redundant.",
  "memory_type": "decision",
  "importance": 0.85,
  "tags": ["architecture", "transport", "human-verified"],
  "namespace": "/path/to/project"
}
```

Always add the `human-verified` tag when a finding was clarified by user input. This signals higher confidence to future recall.

## What to do if you can't ask (non-interactive)

If running in a non-interactive context (e.g., automated pipeline), store the finding with low importance and a `needs-review` tag instead of guessing:

```
store_memory {
  "content": "NEEDS REVIEW: 12 public functions in codemem-mcp/src/http.rs have zero internal callers. They may be external API endpoints or dead code — could not determine intent without human input.",
  "memory_type": "context",
  "importance": 0.3,
  "tags": ["needs-review", "dead-code-candidate", "external-api-candidate"],
  "namespace": "/path/to/project"
}
```

## General principle

**Confidence threshold**: If you're less than ~70% sure about the _intent_ behind a pattern (not just what the code does, but _why_ it's that way), ask. Code structure tells you WHAT; only humans can confirm WHY.
