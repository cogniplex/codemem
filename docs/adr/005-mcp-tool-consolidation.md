# ADR-005: MCP Tool Consolidation (40 to 26 Tools)

**Date:** 2026-03-04
**Status:** Accepted

## Context

The MCP server grew organically to 40 tools. Many were narrow variants of the same operation:
- `recall`, `recall_expanded`, `recall_with_impact` — three tools for memory search with different options
- `consolidate`, `consolidate_namespace` — same operation with different scoping
- `search_code`, `index_codebase` — related but with overlapping parameters
- `enrich_codebase`, `enrich_git_history`, `analyze_codebase` — three enrichment entry points

For AI agents, 40 tools means a large tool-use prompt and decision fatigue. Agents performed worse at tool selection with more options.

## Decision

Consolidate to fewer tools with richer parameters:

- **Merge recall variants** into a single `recall` tool with `expand` and `include_impact` boolean parameters.
- **Merge enrichment tools** into the `analyze` CLI command pipeline. Enrichment is not a separate MCP call — it runs as part of indexing.
- **Remove `index_codebase` from MCP** — indexing is a CLI operation (`codemem analyze`), not something agents should trigger mid-conversation.
- **Remove `enrich_codebase`, `enrich_git_history`** — folded into the analyze pipeline.

The tool count went from 40 → 28 → 26 (after removing LSP-related tools in the SCIP migration).

## Consequences

- Agents select tools more accurately with fewer options.
- Each tool does more, controlled by parameters — but parameter documentation must be clear since agents read tool schemas.
- Breaking change for any external consumers of the MCP protocol. Mitigated by the fact that codemem is not yet widely deployed.
- The `analyze` command is now the single entry point for indexing + enrichment + PageRank + clustering, simplifying the mental model.
