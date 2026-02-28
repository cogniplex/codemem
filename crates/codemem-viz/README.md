# codemem-viz

Interactive REST dashboard with PCA projection of memory embeddings.

## Overview

Axum-based REST API with an embedded HTML/JS frontend. Projects 768-dim memory embeddings to 3D via PCA for interactive visualization, with graph edges overlaid.

## API Routes

- `GET /api/stats` — Database statistics
- `GET /api/namespaces` — List namespaces
- `GET /api/memories` — List memories (filterable by namespace, type)
- `GET /api/memory/:id` — Memory detail
- `GET /api/vectors` — Embedding vectors (PCA-projected)
- `GET /api/graph/nodes` — Graph nodes
- `GET /api/graph/edges` — Graph edges
- `GET /api/graph/browse` — Paginated graph browsing
- `GET /api/graph/d3` — D3-compatible graph format (nodes + links)
- `GET /api/search` — Memory search
- `GET /api/timeline` — Memory creation timeline (grouped by day and type)
- `GET /api/distribution` — Memory type distribution stats

## Usage

```bash
codemem viz                  # Start on port 4242, auto-open browser
codemem viz --port 8080      # Custom port
codemem viz --no-open        # Don't auto-open browser
```
