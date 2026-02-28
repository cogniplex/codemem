# codemem-embeddings

Pluggable embedding providers with Candle (default), Ollama, and OpenAI-compatible backends.

## Overview

Defines the `EmbeddingProvider` trait and a `from_env()` factory that selects the provider at runtime via environment variables.

## Providers

| Provider | Model | Use Case |
|----------|-------|----------|
| `candle` (default) | BAAI/bge-base-en-v1.5 (768-dim) | Fully offline, pure Rust ML, Metal/CUDA GPU |
| `ollama` | nomic-embed-text | Local server, swap models freely |
| `openai` | text-embedding-3-small | OpenAI, Voyage AI, Together, Azure, any compatible API |

## Configuration

| Variable | Default |
|----------|---------|
| `CODEMEM_EMBED_PROVIDER` | `candle` |
| `CODEMEM_EMBED_MODEL` | provider default |
| `CODEMEM_EMBED_URL` | provider default |
| `CODEMEM_EMBED_API_KEY` | reads `OPENAI_API_KEY` |
| `CODEMEM_EMBED_DIMENSIONS` | `768` |

## Key Features

- `CachedProvider` wrapper adds LRU cache (10K entries) to any provider
- `embed_batch()` for efficient bulk embedding
- Safe concurrency — all lock acquisitions return `CodememError::LockPoisoned` instead of panicking
