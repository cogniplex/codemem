# Changelog

## [0.16.0](https://github.com/cogniplex/codemem/compare/v0.15.0...v0.16.0) (2026-03-27)


### Features

* add get_node_memories + node_coverage tools, rewrite code-mapper agent ([e2d10a3](https://github.com/cogniplex/codemem/commit/e2d10a37ab757eabde818bd0e380cdba0c3f2725))
* configurable embedding model, dtype, and batch size ([#31](https://github.com/cogniplex/codemem/issues/31)) ([6dfbfce](https://github.com/cogniplex/codemem/commit/6dfbfce0377b5e46c0a2de8907ac11a53f19e490))
* **core:** add EnrichmentConfig and ChunkingConfig compaction settings ([20969b2](https://github.com/cogniplex/codemem/commit/20969b2e0eda88e90f1f1b9f5e1b725757c3bc6d))
* **core:** extend types, config, and traits for hooks, indexing, and storage ([3d27cf7](https://github.com/cogniplex/codemem/commit/3d27cf7a284ab146256fdbfbc94883d4349a7564))
* graph accuracy improvements and cross-service API surface detection ([#49](https://github.com/cogniplex/codemem/issues/49)) ([e892375](https://github.com/cogniplex/codemem/commit/e892375a253ff89fc6aaffdf4f142775b21afdf9))
* graph intelligence — SCIP noise filtering, test impact, co-change detection ([#63](https://github.com/cogniplex/codemem/issues/63)) ([5723bc6](https://github.com/cogniplex/codemem/commit/5723bc6df45b63be8bd6c2684513ee0b18c1b26a))
* graph quality improvements — blocklist, callbacks, dead code ([#58](https://github.com/cogniplex/codemem/issues/58)) ([42e1486](https://github.com/cogniplex/codemem/commit/42e1486f4d65da0b707738989fcbaa2687645a5d))
* **graph:** add weighted edges, filtered traversal, and memory-bridged graph_strength ([4cfd362](https://github.com/cogniplex/codemem/commit/4cfd362f3ff2df940db86416f00c8654bfd381ef))
* **graph:** tag-based auto-linking and memory-neighbor scoring ([1403797](https://github.com/cogniplex/codemem/commit/1403797ebca46847e617e8d9a2faa9da31565e5a))
* incremental re-indexing with symbol-level diff ([#26](https://github.com/cogniplex/codemem/issues/26)) ([872b10f](https://github.com/cogniplex/codemem/commit/872b10f05cbe35d44e04f87767dcd18c5f5c8ba7))
* LSP enrichment + cross-repo linking pipeline ([#33](https://github.com/cogniplex/codemem/issues/33)) ([a74bde5](https://github.com/cogniplex/codemem/commit/a74bde595b567f6c79c58ff19c134f580258348a))
* memory expiration with opportunistic cleanup ([#41](https://github.com/cogniplex/codemem/issues/41)) ([cf8e995](https://github.com/cogniplex/codemem/commit/cf8e995df43f908ee77dea90520d3483b5a059ca))
* replace LSP enrichment with SCIP integration ([#39](https://github.com/cogniplex/codemem/issues/39)) ([b50dd29](https://github.com/cogniplex/codemem/commit/b50dd29f0c0c12718b1191d6eaf178937d41c33f))
* scope context with repo/branch/user awareness ([#45](https://github.com/cogniplex/codemem/issues/45)) ([d6ec98f](https://github.com/cogniplex/codemem/commit/d6ec98f7f20a4246a0b1ac63b33164710525a244))
* session continuity and persistence pipeline improvements ([#10](https://github.com/cogniplex/codemem/issues/10)) ([970b6f8](https://github.com/cogniplex/codemem/commit/970b6f899dccee81e84143ad6a8ac96f8965307c))
* temporal graph layer — commit/PR nodes, ModifiedBy edges, temporal queries ([#52](https://github.com/cogniplex/codemem/issues/52)) ([3679b22](https://github.com/cogniplex/codemem/commit/3679b2221bf1ac7c8bcbd179e35d9726908e494b))
* v0.4.0 production hardening — zero unwraps, safe concurrency, config persistence, schema migrations ([7a81665](https://github.com/cogniplex/codemem/commit/7a816651580ff7b891a40a9bc41322373fcddb15))
* v0.5.0 smarter memory — temporal edges, semantic consolidation, self-editing tools, LLM summarization ([3ea6f68](https://github.com/cogniplex/codemem/commit/3ea6f68f9e6ead80b4e62d5ea2c04ff65e3e322f))
* v0.6.0 — 8 new language parsers, operational metrics, CLI commands, test coverage ([917d940](https://github.com/cogniplex/codemem/commit/917d940da2770b052e6f65b618df5bbd95ca75db))


### Bug Fixes

* **ci:** resolve all formatting, clippy, and eslint failures ([80cd400](https://github.com/cogniplex/codemem/commit/80cd400d02904c15de8937604b53abe733b05386))
* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* namespace-scoped file hashes, configurable embed batch size, hook storage ([#35](https://github.com/cogniplex/codemem/issues/35)) ([1c5bb9c](https://github.com/cogniplex/codemem/commit/1c5bb9c30ee9cce6868e60fb7a9400c2474a5d1c))
* namespace-scoped PageRank to prevent cross-project score pollution ([#61](https://github.com/cogniplex/codemem/issues/61)) ([5316a89](https://github.com/cogniplex/codemem/commit/5316a892f26dc49ff3d067b55faf9507ab57a320))
* review findings — config validation, cascade delete, BM25 consistency, scoring hardening, and engine refinements ([f0fa69d](https://github.com/cogniplex/codemem/commit/f0fa69d66a9e382c9ed3385cae508033cc21902a))
* storage audit fixes and 4 critical bug fixes ([96b3916](https://github.com/cogniplex/codemem/commit/96b391674be8f3e0af67b77702c999693fb0d7ef))
* use full persist pipeline in store_pattern_memory, remove dead code ([a262533](https://github.com/cogniplex/codemem/commit/a2625331cb876c869cd2b921621c8d38b4c77f81))

## [0.15.0](https://github.com/cogniplex/codemem/compare/v0.14.0...v0.15.0) (2026-03-18)


### Features

* graph quality improvements — blocklist, callbacks, dead code ([#58](https://github.com/cogniplex/codemem/issues/58)) ([42e1486](https://github.com/cogniplex/codemem/commit/42e1486f4d65da0b707738989fcbaa2687645a5d))

## [0.14.0](https://github.com/cogniplex/codemem/compare/v0.13.0...v0.14.0) (2026-03-18)


### Features

* temporal graph layer — commit/PR nodes, ModifiedBy edges, temporal queries ([#52](https://github.com/cogniplex/codemem/issues/52)) ([3679b22](https://github.com/cogniplex/codemem/commit/3679b2221bf1ac7c8bcbd179e35d9726908e494b))

## [0.13.0](https://github.com/cogniplex/codemem/compare/v0.12.0...v0.13.0) (2026-03-16)


### Features

* graph accuracy improvements and cross-service API surface detection ([#49](https://github.com/cogniplex/codemem/issues/49)) ([e892375](https://github.com/cogniplex/codemem/commit/e892375a253ff89fc6aaffdf4f142775b21afdf9))
* memory expiration with opportunistic cleanup ([#41](https://github.com/cogniplex/codemem/issues/41)) ([cf8e995](https://github.com/cogniplex/codemem/commit/cf8e995df43f908ee77dea90520d3483b5a059ca))
* scope context with repo/branch/user awareness ([#45](https://github.com/cogniplex/codemem/issues/45)) ([d6ec98f](https://github.com/cogniplex/codemem/commit/d6ec98f7f20a4246a0b1ac63b33164710525a244))

## [0.12.0](https://github.com/cogniplex/codemem/compare/v0.11.0...v0.12.0) (2026-03-13)


### Features

* LSP enrichment + cross-repo linking pipeline ([#33](https://github.com/cogniplex/codemem/issues/33)) ([a74bde5](https://github.com/cogniplex/codemem/commit/a74bde595b567f6c79c58ff19c134f580258348a))
* replace LSP enrichment with SCIP integration ([#39](https://github.com/cogniplex/codemem/issues/39)) ([b50dd29](https://github.com/cogniplex/codemem/commit/b50dd29f0c0c12718b1191d6eaf178937d41c33f))


### Bug Fixes

* namespace-scoped file hashes, configurable embed batch size, hook storage ([#35](https://github.com/cogniplex/codemem/issues/35)) ([1c5bb9c](https://github.com/cogniplex/codemem/commit/1c5bb9c30ee9cce6868e60fb7a9400c2474a5d1c))

## [0.11.0](https://github.com/cogniplex/codemem/compare/v0.10.1...v0.11.0) (2026-03-11)


### Features

* configurable embedding model, dtype, and batch size ([#31](https://github.com/cogniplex/codemem/issues/31)) ([6dfbfce](https://github.com/cogniplex/codemem/commit/6dfbfce0377b5e46c0a2de8907ac11a53f19e490))
* incremental re-indexing with symbol-level diff ([#26](https://github.com/cogniplex/codemem/issues/26)) ([872b10f](https://github.com/cogniplex/codemem/commit/872b10f05cbe35d44e04f87767dcd18c5f5c8ba7))

## [0.10.1](https://github.com/cogniplex/codemem/compare/v0.10.0...v0.10.1) (2026-03-09)


### Refactoring

* tier 1 quick wins — dead code, wiring, visibility, dedup ([#13](https://github.com/cogniplex/codemem/issues/13)) ([56a469a](https://github.com/cogniplex/codemem/commit/56a469a25ad57a00cfeee6714b76e8c582f33ace))
* tier 2 quick wins — constructors, dedup helpers, storage ergonomics ([#14](https://github.com/cogniplex/codemem/issues/14)) ([e7243b7](https://github.com/cogniplex/codemem/commit/e7243b7f79af93662fcf644b31ea091dd863d0b4))
* tier 3 — domain logic to engine, drop binary storage/embeddings deps ([#15](https://github.com/cogniplex/codemem/issues/15)) ([a92b846](https://github.com/cogniplex/codemem/commit/a92b8463e2660b0318c01fa03f18f9ac864ddc39))

## [0.10.0](https://github.com/cogniplex/codemem/compare/v0.9.0...v0.10.0) (2026-03-08)


### Features

* session continuity and persistence pipeline improvements ([#10](https://github.com/cogniplex/codemem/issues/10)) ([970b6f8](https://github.com/cogniplex/codemem/commit/970b6f899dccee81e84143ad6a8ac96f8965307c))


### Documentation

* update stale docs — remove volatile numbers, fix counts, rewrite CONTRIBUTING ([#11](https://github.com/cogniplex/codemem/issues/11)) ([c481a12](https://github.com/cogniplex/codemem/commit/c481a12d833d02fe12ac86e6de2d09bba7e99158))

## [0.9.0](https://github.com/cogniplex/codemem/compare/v0.8.0...v0.9.0) (2026-03-08)


### Features

* **graph:** tag-based auto-linking and memory-neighbor scoring ([1403797](https://github.com/cogniplex/codemem/commit/1403797ebca46847e617e8d9a2faa9da31565e5a))


### Bug Fixes

* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* use full persist pipeline in store_pattern_memory, remove dead code ([a262533](https://github.com/cogniplex/codemem/commit/a2625331cb876c869cd2b921621c8d38b4c77f81))


### Refactoring

* remove dead code, consolidate utilities, wire config to backends ([f22efcc](https://github.com/cogniplex/codemem/commit/f22efcccd63005e08ce5b82e35f14b0a6cc7a984))
* remove double LRU cache from EmbeddingService, make batch_size configurable ([6a9aec3](https://github.com/cogniplex/codemem/commit/6a9aec3c85698b0415528e268ed4925b15f36340))


### Tests

* add comprehensive test coverage across all crates (~300 tests) ([e758f05](https://github.com/cogniplex/codemem/commit/e758f0585d31ada1b599b5db950e711c02552116))
