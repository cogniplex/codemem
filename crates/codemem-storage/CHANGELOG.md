# Changelog

## [0.17.0](https://github.com/cogniplex/codemem/compare/v0.16.0...v0.17.0) (2026-04-13)


### Features

* UI overhaul — 3-page layout, graph code viewer ([#78](https://github.com/cogniplex/codemem/issues/78)) ([6e22545](https://github.com/cogniplex/codemem/commit/6e225452aff3ef85e6e9cfbfc45c2ca0c909a071))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.18.0 to 0.19.0

## [0.16.0](https://github.com/cogniplex/codemem/compare/v0.15.1...v0.16.0) (2026-04-13)


### Features

* graph accuracy improvements and cross-service API surface detection ([#49](https://github.com/cogniplex/codemem/issues/49)) ([e892375](https://github.com/cogniplex/codemem/commit/e892375a253ff89fc6aaffdf4f142775b21afdf9))
* graph intelligence — SCIP noise filtering, test impact, co-change detection ([#63](https://github.com/cogniplex/codemem/issues/63)) ([5723bc6](https://github.com/cogniplex/codemem/commit/5723bc6df45b63be8bd6c2684513ee0b18c1b26a))
* **graph:** tag-based auto-linking and memory-neighbor scoring ([1403797](https://github.com/cogniplex/codemem/commit/1403797ebca46847e617e8d9a2faa9da31565e5a))
* incremental re-indexing with symbol-level diff ([#26](https://github.com/cogniplex/codemem/issues/26)) ([872b10f](https://github.com/cogniplex/codemem/commit/872b10f05cbe35d44e04f87767dcd18c5f5c8ba7))
* LSP enrichment + cross-repo linking pipeline ([#33](https://github.com/cogniplex/codemem/issues/33)) ([a74bde5](https://github.com/cogniplex/codemem/commit/a74bde595b567f6c79c58ff19c134f580258348a))
* memory expiration with opportunistic cleanup ([#41](https://github.com/cogniplex/codemem/issues/41)) ([cf8e995](https://github.com/cogniplex/codemem/commit/cf8e995df43f908ee77dea90520d3483b5a059ca))
* scope context with repo/branch/user awareness ([#45](https://github.com/cogniplex/codemem/issues/45)) ([d6ec98f](https://github.com/cogniplex/codemem/commit/d6ec98f7f20a4246a0b1ac63b33164710525a244))
* session continuity and persistence pipeline improvements ([#10](https://github.com/cogniplex/codemem/issues/10)) ([970b6f8](https://github.com/cogniplex/codemem/commit/970b6f899dccee81e84143ad6a8ac96f8965307c))
* **storage:** add repository and session activity migrations and queries ([a0dae58](https://github.com/cogniplex/codemem/commit/a0dae5865eac6f341abf6fe8c07f912c015eb44e))
* temporal graph layer — commit/PR nodes, ModifiedBy edges, temporal queries ([#52](https://github.com/cogniplex/codemem/issues/52)) ([3679b22](https://github.com/cogniplex/codemem/commit/3679b2221bf1ac7c8bcbd179e35d9726908e494b))
* UI overhaul — 3-page layout, graph code viewer ([#78](https://github.com/cogniplex/codemem/issues/78)) ([6e22545](https://github.com/cogniplex/codemem/commit/6e225452aff3ef85e6e9cfbfc45c2ca0c909a071))
* v0.4.0 production hardening — zero unwraps, safe concurrency, config persistence, schema migrations ([7a81665](https://github.com/cogniplex/codemem/commit/7a816651580ff7b891a40a9bc41322373fcddb15))
* v0.5.0 smarter memory — temporal edges, semantic consolidation, self-editing tools, LLM summarization ([3ea6f68](https://github.com/cogniplex/codemem/commit/3ea6f68f9e6ead80b4e62d5ea2c04ff65e3e322f))
* v0.6.0 — 8 new language parsers, operational metrics, CLI commands, test coverage ([917d940](https://github.com/cogniplex/codemem/commit/917d940da2770b052e6f65b618df5bbd95ca75db))


### Bug Fixes

* 12 correctness, safety, and pipeline fixes from code review ([#47](https://github.com/cogniplex/codemem/issues/47)) ([71d628f](https://github.com/cogniplex/codemem/commit/71d628f3f76c342c0087e107b1faad2682222068))
* avoid SQLITE_BUSY race between hooks and MCP server on startup ([67266a0](https://github.com/cogniplex/codemem/commit/67266a0e7d493e201cd6c5d2d647a973606dcfc8))
* **ci:** resolve all formatting, clippy, and eslint failures ([80cd400](https://github.com/cogniplex/codemem/commit/80cd400d02904c15de8937604b53abe733b05386))
* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))
* deduplicate content_hash before adding UNIQUE index in migration 006 ([#54](https://github.com/cogniplex/codemem/issues/54)) ([26aecb6](https://github.com/cogniplex/codemem/commit/26aecb6f89a82ebf6b968bec3029ab4a6fae26a7)), closes [#53](https://github.com/cogniplex/codemem/issues/53)
* embedding memory leaks, silent batch drops, HNSW ghost compaction, BM25 persistence ([7a41a36](https://github.com/cogniplex/codemem/commit/7a41a367d7031ac423743f6a36bb317fca0fb054))
* namespace-scoped file hashes, configurable embed batch size, hook storage ([#35](https://github.com/cogniplex/codemem/issues/35)) ([1c5bb9c](https://github.com/cogniplex/codemem/commit/1c5bb9c30ee9cce6868e60fb7a9400c2474a5d1c))
* namespace-scoped PageRank to prevent cross-project score pollution ([#61](https://github.com/cogniplex/codemem/issues/61)) ([5316a89](https://github.com/cogniplex/codemem/commit/5316a892f26dc49ff3d067b55faf9507ab57a320))
* post-review fixes for temporal graph, Gemini provider, and migration safety ([#56](https://github.com/cogniplex/codemem/issues/56)) ([f345dc6](https://github.com/cogniplex/codemem/commit/f345dc65550d2281d204856b1757a901f0a1f338))
* review findings — config validation, cascade delete, BM25 consistency, scoring hardening, and engine refinements ([f0fa69d](https://github.com/cogniplex/codemem/commit/f0fa69d66a9e382c9ed3385cae508033cc21902a))
* storage audit fixes and 4 critical bug fixes ([96b3916](https://github.com/cogniplex/codemem/commit/96b391674be8f3e0af67b77702c999693fb0d7ef))
* **storage:** scope content_hash dedup to namespace ([6c5b081](https://github.com/cogniplex/codemem/commit/6c5b081f7c5c1754a39b4839db47769c9da83fe6))
* use full persist pipeline in store_pattern_memory, remove dead code ([a262533](https://github.com/cogniplex/codemem/commit/a2625331cb876c869cd2b921621c8d38b4c77f81))


### Performance Improvements

* lazy init for vector/BM25/embeddings in CodememEngine ([#28](https://github.com/cogniplex/codemem/issues/28)) ([823cbc1](https://github.com/cogniplex/codemem/commit/823cbc1ed798fc0988c07e63510a51eddc6c0fb6))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.17.0 to 0.18.0

## [0.15.1](https://github.com/cogniplex/codemem/compare/v0.15.0...v0.15.1) (2026-04-07)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.16.0 to 0.17.0

## [0.15.0](https://github.com/cogniplex/codemem/compare/v0.14.1...v0.15.0) (2026-03-27)


### Features

* graph intelligence — SCIP noise filtering, test impact, co-change detection ([#63](https://github.com/cogniplex/codemem/issues/63)) ([5723bc6](https://github.com/cogniplex/codemem/commit/5723bc6df45b63be8bd6c2684513ee0b18c1b26a))


### Bug Fixes

* namespace-scoped PageRank to prevent cross-project score pollution ([#61](https://github.com/cogniplex/codemem/issues/61)) ([5316a89](https://github.com/cogniplex/codemem/commit/5316a892f26dc49ff3d067b55faf9507ab57a320))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.15.0 to 0.16.0

## [0.14.1](https://github.com/cogniplex/codemem/compare/v0.14.0...v0.14.1) (2026-03-18)


### Bug Fixes

* post-review fixes for temporal graph, Gemini provider, and migration safety ([#56](https://github.com/cogniplex/codemem/issues/56)) ([f345dc6](https://github.com/cogniplex/codemem/commit/f345dc65550d2281d204856b1757a901f0a1f338))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.14.0 to 0.15.0

## [0.14.0](https://github.com/cogniplex/codemem/compare/v0.13.0...v0.14.0) (2026-03-18)


### Features

* temporal graph layer — commit/PR nodes, ModifiedBy edges, temporal queries ([#52](https://github.com/cogniplex/codemem/issues/52)) ([3679b22](https://github.com/cogniplex/codemem/commit/3679b2221bf1ac7c8bcbd179e35d9726908e494b))


### Bug Fixes

* deduplicate content_hash before adding UNIQUE index in migration 006 ([#54](https://github.com/cogniplex/codemem/issues/54)) ([26aecb6](https://github.com/cogniplex/codemem/commit/26aecb6f89a82ebf6b968bec3029ab4a6fae26a7)), closes [#53](https://github.com/cogniplex/codemem/issues/53)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.13.0 to 0.14.0

## [0.13.0](https://github.com/cogniplex/codemem/compare/v0.12.0...v0.13.0) (2026-03-16)


### Features

* graph accuracy improvements and cross-service API surface detection ([#49](https://github.com/cogniplex/codemem/issues/49)) ([e892375](https://github.com/cogniplex/codemem/commit/e892375a253ff89fc6aaffdf4f142775b21afdf9))
* memory expiration with opportunistic cleanup ([#41](https://github.com/cogniplex/codemem/issues/41)) ([cf8e995](https://github.com/cogniplex/codemem/commit/cf8e995df43f908ee77dea90520d3483b5a059ca))
* scope context with repo/branch/user awareness ([#45](https://github.com/cogniplex/codemem/issues/45)) ([d6ec98f](https://github.com/cogniplex/codemem/commit/d6ec98f7f20a4246a0b1ac63b33164710525a244))


### Bug Fixes

* 12 correctness, safety, and pipeline fixes from code review ([#47](https://github.com/cogniplex/codemem/issues/47)) ([71d628f](https://github.com/cogniplex/codemem/commit/71d628f3f76c342c0087e107b1faad2682222068))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.12.0 to 0.13.0

## [0.12.0](https://github.com/cogniplex/codemem/compare/v0.11.0...v0.12.0) (2026-03-13)


### Features

* LSP enrichment + cross-repo linking pipeline ([#33](https://github.com/cogniplex/codemem/issues/33)) ([a74bde5](https://github.com/cogniplex/codemem/commit/a74bde595b567f6c79c58ff19c134f580258348a))


### Bug Fixes

* namespace-scoped file hashes, configurable embed batch size, hook storage ([#35](https://github.com/cogniplex/codemem/issues/35)) ([1c5bb9c](https://github.com/cogniplex/codemem/commit/1c5bb9c30ee9cce6868e60fb7a9400c2474a5d1c))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.11.0 to 0.12.0

## [0.11.0](https://github.com/cogniplex/codemem/compare/v0.10.1...v0.11.0) (2026-03-11)


### Features

* incremental re-indexing with symbol-level diff ([#26](https://github.com/cogniplex/codemem/issues/26)) ([872b10f](https://github.com/cogniplex/codemem/commit/872b10f05cbe35d44e04f87767dcd18c5f5c8ba7))


### Performance Improvements

* lazy init for vector/BM25/embeddings in CodememEngine ([#28](https://github.com/cogniplex/codemem/issues/28)) ([823cbc1](https://github.com/cogniplex/codemem/commit/823cbc1ed798fc0988c07e63510a51eddc6c0fb6))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.10.1 to 0.11.0

## [0.10.1](https://github.com/cogniplex/codemem/compare/v0.10.0...v0.10.1) (2026-03-09)


### Refactoring

* tier 1 quick wins — dead code, wiring, visibility, dedup ([#13](https://github.com/cogniplex/codemem/issues/13)) ([56a469a](https://github.com/cogniplex/codemem/commit/56a469a25ad57a00cfeee6714b76e8c582f33ace))
* tier 2 quick wins — constructors, dedup helpers, storage ergonomics ([#14](https://github.com/cogniplex/codemem/issues/14)) ([e7243b7](https://github.com/cogniplex/codemem/commit/e7243b7f79af93662fcf644b31ea091dd863d0b4))
* tier 3 — domain logic to engine, drop binary storage/embeddings deps ([#15](https://github.com/cogniplex/codemem/issues/15)) ([a92b846](https://github.com/cogniplex/codemem/commit/a92b8463e2660b0318c01fa03f18f9ac864ddc39))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.10.0 to 0.10.1

## [0.10.0](https://github.com/cogniplex/codemem/compare/v0.9.0...v0.10.0) (2026-03-08)


### Features

* session continuity and persistence pipeline improvements ([#10](https://github.com/cogniplex/codemem/issues/10)) ([970b6f8](https://github.com/cogniplex/codemem/commit/970b6f899dccee81e84143ad6a8ac96f8965307c))


### Bug Fixes

* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.9.0 to 0.10.0

## [0.9.0](https://github.com/cogniplex/codemem/compare/v0.8.0...v0.9.0) (2026-03-08)


### Features

* **graph:** tag-based auto-linking and memory-neighbor scoring ([1403797](https://github.com/cogniplex/codemem/commit/1403797ebca46847e617e8d9a2faa9da31565e5a))


### Bug Fixes

* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* **storage:** scope content_hash dedup to namespace ([6c5b081](https://github.com/cogniplex/codemem/commit/6c5b081f7c5c1754a39b4839db47769c9da83fe6))
* use full persist pipeline in store_pattern_memory, remove dead code ([a262533](https://github.com/cogniplex/codemem/commit/a2625331cb876c869cd2b921621c8d38b4c77f81))


### Refactoring

* remove dead code, consolidate utilities, wire config to backends ([f22efcc](https://github.com/cogniplex/codemem/commit/f22efcccd63005e08ce5b82e35f14b0a6cc7a984))


### Tests

* add comprehensive test coverage across all crates (~300 tests) ([e758f05](https://github.com/cogniplex/codemem/commit/e758f0585d31ada1b599b5db950e711c02552116))
